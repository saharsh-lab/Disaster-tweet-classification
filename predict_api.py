import os
import sys
from flask import Flask, request, jsonify, render_template, send_file

import data_service
from flarewatch.inference import Classifier, EMERGENCY_KEYWORDS, THRESHOLD, clean_text, extract_resource_keywords
from flarewatch.live import LiveFeed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NPZ = os.path.join(BASE_DIR, "model_weights.npz")
MODEL_H5 = os.path.join(BASE_DIR, "disaster_rnn_model.h5")
TOKENIZER_PKL = os.path.join(BASE_DIR, "tokenizer.pkl")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
MAX_LEN = 30

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# Dual inference: pure NumPy weights when present (fast, same code as Vercel), Keras otherwise
if os.path.exists(MODEL_NPZ):
    classifier = Classifier([BASE_DIR])
    live_feed = LiveFeed(classifier)

    def classify(tweet):
        return classifier.classify(tweet)
else:
    import pickle
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    with open(TOKENIZER_PKL, "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(MODEL_H5)
    live_feed = None

    def classify(tweet):
        cleaned = clean_text(tweet)
        seq = tokenizer.texts_to_sequences([cleaned])
        prob = float(model.predict(pad_sequences(seq, maxlen=MAX_LEN, padding='post'), verbose=0)[0][0])
        is_request = prob >= THRESHOLD
        resource = extract_resource_keywords(tweet) if is_request else "None"
        tokens = cleaned.split()
        return {
            'prediction': f"{resource.capitalize()} is required in that area ({prob:.2f})" if is_request else f"NOT A REQUEST ({prob:.2f})",
            'tweet': tweet, 'cleaned': cleaned, 'tokens': tokens,
            'emergency_signals': [t for t in tokens if t in EMERGENCY_KEYWORDS],
            'is_request': is_request, 'probability': round(prob, 4), 'confidence_percent': round(prob * 100, 1),
            'resource': resource, 'resources': [], 'locations': [], 'explanation': [],
            'status': 'High confidence' if prob > 0.8 else ('Moderate' if prob >= THRESHOLD else 'Non-Disaster'),
        }


def _get_context_data():
    return {
        'initial_stats': data_service.get_stats(),
        'initial_tweets': data_service.query_tweets(page=1, page_size=25),
        'initial_model_info': data_service.get_model_performance(),
        'initial_dataset': data_service.query_dataset(page=1, page_size=25)
    }

@app.route('/')
def home():
    ctx = _get_context_data()
    return render_template('index.html',
                           prediction=None,
                           tweet='',
                           is_request=None,
                           resource=None,
                           confidence=None,
                           prob=None,
                           cleaned=None,
                           **ctx)

@app.route('/predict', methods=['POST'])
def predict():
    ctx = _get_context_data()
    tweet = request.form.get('tweet', '')
    if not tweet.strip():
        return render_template('index.html',
                               prediction="Please enter or select a tweet to classify.",
                               tweet='',
                               is_request=None,
                               resource=None,
                               confidence=None,
                               prob=None,
                               cleaned=None,
                               **ctx)

    res = classify(tweet)
    return render_template('index.html',
                           prediction=res['prediction'],
                           tweet=tweet,
                           cleaned=res['cleaned'],
                           is_request=res['is_request'],
                           resource=res['resource'],
                           confidence=res['confidence_percent'],
                           prob=round(res['probability'], 3),
                           **ctx)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    tweet = data.get('tweet', '')
    if not tweet.strip():
        return jsonify({'error': 'No tweet text provided'}), 400
    res = classify(tweet)
    return jsonify({
        'tweet': tweet,
        'cleaned': res['cleaned'],
        'tokens': res['tokens'],
        'emergency_signals': res['emergency_signals'],
        'is_request': res['is_request'],
        'probability': res['probability'],
        'confidence_percent': res['confidence_percent'],
        'resource': res['resource'],
        'resources': res['resources'],
        'locations': res['locations'],
        'explanation': res.get('explanation', []),
        'status': res['status']
    })

@app.route('/api/live', methods=['GET'])
def api_live():
    if live_feed is None:
        return jsonify({'status': 'unavailable', 'posts': [], 'errors': ['live feed needs model_weights.npz']}), 503
    return jsonify(live_feed.get())

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'healthy', 'service': 'disaster-tweet-classifier', 'runtime': 'flask-local'})

@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(data_service.get_stats())

@app.route('/api/tweets', methods=['GET'])
def api_tweets():
    page = int(request.args.get('page', 1))
    page_size = min(int(request.args.get('page_size', 25)), 100)
    search = request.args.get('search', '')
    filter_type = request.args.get('filter_type', 'all')
    resource = request.args.get('resource', 'all')
    sort_by = request.args.get('sort_by', 'default')
    result = data_service.query_tweets(
        page=page,
        page_size=page_size,
        search=search,
        filter_type=filter_type,
        resource=resource,
        sort_by=sort_by
    )
    return jsonify(result)

@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    page = int(request.args.get('page', 1))
    page_size = min(int(request.args.get('page_size', 25)), 100)
    search = request.args.get('search', '')
    label = request.args.get('label', 'all')
    result = data_service.query_dataset(
        page=page,
        page_size=page_size,
        search=search,
        label=label
    )
    return jsonify(result)

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify(data_service.get_model_performance())

@app.route('/export/<file_type>', methods=['GET'])
def export_file(file_type):
    if file_type == 'results.csv' or file_type == 'classification_results.csv':
        filepath = os.path.join(BASE_DIR, 'classification_results.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='classification_results.csv', mimetype='text/csv')
    elif file_type == 'dataset.csv' or file_type == 'dataset_cleaned.csv':
        filepath = os.path.join(BASE_DIR, 'dataset_cleaned.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='dataset_cleaned.csv', mimetype='text/csv')
    elif file_type == 'stats.json':
        stats = data_service.get_stats()
        return jsonify(stats)
    return jsonify({'error': 'File not found'}), 404

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1].replace('--port=', ''))
        except Exception:
            pass
    else:
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', 5000)) == 0:
                    port = 5001
        except Exception:
            pass
    print(f"🔥 Flarewatch running on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
