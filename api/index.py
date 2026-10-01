import os
import sys
import re
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

def resolve_file(filename):
    p1 = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PARENT_DIR, filename)
    if os.path.exists(p2):
        return p2
    return p1

MODEL_NPZ = resolve_file("model_weights.npz")
WORD_INDEX_JSON = resolve_file("word_index.json")

templates_dir = os.path.join(CURRENT_DIR, "templates")
if not os.path.exists(templates_dir):
    templates_dir = os.path.join(PARENT_DIR, "templates")

app = Flask(__name__, template_folder=templates_dir)

# Load word index
word_index = {}
if os.path.exists(WORD_INDEX_JSON):
    with open(WORD_INDEX_JSON, "r") as f:
        word_index = json.load(f)
oov_id = word_index.get('<OOV>', 1)
MAX_LEN = 30

# Load pure NumPy model weights
weights = np.load(MODEL_NPZ)
w_emb = weights['w_emb']
w_f_i = weights['w_f_i']
w_f_h = weights['w_f_h']
b_f = weights['b_f']
w_b_i = weights['w_b_i']
w_b_h = weights['w_b_h']
b_b = weights['b_b']
w_d1 = weights['w_d1']
b_d1 = weights['b_d1']
w_d2 = weights['w_d2']
b_d2 = weights['b_d2']

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _run_lstm(x_seq, w_i, w_h, b):
    units = w_h.shape[0]
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    for x_t in x_seq:
        gates = np.dot(x_t, w_i) + np.dot(h, w_h) + b
        i_gate = _sigmoid(gates[:units])
        f_gate = _sigmoid(gates[units:2*units])
        c_cand = np.tanh(gates[2*units:3*units])
        o_gate = _sigmoid(gates[3*units:4*units])
        c = f_gate * c + i_gate * c_cand
        h = o_gate * np.tanh(c)
    return h

def predict_probability(cleaned_text):
    seq = []
    for w in cleaned_text.split():
        idx = word_index.get(w, oov_id)
        if idx >= 5000:
            idx = oov_id
        seq.append(idx)
    pad = np.zeros(MAX_LEN, dtype=np.int32)
    pad[:min(len(seq), MAX_LEN)] = seq[:min(len(seq), MAX_LEN)]
    
    emb = w_emb[pad]
    h_f = _run_lstm(emb, w_f_i, w_f_h, b_f)
    h_b = _run_lstm(emb[::-1], w_b_i, w_b_h, b_b)
    h_bidi = np.concatenate([h_f, h_b])
    d1 = np.maximum(0, np.dot(h_bidi, w_d1) + b_d1)
    prob = float(_sigmoid(np.dot(d1, w_d2) + b_d2)[0])
    return prob

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'from', 'in', 'on', 'for', 'of', 'at',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

def extract_resource(text):
    text = text.lower()
    if "food" in text:
        return "food"
    elif "water" in text:
        return "water"
    elif "shelter" in text or "accommodation" in text:
        return "shelter"
    elif "medical" in text or "medicine" in text or "doctor" in text:
        return "medical aid"
    elif "clothes" in text or "clothing" in text:
        return "clothing"
    elif "rescue" in text:
        return "rescue"
    else:
        return "unspecified resource"

def process_classification(tweet):
    cleaned = clean_text(tweet)
    prob = predict_probability(cleaned)
    is_request = prob >= 0.5
    confidence_pct = round(prob * 100, 1)

    if is_request:
        resource = extract_resource(tweet)
        result = f"{resource.capitalize()} is required in that area ({prob:.2f})"
    else:
        resource = "None"
        result = f"NOT A REQUEST ({prob:.2f})"
        
    tokens = cleaned.split()
    emergency_signals = [t for t in tokens if t in ['flood', 'earthquake', 'fire', 'trapped', 'water', 'food', 'shelter', 'doctor', 'medical', 'hospital', 'rescue', 'evacuation', 'urgent', 'emergency', 'help', 'collapsed', 'damage', 'injured', 'cyclone', 'drought', 'storm']]
    return {
        'prediction': result,
        'tweet': tweet,
        'cleaned': cleaned,
        'tokens': tokens,
        'emergency_signals': emergency_signals,
        'is_request': is_request,
        'resource': resource,
        'confidence': confidence_pct,
        'prob': round(prob, 3),
        'status': 'High confidence' if prob > 0.8 else ('Moderate' if prob >= 0.5 else 'Non-Disaster')
    }

try:
    import data_service
except ImportError:
    sys.path.insert(0, PARENT_DIR)
    import data_service

def _get_context_data():
    return {
        'initial_stats': data_service.get_stats(),
        'initial_tweets': data_service.query_tweets(page=1, page_size=25),
        'initial_model_info': data_service.get_model_performance(),
        'initial_dataset': data_service.query_dataset(page=1, page_size=25)
    }

# Explicit JSON API Routes
@app.route('/api/predict', methods=['POST'])
@app.route('/api/index/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    tweet = data.get('tweet', '')
    if not tweet.strip():
        return jsonify({'error': 'No tweet text provided'}), 400
    res = process_classification(tweet)
    return jsonify({
        'tweet': res['tweet'],
        'cleaned': res['cleaned'],
        'tokens': res['tokens'],
        'emergency_signals': res['emergency_signals'],
        'is_request': res['is_request'],
        'probability': res['prob'],
        'confidence_percent': res['confidence'],
        'resource': res['resource'],
        'status': res['status']
    })

@app.route('/api/stats', methods=['GET'])
@app.route('/api/index/api/stats', methods=['GET'])
def api_stats():
    return jsonify(data_service.get_stats())

@app.route('/api/tweets', methods=['GET'])
@app.route('/api/index/api/tweets', methods=['GET'])
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
@app.route('/api/index/api/dataset', methods=['GET'])
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
@app.route('/api/index/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify(data_service.get_model_performance())

@app.route('/api/health', methods=['GET'])
@app.route('/api/index/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'healthy', 'service': 'disaster-tweet-classifier', 'runtime': 'vercel-serverless'})

@app.route('/export/<file_type>', methods=['GET'])
@app.route('/api/index/export/<file_type>', methods=['GET'])
def export_file(file_type):
    if file_type in ('results.csv', 'classification_results.csv'):
        filepath = resolve_file('classification_results.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='classification_results.csv', mimetype='text/csv')
    elif file_type in ('dataset.csv', 'dataset_cleaned.csv'):
        filepath = resolve_file('dataset_cleaned.csv')
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name='dataset_cleaned.csv', mimetype='text/csv')
    elif file_type == 'stats.json':
        stats = data_service.get_stats()
        return jsonify(stats)
    return jsonify({'error': 'File not found'}), 404

def _dispatch_subroute():
    """Dispatches requests when Vercel rewrites all paths to /api/index"""
    candidates = [
        request.args.get('__path__'),
        request.headers.get('x-matched-path'),
        request.headers.get('x-forwarded-uri'),
        request.environ.get('HTTP_X_MATCHED_PATH'),
        request.environ.get('PATH_INFO'),
        request.path
    ]
    target_path = ''
    for c in candidates:
        if c:
            c_clean = c.split('?')[0].rstrip('/')
            if any(endpoint in c_clean for endpoint in ['/api/', '/export/', '/predict']):
                target_path = c_clean
                break

    if not target_path:
        return None

    if target_path.endswith('/api/predict') or (request.is_json and '/api/' in target_path):
        return api_predict()
    elif target_path.endswith('/api/stats') or target_path.endswith('/stats'):
        return api_stats()
    elif target_path.endswith('/api/tweets') or target_path.endswith('/tweets'):
        return api_tweets()
    elif target_path.endswith('/api/dataset') or target_path.endswith('/dataset'):
        return api_dataset()
    elif target_path.endswith('/api/model-info') or target_path.endswith('/model-info'):
        return api_model_info()
    elif target_path.endswith('/api/health') or target_path.endswith('/health'):
        return api_health()
    elif '/export/' in target_path:
        file_type = target_path.split('/export/')[-1]
        return export_file(file_type)

    return None

# Unified handler for all web interface routes
@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
@app.route('/api', methods=['GET', 'POST'])
@app.route('/api/index', methods=['GET', 'POST'])
@app.route('/api/index.py', methods=['GET', 'POST'])
def index():
    # 1. First check if this was a rewritten sub-route call
    dispatched = _dispatch_subroute()
    if dispatched is not None:
        return dispatched

    # 2. Handle Web Form POST request
    if request.method == 'POST':
        ctx = _get_context_data()
        tweet = request.form.get('tweet', '')
        if not tweet.strip():
            return render_template('index.html', 
                                   prediction="Please enter or select a tweet to classify.",
                                   tweet='',
                                   cleaned=None,
                                   is_request=None,
                                   resource=None,
                                   confidence=None,
                                   prob=None,
                                   **ctx)
        res = process_classification(tweet)
        return render_template('index.html', 
                               prediction=res['prediction'], 
                               tweet=res['tweet'],
                               cleaned=res['cleaned'],
                               is_request=res['is_request'],
                               resource=res['resource'],
                               confidence=res['confidence'],
                               prob=res['prob'],
                               **ctx)

    # 3. Default GET request -> Render main dashboard UI
    ctx = _get_context_data()
    return render_template('index.html', 
                           prediction=None, 
                           tweet='', 
                           cleaned=None, 
                           is_request=None, 
                           resource=None, 
                           confidence=None,
                           prob=None, 
                           **ctx)

class VercelMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_FORWARDED_URI')
        if matched:
            path = matched.split('?')[0]
            if path and not path.endswith('/api/index') and not path.endswith('/api/index.py'):
                environ['PATH_INFO'] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelMiddleware(app.wsgi_app)
