from flask import Flask, request, jsonify, render_template
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# Load model and tokenizer
model = load_model("disaster_rnn_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
MAX_LEN = 30

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html', prediction=None, tweet='', is_request=None, resource=None, confidence=None)

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form.get('tweet', '')
    if not tweet.strip():
        return render_template('index.html', 
                               prediction="Please enter or select a tweet to classify.",
                               tweet='',
                               is_request=None,
                               resource=None,
                               confidence=None)
    
    cleaned = clean_text(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = float(model.predict(padded)[0][0])
    is_request = prob >= 0.5
    confidence_pct = round(prob * 100, 1)

    if is_request:
        resource = extract_resource(tweet)
        result = f"{resource.capitalize()} is required in that area ({prob:.2f})"
    else:
        resource = "None"
        result = f"NOT A REQUEST ({prob:.2f})"
        
    return render_template('index.html', 
                           prediction=result, 
                           tweet=tweet,
                           is_request=is_request,
                           resource=resource,
                           confidence=confidence_pct,
                           prob=round(prob, 3))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    tweet = data.get('tweet', '')
    if not tweet.strip():
        return jsonify({'error': 'No tweet text provided'}), 400
    cleaned = clean_text(tweet)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = float(model.predict(padded)[0][0])
    is_request = prob >= 0.5
    resource = extract_resource(tweet) if is_request else None
    return jsonify({
        'tweet': tweet,
        'cleaned': cleaned,
        'is_request': is_request,
        'probability': round(prob, 4),
        'confidence_percent': round(prob * 100, 1),
        'resource': resource
    })

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1].replace('--port=', ''))
        except Exception:
            pass
    else:
        # Check if default port 5000 is occupied (e.g. macOS AirPlay Receiver)
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', 5000)) == 0:
                    print("Port 5000 is already in use (e.g. macOS AirPlay). Using port 5001 instead.")
                    port = 5001
        except Exception:
            pass
    print(f"🚀 Disaster Tweet Classifier Web App running on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
