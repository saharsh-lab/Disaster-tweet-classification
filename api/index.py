import os
import sys
import re
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Resolve file paths (check current dir first, then parent dir)
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

# Template folder
templates_dir = os.path.join(CURRENT_DIR, "templates")
if not os.path.exists(templates_dir):
    templates_dir = os.path.join(PARENT_DIR, "templates")

app = Flask(__name__, template_folder=templates_dir)

# Load word index (pure JSON, zero Keras/TensorFlow requirement)
word_index = {}
if os.path.exists(WORD_INDEX_JSON):
    with open(WORD_INDEX_JSON, "r") as f:
        word_index = json.load(f)
oov_id = word_index.get('<OOV>', 1)
MAX_LEN = 30

# Load pure NumPy model weights (instant cold start, < 250MB limit)
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
    prob = predict_probability(cleaned)
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
    prob = predict_probability(cleaned)
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

# Export WSGI application for Vercel
app = app
