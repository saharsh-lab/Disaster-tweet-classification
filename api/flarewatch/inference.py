"""
Pure-NumPy inference for Flarewatch.

- RequestModel: the Bi-LSTM request classifier (weights exported by train_request_model.py),
  with occlusion-based explanations.
- NeedsModel: multi-label TF-IDF + logistic-regression model for the resources a message needs
  (weights exported by train_needs_model.py), with per-word contributions.
- Classifier: combines both and produces the response used by the Flask apps.

No TensorFlow at runtime, so the same code runs locally and in the Vercel serverless function.
"""
import json
import os
import re

import numpy as np

MAX_LEN = 30
VOCAB_SIZE = 5000
THRESHOLD = 0.5

# Keep in sync with the training pipeline: train and serve must clean text identically
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'to', 'from', 'in', 'on', 'for', 'of', 'at',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
}

EMERGENCY_KEYWORDS = {
    'flood', 'earthquake', 'fire', 'trapped', 'water', 'food', 'shelter', 'doctor', 'medical', 'hospital', 'rescue',
    'evacuation', 'urgent', 'emergency', 'help', 'collapsed', 'damage', 'injured', 'cyclone', 'drought', 'storm'
}

# Dataset column -> resource name used across the app
RESOURCE_LABELS = [
    ("water", "water"), ("food", "food"), ("shelter", "shelter"), ("clothing", "clothing"),
    ("medical_help", "medical aid"), ("search_and_rescue", "rescue"),
]


# Keyword triggers per resource (multi-label version of the original rules)
RESOURCE_KEYWORDS = {
    "water": ("water",),
    "food": ("food",),
    "shelter": ("shelter", "accommodation"),
    "clothing": ("clothes", "clothing"),
    "medical aid": ("medical", "medicine", "doctor"),
    "rescue": ("rescue",),
}


def keyword_hits(text):
    """Boolean per resource, in RESOURCE_LABELS order."""
    lowered = str(text).lower()
    return [any(k in lowered for k in RESOURCE_KEYWORDS[name]) for _, name in RESOURCE_LABELS]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    return ' '.join([w for w in text.split() if w not in STOPWORDS])


def extract_resource_keywords(text):
    """Original keyword rules; kept as a fallback when the needs model is unavailable."""
    text = str(text).lower()
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
    return "unspecified resource"


def find_file(filename, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Request classifier (Bi-LSTM)
# ---------------------------------------------------------------------------
class RequestModel:
    def __init__(self, weights_path, word_index_path):
        w = np.load(weights_path)
        self.w = {k: w[k] for k in w.files}
        with open(word_index_path, "r", encoding="utf-8") as f:
            self.word_index = json.load(f)
        self.oov_id = self.word_index.get('<OOV>', 1)

    def token_ids(self, tokens):
        ids = []
        for t in tokens:
            idx = self.word_index.get(t, self.oov_id)
            ids.append(self.oov_id if idx >= VOCAB_SIZE else idx)
        return ids

    def _lstm(self, x, w_i, w_h, b):
        # x: (batch, steps, emb) -> final hidden state (batch, units)
        units = w_h.shape[0]
        h = np.zeros((x.shape[0], units), dtype=np.float32)
        c = np.zeros((x.shape[0], units), dtype=np.float32)
        for t in range(x.shape[1]):
            gates = x[:, t] @ w_i + h @ w_h + b
            i_gate = _sigmoid(gates[:, :units])
            f_gate = _sigmoid(gates[:, units:2 * units])
            c_cand = np.tanh(gates[:, 2 * units:3 * units])
            o_gate = _sigmoid(gates[:, 3 * units:])
            c = f_gate * c + i_gate * c_cand
            h = o_gate * np.tanh(c)
        return h

    def predict_ids_batch(self, id_lists):
        """Probabilities for several token-id sequences in one vectorised pass."""
        pad = np.zeros((len(id_lists), MAX_LEN), dtype=np.int32)
        for row, ids in enumerate(id_lists):
            ids = ids[:MAX_LEN]
            pad[row, :len(ids)] = ids
        w = self.w
        emb = w['w_emb'][pad]
        h_f = self._lstm(emb, w['w_f_i'], w['w_f_h'], w['b_f'])
        h_b = self._lstm(emb[:, ::-1], w['w_b_i'], w['w_b_h'], w['b_b'])
        d1 = np.maximum(0, np.concatenate([h_f, h_b], axis=1) @ w['w_d1'] + w['b_d1'])
        return _sigmoid(d1 @ w['w_d2'] + w['b_d2'])[:, 0].astype(float)

    def predict(self, cleaned):
        return float(self.predict_ids_batch([self.token_ids(cleaned.split())])[0])

    def predict_many(self, cleaned_texts, batch_size=256):
        out = []
        for start in range(0, len(cleaned_texts), batch_size):
            chunk = cleaned_texts[start:start + batch_size]
            out.extend(self.predict_ids_batch([self.token_ids(t.split()) for t in chunk]))
        return [float(p) for p in out]

    def explain(self, cleaned):
        """
        Occlusion: drop each token in turn and measure the change in P(request).
        Positive weight = the word pushed the message towards "request".
        Tokens beyond the model's 30-token window cannot influence it and get weight 0.
        """
        tokens = cleaned.split()
        ids = self.token_ids(tokens)
        window = min(len(ids), MAX_LEN)
        variants = [ids] + [ids[:i] + ids[i + 1:] for i in range(window)]
        probs = self.predict_ids_batch(variants)
        base = float(probs[0])
        explanation = []
        for i, tok in enumerate(tokens):
            weight = base - float(probs[i + 1]) if i < window else 0.0
            explanation.append({"token": tok, "weight": round(weight, 4), "in_vocab": ids[i] != self.oov_id})
        return base, explanation


# ---------------------------------------------------------------------------
# Needs classifier (multi-label TF-IDF + logistic regression)
# ---------------------------------------------------------------------------
def ngrams(tokens):
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


class TfidfFeaturizer:
    def __init__(self, vocab, idf):
        self.vocab = list(vocab)
        self.index = {term: i for i, term in enumerate(self.vocab)}
        self.idf = np.asarray(idf, dtype=np.float32)

    @classmethod
    def fit(cls, cleaned_texts, max_features=8000, min_df=2):
        df = {}
        for text in cleaned_texts:
            for term in set(ngrams(text.split())):
                df[term] = df.get(term, 0) + 1
        terms = [t for t, c in df.items() if c >= min_df]
        terms.sort(key=lambda t: (-df[t], t))
        terms = terms[:max_features]
        n = len(cleaned_texts)
        idf = [np.log((1 + n) / (1 + df[t])) + 1.0 for t in terms]
        return cls(terms, idf)

    def sparse(self, cleaned):
        """L2-normalised TF-IDF as (indices, values)."""
        counts = {}
        for term in ngrams(cleaned.split()):
            j = self.index.get(term)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        idx = np.fromiter(counts.keys(), dtype=np.int64)
        vals = np.fromiter(counts.values(), dtype=np.float32) * self.idf[idx]
        vals /= np.linalg.norm(vals)
        return idx, vals

    def dense(self, cleaned_texts):
        X = np.zeros((len(cleaned_texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(cleaned_texts):
            idx, vals = self.sparse(text)
            X[row, idx] = vals
        return X


class NeedsModel:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.labels = [str(x) for x in data["labels"]]
        self.featurizer = TfidfFeaturizer([str(x) for x in data["vocab"]], data["idf"])
        self.W = data["W"].astype(np.float32)
        self.b = data["b"].astype(np.float32)
        self.thresholds = data["thresholds"].astype(np.float32)
        self.use_keywords = data["use_keywords"].astype(bool) if "use_keywords" in data.files else np.zeros(len(self.labels), dtype=bool)

    def predict(self, cleaned, top_words=3, raw_text=None):
        idx, vals = self.featurizer.sparse(cleaned)
        logits = (vals @ self.W[idx] if len(idx) else np.zeros(len(self.labels), dtype=np.float32)) + self.b
        probs = _sigmoid(logits)
        hits = keyword_hits(raw_text) if raw_text is not None else [False] * len(self.labels)
        results = []
        for k, label in enumerate(self.labels):
            contrib = vals * self.W[idx, k] if len(idx) else np.zeros(0)
            order = np.argsort(-contrib)[:top_words]
            evidence = [self.featurizer.vocab[idx[j]] for j in order if contrib[j] > 0]
            by_model = bool(probs[k] >= self.thresholds[k])
            by_keyword = bool(self.use_keywords[k] and hits[k])
            results.append({
                "name": label,
                "probability": round(float(probs[k]), 4),
                "threshold": round(float(self.thresholds[k]), 2),
                "detected": by_model or by_keyword,
                "source": "model" if by_model else ("keyword" if by_keyword else None),
                "evidence": evidence,
            })
        results.sort(key=lambda r: -r["probability"])
        return results


# ---------------------------------------------------------------------------
# Combined classifier used by the apps
# ---------------------------------------------------------------------------
class Classifier:
    def __init__(self, search_dirs=None):
        if search_dirs is None:
            from flarewatch.paths import MODELS_DIR
            search_dirs = [MODELS_DIR]
        weights = find_file("model_weights.npz", search_dirs)
        word_index = find_file("word_index.json", search_dirs)
        if not weights or not word_index:
            raise FileNotFoundError("model_weights.npz / word_index.json not found in " + ", ".join(search_dirs))
        self.request_model = RequestModel(weights, word_index)
        needs_path = find_file("needs_model.npz", search_dirs)
        self.needs_model = NeedsModel(needs_path) if needs_path else None

    def resources_for(self, text, cleaned):
        """Detected needs (model first, keyword rules as fallback). Returns (primary, list)."""
        needs = self.needs_model.predict(cleaned, raw_text=text) if self.needs_model else []
        detected = [n for n in needs if n["detected"]]
        if detected:
            return detected[0]["name"], detected
        keyword = extract_resource_keywords(text)
        if keyword != "unspecified resource":
            return keyword, [{"name": keyword, "probability": None, "detected": True, "evidence": [], "source": "keyword"}]
        return "unspecified resource", []

    def classify(self, text, explain=True):
        cleaned = clean_text(text)
        if explain:
            prob, explanation = self.request_model.explain(cleaned)
        else:
            prob, explanation = self.request_model.predict(cleaned), None
        is_request = prob >= THRESHOLD
        if is_request:
            resource, resources = self.resources_for(text, cleaned)
            prediction = f"{resource.capitalize()} is required in that area ({prob:.2f})"
        else:
            resource, resources = "None", []
            prediction = f"NOT A REQUEST ({prob:.2f})"
        tokens = cleaned.split()
        from flarewatch.geo import find_locations  # local import keeps inference usable without the gazetteer
        result = {
            "prediction": prediction,
            "tweet": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "emergency_signals": [t for t in tokens if t in EMERGENCY_KEYWORDS],
            "is_request": bool(is_request),
            "probability": round(prob, 4),
            "confidence_percent": round(prob * 100, 1),
            "resource": resource,
            "resources": resources,
            "locations": find_locations(text),
            "status": "High confidence" if prob > 0.8 else ("Moderate" if prob >= THRESHOLD else "Non-Disaster"),
        }
        if explanation is not None:
            result["explanation"] = explanation
        return result
