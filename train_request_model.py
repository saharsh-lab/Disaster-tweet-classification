"""
Train the Flarewatch request classifier (Bi-LSTM) on the Disaster Response Messages dataset.

Why this exists: the original model (train_model.ipynb) was trained on ~3.5k tornado tweets whose
labels mean "about the disaster", not "asking for help", and it cleaned text differently from the
serving code. This script trains the *same architecture* on messages labelled for requests, using
the exact preprocessing the Flask/Vercel apps use, and exports every artifact the apps load.

Usage:
    venv/bin/python train_request_model.py                 # train + evaluate into models/request_v2/
    venv/bin/python train_request_model.py --install       # also copy artifacts to the served locations

Data: training_data/disaster_response/{train,validation,test}.parquet
      (Figure Eight / Appen "Multilingual Disaster Response Messages", Hugging Face mirror
       community-datasets/disaster_response_messages).
"""
import argparse
import json
import os
import pickle
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from flarewatch.inference import clean_text  # the exact preprocessing used at serving time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_data", "disaster_response")
OUT_DIR = os.path.join(BASE_DIR, "models", "request_v2")
SEED = 42

# Architecture: identical to the deployed model so the NumPy forward pass needs no changes
VOCAB_SIZE = 5000
EMBEDDING_DIM = 64
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT = 0.5
MAX_LEN = 30
THRESHOLD = 0.5

# Hand-written sanity probes (never used for training). Not a metric — a smoke test of intent.
PROBES = [
    ("Shelter required for family of 4 in Mumbai", 1),
    ("We need clean drinking water in our village, please help", 1),
    ("My grandmother is trapped under the rubble, send a rescue team", 1),
    ("No food for three days, the children are hungry, please send supplies", 1),
    ("Need doctors and medicine at the camp near the river", 1),
    ("Please send tents and blankets, we lost our home in the flood", 1),
    ("Enjoying the sunshine and having an iced latte with friends downtown", 0),
    ("The earthquake was magnitude 6.1 according to USGS", 0),
    ("Thoughts and prayers to everyone affected by the hurricane", 0),
    ("Watching a documentary about tornadoes tonight", 0),
    ("Government announces a new relief fund for flood victims", 0),
    ("Great match yesterday, what a goal!", 0),
]

from flarewatch.inference import RESOURCE_LABELS  # noqa: E402


# ---------------------------------------------------------------------------
# NumPy forward pass — a copy of the serving implementation, used for parity checks
# and for evaluating the legacy weights without TensorFlow.
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _run_lstm(x_seq, w_i, w_h, b):
    units = w_h.shape[0]
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    for x_t in x_seq:
        gates = np.dot(x_t, w_i) + np.dot(h, w_h) + b
        i_gate = _sigmoid(gates[:units])
        f_gate = _sigmoid(gates[units:2 * units])
        c_cand = np.tanh(gates[2 * units:3 * units])
        o_gate = _sigmoid(gates[3 * units:4 * units])
        c = f_gate * c + i_gate * c_cand
        h = o_gate * np.tanh(c)
    return h


def numpy_predict(cleaned_text, weights, word_index):
    oov_id = word_index.get('<OOV>', 1)
    seq = []
    for w in cleaned_text.split():
        idx = word_index.get(w, oov_id)
        if idx >= VOCAB_SIZE:
            idx = oov_id
        seq.append(idx)
    pad = np.zeros(MAX_LEN, dtype=np.int32)
    pad[:min(len(seq), MAX_LEN)] = seq[:min(len(seq), MAX_LEN)]
    emb = weights['w_emb'][pad]
    h_f = _run_lstm(emb, weights['w_f_i'], weights['w_f_h'], weights['b_f'])
    h_b = _run_lstm(emb[::-1], weights['w_b_i'], weights['w_b_h'], weights['b_b'])
    d1 = np.maximum(0, np.dot(np.concatenate([h_f, h_b]), weights['w_d1']) + weights['b_d1'])
    return float(_sigmoid(np.dot(d1, weights['w_d2']) + weights['b_d2'])[0])


# ---------------------------------------------------------------------------
# Metrics (NumPy only)
# ---------------------------------------------------------------------------
def binary_metrics(y_true, probs, threshold=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(probs) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p1, r1, f1 = prf(tp, fp, fn)
    p0, r0, f0 = prf(tn, fn, fp)
    n1, n0 = tp + fn, tn + fp
    total = n1 + n0
    return {
        "accuracy": round(100 * (tp + tn) / total, 2),
        "precision_disaster": round(100 * p1, 1), "recall_disaster": round(100 * r1, 1), "f1_disaster": round(100 * f1, 1),
        "precision_normal": round(100 * p0, 1), "recall_normal": round(100 * r0, 1), "f1_normal": round(100 * f0, 1),
        "macro_avg_f1": round(100 * (f1 + f0) / 2, 1),
        "weighted_avg_f1": round(100 * (f1 * n1 + f0 * n0) / total, 1),
        "roc_auc": round(100 * roc_auc(y_true, probs), 1),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp, "total": total},
    }


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.0
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(name):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{name}.parquet"))
    df = df[df["message"].notna()].copy()
    df["clean_text"] = df["message"].map(clean_text)
    df["label"] = df["request"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", action="store_true", help="copy artifacts to the locations the apps load")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tf.random.set_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    train, val, test = load_split("train"), load_split("validation"), load_split("test")
    print(f"train={len(train)} val={len(val)} test={len(test)} | request rate train={train.label.mean():.3f}")

    # ---- Legacy model on the same test set (before overwriting anything) ----
    legacy = None
    legacy_npz = os.path.join(BASE_DIR, "models", "legacy_v1", "model_weights.npz")
    legacy_wi = os.path.join(BASE_DIR, "models", "legacy_v1", "word_index.json")
    if os.path.exists(legacy_npz) and os.path.exists(legacy_wi):
        lw = dict(np.load(legacy_npz))
        lwi = json.load(open(legacy_wi))
        legacy_probs = np.array([numpy_predict(t, lw, lwi) for t in test.clean_text])
        legacy = {"metrics": binary_metrics(test.label, legacy_probs),
                  "probes": [(t, y, round(numpy_predict(clean_text(t), lw, lwi), 3)) for t, y in PROBES]}
        print("legacy test:", {k: legacy["metrics"][k] for k in ("accuracy", "f1_disaster", "recall_disaster", "precision_disaster")})

    # ---- Tokenizer (fit on training split only) ----
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train.clean_text)

    def encode(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, X_val, X_test = encode(train.clean_text), encode(val.clean_text), encode(test.clean_text)
    y_train, y_val, y_test = train.label.values, val.label.values, test.label.values

    # Requests are ~17% of messages: weight classes so the 0.5 threshold is balanced
    pos = y_train.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    # ---- Bi-LSTM (same architecture as deployed) ----
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,), dtype="int32"),
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dropout(DROPOUT),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=32,
        class_weight=class_weight, verbose=2,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    test_probs = model.predict(X_test, verbose=0).ravel()
    new_metrics = binary_metrics(y_test, test_probs)
    print("bi-lstm test:", {k: new_metrics[k] for k in ("accuracy", "f1_disaster", "recall_disaster", "precision_disaster")})

    # ---- Baseline: TF-IDF (uni+bigrams) + logistic regression, same data and threshold ----
    vectorizer = layers.TextVectorization(max_tokens=20000, ngrams=2, output_mode="tf_idf")
    vectorizer.adapt(train.clean_text.values)
    Xb_train = vectorizer(train.clean_text.values)
    Xb_val = vectorizer(val.clean_text.values)
    Xb_test = vectorizer(test.clean_text.values)
    logreg = models.Sequential([layers.Input(shape=(Xb_train.shape[1],)), layers.Dense(1, activation="sigmoid",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-4))])
    logreg.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    logreg.fit(Xb_train, y_train, validation_data=(Xb_val, y_val), epochs=20, batch_size=64, class_weight=class_weight,
               verbose=0, callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
    baseline_metrics = binary_metrics(y_test, logreg.predict(Xb_test, verbose=0).ravel())
    print("tf-idf baseline test:", {k: baseline_metrics[k] for k in ("accuracy", "f1_disaster", "recall_disaster", "precision_disaster")})

    # ---- Export NumPy weights in the serving format ----
    emb_layer, bi_layer, _, d1_layer, d2_layer = model.layers
    f_kernel, f_rec, f_bias = bi_layer.forward_layer.get_weights()
    b_kernel, b_rec, b_bias = bi_layer.backward_layer.get_weights()
    w_d1, b_d1 = d1_layer.get_weights()
    w_d2, b_d2 = d2_layer.get_weights()
    np_weights = {
        "w_emb": emb_layer.get_weights()[0], "w_f_i": f_kernel, "w_f_h": f_rec, "b_f": f_bias,
        "w_b_i": b_kernel, "w_b_h": b_rec, "b_b": b_bias, "w_d1": w_d1, "b_d1": b_d1, "w_d2": w_d2, "b_d2": b_d2,
    }
    np_weights = {k: v.astype(np.float32) for k, v in np_weights.items()}
    word_index = tokenizer.word_index

    # Parity: the NumPy forward pass must reproduce Keras exactly
    sample = test.clean_text.values[:300]
    keras_p = model.predict(encode(sample), verbose=0).ravel()
    numpy_p = np.array([numpy_predict(t, np_weights, word_index) for t in sample])
    parity = float(np.max(np.abs(keras_p - numpy_p)))
    print(f"numpy/keras parity max abs diff = {parity:.2e}")
    assert parity < 1e-4, "NumPy export does not match Keras — refusing to export"

    probes = [(t, y, round(numpy_predict(clean_text(t), np_weights, word_index), 3)) for t, y in PROBES]

    # ---- Write artifacts ----
    np.savez(os.path.join(OUT_DIR, "model_weights.npz"), **np_weights)
    with open(os.path.join(OUT_DIR, "word_index.json"), "w") as f:
        json.dump(word_index, f)
    with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    model.save(os.path.join(OUT_DIR, "disaster_rnn_model.h5"))

    hist = history.history
    metrics_doc = {
        "model_name": "Bidirectional LSTM Request Classifier",
        "version": "2.0",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "Is this message a request for help? (Disaster Response Messages 'request' label)",
        "threshold": THRESHOLD,
        "architecture": {
            "vocab_size": VOCAB_SIZE, "embedding_dim": EMBEDDING_DIM, "sequence_length": MAX_LEN,
            "recurrent_units": LSTM_UNITS, "recurrent_type": f"Bidirectional LSTM ({2 * LSTM_UNITS} concatenated)",
            "dropout": DROPOUT, "dense_units": DENSE_UNITS, "activation_dense": "ReLU", "activation_output": "Sigmoid",
            "loss_function": "Binary Cross-Entropy (class-weighted)", "optimizer": "Adam (lr=0.001)",
            "batch_size": 32, "epochs": len(hist["loss"]), "best_epoch": best_epoch,
        },
        "dataset": {
            "name": "Disaster Response Messages (Figure Eight / Appen)",
            "source": "huggingface.co/datasets/community-datasets/disaster_response_messages",
            "train_requests": int(y_train.sum()), "train_non_requests": int(len(y_train) - y_train.sum()),
        },
        "dataset_split": {
            "total_samples": int(len(train) + len(val) + len(test)),
            "train_samples": int(len(train)), "val_samples": int(len(val)), "test_samples": int(len(test)),
            "split_ratio": "Official train / validation / test splits",
        },
        "metrics": {k: v for k, v in new_metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": new_metrics["confusion_matrix"],
        "training_history": [
            {"epoch": i + 1, "loss": round(hist["loss"][i], 4), "accuracy": round(100 * hist["accuracy"][i], 2),
             "val_loss": round(hist["val_loss"][i], 4), "val_accuracy": round(100 * hist["val_accuracy"][i], 2)}
            for i in range(len(hist["loss"]))
        ],
        "comparison": {
            "bilstm_v2": {k: new_metrics[k] for k in ("accuracy", "precision_disaster", "recall_disaster", "f1_disaster", "roc_auc")},
            "tfidf_logreg_baseline": {k: baseline_metrics[k] for k in ("accuracy", "precision_disaster", "recall_disaster", "f1_disaster", "roc_auc")},
            **({"legacy_bilstm_v1": {k: legacy["metrics"][k] for k in ("accuracy", "precision_disaster", "recall_disaster", "f1_disaster", "roc_auc")}} if legacy else {}),
        },
        "probes": [{"text": t, "expected": y, "probability": p} for t, y, p in probes],
        "legacy_probes": [{"text": t, "expected": y, "probability": p} for t, y, p in legacy["probes"]] if legacy else [],
        "numpy_keras_parity_max_abs_diff": parity,
    }
    # Keep sections written by other training scripts (e.g. "needs" from train_needs_model.py)
    installed = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(installed):
        previous = json.load(open(installed))
        if "needs" in previous:
            metrics_doc["needs"] = previous["needs"]
    with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics_doc, f, indent=2)

    # Corpus for the Dataset tab, in the schema data_service.py reads
    corpus = pd.concat([train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")], ignore_index=True)
    corpus_out = pd.DataFrame({
        "num": range(len(corpus)),
        "text": corpus["message"].str.strip(),
        "clean_text": corpus["clean_text"],
        "label": corpus["label"],
        "location": "N/A",
        "resources": [str([name for col, name in RESOURCE_LABELS if row[col] == 1]) for _, row in corpus.iterrows()],
        "timestamp": "",
        "genre": corpus["genre"],
        "split": corpus["split"],
    })
    corpus_out.to_csv(os.path.join(OUT_DIR, "dataset_cleaned.csv"), index=False)

    print("\nProbe predictions (legacy -> new):")
    legacy_lookup = {t: p for t, _, p in (legacy["probes"] if legacy else [])}
    for t, y, p in probes:
        print(f"  expected={y}  legacy={legacy_lookup.get(t, float('nan')):.3f}  new={p:.3f}  {t}")
    print(f"\nArtifacts written to {OUT_DIR}")

    if args.install:
        targets = {
            "model_weights.npz": [BASE_DIR, os.path.join(BASE_DIR, "api")],
            "word_index.json": [BASE_DIR, os.path.join(BASE_DIR, "api")],
            "model_metrics.json": [BASE_DIR, os.path.join(BASE_DIR, "api")],
            "dataset_cleaned.csv": [BASE_DIR, os.path.join(BASE_DIR, "api")],
            "tokenizer.pkl": [BASE_DIR],
            "disaster_rnn_model.h5": [BASE_DIR],
        }
        targets = {name: [BASE_DIR] for name in targets}  # api/ copies come from scripts/sync_vercel_bundle.py
        for name, dirs in targets.items():
            for d in dirs:
                shutil.copy2(os.path.join(OUT_DIR, name), os.path.join(d, name))
                print(f"installed {name} -> {os.path.relpath(d, BASE_DIR) or '.'}")


if __name__ == "__main__":
    main()
