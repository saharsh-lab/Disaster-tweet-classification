"""
Train the Flarewatch needs model: which resources does a message ask for?

Multi-label TF-IDF (unigrams + bigrams) + logistic regression, one output per resource
(water, food, shelter, clothing, medical aid, rescue). Trained in plain NumPy with the same
featurizer the apps use at inference time (flarewatch.inference), so there is no train/serve skew.
Per-label decision thresholds are tuned on the validation split; metrics are reported on the test
split next to the original keyword rules.

Usage:
    venv/bin/python train_needs_model.py            # writes models/needs_v1/
    venv/bin/python train_needs_model.py --install  # also installs needs_model.npz and updates model_metrics.json
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from flarewatch.inference import RESOURCE_KEYWORDS, RESOURCE_LABELS, TfidfFeaturizer, clean_text, extract_resource_keywords, keyword_hits

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_data", "disaster_response")
OUT_DIR = os.path.join(BASE_DIR, "models", "needs_v1")
SEED = 42
COLUMNS = [c for c, _ in RESOURCE_LABELS]
NAMES = [n for _, n in RESOURCE_LABELS]


def load(split):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{split}.parquet"))
    df = df[df["message"].notna()].copy()
    df["clean_text"] = df["message"].map(clean_text)
    return df


def prf(y, p):
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def evaluate(Y, P):
    """Per-label and micro/macro scores for boolean matrices (n, labels)."""
    per = {}
    tps = fps = fns = 0
    for k, name in enumerate(NAMES):
        p, r, f, tp, fp, fn = prf(Y[:, k], P[:, k])
        per[name] = {"precision": round(100 * p, 1), "recall": round(100 * r, 1), "f1": round(100 * f, 1), "support": int(Y[:, k].sum())}
        tps += tp; fps += fp; fns += fn
    micro_p = tps / (tps + fps) if tps + fps else 0.0
    micro_r = tps / (tps + fns) if tps + fns else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "per_label": per,
        "micro_f1": round(100 * micro_f, 1),
        "micro_precision": round(100 * micro_p, 1),
        "micro_recall": round(100 * micro_r, 1),
        "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 1),
    }


def train_logreg(X, Y, X_val, Y_val, epochs=200, lr=0.03, l2=1e-5, batch=512):
    """Class-weighted multi-label logistic regression with Adam and early stopping on validation loss."""
    rng = np.random.default_rng(SEED)
    n, d = X.shape
    k = Y.shape[1]
    W = np.zeros((d, k), dtype=np.float32)
    b = np.log(Y.mean(0) / (1 - Y.mean(0))).astype(np.float32)
    pos_w = ((1 - Y.mean(0)) / Y.mean(0)).astype(np.float32) ** 0.5  # soften the imbalance
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0

    def loss(Xs, Ys):
        p = 1 / (1 + np.exp(-np.clip(Xs @ W + b, -30, 30)))
        return float(-np.mean(Ys * pos_w * np.log(p + 1e-7) + (1 - Ys) * np.log(1 - p + 1e-7)))

    best = (np.inf, W.copy(), b.copy(), 0)
    patience = 0
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            Xb, Yb = X[idx], Y[idx]
            p = 1 / (1 + np.exp(-np.clip(Xb @ W + b, -30, 30)))
            g = (p * (Yb * pos_w + (1 - Yb)) - Yb * pos_w) / len(idx)  # d loss / d logit for weighted BCE
            gW = Xb.T @ g + l2 * W
            gb = g.sum(0)
            step += 1
            for param, grad, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                param -= lr * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + eps)
        val = loss(X_val, Y_val)
        if val < best[0] - 1e-4:
            best = (val, W.copy(), b.copy(), epoch + 1)
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    print(f"best epoch {best[3]} (val loss {best[0]:.4f})")
    return best[1], best[2], best[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    train, val, test = load("train"), load("validation"), load("test")
    featurizer = TfidfFeaturizer.fit(train.clean_text.tolist())
    X, X_val, X_test = (featurizer.dense(d.clean_text.tolist()) for d in (train, val, test))
    Y, Y_val, Y_test = (d[COLUMNS].values.astype(np.float32) for d in (train, val, test))
    print(f"features={len(featurizer.vocab)} train={len(train)} label rates={dict(zip(NAMES, Y.mean(0).round(3)))}")

    W, b, best_epoch = train_logreg(X, Y, X_val, Y_val)

    def probs(Xs):
        return 1 / (1 + np.exp(-np.clip(Xs @ W + b, -30, 30)))

    # Per-label thresholds that maximise F1 on validation
    P_val = probs(X_val)
    thresholds = []
    for k in range(len(NAMES)):
        grid = np.arange(0.1, 0.91, 0.05)
        scores = [prf(Y_val[:, k] > 0.5, P_val[:, k] >= t)[2] for t in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    thresholds = np.array(thresholds, dtype=np.float32)

    # For each need, keep the keyword rule as an extra trigger only if it helps on validation
    KW_val = np.array([keyword_hits(t) for t in val.message], dtype=bool)
    KW_test = np.array([keyword_hits(t) for t in test.message], dtype=bool)
    Yv = Y_val > 0.5
    model_val = P_val >= thresholds
    use_keywords = np.array([
        prf(Yv[:, k], model_val[:, k] | KW_val[:, k])[2] > prf(Yv[:, k], model_val[:, k])[2]
        for k in range(len(NAMES))
    ])

    Yt = Y_test > 0.5
    model_only_eval = evaluate(Yt, probs(X_test) >= thresholds)
    hybrid_test = (probs(X_test) >= thresholds) | (KW_test & use_keywords)
    model_eval = evaluate(Yt, hybrid_test)

    # Keyword rules: single label, first match wins ("food" before "water", ...)
    kw = np.zeros_like(Yt)
    for row, text in enumerate(test.message):
        name = extract_resource_keywords(text)
        if name in NAMES:
            kw[row, NAMES.index(name)] = True
    keyword_eval = evaluate(Yt, kw)

    # On real requests only — the situation the app shows resources for
    req = test.request.values == 1
    model_req = evaluate(Yt[req], hybrid_test[req])
    keyword_req = evaluate(Yt[req], kw[req])

    print(f"test micro-F1: hybrid {model_eval['micro_f1']} | model only {model_only_eval['micro_f1']} | keywords {keyword_eval['micro_f1']}")
    print(f"test micro-F1 on requests: model {model_req['micro_f1']} vs keywords {keyword_req['micro_f1']}")
    for name in NAMES:
        k = NAMES.index(name)
        print(f"  {name:12s} hybrid F1 {model_eval['per_label'][name]['f1']:5.1f}  model {model_only_eval['per_label'][name]['f1']:5.1f}  keywords {keyword_eval['per_label'][name]['f1']:5.1f}  thr {thresholds[k]:.2f}  +kw {bool(use_keywords[k])}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "needs_model.npz")
    np.savez(out_path, labels=np.array(NAMES), vocab=np.array(featurizer.vocab), idf=featurizer.idf,
             W=W.astype(np.float32), b=b.astype(np.float32), thresholds=thresholds, use_keywords=use_keywords)
    report = {
        "model_name": "TF-IDF + logistic regression needs classifier",
        "version": "1.0",
        "labels": NAMES,
        "features": len(featurizer.vocab),
        "best_epoch": best_epoch,
        "thresholds": {n: round(float(t), 2) for n, t in zip(NAMES, thresholds)},
        "keyword_assist": {n: bool(u) for n, u in zip(NAMES, use_keywords)},
        "test_all_messages": {"model": model_eval, "model_without_keywords": model_only_eval, "keyword_rules": keyword_eval},
        "test_requests_only": {"model": model_req, "keyword_rules": keyword_req, "messages": int(req.sum())},
    }
    with open(os.path.join(OUT_DIR, "needs_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"written to {OUT_DIR}")

    if args.install:
        shutil.copy2(out_path, os.path.join(BASE_DIR, "needs_model.npz"))
        metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
        metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
        metrics["needs"] = report
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("installed needs_model.npz and updated model_metrics.json (run scripts/sync_vercel_bundle.py next)")


if __name__ == "__main__":
    main()
