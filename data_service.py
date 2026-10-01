import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

def resolve_file(filename):
    for candidate in [
        os.path.join(BASE_DIR, filename),
        os.path.join(PARENT_DIR, filename),
        os.path.join(BASE_DIR, "datasets", filename),
        os.path.join(PARENT_DIR, "datasets", filename)
    ]:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DIR, filename)

RESULTS_CSV = resolve_file("classification_results.csv")
DATASET_CSV = resolve_file("dataset_cleaned.csv")

_cached_stats = None
_cached_results_df = None
_cached_dataset_df = None

def get_results_df():
    global _cached_results_df
    if _cached_results_df is None and os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            # Ensure boolean
            df['is_request'] = df['is_request'].astype(bool)
            df['confidence'] = df['confidence'].astype(float)
            if 'cleaned_text' not in df.columns or df['cleaned_text'].isnull().any():
                df['cleaned_text'] = df['cleaned_text'].fillna('')
            _cached_results_df = df
        except Exception as e:
            print(f"Error loading classification_results.csv: {e}")
            _cached_results_df = pd.DataFrame()
    return _cached_results_df

def get_dataset_df():
    global _cached_dataset_df
    if _cached_dataset_df is None and os.path.exists(DATASET_CSV):
        try:
            df = pd.read_csv(DATASET_CSV)
            _cached_dataset_df = df
        except Exception as e:
            print(f"Error loading dataset_cleaned.csv: {e}")
            _cached_dataset_df = pd.DataFrame()
    return _cached_dataset_df

def get_stats():
    global _cached_stats
    if _cached_stats is not None:
        return _cached_stats

    df = get_results_df()
    if df is None or len(df) == 0:
        return {
            "total_tweets": 8000,
            "disaster_tweets": 5755,
            "normal_tweets": 2245,
            "avg_confidence": 0.705,
            "high_confidence_requests": 4886,
            "low_confidence_requests": 489,
            "high_confidence_pct": 84.9,
            "model_accuracy": 0.74,
            "resource_breakdown": {
                "unspecified resource": 4855,
                "water": 458,
                "rescue": 221,
                "medical aid": 108,
                "food": 63,
                "shelter": 50
            },
            "confidence_distribution": {
                "0-20%": 1205,
                "20-40%": 820,
                "40-60%": 490,
                "60-80%": 599,
                "80-100%": 4886
            }
        }

    total = len(df)
    req_mask = df['is_request'] == True
    disaster_count = int(req_mask.sum())
    normal_count = total - disaster_count
    avg_conf = float(df['confidence'].mean())
    high_conf = int(((req_mask) & (df['confidence'] > 0.8)).sum())
    low_conf = int(((req_mask) & (df['confidence'] < 0.7)).sum())
    high_conf_pct = round((high_conf / disaster_count * 100), 1) if disaster_count > 0 else 0.0

    # Resource breakdown
    req_df = df[req_mask]
    resource_counts = req_df['resource'].value_counts().to_dict() if len(req_df) > 0 else {}

    # Confidence buckets
    c = df['confidence']
    conf_buckets = {
        "0-20%": int((c < 0.2).sum()),
        "20-40%": int(((c >= 0.2) & (c < 0.4)).sum()),
        "40-60%": int(((c >= 0.4) & (c < 0.6)).sum()),
        "60-80%": int(((c >= 0.6) & (c < 0.8)).sum()),
        "80-100%": int((c >= 0.8).sum())
    }

    _cached_stats = {
        "total_tweets": total,
        "disaster_tweets": disaster_count,
        "normal_tweets": normal_count,
        "avg_confidence": round(avg_conf, 3),
        "high_confidence_requests": high_conf,
        "low_confidence_requests": low_conf,
        "high_confidence_pct": high_conf_pct,
        "model_accuracy": 0.7399,
        "resource_breakdown": resource_counts,
        "confidence_distribution": conf_buckets
    }
    return _cached_stats

def query_tweets(page=1, page_size=25, search="", filter_type="all", resource="all", sort_by="default"):
    df = get_results_df()
    if df is None or len(df) == 0:
        return {"total": 0, "page": page, "page_size": page_size, "records": []}

    filtered = df

    # Search filter
    if search and search.strip():
        q = search.strip()
        filtered = filtered[filtered['tweet'].str.contains(q, case=False, na=False)]

    # Type filter
    if filter_type == "disaster":
        filtered = filtered[filtered['is_request'] == True]
    elif filter_type == "non_disaster":
        filtered = filtered[filtered['is_request'] == False]
    elif filter_type == "high_confidence":
        filtered = filtered[filtered['confidence'] >= 0.8]
    elif filter_type == "low_confidence":
        filtered = filtered[filtered['confidence'] < 0.7]

    # Resource filter
    if resource and resource != "all":
        filtered = filtered[filtered['resource'].str.lower() == resource.lower()]

    # Sort
    if sort_by == "highest_confidence":
        filtered = filtered.sort_values(by="confidence", ascending=False)
    elif sort_by == "lowest_confidence":
        filtered = filtered.sort_values(by="confidence", ascending=True)

    total_matches = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_records = filtered.iloc[start_idx:end_idx].copy()

    records = []
    for idx, row in page_records.iterrows():
        records.append({
            "id": int(idx) + 1,
            "tweet": str(row.get('tweet', '')),
            "cleaned_text": str(row.get('cleaned_text', '')),
            "is_request": bool(row.get('is_request', False)),
            "confidence": round(float(row.get('confidence', 0.0)) * 100, 1),
            "confidence_raw": round(float(row.get('confidence', 0.0)), 4),
            "resource": str(row.get('resource', 'unspecified resource')),
            "timestamp": str(row.get('timestamp', '2026-10-01 12:00:00'))[:19]
        })

    return {
        "total": total_matches,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_matches + page_size - 1) // page_size),
        "records": records
    }

def query_dataset(page=1, page_size=25, search="", label="all"):
    df = get_dataset_df()
    if df is None or len(df) == 0:
        return {"total": 0, "page": page, "page_size": page_size, "records": []}

    filtered = df

    if search and search.strip():
        q = search.strip()
        filtered = filtered[filtered['text'].str.contains(q, case=False, na=False)]

    if label == "disaster" or label == "1":
        filtered = filtered[filtered['label'] == 1]
    elif label == "non_disaster" or label == "0":
        filtered = filtered[filtered['label'] == 0]

    total_matches = len(filtered)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_records = filtered.iloc[start_idx:end_idx].copy()

    records = []
    for idx, row in page_records.iterrows():
        records.append({
            "num": int(row.get('num', idx)),
            "text": str(row.get('text', '')),
            "clean_text": str(row.get('clean_text', '')),
            "label": int(row.get('label', 0)),
            "location": str(row.get('location', 'N/A')),
            "resources": str(row.get('resources', '[]')),
            "timestamp": str(row.get('timestamp', ''))[:19]
        })

    return {
        "total": total_matches,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_matches + page_size - 1) // page_size),
        "records": records
    }

def get_model_performance():
    return {
        "model_name": "Bidirectional LSTM Disaster Tweet Classifier",
        "architecture": {
            "vocab_size": 5000,
            "embedding_dim": 64,
            "sequence_length": 30,
            "recurrent_units": 64,
            "recurrent_type": "Bidirectional LSTM (128 concatenated)",
            "dropout": 0.5,
            "dense_units": 32,
            "activation_dense": "ReLU",
            "activation_output": "Sigmoid",
            "loss_function": "Binary Cross-Entropy",
            "optimizer": "Adam (lr=0.001)",
            "batch_size": 32,
            "epochs": 5
        },
        "dataset_split": {
            "total_samples": 3572,
            "train_samples": 2857,
            "val_samples": 715,
            "split_ratio": "80% Train / 20% Validation (random_state=42)"
        },
        "metrics": {
            "accuracy": 73.99,
            "precision_disaster": 79.2,
            "recall_disaster": 85.5,
            "f1_disaster": 82.2,
            "precision_normal": 57.3,
            "recall_normal": 46.4,
            "f1_normal": 51.3,
            "macro_avg_f1": 66.7,
            "weighted_avg_f1": 73.0
        },
        "confusion_matrix": {
            "true_negative": 98,
            "false_positive": 113,
            "false_negative": 73,
            "true_positive": 431,
            "total": 715
        },
        "training_history": [
            {"epoch": 1, "loss": 0.6347, "accuracy": 69.36, "val_loss": 0.5498, "val_accuracy": 70.77},
            {"epoch": 2, "loss": 0.4950, "accuracy": 75.34, "val_loss": 0.5388, "val_accuracy": 74.55},
            {"epoch": 3, "loss": 0.3375, "accuracy": 85.45, "val_loss": 0.5539, "val_accuracy": 74.69},
            {"epoch": 4, "loss": 0.1902, "accuracy": 92.93, "val_loss": 0.6175, "val_accuracy": 75.24},
            {"epoch": 5, "loss": 0.1307, "accuracy": 95.97, "val_loss": 0.7849, "val_accuracy": 73.99}
        ]
    }
