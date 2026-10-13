"""
Re-score classification_results.csv with the currently deployed models.

Run this after installing a new model so the dashboard's historical signal stream reflects the
models that are actually serving predictions (request classifier + needs model).

Usage:
    venv/bin/python rescore_results.py
    venv/bin/python scripts/sync_vercel_bundle.py   # then refresh the Vercel copies
"""
import os

import pandas as pd

from flarewatch.inference import Classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "classification_results.csv")


def main():
    clf = Classifier([BASE_DIR])
    df = pd.read_csv(RESULTS)
    before = df["is_request"].astype(str).str.lower().isin(["true", "1"]).sum()
    cache = {}
    for i, tweet in df["tweet"].items():
        if tweet not in cache:
            cache[tweet] = clf.classify(str(tweet), explain=False)
        r = cache[tweet]
        df.at[i, "cleaned_text"] = r["cleaned"]
        df.at[i, "confidence"] = r["probability"]
        df.at[i, "is_request"] = r["is_request"]
        df.at[i, "resource"] = r["resource"] if r["is_request"] else "not_applicable"
    df.to_csv(RESULTS, index=False)
    print(f"{len(df)} rows ({len(cache)} unique) · requests {before} -> {int(df['is_request'].sum())}")


if __name__ == "__main__":
    main()
