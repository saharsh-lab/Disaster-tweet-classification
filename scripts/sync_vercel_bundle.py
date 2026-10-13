"""
Copy the canonical app files into api/ for the Vercel serverless function.

The repository root is the single source of truth. Vercel runs api/index.py, which resolves
templates, data and the flarewatch package from api/ first, so those copies are generated here
rather than edited by hand.

Usage:
    venv/bin/python scripts/sync_vercel_bundle.py          # copy
    venv/bin/python scripts/sync_vercel_bundle.py --check  # exit 1 if api/ is out of date (used by tests/CI)
"""
import argparse
import filecmp
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(BASE_DIR, "api")

FILES = {
    "templates/index.html": "templates/index.html",
    "data_service.py": "data_service.py",
    "model_weights.npz": "model_weights.npz",
    "word_index.json": "word_index.json",
    "needs_model.npz": "needs_model.npz",
    "model_metrics.json": "model_metrics.json",
    "dataset_cleaned.csv": "dataset_cleaned.csv",
    "classification_results.csv": "classification_results.csv",
}
PACKAGE = "flarewatch"


def package_files():
    root = os.path.join(BASE_DIR, PACKAGE)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".DS_Store")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), BASE_DIR)
            yield rel, rel


def pairs():
    yield from FILES.items()
    yield from package_files()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stale = []
    for src_rel, dst_rel in pairs():
        src = os.path.join(BASE_DIR, src_rel)
        dst = os.path.join(API_DIR, dst_rel)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
            continue
        stale.append(dst_rel)
        if not args.check:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # Remove package files that no longer exist at the root
    api_pkg = os.path.join(API_DIR, PACKAGE)
    expected = {dst for _, dst in package_files()}
    for dirpath, dirnames, filenames in os.walk(api_pkg):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), API_DIR)
            if rel not in expected and not name.endswith(".pyc"):
                stale.append(f"{rel} (orphan)")
                if not args.check:
                    os.remove(os.path.join(dirpath, name))

    if args.check:
        if stale:
            print("api/ is out of sync with the repository root:\n  " + "\n  ".join(stale))
            print("Run: python scripts/sync_vercel_bundle.py")
            sys.exit(1)
        print("api/ bundle is in sync")
    else:
        print(f"synced {len(stale)} file(s) into api/" + ("".join(f"\n  {s}" for s in stale) if stale else ""))


if __name__ == "__main__":
    main()
