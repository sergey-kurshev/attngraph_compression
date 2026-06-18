"""Download the raw LongBench v1 task files into experiments/longbench/data/data/.

The HuggingFace ``datasets`` script loader for THUDM/LongBench was removed in
datasets 4.x. We work around that by fetching the bundled ``data.zip`` from
the dataset repo directly with ``huggingface_hub.hf_hub_download`` and
unpacking the per-task JSONL files locally.

Usage
-----
From the repo root (the parent of this ``scripts/`` folder):

    python scripts/download_longbench.py            # default destination
    python scripts/download_longbench.py --dest path/to/longbench/data

Result
------
The destination is populated with one ``<task>.jsonl`` per LongBench v1 task,
matching what ``experiments/longbench/select_examples.py`` expects to find at
``experiments/longbench/data/data/<task>.jsonl``.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRODUCE_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_DEST = os.path.join(REPRODUCE_ROOT, "experiments", "longbench", "data", "data")
REPO_ID = "THUDM/LongBench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help="directory to unpack the per-task .jsonl files into")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep data.zip after extraction (otherwise removed)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("       pip install -r requirements.txt", file=sys.stderr)
        return 1

    os.makedirs(args.dest, exist_ok=True)
    print(f"[longbench] downloading data.zip from {REPO_ID} ...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename="data.zip",
                               repo_type="dataset", local_dir=args.dest)
    print(f"[longbench] extracting -> {args.dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)
    # data.zip contains a top-level "data/" folder; flatten so files land in --dest.
    nested = os.path.join(args.dest, "data")
    if os.path.isdir(nested):
        for name in os.listdir(nested):
            src = os.path.join(nested, name)
            dst = os.path.join(args.dest, name)
            if os.path.exists(dst):
                continue
            os.replace(src, dst)
        try:
            os.rmdir(nested)
        except OSError:
            pass
    if not args.keep_zip:
        try:
            os.remove(zip_path)
        except OSError:
            pass

    tasks = sorted(f for f in os.listdir(args.dest) if f.endswith(".jsonl"))
    print(f"[longbench] OK  ({len(tasks)} task files)")
    for t in tasks:
        size_kb = os.path.getsize(os.path.join(args.dest, t)) // 1024
        print(f"            {size_kb:>7d} KB  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
