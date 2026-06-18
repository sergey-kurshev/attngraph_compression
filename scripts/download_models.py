"""Download the proxy model used by the H1/H2 experiments to a local folder.

By default this fetches Qwen2.5-0.5B-Instruct into ``models/qwen2.5-0.5b-instruct/``.
That path is what the experiment runners look up first (each runner falls back
to the HuggingFace hub id ``Qwen/Qwen2.5-0.5B-Instruct`` if the folder is
absent, so this script is technically optional but makes runs reproducible
and air-gappable).

The 4-bit reader for the QA campaign (``Qwen/Qwen2.5-7B-Instruct``) is loaded
*from the HuggingFace cache* by ``transformers`` at runtime — no local-dir copy
needed. Pass ``--reader`` to pre-warm the cache anyway (~15 GB download).

Usage
-----
    python scripts/download_models.py                  # proxy only (~1 GB)
    python scripts/download_models.py --reader         # also pre-warm reader (~15 GB)
    python scripts/download_models.py --dest path/to/models
"""

from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRODUCE_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_DEST = os.path.join(REPRODUCE_ROOT, "models")

PROXY_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROXY_DIR_NAME = "qwen2.5-0.5b-instruct"
READER_REPO = "Qwen/Qwen2.5-7B-Instruct"


def _snapshot(repo_id: str, local_dir: str | None = None) -> str:
    from huggingface_hub import snapshot_download
    kwargs: dict = {"repo_id": repo_id}
    if local_dir is not None:
        kwargs["local_dir"] = local_dir
        kwargs["local_dir_use_symlinks"] = False
    return snapshot_download(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help="parent folder for vendored model directories")
    ap.add_argument("--reader", action="store_true",
                    help="also pre-warm the 7B reader into the HF cache (~15 GB)")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("       pip install -r requirements.txt", file=sys.stderr)
        return 1

    proxy_dir = os.path.join(args.dest, PROXY_DIR_NAME)
    os.makedirs(proxy_dir, exist_ok=True)
    print(f"[models] proxy   {PROXY_REPO}  ->  {proxy_dir}")
    _snapshot(PROXY_REPO, local_dir=proxy_dir)
    print(f"[models] proxy   OK")

    if args.reader:
        print(f"[models] reader  {READER_REPO}  ->  HF cache")
        _snapshot(READER_REPO, local_dir=None)
        print(f"[models] reader  OK")
    else:
        print(f"[models] reader  skipped (will download into the HF cache on first QA run)")
        print(f"                 pass --reader to pre-warm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
