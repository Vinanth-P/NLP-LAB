"""
download_data.py
================
Utility functions to stream data from HuggingFace datasets.

Datasets used:
  1. ai4bharat/IndicCorpV2  — Hindi subset (hin_Deva)
  2. oscar-corpus/OSCAR-2301 — Hindi subset (hi)

Streaming is used so the full dataset is never downloaded at once.
Each function is a generator that yields one text string at a time.

Usage
-----
    from download_data import stream_indiccorp, stream_oscar

    for text in stream_indiccorp(max_docs=1000):
        print(text[:100])
"""

import os
import sys

# Ensure UTF-8 is used for all I/O on Windows (fix for cp1252 issues)
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from datasets import load_dataset, IterableDataset
except ImportError:
    print("[ERROR] 'datasets' package not found. Run: pip install datasets")
    sys.exit(1)

from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# IndicCorpV2  (ai4bharat)
# ---------------------------------------------------------------------------

def stream_indiccorp(
    max_docs: Optional[int] = None,
    language: str = "hin_Deva",
) -> Iterator[str]:
    """
    Stream text documents from IndicCorpV2.

    Parameters
    ----------
    max_docs : int, optional
        Maximum number of documents to yield. None = stream everything.
    language : str
        Language split key, e.g. 'hin_Deva' for Hindi.

    Yields
    ------
    str
        Raw text of each document.
    """
    print(f"[IndicCorpV2] Loading stream for language={language} ...")
    try:
        dataset: IterableDataset = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split=language,
            streaming=True,
        )
    except Exception as e:
        print(f"[IndicCorpV2] Failed to load dataset: {e}")
        print("  Tip: Make sure you have internet access and the 'datasets' package installed.")
        return

    count = 0
    for example in dataset:
        text = example.get("text", "")
        if text and text.strip():
            yield text.strip()
            count += 1
            if max_docs is not None and count >= max_docs:
                break

    print(f"[IndicCorpV2] Streamed {count} documents.")


# ---------------------------------------------------------------------------
# OSCAR-2301  (oscar-corpus)
# ---------------------------------------------------------------------------

def stream_oscar(
    max_docs: Optional[int] = None,
    language: str = "hi",
) -> Iterator[str]:
    """
    Stream text documents from OSCAR-2301.

    ⚠️  OSCAR-2301 is a gated dataset.
    Before running, authenticate via:
        huggingface-cli login
    or set the environment variable HF_TOKEN.

    Parameters
    ----------
    max_docs : int, optional
        Maximum number of documents to yield. None = stream everything.
    language : str
        Language code, e.g. 'hi' for Hindi.

    Yields
    ------
    str
        Raw text of each document.
    """
    import os
    token = os.environ.get("HF_TOKEN", None)

    print(f"[OSCAR-2301] Loading stream for language={language} ...")
    if token is None:
        print("  [OSCAR-2301] No HF_TOKEN env var found. Attempting without token.")
        print("  If this fails, run: huggingface-cli login")

    try:
        dataset: IterableDataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            language=language,
            split="train",
            streaming=True,
            token=token,
        )
    except Exception as e:
        print(f"[OSCAR-2301] Failed to load dataset: {e}")
        print("  Tip: Run 'huggingface-cli login' or set HF_TOKEN env variable.")
        print("  Also ensure you have accepted the dataset license on HuggingFace.")
        return

    count = 0
    for example in dataset:
        # OSCAR records have a 'content' field
        text = example.get("content", example.get("text", ""))
        if text and text.strip():
            yield text.strip()
            count += 1
            if max_docs is not None and count >= max_docs:
                break

    print(f"[OSCAR-2301] Streamed {count} documents.")


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset streaming")
    parser.add_argument(
        "--dataset", choices=["indiccorp", "oscar"], default="indiccorp",
        help="Which dataset to test-stream"
    )
    parser.add_argument(
        "--max-docs", type=int, default=5,
        help="Number of documents to fetch"
    )
    args = parser.parse_args()

    gen = stream_indiccorp(max_docs=args.max_docs) if args.dataset == "indiccorp" \
          else stream_oscar(max_docs=args.max_docs)

    for i, text in enumerate(gen, 1):
        print(f"\n--- Doc {i} ---")
        print(text[:300])
        print("...")
