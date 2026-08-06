"""
stats.py
========
Compute corpus statistics from tokenized Parquet files.

Statistics computed:
  i.   Total number of sentences
  ii.  Total number of words (tokens)
  iii. Total number of characters (across all tokens, not counting spaces)
  iv.  Average sentence length (avg words per sentence)
  v.   Average word length (avg characters per word)
  vi.  Type/Token Ratio (TTR) = unique tokens / total tokens

Usage (standalone)
------------------
    python stats.py --input output/indiccorp --name "IndicCorpV2 Hindi"
    python stats.py --input output/oscar     --name "OSCAR-2301 Hindi"
"""

import os
import sys
import argparse
import glob
from typing import Dict, Any

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] Missing dependencies. Run: pip install pandas pyarrow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Core statistics computation
# ---------------------------------------------------------------------------

def compute_stats(parquet_dir: str, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Read all parquet files in a directory and compute corpus statistics.

    Parameters
    ----------
    parquet_dir : str
        Path to directory containing tokenized parquet files.
    dataset_name : str
        Label for display purposes.

    Returns
    -------
    dict with keys:
        total_sentences, total_words, total_chars,
        avg_sentence_length, avg_word_length, ttr,
        unique_types, dataset_name
    """
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    if not parquet_files:
        print(f"[WARN] No parquet files found in: {parquet_dir}")
        return {}

    print(f"\n[{dataset_name}] Computing stats from {len(parquet_files)} parquet file(s)...")

    total_sentences = 0
    total_words = 0
    total_chars = 0
    # For TTR we need to track unique tokens — use a set
    # For very large corpora this may use significant memory.
    # An alternative would be HyperLogLog, but a set is exact and fine for 100K docs.
    unique_tokens: set = set()

    file_iter = tqdm(parquet_files, desc="Reading parquet") if HAS_TQDM else parquet_files

    for fpath in file_iter:
        df = pq.read_table(fpath).to_pandas()

        for sent_str in df["tokenized_sentence"]:
            if not isinstance(sent_str, str) or not sent_str.strip():
                continue

            tokens = sent_str.split()   # space-separated tokens from tokenizer
            num_words = len(tokens)

            # Character count: sum of lengths of individual tokens (no spaces)
            num_chars = sum(len(t) for t in tokens)

            total_sentences += 1
            total_words += num_words
            total_chars += num_chars
            unique_tokens.update(tokens)

    unique_types = len(unique_tokens)

    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0.0
    avg_word_length = total_chars / total_words if total_words > 0 else 0.0
    ttr = unique_types / total_words if total_words > 0 else 0.0

    stats = {
        "dataset_name":         dataset_name,
        "total_sentences":      total_sentences,
        "total_words":          total_words,
        "total_chars":          total_chars,
        "avg_sentence_length":  round(avg_sentence_length, 4),
        "avg_word_length":      round(avg_word_length, 4),
        "unique_types":         unique_types,
        "ttr":                  round(ttr, 6),
    }

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Pretty-print a stats dictionary."""
    if not stats:
        print("No stats to display.")
        return

    name = stats.get("dataset_name", "Dataset")
    sep = "=" * 55

    print(f"\n{sep}")
    print(f"  Corpus Statistics — {name}")
    print(sep)
    print(f"  {'Statistic':<35} {'Value':>15}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'i.  Total Sentences':<35} {stats['total_sentences']:>15,}")
    print(f"  {'ii. Total Words (Tokens)':<35} {stats['total_words']:>15,}")
    print(f"  {'iii.Total Characters':<35} {stats['total_chars']:>15,}")
    print(f"  {'iv. Avg Sentence Length (words)':<35} {stats['avg_sentence_length']:>15.4f}")
    print(f"  {'v.  Avg Word Length (chars)':<35} {stats['avg_word_length']:>15.4f}")
    print(f"  {'vi. Unique Types':<35} {stats['unique_types']:>15,}")
    print(f"  {'    Type/Token Ratio (TTR)':<35} {stats['ttr']:>15.6f}")
    print(sep)


def save_stats_to_file(stats: Dict[str, Any], output_path: str) -> None:
    """Save stats as a CSV file for easy reference."""
    df = pd.DataFrame([stats])
    df.to_csv(output_path, index=False)
    print(f"\n  Stats saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute corpus statistics from tokenized Parquet files."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to directory containing tokenized parquet files."
    )
    parser.add_argument(
        "--name", type=str, default="Dataset",
        help="Display name for the dataset."
    )
    parser.add_argument(
        "--save-csv", type=str, default=None,
        help="Optional path to save statistics as a CSV file."
    )
    args = parser.parse_args()

    stats = compute_stats(parquet_dir=args.input, dataset_name=args.name)
    print_stats(stats)

    if args.save_csv:
        save_stats_to_file(stats, args.save_csv)


if __name__ == "__main__":
    main()
