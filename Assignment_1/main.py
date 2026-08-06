"""
main.py
=======
Orchestrator for NLP Assignment 1 — Text Preprocessing Pipeline.

Runs the full pipeline for both datasets:
  1. IndicCorpV2 (Hindi — hin_Deva)
  2. OSCAR-2301  (Hindi — hi)

Steps per dataset:
  a. Stream raw text from HuggingFace
  b. Tokenize into sentences + words
  c. Save tokenized sentences to Parquet files
  d. Compute and display corpus statistics

Usage
-----
    # Full run (100K docs each)
    python main.py

    # Quick test run (1K docs each)
    python main.py --max-docs 1000

    # Only one dataset
    python main.py --only indiccorp
    python main.py --only oscar

    # Custom output directory
    python main.py --output-dir my_output

Environment variables
---------------------
    HF_TOKEN   — HuggingFace access token (required for OSCAR-2301 gated access)
                 Alternative: run `huggingface-cli login` before this script.
"""

import os
import sys
import argparse
import time

# Make sure we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import stream_indiccorp, stream_oscar
from preprocess import preprocess_and_save, BATCH_SIZE
from stats import compute_stats, print_stats, save_stats_to_file


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_DOCS = 100_000   # Documents to process per dataset
DEFAULT_OUTPUT   = "output"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    dataset: str,
    output_dir: str,
    max_docs: int,
    batch_size: int,
) -> dict:
    """
    Run the full pipeline for one dataset.

    Parameters
    ----------
    dataset : str
        'indiccorp' or 'oscar'
    output_dir : str
        Root output directory; a subdirectory will be created.
    max_docs : int
        Max source documents to process.
    batch_size : int
        Sentences per parquet file.

    Returns
    -------
    dict
        Corpus statistics.
    """
    if dataset == "indiccorp":
        parquet_dir  = os.path.join(output_dir, "indiccorp")
        display_name = "IndicCorpV2 — Hindi (hin_Deva)"
        text_gen     = stream_indiccorp(max_docs=max_docs)
    else:
        parquet_dir  = os.path.join(output_dir, "oscar")
        display_name = "OSCAR-2301 — Hindi (hi)"
        text_gen     = stream_oscar(max_docs=max_docs)

    header = f"\n{'#'*60}\n  Pipeline: {display_name}\n{'#'*60}"
    print(header)

    # ---- Step 1 + 2 + 3: Stream → Tokenize → Save ----
    t0 = time.time()
    preprocess_and_save(
        text_generator=text_gen,
        output_dir=parquet_dir,
        max_docs=None,          # already capped in generator
        batch_size=batch_size,
        dataset_name=display_name,
    )
    elapsed_preprocess = time.time() - t0
    print(f"  Preprocessing time: {elapsed_preprocess:.1f}s")

    # ---- Step 4: Compute statistics ----
    t0 = time.time()
    stats = compute_stats(parquet_dir=parquet_dir, dataset_name=display_name)
    elapsed_stats = time.time() - t0

    print_stats(stats)
    print(f"  Stats computation time: {elapsed_stats:.1f}s")

    # Save stats CSV alongside parquet files
    csv_path = os.path.join(parquet_dir, "corpus_stats.csv")
    save_stats_to_file(stats, csv_path)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NLP Assignment 1 — Full Text Preprocessing Pipeline (Hindi)"
    )
    parser.add_argument(
        "--max-docs", type=int, default=DEFAULT_MAX_DOCS,
        help=f"Max source documents per dataset (default: {DEFAULT_MAX_DOCS}). "
             "Set to 0 for no limit."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help=f"Root directory for output parquet files (default: '{DEFAULT_OUTPUT}')."
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sentences per parquet file (default: {BATCH_SIZE})."
    )
    parser.add_argument(
        "--only", choices=["indiccorp", "oscar"], default=None,
        help="Run only one dataset's pipeline. Omit to run both."
    )
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs > 0 else None

    all_stats = {}

    t_start = time.time()

    datasets_to_run = [args.only] if args.only else ["indiccorp", "oscar"]

    for ds in datasets_to_run:
        stats = run_pipeline(
            dataset=ds,
            output_dir=args.output_dir,
            max_docs=max_docs,
            batch_size=args.batch_size,
        )
        all_stats[ds] = stats

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All pipelines complete in {total_time:.1f}s")
    print(f"  Output saved to: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}\n")

    # Final summary comparison (if both ran)
    if len(all_stats) == 2:
        print("\n  SUMMARY COMPARISON")
        print(f"  {'Metric':<35} {'IndicCorpV2':>15} {'OSCAR-2301':>15}")
        print(f"  {'-'*35} {'-'*15} {'-'*15}")
        metrics = [
            ("total_sentences",     "Total Sentences"),
            ("total_words",         "Total Words"),
            ("total_chars",         "Total Characters"),
            ("avg_sentence_length", "Avg Sentence Length"),
            ("avg_word_length",     "Avg Word Length"),
            ("ttr",                 "Type/Token Ratio"),
        ]
        ic = all_stats.get("indiccorp", {})
        os_ = all_stats.get("oscar", {})

        def _fmt(v) -> str:
            """Format a stat value for display."""
            if isinstance(v, float):
                return f"{v:>15.4f}"
            elif isinstance(v, int):
                return f"{v:>15,}"
            else:
                return f"{'N/A':>15}"

        for key, label in metrics:
            v1 = ic.get(key, "N/A")
            v2 = os_.get(key, "N/A")
            print(f"  {label:<35} {_fmt(v1)} {_fmt(v2)}")
        print()


if __name__ == "__main__":
    main()
