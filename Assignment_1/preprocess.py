"""
preprocess.py
=============
Runs the full preprocessing pipeline:
  1. Streams raw text from a dataset generator
  2. Tokenizes each document into sentences, then words
  3. Writes tokenized sentences to Parquet files in batches

Output format (one Parquet file per batch of BATCH_SIZE sentences):
  Schema:
    sentence_id        : int64  — global monotonically increasing id
    tokenized_sentence : string — words joined by spaces

Usage (standalone)
------------------
    python preprocess.py --dataset indiccorp --max-docs 100000 --output output/indiccorp
    python preprocess.py --dataset oscar     --max-docs 100000 --output output/oscar
"""

import os
import sys
import argparse
from typing import Iterator, List, Optional

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] Missing dependencies. Run: pip install pandas pyarrow")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from tokenizer import tokenize_paragraph_to_strings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 10_000   # sentences per Parquet file


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def preprocess_and_save(
    text_generator: Iterator[str],
    output_dir: str,
    max_docs: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    dataset_name: str = "dataset",
) -> int:
    """
    Stream text → tokenize → write parquet in batches.

    Parameters
    ----------
    text_generator : Iterator[str]
        Generator that yields raw document/paragraph strings.
    output_dir : str
        Directory path where parquet files will be written.
    max_docs : int, optional
        Max number of source documents to process. None = all.
    batch_size : int
        Number of tokenized sentences per parquet file.
    dataset_name : str
        Label used in progress messages.

    Returns
    -------
    int
        Total number of tokenized sentences written.
    """
    os.makedirs(output_dir, exist_ok=True)

    batch: List[dict] = []
    total_sentences = 0
    file_index = 0
    doc_count = 0

    # PyArrow schema
    schema = pa.schema([
        pa.field("sentence_id", pa.int64()),
        pa.field("tokenized_sentence", pa.string()),
    ])

    def flush_batch(b: List[dict], fidx: int) -> str:
        """Write a batch to a parquet file and return the filename."""
        df = pd.DataFrame(b, columns=["sentence_id", "tokenized_sentence"])
        fname = os.path.join(output_dir, f"tokenized_{fidx:05d}.parquet")
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, fname, compression="snappy")
        return fname

    print(f"\n[{dataset_name}] Starting preprocessing → {output_dir}")

    gen = tqdm(text_generator, desc=f"[{dataset_name}] Docs", unit="doc") \
          if HAS_TQDM else text_generator

    for doc_text in gen:
        doc_count += 1
        tokenized_sentences = tokenize_paragraph_to_strings(doc_text)

        for sent_str in tokenized_sentences:
            if not sent_str.strip():
                continue
            batch.append({
                "sentence_id": total_sentences,
                "tokenized_sentence": sent_str,
            })
            total_sentences += 1

            # Flush when batch is full
            if len(batch) >= batch_size:
                fname = flush_batch(batch, file_index)
                print(f"  Wrote {fname}  ({len(batch)} sentences, cumulative: {total_sentences})")
                file_index += 1
                batch = []

        if max_docs is not None and doc_count >= max_docs:
            break

    # Flush remaining
    if batch:
        fname = flush_batch(batch, file_index)
        print(f"  Wrote {fname}  ({len(batch)} sentences, cumulative: {total_sentences})")

    print(f"\n[{dataset_name}] Done. Processed {doc_count} docs → {total_sentences} sentences "
          f"across {file_index + (1 if batch or total_sentences > 0 else 0)} files.")
    return total_sentences


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize and save IndicCorpV2 or OSCAR-2301 Hindi data to Parquet."
    )
    parser.add_argument(
        "--dataset", choices=["indiccorp", "oscar"], required=True,
        help="Which dataset to preprocess."
    )
    parser.add_argument(
        "--max-docs", type=int, default=100_000,
        help="Max number of source documents to process (default: 100000). "
             "Use 0 for no limit."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for parquet files. "
             "Defaults to output/indiccorp or output/oscar."
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Sentences per parquet file (default: {BATCH_SIZE})."
    )
    args = parser.parse_args()

    max_docs = args.max_docs if args.max_docs > 0 else None

    # Import lazily (so tokenizer tests don't need datasets installed)
    from download_data import stream_indiccorp, stream_oscar

    if args.dataset == "indiccorp":
        output_dir = args.output or os.path.join("output", "indiccorp")
        gen = stream_indiccorp(max_docs=max_docs)
        dataset_name = "IndicCorpV2 (hin_Deva)"
    else:
        output_dir = args.output or os.path.join("output", "oscar")
        gen = stream_oscar(max_docs=max_docs)
        dataset_name = "OSCAR-2301 (hi)"

    preprocess_and_save(
        text_generator=gen,
        output_dir=output_dir,
        max_docs=None,        # already limited in generator
        batch_size=args.batch_size,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()
