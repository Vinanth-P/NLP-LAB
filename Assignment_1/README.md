# Assignment 1 — Text Preprocessing

**Course**: Natural Language Processing  
**Dataset 1**: [ai4bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) — Hindi (`hin_Deva`)  
**Dataset 2**: [oscar-corpus/OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) — Hindi (`hi`)

---

## Overview

This assignment implements a complete text preprocessing pipeline for Hindi monolingual corpora:

1. **Data Download** — Stream large datasets from HuggingFace without downloading them fully
2. **Sentence Tokenization** — Split paragraphs into sentences (handles Devanagari danda `।`)
3. **Word Tokenization** — Tokenize sentences into words, with special handling for:
   - URLs (`https://example.com`)
   - Email IDs (`user@domain.com`)
   - Dates (`15/08/1947`, `2024-03-15`)
   - Decimal numbers (`3.14`, `36.5`)
   - Integer numbers (`2024`, `१०`)
   - Hindi/Devanagari words (`नमस्ते`)
   - ASCII words (`India`, `GDP`)
   - Punctuation (each as a separate token)
4. **Parquet Storage** — Save tokenized data in compressed Parquet format (Snappy compression)
5. **Corpus Statistics** — Compute 6 key statistics on the tokenized corpus

---

## File Structure

```
Assignment_1/
├── tokenizer.py        # Core sentence + word tokenizer (no external NLP libs)
├── download_data.py    # HuggingFace streaming data downloader
├── preprocess.py       # Tokenization + Parquet writing pipeline
├── stats.py            # Corpus statistics computation
├── main.py             # Orchestrator — runs everything end-to-end
├── output/
│   ├── indiccorp/      # Parquet files for IndicCorpV2 Hindi
│   │   ├── tokenized_00000.parquet
│   │   ├── tokenized_00001.parquet
│   │   └── corpus_stats.csv
│   └── oscar/          # Parquet files for OSCAR-2301 Hindi
│       ├── tokenized_00000.parquet
│       └── corpus_stats.csv
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install datasets pandas pyarrow tqdm huggingface_hub
```

### 2. Authenticate with HuggingFace (required for OSCAR-2301)

OSCAR-2301 is a gated dataset. You must:
1. Create a HuggingFace account at https://huggingface.co
2. Request access to the dataset at https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
3. Authenticate locally:

```bash
huggingface-cli login
# OR set the environment variable:
set HF_TOKEN=your_token_here   # Windows
```

> IndicCorpV2 does **not** require authentication.

---

## How to Run

### Quick test (1,000 documents — runs in ~1 minute)

```bash
$env:PYTHONUTF8="1"
python main.py --max-docs 1000
```

### Full run (100,000 documents — recommended for submission)

```bash
python main.py --max-docs 100000
```

### Run only one dataset

```bash
python main.py --only indiccorp --max-docs 100000
python main.py --only oscar     --max-docs 100000
```

### Run individual modules

```bash
# Test the tokenizer
python tokenizer.py

# Test streaming (fetches 5 documents)
python download_data.py --dataset indiccorp --max-docs 5

# Preprocess and save to parquet
python preprocess.py --dataset indiccorp --max-docs 100000

# Compute stats on already-processed data
python stats.py --input output/indiccorp --name "IndicCorpV2 Hindi"
```

---

## Output Format

Each parquet file contains tokenized sentences in the following schema:

| Column               | Type   | Description                                   |
|----------------------|--------|-----------------------------------------------|
| `sentence_id`        | int64  | Global monotonically increasing sentence index |
| `tokenized_sentence` | string | Space-separated tokens                        |

### Example rows

| sentence_id | tokenized_sentence                                          |
|-------------|-------------------------------------------------------------|
| 0           | `भारत की राजधानी नई दिल्ली है ।`                          |
| 1           | `जनसंख्या 1.4 अरब से अधिक है ।`                           |
| 2           | `वेबसाइट https://www.india.gov.in पर जाएं ।`              |
| 3           | `स्वतंत्रता दिवस 15/08/1947 को मनाया जाता है ।`          |

### Reading parquet files in Python

```python
import pandas as pd

df = pd.read_parquet("output/indiccorp/tokenized_00000.parquet")
print(df.head())
print(f"Sentences in this file: {len(df)}")
```

---

## Corpus Statistics

Statistics are computed after tokenization and also saved as `corpus_stats.csv` in each output folder.

### Metrics

| # | Statistic              | Description                                        |
|---|------------------------|----------------------------------------------------|
| i | Total Sentences        | Count of all tokenized sentences                   |
| ii | Total Words           | Total number of tokens across all sentences        |
| iii | Total Characters     | Sum of character lengths of all tokens (no spaces) |
| iv | Avg Sentence Length   | Total words ÷ Total sentences                      |
| v | Avg Word Length        | Total characters ÷ Total words                     |
| vi | Type/Token Ratio (TTR) | Unique tokens ÷ Total tokens                     |

### Sample output

```
=======================================================
  Corpus Statistics — IndicCorpV2 — Hindi (hin_Deva)
=======================================================
  Statistic                             Value
  ----------------------------------- ---------------
  i.  Total Sentences                     2,341,087
  ii. Total Words (Tokens)               28,912,340
  iii.Total Characters                  138,430,512
  iv. Avg Sentence Length (words)             12.3497
  v.  Avg Word Length (chars)                  4.7882
  vi. Unique Types                        1,240,567
      Type/Token Ratio (TTR)               0.042909
=======================================================
```

---

## Tokenizer Design

The tokenizer (`tokenizer.py`) is entirely rule-based using Python `re` (no NLTK, no spaCy).

### Sentence Tokenizer

Uses a regex that splits **after** sentence-ending punctuation:
- `।` — Devanagari danda (primary Hindi sentence terminator)
- `?` — question mark
- `!` — exclamation mark
- `.` — full stop, **but only** when not part of a decimal number or abbreviation

### Word Tokenizer

Regex patterns applied in **priority order** (most specific first):

```
Priority  Pattern type      Example match
--------  ----------------  ---------------------------
1         URL               https://www.example.com
2         Email             user@domain.com
3         Date              15/08/1947  or  2024-03-15
4         Decimal number    3.14  or  36.5
5         Integer number    2024  or  १०
6         Hindi word        नमस्ते
7         ASCII word        India
8         Punctuation       ,  ।  !  ?  (  )
```

---

## About Parquet Format

[Apache Parquet](https://parquet.apache.org/) is a columnar storage format designed for large datasets. Advantages:
- **Compression**: Snappy compression reduces file size by ~5–10× vs plain text
- **Fast reads**: Columnar layout allows reading only the columns you need
- **Interoperability**: Works with pandas, PySpark, DuckDB, and more

Files are written in batches of **10,000 sentences** each, making them easy to process incrementally.

---

## References

1. Doddapaneni et al. (2023). *Towards Leaving No Indic Language Behind*. ACL 2023.  
   Dataset: https://huggingface.co/datasets/ai4bharat/IndicCorpV2

2. Abadji et al. (2022). *Towards a Cleaner Document-Oriented Multilingual Crawled Corpus*. LREC 2022.  
   Dataset: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
