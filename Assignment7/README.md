# Assignment 7: Hindi POS Tagging

This assignment implements and evaluates four Hindi part-of-speech (POS) tagging
approaches using the provided train, development, and test splits:

1. A Naive Bayes classifier.
2. Standard forward Viterbi decoding for a Hidden Markov Model (HMM).
3. Backward Viterbi decoding for the same HMM.
4. Forward-backward HMM decoding.

The complete implementation is in
[`Hindi_POS_Tagging_Assignment7.ipynb`](./Hindi_POS_Tagging_Assignment7.ipynb).
The assignment requirements are provided in
[`NLP-Lab-Assignment-7-2026.pdf`](./NLP-Lab-Assignment-7-2026.pdf).

## Dataset

The data is in `word<TAB>POS-tag` format, with blank lines separating
sentences. The split sizes used by the notebook are:

| Split | Sentences | Tokens |
|---|---:|---:|
| Train | 15,087 | 325,710 |
| Development | 1,866 | 40,762 |
| Test | 1,910 | 41,415 |

The training data contains 24 distinct POS tags. The same training-derived tag
inventory is used when calculating development and test metrics.

## Methodology

### 1. Naive Bayes classifier

The classifier predicts one tag at a time using these features:

- Word length category: `MORE` when the word length is greater than 4,
  otherwise `LESS`.
- The current word.
- The previous word.
- The next word.
- The previous tag.

The categorical features are one-hot encoded and classified with
`MultinomialNB(alpha=0.1)`. Decoding is greedy: at position `i`, the previous
tag feature is the tag predicted at position `i-1`; the first token uses
`<BOS>`.

### 2. HMM tagger

The HMM estimates:

- Initial tag probabilities.
- Tag-to-tag transition probabilities.
- Tag-to-word emission probabilities.
- End-of-sentence probabilities.

Additive smoothing with `alpha=0.1` is applied to initial, transition,
emission, and end probabilities. Words occurring once or less in the training
data are mapped to `<UNK>` so that rare and unseen words receive a valid
emission distribution.

The three HMM decoders operate in log space:

- **Forward Viterbi:** max-product dynamic programming from left to right.
- **Backward Viterbi:** max-product dynamic programming from right to left,
  followed by path reconstruction.
- **Forward-backward:** sum-product forward and backward probabilities are
  combined to obtain token-level posterior scores; the highest-posterior tag
  is selected at each position.

## Evaluation

Each model is evaluated on exactly the same gold tokens and tag inventory.
The notebook reports token accuracy, macro-averaged precision/recall/F1,
weighted precision/recall/F1, and precision/recall/F1 for every tag.

### Overall results

Values below are rounded to four decimal places. The full-precision values are
available in [`summary_metrics.csv`](./pos_tagger_outputs/summary_metrics.csv).

| Split | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Dev | Naive Bayes | 0.9314 | 0.8848 | 0.8379 | 0.8528 | 0.9310 |
| Dev | Forward Viterbi | 0.9352 | 0.9351 | 0.9362 | 0.9352 | 0.9353 |
| Dev | Backward Viterbi | 0.9352 | 0.9351 | 0.9362 | 0.9352 | 0.9353 |
| Dev | Forward-Backward | **0.9357** | **0.9380** | **0.9357** | **0.9363** | **0.9358** |
| Test | Naive Bayes | 0.9319 | 0.8895 | 0.8410 | 0.8562 | 0.9316 |
| Test | Forward Viterbi | 0.9347 | 0.9054 | **0.8978** | **0.9004** | 0.9349 |
| Test | Backward Viterbi | 0.9347 | 0.9054 | **0.8978** | **0.9004** | 0.9349 |
| Test | Forward-Backward | **0.9351** | **0.9072** | 0.8921 | 0.8985 | **0.9353** |

### Per-tag results

The complete per-tag table, including precision, recall, F1, and support for
all 24 tags on both splits, is available in
[`per_tag_metrics.csv`](./pos_tagger_outputs/per_tag_metrics.csv).

Some notable observations from that table:

- On the development set, forward-backward decoding has the highest macro-F1
  at `0.9363`.
- On the test set, forward Viterbi and backward Viterbi tie for the highest
  macro-F1 at `0.9004`.
- Forward-backward obtains the highest test accuracy (`0.9351`) and weighted
  F1 (`0.9353`).
- Forward and backward Viterbi produce identical aggregate and per-tag scores
  for these data and model parameters.
- Naive Bayes has competitive token accuracy because frequent tags dominate
  the corpus, but its macro-F1 is lower (`0.8528` on development and
  `0.8562` on test), indicating weaker performance on less frequent tags.
- Frequent tags such as `RD_PUNC`, `PSP`, and `V_VAUX` receive consistently
  high F1 scores. Rare tags such as `RP_INJ`, `PR_PRQ`, and `RP_INTF` are
  more variable; their scores should be interpreted together with their small
  support.

### Charts

The notebook generates bar charts comparing the models on macro-F1 and
weighted-F1 for the development and test splits. These charts are displayed
when the notebook is run; the current output directory stores the numerical
CSV results rather than separate image files.

## Output files

| File | Description |
|---|---|
| [`Hindi_POS_Tagging_Assignment7.ipynb`](./Hindi_POS_Tagging_Assignment7.ipynb) | Source code, evaluation, tables, and plots |
| [`Hindi-POS-Data/hindi-train-pos.txt`](./Hindi-POS-Data/hindi-train-pos.txt) | Training split |
| [`Hindi-POS-Data/hindi-dev-pos.txt`](./Hindi-POS-Data/hindi-dev-pos.txt) | Development split |
| [`Hindi-POS-Data/hindi-test-pos.txt`](./Hindi-POS-Data/hindi-test-pos.txt) | Test split |
| [`pos_tagger_outputs/summary_metrics.csv`](./pos_tagger_outputs/summary_metrics.csv) | Overall metrics for every model and split |
| [`pos_tagger_outputs/per_tag_metrics.csv`](./pos_tagger_outputs/per_tag_metrics.csv) | Per-tag precision, recall, F1, and support |

## How to run

From this `Assignment7` directory, open and run all cells in
`Hindi_POS_Tagging_Assignment7.ipynb`. The notebook expects the three data
files under `Hindi-POS-Data/` and creates or updates the CSV files under
`pos_tagger_outputs/`.

The notebook uses:

- Python
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib

The required packages can be installed with:

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## Conclusion

The HMM-based methods outperform the feature-based Naive Bayes model on
macro-averaged metrics because they model tag-sequence dependencies directly.
For this run, forward-backward decoding is the strongest overall choice on the
development set and gives the best test accuracy and weighted F1. Forward and
backward Viterbi are equally effective on the supplied data, while Naive Bayes
remains a strong baseline for token accuracy but is less balanced across POS
tags.
