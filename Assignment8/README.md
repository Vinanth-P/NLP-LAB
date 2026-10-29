# Assignment 8: Spell Correction and NLP Programs

This assignment contains:

1. Programs solving the five questions from the supplied NLP midsemester
   question paper.
2. A noisy-channel spell-correction model evaluated on the labelled typo pairs
   provided with the assignment.

The complete implementation and outputs are in
[`NLP_Assignment8_Solutions_Optimized.ipynb`](./NLP_Assignment8_Solutions_Optimized.ipynb).
The assignment brief is available in
[`NLP-Lab-Assignment-8-2026.pdf`](./NLP-Lab-Assignment-8-2026.pdf), and the
midsemester questions are in
[`NLP-Midsem-Questions-Monsoon-2026.pdf`](./NLP-Midsem-Questions-Monsoon-2026.pdf).

## Part A: Midsemester question programs

### Q1. Finite-state transducer for English noun number

The FST supports lexical-to-surface generation and surface-to-lexical
analysis. It copies the stem and marks the lexical representation with
`+N+SG` or `+N+PL`.

The plural rule is:

- Append `es` when the stem ends in `s`, `x`, `z`, `ch`, or `sh`.
- Append `s` otherwise.

| State | Input or condition | Next state | Output or action |
|---|---|---|---|
| `q0` | Stem characters | `q1` | Copy characters |
| `q1` | `+N+SG` | `qf` | Emit nothing; singular stem is unchanged |
| `q1` | `+N+PL` | `q_regular` | Inspect the stem ending |
| `q_regular` | Stem ends in `s/x/z/ch/sh` | `qf` | Emit `es` |
| `q_regular` | Otherwise | `qf` | Emit `s` |

Generated examples:

| Stem | Singular | Plural |
|---|---|---|
| `cat` | `cat` | `cats` |
| `bus` | `bus` | `buses` |
| `box` | `box` | `boxes` |
| `quiz` | `quiz` | `quizes` |
| `church` | `church` | `churches` |
| `dish` | `dish` | `dishes` |

The `quiz -> quizes` output follows the assignment's stated suffix rule; it
does not attempt to model every irregular English spelling convention.

### Q2. WordPiece learning and tokenization

The five-line corpus from the question is represented initially as character
tokens. Non-initial characters use the `##` continuation marker. Two adjacent
tokens are merged at each step using:

```text
score(left, right) =
    frequency(left, right) /
    (frequency(left) * frequency(right))
```

Ties are resolved deterministically using pair count and then lexicographic
pair order.

The learned merges were:

| Merge | Operation | Pair count | Score |
|---:|---|---:|---:|
| 1 | `##s + ##t -> ##st` | 3 | 0.333333 |
| 2 | `l + ##o -> lo` | 10 | 0.100000 |

The resulting greedy tokenization of `newest` is:

```text
['n', '##e', '##w', '##e', '##st']
```

### Q3. Raw-count TF-IDF and closest sentences

For each sentence, the implementation uses raw term frequency and unsmoothed
inverse document frequency:

```text
TF(t, d) = count of t in sentence d
IDF(t) = ln(N / df(t))
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

The resulting IDF values are:

| Term | `animal` | `cat` | `dog` | `pet` |
|---|---:|---:|---:|---:|
| IDF | 0.693147 | 0.287682 | 0.287682 | 0.287682 |

TF-IDF vectors:

| Sentence | `animal` | `cat` | `dog` | `pet` |
|---|---:|---:|---:|---:|
| S1 | 0.693147 | 0.287682 | 0.287682 | 0.000000 |
| S2 | 0.000000 | 0.575364 | 0.000000 | 0.287682 |
| S3 | 0.693147 | 0.000000 | 0.287682 | 0.287682 |
| S4 | 0.000000 | 0.287682 | 0.287682 | 0.287682 |

Cosine similarities:

| Sentence pair | Cosine similarity |
|---|---:|
| S1, S3 | **0.871882** |
| S2, S4 | 0.774597 |
| S3, S4 | 0.413308 |
| S1, S4 | 0.413308 |
| S1, S2 | 0.320147 |
| S2, S3 | 0.160074 |

The closest pair is **S1 and S3**, with cosine similarity `0.871882`.

### Q4. Bernoulli Naive Bayes spam classification

The question classifies `E8 = (0, 0, 0)` for the binary features
`free`, `win`, and `offer`. Since each feature represents presence or absence,
the implementation uses Bernoulli Naive Bayes without smoothing.

| Class | Prior | `P(free=0)` | `P(win=0)` | `P(offer=0)` | Unnormalized posterior |
|---|---:|---:|---:|---:|---:|
| Spam | 0.571429 | 0.250000 | 0.250000 | 0.250000 | 0.008929 |
| NotSpam | 0.428571 | 0.666667 | 0.666667 | 0.666667 | 0.126984 |

Normalized posteriors:

| Class | Posterior |
|---|---:|
| Spam | 0.065693 |
| NotSpam | **0.934307** |

**Prediction for E8:** `NotSpam`.

### Q5. Kneser–Ney smoothing

The implementation applies interpolated Kneser–Ney smoothing with discount
`d = 0.5` to the supplied bigram examples:

```text
P_KN(w | h) =
    max(c(h,w) - d, 0) / c(h)
    + lambda(h) * P_cont(w)

lambda(h) = d * number_of_unique_followers(h) / c(h)
P_cont(w) = number_of_unique_predecessors(w) / number_of_distinct_bigrams
```

There are 8 distinct bigram types. The continuation probabilities for `AI`
and `NLP` are both `0.25`.

| Probability | Value |
|---|---:|
| `P_KN(AI \| love)` | **0.375000** |
| `P_KN(NLP \| like)` | **0.583333** |

## Part B: Noisy-channel spell correction

### Model

For an observed misspelling `x`, the correct word is selected using:

```text
w_hat = argmax_w P(x | w) * P(w)
```

- `P(w)` is estimated from word frequencies in `count_unigram.txt`.
- `P(x | w)` is estimated from edit-operation counts in
  `count_1edit.txt`.
- Candidate corrections are generated at edit distance 1 and retained only
  when they occur in the unigram vocabulary.
- Candidates are ranked by the noisy-channel log score. Ties are resolved
  lexicographically.

The edit-count file is interpreted as `source|target<TAB>count`, for example:

```text
e|i    917
```

The evaluator also expands entries such as `fore*5` in
`spell-errors.txt` into repeated labelled typo pairs.

### Input data and evaluation

| Data file | Entries or size |
|---|---:|
| `count_unigram.txt` | 333,333 vocabulary entries |
| `count_1edit.txt` | 1,512 edit-operation entries |
| `spell-errors.txt` | 41,718 labelled correct-word/typo pairs evaluated |

The sample demonstration produces:

```text
rainning -> raining
```

### Results

The measured result is:

| Metric | Result |
|---|---:|
| Vocabulary size | 333,333 |
| Labelled pairs evaluated | 41,718 |
| Top-1 accuracy | **18.37%** |

This is top-1 accuracy on the labelled typo pairs in `spell-errors.txt`. The
provided materials do not define a separate train/dev/test split, so this
should not be interpreted as performance on an independent held-out test set.

Example behaviours from the notebook include:

| Correct word | Misspelling | Prediction | Correct? |
|---|---|---|---|
| `raining` | `rainning` | `rainning` | No |
| `raining` | `raning` | `ranking` | No |
| `writings` | `writtings` | `writings` | Yes |
| `disparagingly` | `disparingly` | No candidate | No |
| `woods` | `woodes` | `woods` | Yes |
| `hanging` | `haing` | `having` | No |

The low accuracy reflects the limitations of restricting candidates to
single-edit vocabulary words and estimating the channel from aggregate edit
operation counts. A frequent word or an incorrect candidate can outrank the
gold correction, and some typos have no candidate within the implemented
search space.

## Output files

| File | Description |
|---|---|
| [`NLP_Assignment8_Solutions_Optimized.ipynb`](./NLP_Assignment8_Solutions_Optimized.ipynb) | All source code, outputs, tables, and plots |
| [`Spell-Correction/count_unigram.txt`](./Spell-Correction/count_unigram.txt) | Word-frequency vocabulary used for priors |
| [`Spell-Correction/count_1edit.txt`](./Spell-Correction/count_1edit.txt) | Edit-operation counts used for channel probabilities |
| [`Spell-Correction/spell-errors.txt`](./Spell-Correction/spell-errors.txt) | Labelled correct-word and misspelling pairs |

The notebook displays a spell-correction correct/incorrect count chart after
the evaluation. The numerical results are printed in the notebook output; no
separate result CSV is generated.

## How to run

From the `Assignment8` directory, open and run all cells in
`NLP_Assignment8_Solutions_Optimized.ipynb`. The notebook automatically checks
for the spell-correction files under `Spell-Correction/`. If it is launched
from another directory, update the `DATA_DIR` path in the first setup section.

Required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

The five midsemester question programs are self-contained. The spell
correction section requires all three files in `Spell-Correction/`.

## Conclusion

The notebook provides reproducible implementations for finite-state
morphology, WordPiece learning, TF-IDF similarity, Bernoulli Naive Bayes, and
Kneser–Ney smoothing. The noisy-channel spell corrector demonstrates the
complete candidate-generation, scoring, and evaluation pipeline, achieving
18.37% top-1 accuracy on the supplied labelled typo pairs. Its main
limitations are the single-edit candidate restriction, aggregate channel
counts, and the absence of an independent evaluation split.
