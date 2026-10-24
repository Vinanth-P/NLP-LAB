# Assignment 8 — Mid-semester questions and spell correction

## 1. Mid-semester questions

`solve_midsem.py` prints the full working for all five questions. The saved run is `outputs/midsem_solutions.txt`.

| Q | Answer |
|---|--------|
| 1 | FST that copies a noun stem and writes `es` after `s/x/z/ch/sh`, otherwise `s`. `fox+N+PL` → `foxes`, `cat+N+PL` → `cats`, `church+N+PL` → `churches`. Without a lexicon the analyser is ambiguous: `cats` is both `cat+N+PL` and `cats+N+SG`. `foxs` is not the plural of `fox`. |
| 2 | WordPiece score of a pair is \(\mathrm{count}(ab)/(\mathrm{count}(a)\mathrm{count}(b))\). Merge 1 is `s+t → st` (score 0.333). Merge 2 is `l+o → lo` (score 0.100). **`newest` → `n e w e st`.** BPE, which merges by raw count instead, merges `lo` then `low` and leaves `newest` as six characters. The question asks for WordPiece, so the first of those is the answer. |
| 3 | Unsmoothed IDF is \(\log_{10}(4/\mathrm{df})\). Cosine similarity is highest for **d1 `cat dog animal` and d3 `dog pet animal`**, at 0.872. |
| 4 | E8 = (free=0, win=0, offer=0). \(P(\text{Spam})P(E8\mid\text{Spam}) = 1/112\), \(P(\text{NotSpam})P(E8\mid\text{NotSpam}) = 8/63\). **Predicted NotSpam.** No smoothing is needed: none of these feature values has a zero count. |
| 5 | No start or end markers, \(d = 0.5\). **\(P_{KN}(\text{AI}\mid\text{love}) = 3/8\)**, **\(P_{KN}(\text{NLP}\mid\text{like}) = 7/12\)**. |

## 2. Noisy-channel spell correction

`spell-errors.txt` pairs a correct word with the ways it was mistyped. A trailing `*N` means that mistake was seen N times. `count_1edit.txt` counts one edit at a time; the left side of `X|Y` is the correct fragment and the right side is what was typed (`e|i`, `re|r`, `t|te`, `er|re`). `>` is the start-of-word marker. `count_unigram.txt` is the word frequency list.

\[
\hat{c} = \arg\max_c P(w \mid c)\, P(c)
\]

\(P(c)\) is the unigram frequency. \(P(w \mid c)\) is the probability of the single edit that turns \(c\) into the typed word:

\[
P(w \mid c) = \frac{\mathrm{count}(\text{edit}) + 1}{\mathrm{count}(\text{correct letters}) + 1}
\]

The denominator is a character unigram for a substitution or insertion, and a character bigram for a deletion or transposition, counted from the unigram word list with `>` in front of every word. Candidates are dictionary words at Damerau-Levenshtein distance 1. The typed string is not itself a candidate, because every row of the error file is a mistake.

### Accuracy

| Set | Instances | Accuracy |
|-----|----------:|---------:|
| All error instances | 41,718 | 29.3% |
| One edit away from the gold word | 15,751 | **77.7%** |
| Same one-edit set, most frequent neighbour only | 15,751 | 71.4% |

Most of the file is more than one edit away (`fore` for `four`, `disparingly` for `disparagingly`). A one-edit channel cannot reach those, so the headline number is the one-edit accuracy. On that set the channel beats picking the most common dictionary neighbour by 6.3 points.

Some one-edit mistakes are still missed because `count_1edit.txt` has no row for them. Doubling a letter (`rainning`, `writtings`, `loking`) is the common case: the edit `n|nn` or `oo|o` is absent, so an attested edit into a frequent word wins instead (`rainning` → `trainning`).

## Run

```bash
cd Assignment_8
python solve_midsem.py
python spellcheck.py
```

The unigram file is git-ignored. It is inside `Lab-Assignment-8.zip` under `Spell-Correction/`. The script reads `data/count_unigram.txt`, `data/count_1edit.txt` and `data/spell-errors.txt`.

## Files

| File | What it does |
|------|----------------|
| `solve_midsem.py` | Questions 1–5 |
| `spellcheck.py` | Noisy-channel corrector and the accuracy evaluation |
| `outputs/midsem_solutions.txt` | Full written solutions |
| `outputs/spell_results.json` | Accuracies and a sample of mistakes |
| `outputs/spell_examples.txt` | First few error rows with the prediction |
