# Assignment 7 — Hindi POS tagging

Tag each word of a Hindi sentence. The data is the IIIT-Hyderabad style tagset (24 tags) with a blank line between sentences.

| Split | Sentences | Tokens |
|-------|----------:|-------:|
| Train | 15,087 | 325,710 |
| Dev | 1,866 | 40,762 |
| Test | 1,910 | 41,415 |

Both models use add-one smoothing. A word that never occurs in training is mapped to `<UNK>`.

## Naive Bayes

The class is the tag. The five features from the spec are:

| Feature | Values |
|---------|--------|
| Length | `MORE` if the word has more than 4 characters, otherwise `LESS` |
| Word | the word itself |
| Previous word | `<s>` at the start of the sentence |
| Next word | `</s>` at the end of the sentence |
| Previous tag | `<s>` for the first word |

\[
P(t \mid \text{features}) \propto P(t)\, P(\text{len}\mid t)\, P(w\mid t)\, P(w_{prev}\mid t)\, P(w_{next}\mid t)\, P(t_{prev}\mid t)
\]

Tags are chosen left to right, so the previous-tag feature at test time is the tag just predicted, not the gold tag.

## HMM

States are tags, observations are words. Transitions are \(P(t_i \mid t_{i-1})\) with a start tag `<s>` and an end tag `</s>`.

- **Forward Viterbi** is the usual left-to-right decode. The best final tag also includes \(P({</s>}\mid t_n)\).
- **Backward Viterbi** runs the same algorithm on the reversed sentence. Its transitions are estimated on reversed tag sequences, so they are \(P(t_i \mid t_{i+1})\).
- **Forward-backward Viterbi** does not return one path. At each position it scores a tag by the best path reaching it from the left plus the best path reaching it from the right, subtracting the emission once because both sides include it, and picks the tag on its own.

Forward Viterbi was checked against an exhaustive search on a 3-word sentence.

## Results

| Method | Dev accuracy | Dev macro-F1 | Test accuracy | Test macro-F1 |
|--------|-------------:|-------------:|--------------:|--------------:|
| Naive Bayes | 91.02% | 74.34% | 91.04% | 75.99% |
| HMM forward | 91.07% | 84.31% | 91.19% | 84.81% |
| HMM backward | 91.13% | 84.42% | **91.24%** | **85.06%** |
| HMM forward-backward | 89.16% | 67.55% | 89.35% | 67.87% |

Accuracy is close for all four, because the common tags (`N_NN`, `PSP`, `N_NNP`, `V_VM`) dominate the token count. Macro-F1 is the fairer comparison. The two directional Viterbi decoders win there: the tag of a rare word is much easier to get right when the neighbouring tag is part of one consistent path. Forward-backward tagging chooses each position independently, and a rare tag has to be on the best path from both directions at once, so tags such as `RB`, `PR_PRL`, `QT_QTO` and `RP_INTF` collapse. Naive Bayes is close to the HMM on frequent tags and weaker on the rare ones, which is what the length and lexical features can and cannot see.

Per-tag test F1 is in `outputs/pos_f1_per_tag.txt`. Charts:

- `outputs/accuracy_comparison.svg`
- `outputs/f1_per_tag_test.svg`

## Run

```bash
cd Assignment_7
# data/*.txt is extracted from Lab-Assignment-7.zip (git-ignored)
python run_assignment7.py
```

Needs numpy.
