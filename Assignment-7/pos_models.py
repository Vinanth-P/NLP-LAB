"""Hindi POS taggers: Naive Bayes and HMM Viterbi (forward, backward, both).

Naive Bayes uses five features from the spec: word length (MORE if the word
has more than 4 characters, otherwise LESS), the word, the previous word, the
next word, and the previous tag. At test time the previous tag is the tag the
model predicted for the previous word.

The HMM is the standard tag-bigram model with add-one smoothing.
- Forward Viterbi is the usual left-to-right MAP decode.
- Backward Viterbi is the same algorithm on the reversed sentence, with
  transitions estimated on reversed tag sequences.
- Forward-backward Viterbi scores each tag at each position by the sum of the
  best left-context score and the best right-context score (the emission at
  that position is counted only once) and picks the tag independently.
"""

from __future__ import annotations

import numpy as np

START_W, END_W, UNK = "<s>", "</s>", "<UNK>"
LESS, MORE = "LESS", "MORE"


def load_sents(path: str) -> list[list[tuple[str, str]]]:
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            word, tag = line.split("\t", 1)
            cur.append((word, tag))
    if cur:
        sents.append(cur)
    return sents


class POSModels:
    def __init__(self, train: list[list[tuple[str, str]]]):
        tags = sorted({t for s in train for _, t in s})
        self.tags = tags
        self.T = len(tags)
        self.tag_id = {t: i for i, t in enumerate(tags)}
        words = {UNK, START_W, END_W}
        for s in train:
            words.update(w for w, _ in s)
        self.words = sorted(words)
        self.V = len(self.words)
        self.word_id = {w: i for i, w in enumerate(self.words)}
        self.unk = self.word_id[UNK]
        self._train_nb(train)
        self._train_hmm(train)

    def wid(self, word: str) -> int:
        return self.word_id.get(word, self.unk)

    # ---------------------------------------------------------------- NB
    def _train_nb(self, train) -> None:
        T, V = self.T, self.V
        self.nb_word = np.zeros((T, V))
        self.nb_prev_w = np.zeros((T, V))
        self.nb_next_w = np.zeros((T, V))
        self.nb_len = np.zeros((T, 2))
        self.nb_ptag = np.zeros((T, T + 1))  # last column is <s>
        self.nb_tag = np.zeros(T)
        start_col = T
        for sent in train:
            prev_w, prev_t = START_W, start_col
            for i, (w, tname) in enumerate(sent):
                t = self.tag_id[tname]
                nxt = sent[i + 1][0] if i + 1 < len(sent) else END_W
                self.nb_tag[t] += 1
                self.nb_word[t, self.wid(w)] += 1
                self.nb_prev_w[t, self.wid(prev_w)] += 1
                self.nb_next_w[t, self.wid(nxt)] += 1
                self.nb_len[t, 1 if len(w) > 4 else 0] += 1
                self.nb_ptag[t, prev_t] += 1
                prev_w, prev_t = w, t
        n = self.nb_tag.sum()
        self.log_prior = np.log(self.nb_tag / n)
        self.log_word = self._smooth(self.nb_word, self.nb_tag, V)
        self.log_prev_w = self._smooth(self.nb_prev_w, self.nb_tag, V)
        self.log_next_w = self._smooth(self.nb_next_w, self.nb_tag, V)
        self.log_len = self._smooth(self.nb_len, self.nb_tag, 2)
        self.log_ptag = self._smooth(self.nb_ptag, self.nb_tag, T + 1)

    @staticmethod
    def _smooth(counts: np.ndarray, row_tot: np.ndarray, nvals: int) -> np.ndarray:
        return np.log((counts + 1.0) / (row_tot[:, None] + nvals))

    def nb_tag_sentence(self, words: list[str]) -> list[int]:
        prev = self.T  # <s> column
        out = []
        for i, w in enumerate(words):
            pw = words[i - 1] if i else START_W
            nw = words[i + 1] if i + 1 < len(words) else END_W
            scores = (
                self.log_prior
                + self.log_len[:, 1 if len(w) > 4 else 0]
                + self.log_word[:, self.wid(w)]
                + self.log_prev_w[:, self.wid(pw)]
                + self.log_next_w[:, self.wid(nw)]
                + self.log_ptag[:, prev]
            )
            prev = int(scores.argmax())
            out.append(prev)
        return out

    # --------------------------------------------------------------- HMM
    def _train_hmm(self, train) -> None:
        T, V = self.T, self.V
        START = T
        # trans[prev, tag], prev == START for the sentence-initial tag
        trans = np.zeros((T + 1, T))
        end = np.zeros(T)
        emit = np.zeros((T, V))
        for sent in train:
            prev = START
            for w, tname in sent:
                t = self.tag_id[tname]
                trans[prev, t] += 1
                emit[t, self.wid(w)] += 1
                prev = t
            end[prev] += 1
        tag_count = emit.sum(axis=1)
        n_sent = len(train)
        self.log_emit = np.log((emit + 1.0) / (tag_count[:, None] + V))
        self.log_tfwd = np.empty((T + 1, T))
        denom_tag = tag_count + T + 1  # next tag or </s>
        self.log_tfwd[:T] = np.log((trans[:T] + 1.0) / denom_tag[:, None])
        self.log_tfwd[START] = np.log((trans[START] + 1.0) / (n_sent + T))
        self.log_end = np.log((end + 1.0) / denom_tag)

        # Reversed sentences: P(tag to the left | tag to the right).
        # rev[right, left]; rev[START, tag] = P(tag | </s>) i.e. a good final tag.
        self.log_trev = np.empty((T + 1, T))
        self.log_trev[:T] = np.log((trans[:T].T + 1.0) / (tag_count[:, None] + T + 1))
        self.log_trev[START] = np.log((end + 1.0) / (n_sent + T))

    def _viterbi(self, emit_rows: np.ndarray, log_trans: np.ndarray, log_end):
        """emit_rows is (n, T). Returns (tag ids, dp matrix)."""
        n, T = emit_rows.shape
        dp = np.empty((n, T))
        bp = np.empty((n, T), dtype=np.int16)
        dp[0] = log_trans[T] + emit_rows[0]
        bp[0] = -1
        for i in range(1, n):
            scores = dp[i - 1][:, None] + log_trans[:T]
            bp[i] = scores.argmax(axis=0).astype(np.int16)
            dp[i] = scores[bp[i], np.arange(T)] + emit_rows[i]
        last = int((dp[-1] + log_end).argmax()) if log_end is not None else int(dp[-1].argmax())
        path = [last]
        for i in range(n - 1, 0, -1):
            last = int(bp[i, last])
            path.append(last)
        path.reverse()
        return path, dp

    def _emit_rows(self, words: list[str]) -> np.ndarray:
        ids = [self.wid(w) for w in words]
        return self.log_emit[:, ids].T

    def hmm_forward(self, words: list[str]) -> list[int]:
        path, _ = self._viterbi(self._emit_rows(words), self.log_tfwd, self.log_end)
        return path

    def hmm_backward(self, words: list[str]) -> list[int]:
        rows = self._emit_rows(words)[::-1]
        path, _ = self._viterbi(rows, self.log_trev, None)
        return path[::-1]

    def hmm_forward_backward(self, words: list[str]) -> list[int]:
        rows = self._emit_rows(words)
        _, alpha = self._viterbi(rows, self.log_tfwd, None)
        _, beta_rev = self._viterbi(rows[::-1], self.log_trev, None)
        beta = beta_rev[::-1]
        scores = alpha + beta - rows
        return scores.argmax(axis=1).tolist()

    def decode(self, words: list[str]) -> dict[str, list[str]]:
        ids = {
            "naive_bayes": self.nb_tag_sentence(words),
            "hmm_forward": self.hmm_forward(words),
            "hmm_backward": self.hmm_backward(words),
            "hmm_forward_backward": self.hmm_forward_backward(words),
        }
        return {name: [self.tags[i] for i in seq] for name, seq in ids.items()}


def prf(gold: list[str], pred: list[str], tags: list[str]) -> dict:
    index = {t: i for i, t in enumerate(tags)}
    k = len(tags)
    tp = np.zeros(k)
    fp = np.zeros(k)
    fn = np.zeros(k)
    for g, p in zip(gold, pred):
        gi, pi = index[g], index[p]
        if gi == pi:
            tp[gi] += 1
        else:
            fp[pi] += 1
            fn[gi] += 1
    support = tp + fn
    precision = np.divide(tp, tp + fp, out=np.zeros(k), where=(tp + fp) > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(k), where=(precision + recall) > 0)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / max(len(gold), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
        "per_tag": {
            tags[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(k)
        },
    }
