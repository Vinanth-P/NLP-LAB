
import re
import math
import random
import os
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "original_kannada.txt"

TRAIN_SIZE = 1_000_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000

TOTAL_SENTENCES = TRAIN_SIZE + DEV_SIZE + TEST_SIZE

RANDOM_SEED = 42
OUTPUT_DIR = "q1_output"

# Interpolation weights.
# λ1 + λ2 + ... + λn = 1.
INTERPOLATION_WEIGHTS = {
    2: [0.40, 0.60],              # unigram + bigram
    3: [0.20, 0.30, 0.50],       # unigram + bigram + trigram
    4: [0.10, 0.20, 0.30, 0.40]  # unigram + bigram + trigram + quadgram
}

GOOD_TURING_CUTOFF = 5
KATZ_CUTOFF = 5
STUPID_BACKOFF_ALPHA = 0.4
KNESER_NEY_DISCOUNT = 0.75

EPSILON = 1e-15


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TOKENIZATION
# ============================================================

def sentence_tokenize(text):
    """
    Sentence tokenizer for Kannada + English text.

    Sentence boundaries:
        .
        !
        ?
        ।
        ॥

    Newline is also treated as a sentence boundary.
    """
    text = text.replace("॥", "।")
    text = re.sub(r"([.!?।])", r"\1\n", text)

    sentences = text.splitlines()

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def word_tokenize(sentence):
    """
    Tokenize:
        Kannada words
        English words
        Integers / decimal numbers
    """
    tokens = re.findall(
        r"[\u0C80-\u0CFF]+|[A-Za-z]+|\d+(?:[.,]\d+)?",
        sentence
    )
    return tokens


# ============================================================
# DATA LOADING
# ============================================================

print("=" * 70)
print("READING CORPUS")
print("=" * 70)

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print(f"ERROR: {INPUT_FILE} not found.")
    print("Place original_kannada.txt in the same folder as this script.")
    raise SystemExit(1)

print("Characters:", f"{len(text):,}")


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

print("\nTokenizing sentences...")

raw_sentences = sentence_tokenize(text)

print("Raw extracted sentences:", f"{len(raw_sentences):,}")


# ============================================================
# WORD TOKENIZATION
# ============================================================

print("\nTokenizing words...")

tokenized_sentences = []
empty_sentences = 0

for sentence in raw_sentences:
    tokens = word_tokenize(sentence)

    if tokens:
        tokenized_sentences.append(tokens)
    else:
        empty_sentences += 1

print("Valid tokenized sentences:", f"{len(tokenized_sentences):,}")
print("Empty sentences removed:", f"{empty_sentences:,}")


# ============================================================
# SIZE CHECK
# ============================================================

if len(tokenized_sentences) < TOTAL_SENTENCES:
    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)

    print(f"Required valid sentences : {TOTAL_SENTENCES:,}")
    print(f"Available valid sentences: {len(tokenized_sentences):,}")
    print(f"Short by                 : {TOTAL_SENTENCES - len(tokenized_sentences):,}")

    raise SystemExit(1)


# ============================================================
# SELECT + SHUFFLE
# ============================================================

tokenized_sentences = tokenized_sentences[:TOTAL_SENTENCES]

print(f"\nUsing exactly {len(tokenized_sentences):,} sentences.")

print("\nShuffling sentences...")
random.seed(RANDOM_SEED)
random.shuffle(tokenized_sentences)


# ============================================================
# TRAIN / DEV / TEST SPLIT
# ============================================================

train_sentences = tokenized_sentences[:TRAIN_SIZE]

dev_sentences = tokenized_sentences[
    TRAIN_SIZE:
    TRAIN_SIZE + DEV_SIZE
]

test_sentences = tokenized_sentences[
    TRAIN_SIZE + DEV_SIZE:
    TOTAL_SENTENCES
]

print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print(f"Training     : {len(train_sentences):,}")
print(f"Development  : {len(dev_sentences):,}")
print(f"Test         : {len(test_sentences):,}")
print(
    f"Total        : "
    f"{len(train_sentences) + len(dev_sentences) + len(test_sentences):,}"
)


# ============================================================
# SAVE SENTENCE SPLITS
# ============================================================

def save_sentences(filename, sentences):
    with open(filename, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(" ".join(sentence) + "\n")


print("\nSaving dataset splits...")

save_sentences(
    os.path.join(OUTPUT_DIR, "train_kannada.txt"),
    train_sentences
)

save_sentences(
    os.path.join(OUTPUT_DIR, "dev_kannada.txt"),
    dev_sentences
)

save_sentences(
    os.path.join(OUTPUT_DIR, "test_kannada.txt"),
    test_sentences
)


# ============================================================
# VOCABULARY
# ============================================================

print("\nBuilding vocabulary...")

word_counts = Counter()

for sentence in train_sentences:
    word_counts.update(sentence)

vocab = set(word_counts.keys())

vocab.add("<UNK>")
vocab.add("<s>")
vocab.add("</s>")

print("Vocabulary size:", f"{len(vocab):,}")


# ============================================================
# UNKNOWN WORD CONVERSION
# ============================================================

def convert_to_vocab(sentence):
    return [
        word if word in vocab else "<UNK>"
        for word in sentence
    ]


train_sentences = [
    convert_to_vocab(sentence)
    for sentence in train_sentences
]

dev_sentences = [
    convert_to_vocab(sentence)
    for sentence in dev_sentences
]

test_sentences = [
    convert_to_vocab(sentence)
    for sentence in test_sentences
]


# ============================================================
# SENTENCE BOUNDARIES
# ============================================================

def add_boundaries(sentence, n):
    """
    Adds n-1 start tokens and one end token.

    n=1:  w1 w2 ... </s>
    n=2:  <s> w1 w2 ... </s>
    n=3:  <s> <s> w1 w2 ... </s>
    n=4:  <s> <s> <s> w1 w2 ... </s>
    """
    return (
        ["<s>"] * (n - 1)
        + sentence
        + ["</s>"]
    )


# ============================================================
# BASE MLE N-GRAM MODEL
# ============================================================

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.total_tokens = 0

    def train(self, sentences):
        for sentence in sentences:
            tokens = add_boundaries(sentence, self.n)

            self.total_tokens += len(tokens)

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                self.ngram_counts[ngram] += 1

                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

        print(
            f"{self.n}-gram model: "
            f"{len(self.ngram_counts):,} unique n-grams"
        )

    def probability(self, ngram):
        ngram = tuple(ngram)
        count = self.ngram_counts.get(ngram, 0)

        if self.n == 1:
            if self.total_tokens == 0:
                return 0.0
            return count / self.total_tokens

        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)

        if context_count == 0:
            return 0.0

        return count / context_count


# ============================================================
# CREATE + TRAIN FOUR MODELS
# ============================================================

print("\n" + "=" * 70)
print("CREATING N-GRAM MODELS")
print("=" * 70)

unigram_model = NGramModel(1)
bigram_model = NGramModel(2)
trigram_model = NGramModel(3)
quadrigram_model = NGramModel(4)

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

unigram_model.train(train_sentences)
bigram_model.train(train_sentences)
trigram_model.train(train_sentences)
quadrigram_model.train(train_sentences)

models = {
    1: unigram_model,
    2: bigram_model,
    3: trigram_model,
    4: quadrigram_model
}


# ============================================================
# GOOD-TURING COUNT-OF-COUNTS
# ============================================================

class GoodTuringModel:
    """
    Conditional Good-Turing smoothing.

    For observed count c:
        c* = (c + 1) * N(c+1) / N(c)

    When the next count is unavailable, the raw count is retained.

    For unseen events:
        c*(0) = N1 / N0

    Probabilities are normalized within each seen context so that
    probabilities over the vocabulary sum approximately to one.
    """

    def __init__(self, base_model, lower_model=None, cutoff=5):
        self.model = base_model
        self.n = base_model.n
        self.lower_model = lower_model
        self.cutoff = cutoff

        self.freq_of_freq = Counter(base_model.ngram_counts.values())
        self.total_types = len(base_model.ngram_counts)
        self.total_tokens = sum(base_model.ngram_counts.values())

        # Cache adjusted counts for counts that actually occur.
        self.adjusted_cache = {}
        self.unigram_normalization = 0.0

        # For conditional normalization:
        # context -> sum of adjusted counts of observed continuations
        # context -> number of observed continuation types
        self.context_adjusted_sum = defaultdict(float)
        self.context_seen_types = Counter()

        if self.n > 1:
            for ngram, count in self.model.ngram_counts.items():
                context = ngram[:-1]
                self.context_adjusted_sum[context] += self.adjusted_count(count)
                self.context_seen_types[context] += 1

        # Estimate unseen-event count before any normalization uses it.
        N1 = self.freq_of_freq.get(1, 0)

        # For a unigram model, estimate the number of unseen vocabulary
        # types directly from the model vocabulary represented in the
        # caller's global vocab.
        if self.n == 1:
            observed_types = len(self.model.ngram_counts)
            self.N0 = max(len(vocab) - observed_types, 1)
        else:
            # Global type-space is enormous; this is only used to obtain
            # a positive unseen-count estimate.
            V = max(len(vocab), 1)
            possible_types = V ** self.n
            self.N0 = max(possible_types - self.total_types, 1)

        self.unseen_count_estimate = (
            N1 / self.N0 if N1 > 0 else EPSILON
        )

        if self.n == 1:
            self.unigram_normalization = (
                sum(
                    self.adjusted_count(c)
                    for c in self.model.ngram_counts.values()
                )
                + self.N0 * self.unseen_count_estimate
            )

    def adjusted_count(self, count):
        if count == 0:
            return self.unseen_count_estimate

        if count in self.adjusted_cache:
            return self.adjusted_cache[count]

        nc = self.freq_of_freq.get(count, 0)
        nc1 = self.freq_of_freq.get(count + 1, 0)

        if (
            1 <= count <= self.cutoff
            and nc > 0
            and nc1 > 0
        ):
            c_star = (count + 1) * nc1 / nc
            c_star = max(c_star, EPSILON)
        else:
            c_star = float(count)

        # Avoid an unstable adjusted count larger than the raw count.
        c_star = min(c_star, float(count))

        self.adjusted_cache[count] = c_star
        return c_star

    def unigram_probability(self, word):
        count = self.model.ngram_counts.get((word,), 0)

        if count > 0:
            return self.adjusted_count(count) / max(
                self.unigram_normalization,
                EPSILON
            )

        # Unseen unigram type.
        return self.unseen_count_estimate / max(
            self.unigram_normalization,
            EPSILON
        )

    def probability(self, ngram):
        ngram = tuple(ngram)

        if self.n == 1:
            return self.unigram_probability(ngram[0])

        context = ngram[:-1]
        word = ngram[-1]
        count = self.model.ngram_counts.get(ngram, 0)

        # If the context itself was unseen, fall back recursively.
        if self.context_seen_types.get(context, 0) == 0:
            if self.lower_model is not None:
                lower_ngram = tuple(context[-(self.n - 2):]) + (word,)
                if self.n == 2:
                    lower_ngram = (word,)
                return max(
                    self.lower_model.probability(lower_ngram),
                    EPSILON
                )
            return 1.0 / max(len(vocab), 1)

        seen_types = self.context_seen_types[context]
        unseen_types = max(len(vocab) - seen_types, 0)

        normalization = (
            self.context_adjusted_sum[context]
            + unseen_types * self.unseen_count_estimate
        )

        if count > 0:
            numerator = self.adjusted_count(count)
        else:
            numerator = self.unseen_count_estimate

        p = numerator / max(normalization, EPSILON)

        if p <= 0 and self.lower_model is not None:
            lower_ngram = tuple(context[-(self.n - 2):]) + (word,)
            if self.n == 2:
                lower_ngram = (word,)
            return max(self.lower_model.probability(lower_ngram), EPSILON)

        return max(p, EPSILON)


# ============================================================
# KATZ BACKOFF MODEL
# ============================================================

class KatzBackoffModel:
    """
    Katz backoff with Good-Turing style discounting for low counts.

    Seen n-grams use discounted probabilities.

    Unseen n-grams back off:
        P_Katz(w|h) = alpha(h) * P_lower(w|h')

    where alpha(h) reserves the probability mass not assigned to
    seen continuations.
    """

    def __init__(self, base_model, lower_model=None, cutoff=5):
        self.model = base_model
        self.n = base_model.n
        self.lower_model = lower_model
        self.cutoff = cutoff

        self.freq_of_freq = Counter(base_model.ngram_counts.values())
        self.adjusted_cache = {}
        self.alpha = {}
        self.gt_unigram = None
        self.unigram_normalization = 0.0

        if self.n == 1:
            self.gt_unigram = GoodTuringModel(
                self.model,
                None,
                self.cutoff
            )

        if self.n > 1:
            self._build_backoff_weights()

    def adjusted_count(self, count):
        if count == 0:
            return 0.0

        if count in self.adjusted_cache:
            return self.adjusted_cache[count]

        nc = self.freq_of_freq.get(count, 0)
        nc1 = self.freq_of_freq.get(count + 1, 0)

        if (
            1 <= count <= self.cutoff
            and nc > 0
            and nc1 > 0
        ):
            c_star = (count + 1) * nc1 / nc
            c_star = min(max(c_star, EPSILON), float(count))
        else:
            c_star = float(count)

        self.adjusted_cache[count] = c_star
        return c_star

    def probability(self, ngram):
        ngram = tuple(ngram)

        if self.n == 1:
            # Lower-order base. Good-Turing is used so unseen words
            # receive a positive probability.
            return max(
                self.gt_unigram.probability(ngram),
                EPSILON
            )

        context = ngram[:-1]
        word = ngram[-1]
        count = self.model.ngram_counts.get(ngram, 0)
        context_count = self.model.context_counts.get(context, 0)

        if count > 0 and context_count > 0:
            return max(
                self.adjusted_count(count) / context_count,
                EPSILON
            )

        # Unseen context: directly use lower-order model.
        if context not in self.alpha:
            if self.lower_model is not None:
                lower_ngram = (word,)
                suffix = context[-(self.n - 2):]
                if suffix:
                    lower_ngram = tuple(suffix) + (word,)
                return max(
                    self.lower_model.probability(lower_ngram),
                    EPSILON
                )
            return 1.0 / max(len(vocab), 1)

        lower_ngram = (word,)
        suffix = context[-(self.n - 2):]
        if suffix:
            lower_ngram = tuple(suffix) + (word,)

        lower_p = self.lower_model.probability(lower_ngram)

        return max(
            self.alpha[context] * lower_p,
            EPSILON
        )

    def _build_backoff_weights(self):
        """
        Build alpha(h) for every seen context.

        alpha(h) =
            (1 - discounted mass of seen words)
            /
            (1 - lower-order mass of seen words)
        """
        discounted_mass = defaultdict(float)
        lower_seen_mass = defaultdict(float)

        for ngram, count in self.model.ngram_counts.items():
            context = ngram[:-1]
            word = ngram[-1]

            context_count = self.model.context_counts.get(context, 0)
            if context_count == 0:
                continue

            discounted_mass[context] += (
                self.adjusted_count(count) / context_count
            )

            suffix = context[-(self.n - 2):]
            if suffix:
                lower_ngram = tuple(suffix) + (word,)
            else:
                lower_ngram = (word,)

            lower_seen_mass[context] += (
                self.lower_model.probability(lower_ngram)
            )

        for context in self.model.context_counts:
            seen_discounted = discounted_mass.get(context, 0.0)
            seen_lower = lower_seen_mass.get(context, 0.0)

            numerator = max(1.0 - seen_discounted, EPSILON)
            denominator = max(1.0 - seen_lower, EPSILON)

            self.alpha[context] = numerator / denominator


# ============================================================
# STUPID BACKOFF MODEL
# ============================================================

class StupidBackoffModel:
    """
    Stupid Backoff:

        S(w|h) =
            P_MLE(w|h)              if seen
            alpha * S(w|h')         otherwise

    alpha is normally set to 0.4.

    NOTE:
    Stupid Backoff is not a normalized probability model. Therefore
    the resulting "perplexity" is a score-based / pseudo-perplexity
    metric, not a mathematically normalized language-model perplexity.
    """

    def __init__(self, model, lower_model=None, alpha=0.4):
        self.model = model
        self.n = model.n
        self.lower_model = lower_model
        self.alpha = alpha

    def probability(self, ngram):
        ngram = tuple(ngram)

        if self.n == 1:
            count = self.model.ngram_counts.get(ngram, 0)
            if count > 0:
                return count / self.model.total_tokens
            return EPSILON

        count = self.model.ngram_counts.get(ngram, 0)
        context = ngram[:-1]
        context_count = self.model.context_counts.get(context, 0)

        if count > 0 and context_count > 0:
            return count / context_count

        if self.lower_model is not None:
            word = ngram[-1]
            suffix = context[-(self.n - 2):]
            lower_ngram = (word,)
            if suffix:
                lower_ngram = tuple(suffix) + (word,)

            return max(
                self.alpha * self.lower_model.probability(lower_ngram),
                EPSILON
            )

        return EPSILON


# ============================================================
# KNESER-NEY MODEL
# ============================================================

class KneserNeyModel:
    """
    Interpolated absolute-discount Kneser-Ney.

    For n >= 2:

        P_KN(w|h) =
            max(c(h,w)-D, 0) / c(h)
            + lambda(h) * P_KN(w|h')

        lambda(h) =
            D * N1+(h,*) / c(h)

    Lower-order counts are continuation counts:

        N1+(*, w)
        N1+(*, h, w)
        ...

    For n=2 the base probability is:
        continuation_count(w) / total_unique_bigrams
    """

    def __init__(
        self,
        order,
        raw_models,
        continuation_counts,
        continuation_context_totals,
        discount=0.75
    ):
        self.n = order
        self.raw_models = raw_models
        self.discount = discount
        self.continuation_counts = continuation_counts
        self.continuation_context_totals = continuation_context_totals
        self.unique_continuations = {}
        self._prob_cache = {}

        # Cache N1+(h, *) for the raw top-order model and for all
        # lower-order continuation tables.
        for current_order in range(1, self.n):
            self.unique_continuations[current_order] = Counter()

        # Lower-order continuation tables only go up to n-1.
        # For example, a bigram Kneser-Ney model only needs
        # unigram continuation counts; it must not try to access
        # unique_continuations[2].
        for current_order in range(1, self.n):
            table = self.continuation_counts[current_order]
            for gram in table:
                context = gram[:-1] if current_order > 1 else ()
                self.unique_continuations[current_order][context] += 1

        # Raw top-order observed continuation types per context.
        self.unique_top_continuations = Counter()
        raw_top = self.raw_models[self.n].ngram_counts
        for gram in raw_top:
            self.unique_top_continuations[gram[:-1]] += 1

    def _base_probability(self, word):
        numerator = self.continuation_counts[1].get(
            (word,),
            0
        )
        denominator = sum(self.continuation_counts[1].values())

        if numerator == 0 or denominator == 0:
            return 1.0 / max(len(vocab), 1)

        return numerator / denominator

    def probability(self, ngram):
        ngram = tuple(ngram)

        if ngram in self._prob_cache:
            return self._prob_cache[ngram]

        value = self._probability_recursive(
            self.n,
            ngram
        )

        value = max(value, EPSILON)
        self._prob_cache[ngram] = value
        return value

    def _probability_recursive(self, order, ngram):
        word = ngram[-1]

        if order == 1:
            return self._base_probability(word)

        context = ngram[:-1]

        if order == self.n:
            model = self.raw_models[order]
            count = model.ngram_counts.get(ngram, 0)
            context_total = model.context_counts.get(context, 0)

            if context_total == 0:
                return self._probability_recursive(
                    order - 1,
                    ngram[1:]
                )

            seen_continuations = self.unique_top_continuations.get(
                context,
                0
            )

            first_term = max(count - self.discount, 0.0) / context_total

            lambda_h = (
                self.discount
                * seen_continuations
                / context_total
            )

            backoff = self._probability_recursive(
                order - 1,
                ngram[1:]
            )

            return first_term + lambda_h * backoff

        # Lower-order Kneser-Ney uses continuation counts.
        lower_counts = self.continuation_counts[order]
        context_total = self.continuation_context_totals[order].get(
            context,
            0
        )

        if context_total == 0:
            return self._probability_recursive(
                order - 1,
                ngram[1:]
            )

        count = lower_counts.get(ngram, 0)

        seen_continuations = self.unique_continuations[
            order
        ].get(context, 0)

        first_term = max(count - self.discount, 0.0) / context_total

        lambda_h = (
            self.discount
            * seen_continuations
            / context_total
        )

        backoff = self._probability_recursive(
            order - 1,
            ngram[1:]
        )

        return first_term + lambda_h * backoff


# ============================================================
# BUILD CONTINUATION COUNTS FOR KNESER-NEY
# ============================================================

def build_continuation_counts(raw_models):
    """
    continuation_counts[k] stores N1+(*, k-gram).

    Example:
        continuation_counts[1][(word,)]
            = number of distinct predecessors of word

        continuation_counts[2][(w1, w2)]
            = number of distinct predecessors of the bigram
    """
    continuation_counts = {}

    # Lower-order unigram continuation counts from bigram types.
    continuation_counts[1] = Counter()

    for bigram in raw_models[2].ngram_counts:
        continuation_counts[1][(bigram[1],)] += 1

    # Lower-order bigram continuation counts from trigram types.
    continuation_counts[2] = Counter()

    for trigram in raw_models[3].ngram_counts:
        continuation_counts[2][(trigram[1], trigram[2])] += 1

    # Lower-order trigram continuation counts from quadgram types.
    continuation_counts[3] = Counter()

    for quadgram in raw_models[4].ngram_counts:
        continuation_counts[3][
            (quadgram[1], quadgram[2], quadgram[3])
        ] += 1

    continuation_context_totals = {}

    for order in (1, 2, 3):
        context_totals = Counter()

        for gram, count in continuation_counts[order].items():
            if order == 1:
                context = ()
            else:
                context = gram[:-1]

            context_totals[context] += count

        continuation_context_totals[order] = context_totals

    return (
        continuation_counts,
        continuation_context_totals
    )


# ============================================================
# BUILD ALL SMOOTHED MODELS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING SMOOTHED MODELS")
print("=" * 70)

# Good-Turing
gt_models = {}

for n in range(1, 5):
    lower = gt_models.get(n - 1)
    gt_models[n] = GoodTuringModel(
        models[n],
        lower_model=lower,
        cutoff=GOOD_TURING_CUTOFF
    )


# Katz Backoff
katz_models = {}

# Unigram first
katz_models[1] = KatzBackoffModel(
    models[1],
    lower_model=None,
    cutoff=KATZ_CUTOFF
)

for n in range(2, 5):
    katz_models[n] = KatzBackoffModel(
        models[n],
        lower_model=katz_models[n - 1],
        cutoff=KATZ_CUTOFF
    )


# Stupid Backoff
stupid_backoff_models = {
    1: StupidBackoffModel(models[1], None, STUPID_BACKOFF_ALPHA),
    2: StupidBackoffModel(
        models[2],
        StupidBackoffModel(models[1], None, STUPID_BACKOFF_ALPHA),
        STUPID_BACKOFF_ALPHA
    ),
    3: StupidBackoffModel(
        models[3],
        None,
        STUPID_BACKOFF_ALPHA
    ),
    4: StupidBackoffModel(
        models[4],
        None,
        STUPID_BACKOFF_ALPHA
    )
}

# Rebuild higher-order Stupid Backoff chains correctly.
stupid_backoff_models[2].lower_model = stupid_backoff_models[1]
stupid_backoff_models[3].lower_model = stupid_backoff_models[2]
stupid_backoff_models[4].lower_model = stupid_backoff_models[3]


# Kneser-Ney
continuation_counts, continuation_context_totals = (
    build_continuation_counts(models)
)

kn_models = {}

for n in range(2, 5):
    kn_models[n] = KneserNeyModel(
        n,
        raw_models=models,
        continuation_counts=continuation_counts,
        continuation_context_totals=continuation_context_totals,
        discount=KNESER_NEY_DISCOUNT
    )


# ============================================================
# INTERPOLATED SMOOTHING
# ============================================================

class InterpolatedModel:
    """
    Linear interpolation:

        P(w|h) =
            λ1 P1 +
            λ2 P2 +
            ...
            λn Pn

    The component models are the MLE unigram/bigram/trigram/quadgram
    models.
    """

    def __init__(self, order, component_models, weights):
        self.n = order
        self.component_models = component_models
        self.weights = weights

        if len(weights) != order:
            raise ValueError("Number of interpolation weights must equal order.")

        if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Interpolation weights must sum to 1.")

    def probability_from_components(self, component_probabilities):
        p = 0.0

        for weight, component_probability in zip(
            self.weights,
            component_probabilities
        ):
            p += weight * component_probability

        return max(p, EPSILON)


interpolated_models = {}

for n in (2, 3, 4):
    interpolated_models[n] = InterpolatedModel(
        n,
        component_models=models,
        weights=INTERPOLATION_WEIGHTS[n]
    )


def interpolated_ngram_probability(
    sentence,
    position,
    order
):
    """
    Compute an interpolated probability for one prediction event.

    Each lower-order model receives its own correctly aligned sentence
    boundaries. This matters for the first word because the bigram,
    trigram, and quadgram models use different numbers of <s> tokens.
    """
    probabilities = []

    for k in range(1, order + 1):
        model_tokens = add_boundaries(sentence, k)
        ngram = tuple(
            model_tokens[
                position - (order - k):
                position - (order - k) + k
            ]
        )

        # The alignment above is easiest to reason about by using the
        # prediction index in each model separately below.
        probabilities.append(0.0)

    # More robust implementation: align by prediction count.
    probabilities = []

    for k in range(1, order + 1):
        model_tokens = add_boundaries(sentence, k)

        # Each model has len(sentence)+1 prediction events:
        # w1, w2, ..., </s>
        prediction_index = position

        start = k - 1 + prediction_index
        ngram = tuple(
            model_tokens[start - k + 1:start + 1]
        )

        # Use the Good-Turing unigram as the base component so that
        # an OOV token mapped to <UNK> does not make the interpolated
        # probability collapse to zero.
        component_model = (
            gt_models[1]
            if k == 1
            else models[k]
        )

        probabilities.append(
            component_model.probability(ngram)
        )

    return interpolated_models[order].probability_from_components(
        probabilities
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_ngram_scorer(
    sentences,
    order,
    scorer,
    scorer_name
):
    """
    Evaluate one model on a dataset.

    For each sentence:
        P(sentence) = product of conditional probabilities

    Metrics:
        Cross Entropy = - average log2 probability
        Perplexity    = 2 ** Cross Entropy

    Stupid Backoff produces scores rather than normalized probabilities,
    so its reported perplexity is a pseudo-perplexity.
    """
    total_log_probability = 0.0
    total_predictions = 0
    zero_probability_count = 0

    for sentence in sentences:
        tokens = add_boundaries(sentence, order)

        # Prediction events correspond to actual words + </s>.
        for prediction_index in range(len(sentence) + 1):
            i = order - 1 + prediction_index
            ngram = tuple(tokens[i - order + 1:i + 1])

            p = scorer.probability(ngram)

            total_predictions += 1

            if p <= 0:
                zero_probability_count += 1
            else:
                total_log_probability += math.log2(p)

    if zero_probability_count > 0:
        return {
            "name": scorer_name,
            "cross_entropy": float("inf"),
            "perplexity": float("inf"),
            "zero_count": zero_probability_count,
            "total_predictions": total_predictions
        }

    cross_entropy = (
        -total_log_probability / max(total_predictions, 1)
    )

    perplexity = 2 ** cross_entropy

    return {
        "name": scorer_name,
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "zero_count": zero_probability_count,
        "total_predictions": total_predictions
    }


def evaluate_interpolated(
    sentences,
    order,
    model_name
):
    total_log_probability = 0.0
    total_predictions = 0
    zero_probability_count = 0

    for sentence in sentences:
        # prediction_index:
        #   0 -> first word
        #   ...
        #   len(sentence) -> </s>
        for prediction_index in range(len(sentence) + 1):
            p = interpolated_ngram_probability(
                sentence,
                prediction_index,
                order
            )

            total_predictions += 1

            if p <= 0:
                zero_probability_count += 1
            else:
                total_log_probability += math.log2(p)

    if zero_probability_count > 0:
        return {
            "name": model_name,
            "cross_entropy": float("inf"),
            "perplexity": float("inf"),
            "zero_count": zero_probability_count,
            "total_predictions": total_predictions
        }

    cross_entropy = (
        -total_log_probability / max(total_predictions, 1)
    )

    perplexity = 2 ** cross_entropy

    return {
        "name": model_name,
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
        "zero_count": zero_probability_count,
        "total_predictions": total_predictions
    }


# ============================================================
# COLLECT RESULTS
# ============================================================

def format_result(result):
    if math.isinf(result["cross_entropy"]):
        ce = "INF"
        pp = "INF"
    else:
        ce = f"{result['cross_entropy']:.4f}"
        pp = f"{result['perplexity']:.4f}"

    return (
        f"{result['name']:<34}"
        f"Cross Entropy = {ce:<12}"
        f"Perplexity = {pp:<14}"
        f"Zero = {result['zero_count']:,}/{result['total_predictions']:,}"
    )


def evaluate_dataset(sentences, dataset_name, result_file):
    print("\n" + "=" * 90)
    print(f"{dataset_name.upper()} SET RESULTS")
    print("=" * 90)

    result_file.write("\n" + "=" * 90 + "\n")
    result_file.write(
        f"{dataset_name.upper()} SET RESULTS\n"
    )
    result_file.write("=" * 90 + "\n")

    all_results = []

    # --------------------------------------------------------
    # MLE baselines
    # --------------------------------------------------------
    for n in range(1, 5):
        result = evaluate_ngram_scorer(
            sentences,
            n,
            models[n],
            f"MLE {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    # --------------------------------------------------------
    # Interpolated smoothing
    # --------------------------------------------------------
    for n in (2, 3, 4):
        result = evaluate_interpolated(
            sentences,
            n,
            f"Interpolated {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    # --------------------------------------------------------
    # Good-Turing
    # --------------------------------------------------------
    for n in range(1, 5):
        result = evaluate_ngram_scorer(
            sentences,
            n,
            gt_models[n],
            f"Good-Turing {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    # --------------------------------------------------------
    # Katz Backoff
    # --------------------------------------------------------
    for n in range(1, 5):
        result = evaluate_ngram_scorer(
            sentences,
            n,
            katz_models[n],
            f"Katz Backoff {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    # --------------------------------------------------------
    # Stupid Backoff
    # --------------------------------------------------------
    for n in (2, 3, 4):
        result = evaluate_ngram_scorer(
            sentences,
            n,
            stupid_backoff_models[n],
            f"Stupid Backoff {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    # --------------------------------------------------------
    # Kneser-Ney
    # --------------------------------------------------------
    for n in (2, 3, 4):
        result = evaluate_ngram_scorer(
            sentences,
            n,
            kn_models[n],
            f"Kneser-Ney {n}-gram"
        )
        all_results.append(result)
        line = format_result(result)
        print(line)
        result_file.write(line + "\n")

    return all_results


# ============================================================
# RUN DEVELOPMENT + TEST EVALUATION
# ============================================================

results_file_path = os.path.join(
    OUTPUT_DIR,
    "smoothing_results.txt"
)

print("\nWriting results to:", results_file_path)

with open(
    results_file_path,
    "w",
    encoding="utf-8"
) as result_file:

    # Record configuration.
    result_file.write("SMOOTHING EXPERIMENT CONFIGURATION\n")
    result_file.write("=" * 90 + "\n")
    result_file.write(
        f"Train sentences = {TRAIN_SIZE:,}\n"
    )
    result_file.write(
        f"Development sentences = {DEV_SIZE:,}\n"
    )
    result_file.write(
        f"Test sentences = {TEST_SIZE:,}\n"
    )
    result_file.write(
        f"Good-Turing cutoff = {GOOD_TURING_CUTOFF}\n"
    )
    result_file.write(
        f"Katz cutoff = {KATZ_CUTOFF}\n"
    )
    result_file.write(
        f"Stupid Backoff alpha = {STUPID_BACKOFF_ALPHA}\n"
    )
    result_file.write(
        f"Kneser-Ney D = {KNESER_NEY_DISCOUNT}\n"
    )
    result_file.write(
        f"Interpolation bigram = {INTERPOLATION_WEIGHTS[2]}\n"
    )
    result_file.write(
        f"Interpolation trigram = {INTERPOLATION_WEIGHTS[3]}\n"
    )
    result_file.write(
        f"Interpolation quadgram = {INTERPOLATION_WEIGHTS[4]}\n\n"
    )

    dev_results = evaluate_dataset(
        dev_sentences,
        "Development",
        result_file
    )

    test_results = evaluate_dataset(
        test_sentences,
        "Test",
        result_file
    )

    result_file.write("\n")
    result_file.write("=" * 90 + "\n")
    result_file.write(
        "NOTE: Stupid Backoff is not normalized, so its reported "
        "perplexity is a pseudo-perplexity.\n"
    )
    result_file.write(
        "NOTE: Interpolation weights can be tuned on the development "
        "set before final test evaluation.\n"
    )


# ============================================================
# SAVE N-GRAM COUNTS
# ============================================================

def save_counts(filename, counts):
    with open(filename, "w", encoding="utf-8") as f:
        for ngram, count in counts.items():
            f.write(f"{' '.join(ngram)}\t{count}\n")


print("\nSaving n-gram count files...")

save_counts(
    os.path.join(OUTPUT_DIR, "unigram_counts.txt"),
    unigram_model.ngram_counts
)

save_counts(
    os.path.join(OUTPUT_DIR, "bigram_counts.txt"),
    bigram_model.ngram_counts
)

save_counts(
    os.path.join(OUTPUT_DIR, "trigram_counts.txt"),
    trigram_model.ngram_counts
)

save_counts(
    os.path.join(OUTPUT_DIR, "quadrigram_counts.txt"),
    quadrigram_model.ngram_counts
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"All Q2 output files are in: {OUTPUT_DIR}")
print("\nFiles created:")
print("  train_kannada.txt")
print("  dev_kannada.txt")
print("  test_kannada.txt")
print("  unigram_counts.txt")
print("  bigram_counts.txt")
print("  trigram_counts.txt")
print("  quadrigram_counts.txt")
print("  smoothing_results.txt")
