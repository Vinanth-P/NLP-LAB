import re
import math
import random
import os
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "original_kannada.txt"

TRAIN_SIZE = 1_000_000
DEV_SIZE = 1_000
TEST_SIZE = 1_000

TOTAL_SENTENCES = (
    TRAIN_SIZE +
    DEV_SIZE +
    TEST_SIZE
)

RANDOM_SEED = 42

# Add-K value
K = 0.3

OUTPUT_DIR = "q1_output"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. SENTENCE TOKENIZATION
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
        newline
    """

    # Normalize double danda
    text = text.replace(
        "॥",
        "।"
    )

    # Put newline after sentence-ending punctuation
    text = re.sub(
        r'([.!?।])',
        r'\1\n',
        text
    )

    # Split by newline
    sentences = text.splitlines()

    # Remove empty lines
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# 2. WORD TOKENIZATION
# ============================================================

def word_tokenize(sentence):
    """
    Tokenize Kannada words, English words and numbers.
    """

    tokens = re.findall(
        r"[\u0C80-\u0CFF]+|[A-Za-z]+|\d+(?:[.,]\d+)*",
        sentence
    )

    return tokens


# ============================================================
# 3. LOAD CORPUS
# ============================================================

print("=" * 70)
print("READING CORPUS")
print("=" * 70)

try:

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

except FileNotFoundError:

    print(
        f"ERROR: {INPUT_FILE} not found."
    )

    print(
        "Make sure original_kannada.txt is in "
        "the same folder as addk.py."
    )

    exit()


print(
    "Characters:",
    f"{len(text):,}"
)


# ============================================================
# 4. SENTENCE TOKENIZATION
# ============================================================

print("\nTokenizing sentences...")

raw_sentences = sentence_tokenize(
    text
)

print(
    "Raw extracted sentences:",
    f"{len(raw_sentences):,}"
)


# ============================================================
# 5. WORD TOKENIZATION
# ============================================================

print("\nTokenizing words...")

tokenized_sentences = []

empty_sentences = 0


for sentence in raw_sentences:

    tokens = word_tokenize(
        sentence
    )

    if tokens:

        tokenized_sentences.append(
            tokens
        )

    else:

        empty_sentences += 1


print(
    "Valid tokenized sentences:",
    f"{len(tokenized_sentences):,}"
)

print(
    "Empty sentences removed:",
    f"{empty_sentences:,}"
)


# ============================================================
# 6. CHECK REQUIRED SIZE
# ============================================================

if len(tokenized_sentences) < TOTAL_SENTENCES:

    print("\n" + "=" * 70)
    print("ERROR")
    print("=" * 70)

    print(
        f"Required valid sentences : "
        f"{TOTAL_SENTENCES:,}"
    )

    print(
        f"Available valid sentences: "
        f"{len(tokenized_sentences):,}"
    )

    print(
        f"Short by: "
        f"{TOTAL_SENTENCES - len(tokenized_sentences):,}"
    )

    exit()


# ============================================================
# 7. SELECT EXACTLY 1,002,000 SENTENCES
# ============================================================

tokenized_sentences = tokenized_sentences[
    :TOTAL_SENTENCES
]

print(
    f"\nUsing exactly "
    f"{len(tokenized_sentences):,} sentences."
)


# ============================================================
# 8. SHUFFLE
# ============================================================

print("\nShuffling sentences...")

random.seed(
    RANDOM_SEED
)

random.shuffle(
    tokenized_sentences
)


# ============================================================
# 9. TRAIN / DEV / TEST SPLIT
# ============================================================

train_sentences = tokenized_sentences[
    :TRAIN_SIZE
]

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

print(
    f"Training      : {len(train_sentences):,}"
)

print(
    f"Development   : {len(dev_sentences):,}"
)

print(
    f"Test          : {len(test_sentences):,}"
)

print(
    f"Total         : "
    f"{len(train_sentences) + len(dev_sentences) + len(test_sentences):,}"
)


# ============================================================
# 10. SAVE DATASET SPLITS
# ============================================================

def save_sentences(
    filename,
    sentences
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for sentence in sentences:

            f.write(
                " ".join(sentence)
                + "\n"
            )


print("\nSaving dataset splits...")

save_sentences(
    os.path.join(
        OUTPUT_DIR,
        "train_kannada.txt"
    ),
    train_sentences
)

save_sentences(
    os.path.join(
        OUTPUT_DIR,
        "dev_kannada.txt"
    ),
    dev_sentences
)

save_sentences(
    os.path.join(
        OUTPUT_DIR,
        "test_kannada.txt"
    ),
    test_sentences
)


# ============================================================
# 11. BUILD VOCABULARY FROM TRAINING DATA
# ============================================================

print("\nBuilding vocabulary...")

word_counts = Counter()

for sentence in train_sentences:

    word_counts.update(
        sentence
    )


vocab = set(
    word_counts.keys()
)

# Special tokens
vocab.add("<UNK>")
vocab.add("<s>")
vocab.add("</s>")


VOCAB_SIZE = len(vocab)


print(
    "Vocabulary size:",
    f"{VOCAB_SIZE:,}"
)

print(
    "Add-K value:",
    K
)


# ============================================================
# 12. CONVERT UNKNOWN WORDS
# ============================================================

def convert_to_vocab(
    sentence
):

    return [
        word
        if word in vocab
        else "<UNK>"
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
# 13. ADD SENTENCE BOUNDARIES
# ============================================================

def add_boundaries(
    sentence,
    n
):

    return (
        ["<s>"] * (n - 1)
        + sentence
        + ["</s>"]
    )


# ============================================================
# 14. ADD-K N-GRAM MODEL
# ============================================================

class AddKModel:

    def __init__(
        self,
        n,
        vocab_size,
        k
    ):

        self.n = n
        self.vocab_size = vocab_size
        self.k = k

        # Count of n-grams
        self.ngram_counts = Counter()

        # Count of contexts
        self.context_counts = Counter()

        # Total tokens
        self.total_tokens = 0


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        sentences
    ):

        for sentence in sentences:

            tokens = add_boundaries(
                sentence,
                self.n
            )

            self.total_tokens += len(
                tokens
            )

            # Generate n-grams
            for i in range(
                len(tokens) - self.n + 1
            ):

                ngram = tuple(
                    tokens[
                        i:i + self.n
                    ]
                )

                self.ngram_counts[
                    ngram
                ] += 1

                # Context count
                if self.n > 1:

                    context = ngram[:-1]

                    self.context_counts[
                        context
                    ] += 1


        print(
            f"{self.n}-gram model:"
            f" {len(self.ngram_counts):,}"
            f" unique n-grams"
        )


    # ========================================================
    # ADD-K PROBABILITY
    # ========================================================

    def probability(
        self,
        ngram
    ):

        ngram = tuple(
            ngram
        )

        count = self.ngram_counts.get(
            ngram,
            0
        )


        # ----------------------------------------------------
        # UNIGRAM
        #
        # P(w) =
        # (C(w) + K)
        # ----------------
        # (N + K*V)
        # ----------------------------------------------------

        if self.n == 1:

            denominator = (
                self.total_tokens
                + self.k * self.vocab_size
            )

            return (
                count + self.k
            ) / denominator


        # ----------------------------------------------------
        # BIGRAM / TRIGRAM / QUADRIGRAM
        #
        # P(w|context) =
        # (C(context,w) + K)
        # ----------------------------
        # (C(context) + K*V)
        # ----------------------------------------------------

        context = ngram[:-1]

        context_count = (
            self.context_counts.get(
                context,
                0
            )
        )

        denominator = (
            context_count
            + self.k * self.vocab_size
        )

        return (
            count + self.k
        ) / denominator


    # ========================================================
    # SENTENCE PROBABILITY
    # ========================================================

    def sentence_probability(
        self,
        sentence
    ):

        tokens = add_boundaries(
            sentence,
            self.n
        )

        probability = 1.0


        for i in range(
            len(tokens) - self.n + 1
        ):

            ngram = tuple(
                tokens[
                    i:i + self.n
                ]
            )

            p = self.probability(
                ngram
            )

            probability *= p


        return probability


    # ========================================================
    # LOG PROBABILITY
    # ========================================================

    def sentence_log_probability(
        self,
        sentence
    ):

        tokens = add_boundaries(
            sentence,
            self.n
        )

        log_probability = 0.0


        for i in range(
            len(tokens) - self.n + 1
        ):

            ngram = tuple(
                tokens[
                    i:i + self.n
                ]
            )

            p = self.probability(
                ngram
            )

            log_probability += math.log2(
                p
            )


        return log_probability


    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(
        self,
        sentences
    ):

        total_log_probability = 0.0

        total_predictions = 0


        for sentence in sentences:

            tokens = add_boundaries(
                sentence,
                self.n
            )


            for i in range(
                len(tokens) - self.n + 1
            ):

                ngram = tuple(
                    tokens[
                        i:i + self.n
                    ]
                )

                p = self.probability(
                    ngram
                )

                total_log_probability += math.log2(
                    p
                )

                total_predictions += 1


        # ----------------------------------------------------
        # CROSS ENTROPY
        # ----------------------------------------------------

        cross_entropy = (
            -total_log_probability
            /
            total_predictions
        )


        # ----------------------------------------------------
        # PERPLEXITY
        # ----------------------------------------------------

        perplexity = (
            2 ** cross_entropy
        )


        return (
            cross_entropy,
            perplexity,
            total_predictions
        )


# ============================================================
# 15. CREATE MODELS
# ============================================================

print("\n" + "=" * 70)
print("CREATING ADD-K MODELS")
print("=" * 70)


unigram_model = AddKModel(
    1,
    VOCAB_SIZE,
    K
)

bigram_model = AddKModel(
    2,
    VOCAB_SIZE,
    K
)

trigram_model = AddKModel(
    3,
    VOCAB_SIZE,
    K
)

quadrigram_model = AddKModel(
    4,
    VOCAB_SIZE,
    K
)


# ============================================================
# 16. TRAIN MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)


unigram_model.train(
    train_sentences
)

bigram_model.train(
    train_sentences
)

trigram_model.train(
    train_sentences
)

quadrigram_model.train(
    train_sentences
)


models = {

    "Unigram":
        unigram_model,

    "Bigram":
        bigram_model,

    "Trigram":
        trigram_model,

    "Quadrigram":
        quadrigram_model
}


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as result_file:


    # ========================================================
    # DEVELOPMENT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("DEVELOPMENT SET RESULTS")
    print("=" * 70)


    result_file.write(
        "ADD-K SMOOTHING (K = 0.3)\n\n"
    )

    result_file.write(
        "DEVELOPMENT SET RESULTS\n"
    )

    result_file.write(
        "=" * 70 + "\n"
    )


    for name, model in models.items():

        (
            cross_entropy,
            perplexity,
            total
        ) = model.evaluate(
            dev_sentences
        )


        line = (
            f"{name:<12}"
            f"Cross Entropy = "
            f"{cross_entropy:.4f}   "
            f"Perplexity = "
            f"{perplexity:.4f}\n"
        )


        print(
            line,
            end=""
        )

        result_file.write(
            line
        )


    # ========================================================
    # TEST RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)


    result_file.write(
        "\nTEST SET RESULTS\n"
    )

    result_file.write(
        "=" * 70 + "\n"
    )


    for name, model in models.items():

        (
            cross_entropy,
            perplexity,
            total
        ) = model.evaluate(
            test_sentences
        )


        line = (
            f"{name:<12}"
            f"Cross Entropy = "
            f"{cross_entropy:.4f}   "
            f"Perplexity = "
            f"{perplexity:.4f}\n"
        )


        print(
            line,
            end=""
        )

        result_file.write(
            line
        )


# ============================================================
# 18. SAVE N-GRAM COUNTS
# ============================================================

def save_counts(
    filename,
    counts
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for ngram, count in counts.items():

            f.write(
                f"{' '.join(ngram)}\t{count}\n"
            )


print(
    "\nSaving n-gram count files..."
)


save_counts(
    os.path.join(
        OUTPUT_DIR,
        "unigram_counts.txt"
    ),
    unigram_model.ngram_counts
)


save_counts(
    os.path.join(
        OUTPUT_DIR,
        "bigram_counts.txt"
    ),
    bigram_model.ngram_counts
)


save_counts(
    os.path.join(
        OUTPUT_DIR,
        "trigram_counts.txt"
    ),
    trigram_model.ngram_counts
)


save_counts(
    os.path.join(
        OUTPUT_DIR,
        "quadrigram_counts.txt"
    ),
    quadrigram_model.ngram_counts
)


# ============================================================
# 19. EXAMPLE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE PROBABILITIES")
print("=" * 70)


example_sentence = word_tokenize(
    "ಇಂದು ಗಜಪಯಣದಲ್ಲಿ ಮೈಸೂರಿಗೆ ಆಗಮಿಸಿದ ಆನೆಗಳ ವಿವರ ಇಲ್ಲಿದೆ"
)


example_sentence = convert_to_vocab(
    example_sentence
)


print("\nExample sentence:")

print(
    " ".join(example_sentence)
)


# ============================================================
# UNIGRAM
# ============================================================

print("\nUnigram probabilities:")


for word in example_sentence:

    p = unigram_model.probability(
        (word,)
    )

    print(
        f"P({word}) = {p:.10f}"
    )


# ============================================================
# BIGRAM
# ============================================================

print("\nBigram probabilities:")


tokens = add_boundaries(
    example_sentence,
    2
)


for i in range(
    len(tokens) - 1
):

    bigram = tuple(
        tokens[
            i:i + 2
        ]
    )

    p = bigram_model.probability(
        bigram
    )

    print(
        f"P({bigram[1]} | {bigram[0]}) "
        f"= {p:.10f}"
    )


# ============================================================
# TRIGRAM
# ============================================================

print("\nTrigram probabilities:")


tokens = add_boundaries(
    example_sentence,
    3
)


for i in range(
    len(tokens) - 2
):

    trigram = tuple(
        tokens[
            i:i + 3
        ]
    )

    p = trigram_model.probability(
        trigram
    )

    print(
        f"P({trigram[2]} | "
        f"{trigram[0]}, "
        f"{trigram[1]}) "
        f"= {p:.10f}"
    )


# ============================================================
# QUADRIGRAM
# ============================================================

print("\nQuadrigram probabilities:")


tokens = add_boundaries(
    example_sentence,
    4
)


for i in range(
    len(tokens) - 3
):

    quadrigram = tuple(
        tokens[
            i:i + 4
        ]
    )

    p = quadrigram_model.probability(
        quadrigram
    )

    print(
        f"P({quadrigram[3]} | "
        f"{quadrigram[0]}, "
        f"{quadrigram[1]}, "
        f"{quadrigram[2]}) "
        f"= {p:.10f}"
    )


# ============================================================
# 20. FINISH
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"All Add-K output files are in: "
    f"{OUTPUT_DIR}"
)

print("\nFiles created:")

print("  train_kannada.txt")
print("  dev_kannada.txt")
print("  test_kannada.txt")
print("  unigram_counts.txt")
print("  bigram_counts.txt")
print("  trigram_counts.txt")
print("  quadrigram_counts.txt")
print("  results.txt")