import re


# ============================================================
# Sentence Tokenizer
# ============================================================

def sentence_tokenize(text):
    """
    Splits text into sentences.
    Supports Kannada/Indic danda and English punctuation.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r'(?<=[।.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# Word Tokenizer
# ============================================================

TOKEN_PATTERN = re.compile(
    r'''
    https?://\S+                       | # URL
    www\.\S+                           | # www URL
    [\w\.-]+@[\w\.-]+\.\w+             | # Email
    \d{1,2}[/-]\d{1,2}[/-]\d{2,4}      | # Date
    \d{4}[/-]\d{1,2}[/-]\d{1,2}        | # Date
    \d+\.\d+                           | # Decimal
    \d+                                | # Integer
    [\u0C80-\u0CFF]+                   | # Kannada word
    [^\W\d_]+                          | # English word
    [^\s]                              | # Punctuation
    ''',
    re.VERBOSE | re.UNICODE
)


def word_tokenize(sentence):
    return TOKEN_PATTERN.findall(sentence)


# ============================================================
# Files
# ============================================================

INPUT_FILE = "original_kannada.txt"
OUTPUT_FILE = "tokenized_kannada.txt"


# ============================================================
# Statistics
# ============================================================

total_sentences = 0
total_words = 0
total_characters = 0

unique_tokens = set()


# ============================================================
# Read downloaded data
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    text = fin.read()

    # Sentence tokenization
    sentences = sentence_tokenize(text)

    for sentence in sentences:

        # Word tokenization
        tokens = word_tokenize(sentence)

        if not tokens:
            continue

        # Save tokenized sentence
        fout.write(" ".join(tokens) + "\n")

        # Statistics
        total_sentences += 1
        total_words += len(tokens)

        for token in tokens:
            total_characters += len(token)
            unique_tokens.add(token)


# ============================================================
# Calculate Statistics
# ============================================================

avg_sentence_length = (
    total_words / total_sentences
    if total_sentences else 0
)

avg_word_length = (
    total_characters / total_words
    if total_words else 0
)

ttr = (
    len(unique_tokens) / total_words
    if total_words else 0
)


# ============================================================
# Print Statistics
# ============================================================

print("=" * 45)
print("Kannada Corpus Statistics")
print("=" * 45)

print(f"Total Sentences      : {total_sentences}")
print(f"Total Words          : {total_words}")
print(f"Total Characters     : {total_characters}")
print(f"Average Sent Length  : {avg_sentence_length:.2f}")
print(f"Average Word Length  : {avg_word_length:.2f}")
print(f"Unique Tokens        : {len(unique_tokens)}")
print(f"Type Token Ratio     : {ttr:.4f}")

print("=" * 45)

print(f"Original data : {INPUT_FILE}")
print(f"Tokenized data: {OUTPUT_FILE}")