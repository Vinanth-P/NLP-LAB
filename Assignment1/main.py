import re
from datasets import load_dataset

# -------------------------------
# Sentence Tokenizer
# -------------------------------

def sentence_tokenize(text):
    """
    Splits paragraph into sentences.
    Supports Hindi danda and English punctuation.
    """
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r'(?<=[।.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


# -------------------------------
# Word Tokenizer
# -------------------------------

TOKEN_PATTERN = re.compile(
    r'''
    https?://\S+                      | # URL
    www\.\S+                          | # www URL
    [\w\.-]+@[\w\.-]+\.\w+            | # Email
    \d{1,2}[/-]\d{1,2}[/-]\d{2,4}     | # Dates
    \d{4}[/-]\d{1,2}[/-]\d{1,2}       | # Dates
    \d+\.\d+                          | # Decimal
    \d+                               | # Integer
    [^\W\d_]+                         | # Hindi/English words
    [^\s]                               # Punctuation
    ''',
    re.VERBOSE | re.UNICODE
)


def word_tokenize(sentence):
    return TOKEN_PATTERN.findall(sentence)


# -------------------------------
# Main
# -------------------------------

dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    name="indiccorp_v2",
    split="hin_Deva",
    streaming=True
)

output_file = "tokenized_hindi.txt"

total_sentences = 0
total_words = 0
total_characters = 0
unique_tokens = set()

MAX_PARAGRAPHS = 10000

with open(output_file, "w", encoding="utf-8") as fout:

    for idx, example in enumerate(dataset):

        if idx >= MAX_PARAGRAPHS:
            break

        text = example["text"]

        sentences = sentence_tokenize(text)

        for sentence in sentences:

            tokens = word_tokenize(sentence)

            if len(tokens) == 0:
                continue

            fout.write(" ".join(tokens) + "\n")

            total_sentences += 1
            total_words += len(tokens)

            for token in tokens:
                total_characters += len(token)
                unique_tokens.add(token)

# -------------------------------
# Statistics
# -------------------------------

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

print("=" * 40)
print("Corpus Statistics")
print("=" * 40)
print(f"Total Sentences      : {total_sentences}")
print(f"Total Words          : {total_words}")
print(f"Total Characters     : {total_characters}")
print(f"Average Sent Length  : {avg_sentence_length:.2f}")
print(f"Average Word Length  : {avg_word_length:.2f}")
print(f"Unique Tokens        : {len(unique_tokens)}")
print(f"Type Token Ratio     : {ttr:.4f}")
print("=" * 40)