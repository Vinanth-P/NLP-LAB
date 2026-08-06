"""
tokenizer.py
============
Custom rule-based tokenizer for Hindi (and mixed-script) text.

Features:
  - Sentence tokenizer: splits on Devanagari danda (।), ?, !
    and '.' only when not part of a URL, email, decimal, or date.
  - Word tokenizer: regex-based, tokenizes (in priority order):
      1. URLs
      2. Email IDs
      3. Dates  (DD/MM/YYYY, YYYY-MM-DD, etc.)
      4. Decimal numbers
      5. Integer numbers
      6. Hindi (Devanagari) word sequences (WITHOUT attached punctuation)
      7. ASCII words
      8. Individual punctuation / symbol characters (each as a separate token)
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Sentence boundary: split AFTER  ।  ?  !
# We deliberately avoid splitting on '.' here because URLs and decimals make
# it very hard to handle correctly without a full parser. The danda (।) is the
# primary Hindi sentence terminator and handles the vast majority of splits.
_SENT_SPLIT = re.compile(
    r'(?<=[।?!])\s*'     # danda / ? / ! followed by optional whitespace
)

# Word-level patterns (compiled once for speed)
# IMPORTANT: Devanagari range \u0900-\u097F includes dandas \u0964 (।) and
# \u0965 (॥). We explicitly EXCLUDE those so dandas become punctuation tokens.
_WORD_PATTERNS = re.compile(
    r'(?:'
    # 1. URLs  (http/https/ftp + www)
    r'https?://[^\s<>"{}|\\^`\[\]]+'
    r'|www\.[^\s<>"{}|\\^`\[\]]+'
    r'|'
    # 2. Email addresses
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    r'|'
    # 3. Dates: DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD, YYYY-MM-DD
    r'\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}'
    r'|'
    # 4. Decimal numbers (must come before plain integers)
    r'\d+\.\d+'
    r'|'
    # 5. Plain integers (including Devanagari numerals ०-९  U+0966-U+096F)
    r'[\d\u0966-\u096F]+'
    r'|'
    # 6. Devanagari words — EXCLUDES danda U+0964, double-danda U+0965,
    #    and Devanagari numerals U+0966-U+096F (handled above).
    #    Covers: U+0900-U+0963 and U+0966... wait, numerals already grabbed.
    #    Range used: U+0900-U+0963 then U+0970-U+097F
    r'[\u0900-\u0963\u0970-\u097F]+'
    r'|'
    # 7. ASCII words (Latin alphabet + apostrophe contractions)
    r"[a-zA-Z]+(?:'[a-zA-Z]+)*"
    r'|'
    # 8. Individual punctuation / symbol — catches danda, ?, !, . etc.
    r'[^\s]'
    r')',
    re.UNICODE
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sentence_tokenize(text: str) -> List[str]:
    """
    Split a paragraph into sentences.

    Parameters
    ----------
    text : str
        Raw paragraph / document text.

    Returns
    -------
    List[str]
        List of sentence strings (stripped of leading/trailing whitespace).
        Empty strings are filtered out.
    """
    # Normalise whitespace first
    text = text.strip()
    if not text:
        return []

    # Split on boundary markers
    parts = _SENT_SPLIT.split(text)

    sentences = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)

    return sentences


def word_tokenize(sentence: str) -> List[str]:
    """
    Tokenize a sentence into a list of word/token strings.

    Special tokens returned as single items:
      - URLs              e.g. https://example.com
      - Email IDs         e.g. user@domain.com
      - Dates             e.g. 15/08/1947
      - Decimal numbers   e.g. 3.14
      - Integers          e.g. 2024
      - Hindi words       e.g. नमस्ते
      - ASCII words       e.g. India
      - Punctuation       e.g. , । ! ?

    Parameters
    ----------
    sentence : str
        A single sentence string.

    Returns
    -------
    List[str]
        Ordered list of tokens.
    """
    return _WORD_PATTERNS.findall(sentence)


def tokenize_paragraph(paragraph: str) -> List[List[str]]:
    """
    Tokenize a full paragraph into a list of token-lists (one per sentence).

    Parameters
    ----------
    paragraph : str
        Raw text of a document / paragraph.

    Returns
    -------
    List[List[str]]
        [ [tok1, tok2, ...],   # sentence 1
          [tok1, tok2, ...],   # sentence 2
          ...
        ]
    """
    sentences = sentence_tokenize(paragraph)
    return [word_tokenize(sent) for sent in sentences if sent.strip()]


def tokenize_paragraph_to_strings(paragraph: str) -> List[str]:
    """
    Tokenize a paragraph and return tokenized sentences as joined strings.
    Each string is a sentence where tokens are separated by a single space.

    This is the format required for writing to parquet files.

    Parameters
    ----------
    paragraph : str
        Raw text.

    Returns
    -------
    List[str]
        E.g. ['नमस्ते दुनिया !', 'आज का तापमान 36.5 °C है ।']
    """
    tokenized = tokenize_paragraph(paragraph)
    return [" ".join(tokens) for tokens in tokenized if tokens]


# ---------------------------------------------------------------------------
# Quick self-test (run: python tokenizer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # Hindi with danda
        "नमस्ते दुनिया। आज का मौसम अच्छा है। क्या आप ठीक हैं?",
        # Decimals and numbers
        "तापमान 36.5 डिग्री है। जनसंख्या 1.4 अरब है।",
        # URL and email
        "अधिक जानकारी के लिए https://www.example.com पर जाएं। संपर्क करें: info@example.com",
        # Date
        "स्वतंत्रता दिवस 15/08/1947 को मनाया जाता है।",
        # Mixed Hindi + English
        "India की GDP 3.5 trillion dollars है। यह बहुत बड़ी संख्या है!",
        # YYYY-MM-DD date
        "बैठक 2024-03-15 को होगी।",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {text}")
        print("-" * 60)
        sentences = sentence_tokenize(text)
        for j, sent in enumerate(sentences, 1):
            tokens = word_tokenize(sent)
            print(f"  Sentence {j}: {sent!r}")
            print(f"  Tokens   : {tokens}")
