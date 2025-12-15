"""
Data preparation utilities for the MSIS-822 project.

Note: Function bodies are copied verbatim from the notebook `phase 3 and 4 updated.ipynb`.
"""

import re
import regex as re2
import numpy as np

# NLTK stopwords + stemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Ensure stopwords are available
try:
    _ = stopwords.words("arabic")
except LookupError as e:
    raise LookupError(
        "NLTK stopwords corpus not found. Run: import nltk; nltk.download('stopwords')"
    ) from e

# Initialize stopwords and stemmer (mirrors notebook initialisation)
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

#Define Arabic text cleaning functions
# Remove tashkeel (diacritics)
def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)

# Normalize Arabic text
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # remove non-Arabic chars
    return text

# Full preprocessing pipeline
def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

#Helper functions


def simple_word_tokenize(text):
    """
    Tokenize text into words / symbols with Arabic support.
    """
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)

def sentence_tokenize(text):
    """
    Split text into sentences using Arabic/English punctuation.
    """
    if not isinstance(text, str):
        return []
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def paragraph_tokenize(text):
    """
    Split text into paragraphs based on double newlines.
    """
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]

