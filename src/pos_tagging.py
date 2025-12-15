"""
POS tagging utilities (Stanza).

Note: `run_stanza_pos` function body is copied verbatim from the notebook `phase 3 and 4 updated.ipynb`.
"""

import stanza

def build_stanza_pipeline(use_gpu: bool = False):
    """Convenience wrapper to build the same pipeline used in the notebook."""
    return stanza.Pipeline(lang='ar', processors='tokenize,pos', tokenize_pretokenized=False, use_gpu=use_gpu)

#Function to run Stanza POS tagging
def run_stanza_pos(text, nlp):
    """
    Runs POS tagging on Arabic text using Stanza.
    Returns a list of (word, POS) tuples.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    doc = nlp(text)
    tags = []
    for sent in doc.sentences:
        for word in sent.words:
            tags.append((word.text, word.upos))  # Universal POS tag
    return tags
