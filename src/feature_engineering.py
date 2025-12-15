"""
Feature engineering utilities for the MSIS-822 project.

Note: Function bodies are copied verbatim from the notebook `phase 3 and 4 updated.ipynb`.
"""

import numpy as np
import regex as re2

from .data_preparation import simple_word_tokenize

# Feature 23: Average Syllables per Word (Appplied on original text columns)
#متوسط عدد المقاطع الصوتية في كل كلمة — يُستخدم لتقييم صعوبة النطق أو القراءة.
# Approximate function to estimate syllables using common vowel grouping
def _estimate_syllables(word):
    """Approximate function to estimate syllables using English vowel grouping."""
    # This function is English-centric. For accurate Arabic NLP, a dedicated Arabic syllabification tool is required.
    vowels = "aeiouAEIOU"
    count = 0
    previous_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not previous_char_was_vowel:
                count += 1
                previous_char_was_vowel = True
        else:
            previous_char_was_vowel = False
    return max(count, 1) # Assumes at least 1 syllable per valid word

def _avg_syllables_per_word(words):
    if not words:
        return 0.0
    return np.mean([_estimate_syllables(w) for w in words])



# Feature 46: Number of adverbs (approx) بإستخدام قواعد و قاموس للظرف
def _count_possible_adverbs(words):
    """Approximate count of adverbs based on common suffixes and list."""
    if not isinstance(words, list):
        return 0
    suffixes = ('ًا', 'اً', 'ا')
    common_adverbs = {'الآن', 'دائمًا', 'غالبًا', 'أحيانًا', 'سريعًا', 'متأخرًا',
                      'مبكرًا', 'هناك', 'هنا', 'غدًا', 'اليوم', 'حقًا', 'فعلا'}

    return sum(
        1 for w in words
        if any(w.endswith(suf) for suf in suffixes) or w in common_adverbs
    )

#92. Emotional Valence Score: Mean emotional valence of words.
import regex as re2
import numpy as np

#  مثال على معجم مشاعر عربي (عينة)
# القيم تتراوح بين -1 (سلبي) إلى +1 (إيجابي)
arabic_sentiment_lexicon = {
    "سعيد": 1.0,
    "حزين": -1.0,
    "ممتاز": 0.8,
    "سيء": -0.8,
    "فرح": 0.9,
    "غضب": -0.9,
    # يمكن إضافة آلاف الكلمات لاحقًا
}


#  دالة لحساب متوسط Emotional Valence لكل نص
def emotional_valence_score(text):
    words = simple_word_tokenize(text)
    scores = [arabic_sentiment_lexicon[w] for w in words if w in arabic_sentiment_lexicon]
    if len(scores) == 0:
        return 0
    return np.mean(scores)


