"""
NLP Preprocessing Example
=========================
This script demonstrates common text preprocessing steps used in NLP:
  1. Lowercasing
  2. Tokenization (splitting text into words)
  3. Removing punctuation
  4. Removing stop words
  5. Stemming (reducing words to their root form)

No external libraries are required beyond the Python standard library.
"""

import string

# ---------------------------------------------------------------------------
# 1. Sample text
# ---------------------------------------------------------------------------
text = "The quick brown Fox jumps over the lazy Dog! NLP is really fun, isn't it?"

print("Original text:")
print(text)
print()

# ---------------------------------------------------------------------------
# 2. Lowercasing  – makes matching case-insensitive
# ---------------------------------------------------------------------------
text_lower = text.lower()
print("After lowercasing:")
print(text_lower)
print()

# ---------------------------------------------------------------------------
# 3. Remove punctuation
# ---------------------------------------------------------------------------
text_no_punct = text_lower.translate(str.maketrans("", "", string.punctuation))
print("After removing punctuation:")
print(text_no_punct)
print()

# ---------------------------------------------------------------------------
# 4. Tokenization – split into individual words (tokens)
# ---------------------------------------------------------------------------
tokens = text_no_punct.split()
print("Tokens:")
print(tokens)
print()

# ---------------------------------------------------------------------------
# 5. Remove stop words
#    Stop words are very common words that usually carry little meaning
#    (e.g. "the", "is", "over").  Here we use a small hand-crafted list;
#    in practice you would use a library like NLTK or spaCy.
# ---------------------------------------------------------------------------
stop_words = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "over", "really", "isnt",
}

filtered_tokens = [word for word in tokens if word not in stop_words]
print("After removing stop words:")
print(filtered_tokens)
print()

# ---------------------------------------------------------------------------
# 6. Simple stemming
#    Stemming chops common suffixes so that related word forms are grouped
#    together (e.g. "jumps" → "jump").  This is a simplified rule-based
#    version; a proper stemmer (like Porter Stemmer) handles many more cases.
# ---------------------------------------------------------------------------
def simple_stem(word: str) -> str:
    """Remove simple suffixes to get an approximate word stem."""
    for suffix in ("ing", "tion", "ly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word

stemmed_tokens = [simple_stem(word) for word in filtered_tokens]
print("After stemming:")
print(stemmed_tokens)
print()

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("=== Preprocessing Pipeline Summary ===")
print(f"  Original : {text}")
print(f"  Processed: {' '.join(stemmed_tokens)}")
