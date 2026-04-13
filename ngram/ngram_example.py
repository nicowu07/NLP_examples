"""
N-Gram Language Model Example
==============================
An n-gram model predicts the next word based on the previous (n-1) words.

This script shows:
  1. How to build bigram (2-gram) and trigram (3-gram) counts from a corpus.
  2. How to estimate the probability of the next word.
  3. How to generate new text using the model.

No external libraries are required beyond the Python standard library.
"""

import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Mini corpus (a list of sentences already tokenised into word lists)
# ---------------------------------------------------------------------------
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "cat", "ate", "the", "rat"],
    ["the", "rat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["the", "dog", "ate", "the", "cat"],
    ["the", "cat", "chased", "the", "dog"],
]

# ---------------------------------------------------------------------------
# 2. Build n-gram counts
#    We use special <START> and <END> tokens to mark sentence boundaries.
# ---------------------------------------------------------------------------
def build_ngram_counts(corpus, n):
    """
    Count how often each n-gram appears.

    Returns a dict mapping (n-1)-gram context → {next_word: count}.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        # Pad with start/end tokens
        padded = ["<START>"] * (n - 1) + sentence + ["<END>"]
        for i in range(len(padded) - n + 1):
            context = tuple(padded[i : i + n - 1])   # the (n-1) preceding words
            next_word = padded[i + n - 1]
            counts[context][next_word] += 1
    return counts

bigram_counts  = build_ngram_counts(corpus, n=2)
trigram_counts = build_ngram_counts(corpus, n=3)

# ---------------------------------------------------------------------------
# 3. Compute conditional probabilities  P(word | context)
# ---------------------------------------------------------------------------
def ngram_probability(counts, context, word):
    """Return P(word | context) using maximum-likelihood estimation."""
    context_counts = counts[context]
    total = sum(context_counts.values())
    if total == 0:
        return 0.0
    return context_counts[word] / total

# Example: P("cat" | "the") using bigrams
context = ("the",)
word    = "cat"
prob    = ngram_probability(bigram_counts, context, word)
print(f"Bigram P('{word}' | '{context[0]}') = {prob:.4f}")
print()

# Example: P("sat" | "cat") using bigrams
context = ("cat",)
word    = "sat"
prob    = ngram_probability(bigram_counts, context, word)
print(f"Bigram P('{word}' | '{context[0]}') = {prob:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. Text generation using the bigram model
# ---------------------------------------------------------------------------
def generate_text(counts, n, max_words=20, seed=42):
    """Generate a sentence by sampling from the n-gram model."""
    random.seed(seed)
    context = tuple(["<START>"] * (n - 1))
    sentence = []

    for _ in range(max_words):
        next_word_counts = counts[context]
        if not next_word_counts:
            break
        # Sample proportionally to counts
        words  = list(next_word_counts.keys())
        freqs  = list(next_word_counts.values())
        next_w = random.choices(words, weights=freqs, k=1)[0]
        if next_w == "<END>":
            break
        sentence.append(next_w)
        # Slide the context window
        context = tuple(list(context[1:]) + [next_w])

    return " ".join(sentence)

print("=== Generated sentences ===")
print("Bigram  :", generate_text(bigram_counts,  n=2, seed=0))
print("Bigram  :", generate_text(bigram_counts,  n=2, seed=7))
print("Trigram :", generate_text(trigram_counts, n=3, seed=0))
print("Trigram :", generate_text(trigram_counts, n=3, seed=7))
