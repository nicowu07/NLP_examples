"""
Text Classification Example – Naive Bayes
==========================================
Naive Bayes is one of the simplest and most effective classifiers for text.
It works by estimating P(class | document) using Bayes' theorem and assumes
that each word contributes independently to the probability (the "naive"
assumption).

This script shows:
  1. Building a bag-of-words vocabulary from training data.
  2. Training a Multinomial Naive Bayes classifier from scratch.
  3. Classifying new sentences.

No external libraries are required beyond the Python standard library.
"""

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Training data  (sentence, label) pairs
#    Labels: "positive" or "negative"
# ---------------------------------------------------------------------------
training_data = [
    ("I love this movie it is great",       "positive"),
    ("This film is wonderful and amazing",  "positive"),
    ("Fantastic performance and great plot","positive"),
    ("I enjoyed every moment of this film", "positive"),
    ("Best movie I have ever seen",         "positive"),
    ("I hate this movie it is terrible",    "negative"),
    ("Awful film boring and dull",          "negative"),
    ("Worst movie ever complete waste",     "negative"),
    ("I did not enjoy this film at all",    "negative"),
    ("Terrible acting and poor story",      "negative"),
]

# ---------------------------------------------------------------------------
# 2. Helper: simple tokenizer
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list:
    return text.lower().split()

# ---------------------------------------------------------------------------
# 3. Train Multinomial Naive Bayes
# ---------------------------------------------------------------------------
class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts    = defaultdict(int)         # how many docs per class
        self.word_counts     = defaultdict(lambda: defaultdict(int))  # word freq per class
        self.vocab           = set()
        self.total_docs      = 0

    def train(self, data):
        for text, label in data:
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.total_docs          += 1
            for token in tokens:
                self.word_counts[label][token] += 1
                self.vocab.add(token)

    def _log_prob(self, tokens, label):
        """Compute log P(tokens | label) using Laplace (add-1) smoothing."""
        total_words_in_class = sum(self.word_counts[label].values())
        vocab_size           = len(self.vocab)
        log_p = 0.0
        for token in tokens:
            count   = self.word_counts[label][token]
            # Laplace smoothing: add 1 to every word count
            log_p  += math.log((count + 1) / (total_words_in_class + vocab_size))
        return log_p

    def predict(self, text: str) -> str:
        tokens    = tokenize(text)
        best_label, best_score = None, float("-inf")
        for label, count in self.class_counts.items():
            # Prior: log P(class)
            log_prior = math.log(count / self.total_docs)
            # Likelihood: log P(tokens | class)
            log_likelihood = self._log_prob(tokens, label)
            score = log_prior + log_likelihood
            if score > best_score:
                best_score, best_label = score, label
        return best_label

# ---------------------------------------------------------------------------
# 4. Train and test
# ---------------------------------------------------------------------------
clf = NaiveBayesClassifier()
clf.train(training_data)

test_sentences = [
    "This movie is great and I love it",
    "Terrible film I hated every moment",
    "The acting was fantastic",
    "Boring and dull waste of time",
]

print("=== Naive Bayes Text Classification ===\n")
for sentence in test_sentences:
    label = clf.predict(sentence)
    print(f"  [{label:8s}]  \"{sentence}\"")
