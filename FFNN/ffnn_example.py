"""
Feed-Forward Neural Network (FFNN) for Text Classification
===========================================================
A Feed-Forward Neural Network (also called a Multi-Layer Perceptron, MLP)
is the simplest kind of neural network.  Data flows in one direction: from
input → hidden layer(s) → output, with no loops.

This script shows:
  1. Representing a document as a bag-of-words vector.
  2. A one-hidden-layer FFNN implemented in pure NumPy.
  3. Training with stochastic gradient descent (SGD) + backpropagation.
  4. Predicting sentiment (positive / negative).

Requires: numpy  (install with `pip install numpy`)
"""

import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------
training_data = [
    ("I love this movie it is great",       1),
    ("This film is wonderful and amazing",  1),
    ("Fantastic performance and great plot",1),
    ("I enjoyed every moment of this film", 1),
    ("Best movie I have ever seen",         1),
    ("I hate this movie it is terrible",    0),
    ("Awful film boring and dull",          0),
    ("Worst movie ever complete waste",     0),
    ("I did not enjoy this film at all",    0),
    ("Terrible acting and poor story",      0),
]

def tokenize(text):
    return text.lower().split()

# ---------------------------------------------------------------------------
# 2. Build vocabulary and bag-of-words vectors
# ---------------------------------------------------------------------------
vocab = sorted({token for text, _ in training_data for token in tokenize(text)})
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)  # vocabulary size

def text_to_bow(text):
    """Convert a sentence to a normalised bag-of-words vector."""
    vec = np.zeros(V)
    for token in tokenize(text):
        if token in word2idx:
            vec[word2idx[token]] += 1
    if vec.sum() > 0:
        vec /= vec.sum()          # L1 normalisation
    return vec

X = np.array([text_to_bow(t) for t, _ in training_data])   # shape (N, V)
y = np.array([[label] for _, label in training_data],
             dtype=float)                                    # shape (N, 1)

# ---------------------------------------------------------------------------
# 3. FFNN architecture  (V → hidden_size → 1)
# ---------------------------------------------------------------------------
hidden_size = 8
learning_rate = 0.5
epochs = 300

# Weight initialisation (small random values)
np.random.seed(42)
W1 = np.random.randn(V, hidden_size) * 0.1    # input → hidden
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * 0.1    # hidden → output
b2 = np.zeros((1, 1))

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# ---------------------------------------------------------------------------
# 4. Training loop – forward pass + backpropagation
# ---------------------------------------------------------------------------
print("Training FFNN ...")
for epoch in range(epochs):
    # --- Forward pass ---
    z1    = X @ W1 + b1        # (N, hidden_size)
    a1    = relu(z1)           # hidden activations
    z2    = a1 @ W2 + b2       # (N, 1)
    a2    = sigmoid(z2)        # output: predicted probability

    # --- Binary cross-entropy loss ---
    loss  = -np.mean(y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))

    # --- Backward pass ---
    dL_da2 = -(y / (a2 + 1e-9) - (1 - y) / (1 - a2 + 1e-9)) / len(y)
    dL_dz2 = dL_da2 * sigmoid_deriv(z2)

    dL_dW2 = a1.T @ dL_dz2
    dL_db2 = dL_dz2.sum(axis=0, keepdims=True)
    dL_da1 = dL_dz2 @ W2.T
    dL_dz1 = dL_da1 * relu_deriv(z1)
    dL_dW1 = X.T @ dL_dz1
    dL_db1 = dL_dz1.sum(axis=0, keepdims=True)

    # --- Gradient descent update ---
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch + 1:4d} | Loss: {loss:.4f}")

print()

# ---------------------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------------------
def predict(text):
    x   = text_to_bow(text).reshape(1, -1)
    a1  = relu(x @ W1 + b1)
    a2  = sigmoid(a1 @ W2 + b2)
    prob = float(a2[0, 0])
    label = "positive" if prob >= 0.5 else "negative"
    return label, prob

print("=== FFNN Predictions ===\n")
test_sentences = [
    "This movie is great and I love it",
    "Terrible film I hated every moment",
    "The acting was fantastic",
    "Boring and dull waste of time",
]
for sentence in test_sentences:
    label, prob = predict(sentence)
    print(f"  [{label:8s}] (p={prob:.3f})  \"{sentence}\"")
