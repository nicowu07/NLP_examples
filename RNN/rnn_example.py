"""
Recurrent Neural Network (RNN) – Character-Level Text Generation
=================================================================
An RNN processes sequences one step at a time, maintaining a hidden state
that acts as a "memory" of what it has seen so far.  This makes RNNs
well-suited for text, speech, and other sequential data.

This script trains a minimal Vanilla RNN (from scratch with NumPy) to
predict the next character in a short piece of text, then generates new
text by sampling from the trained model.

Requires: numpy  (install with `pip install numpy`)

This is based on the classic "min-char-rnn" idea introduced by Andrej
Karpathy (https://gist.github.com/karpathy/d4dee566867f8291f086).
"""

import numpy as np
import random

# ---------------------------------------------------------------------------
# 1. Tiny training corpus
# ---------------------------------------------------------------------------
text = (
    "the cat sat on the mat "
    "the dog sat on the log "
    "the cat ate the rat "
    "the rat ran away "
) * 5                            # repeat to give the model more data

chars   = sorted(set(text))
vocab_size = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

print(f"Vocabulary ({vocab_size} chars): {''.join(chars)!r}")
print(f"Corpus length: {len(text)} characters\n")

# ---------------------------------------------------------------------------
# 2. Hyperparameters
# ---------------------------------------------------------------------------
hidden_size   = 32
seq_length    = 15       # length of each training window (BPTT truncation)
learning_rate = 0.01
epochs        = 300

# ---------------------------------------------------------------------------
# 3. Model parameters (Xavier-like initialisation)
# ---------------------------------------------------------------------------
np.random.seed(42)
Wxh = np.random.randn(hidden_size, vocab_size)  * 0.01  # input  → hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden → hidden
Why = np.random.randn(vocab_size,  hidden_size) * 0.01  # hidden → output
bh  = np.zeros((hidden_size, 1))
by  = np.zeros((vocab_size,  1))

# Adagrad memory
mWxh = np.zeros_like(Wxh); mWhh = np.zeros_like(Whh)
mWhy = np.zeros_like(Why); mbh  = np.zeros_like(bh); mby = np.zeros_like(by)

# ---------------------------------------------------------------------------
# 4. Forward and backward pass for one chunk
# ---------------------------------------------------------------------------
def loss_fun(inputs, targets, hprev):
    """
    inputs, targets: lists of integer indices of length seq_length
    hprev:           initial hidden state  shape (hidden_size, 1)
    Returns: loss, gradients, last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = hprev.copy()
    loss = 0.0

    # Forward pass
    for t, idx in enumerate(inputs):
        xs[t] = np.zeros((vocab_size, 1)); xs[t][idx] = 1
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)
        ys[t] = Why @ hs[t] + by                 # unnormalized log probs
        exp_y = np.exp(ys[t] - ys[t].max())      # numerically stable softmax
        ps[t] = exp_y / exp_y.sum()
        loss += -np.log(ps[t][targets[t], 0])    # cross-entropy

    # Backward pass (BPTT)
    dWxh = np.zeros_like(Wxh); dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why); dbh  = np.zeros_like(bh); dby = np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy          = ps[t].copy()
        dy[targets[t]] -= 1                  # softmax gradient
        dWhy       += dy @ hs[t].T
        dby        += dy
        dh          = Why.T @ dy + dhnext
        dhraw       = (1 - hs[t] ** 2) * dh  # tanh derivative
        dbh        += dhraw
        dWxh       += dhraw @ xs[t].T
        dWhh       += dhraw @ hs[t - 1].T
        dhnext      = Whh.T @ dhraw

    # Clip gradients to prevent exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def train():
    global Wxh, Whh, Why, bh, by
    global mWxh, mWhh, mWhy, mbh, mby

    hprev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # initial loss

    for epoch in range(epochs):
        # Reset hidden state each epoch (small corpus, start fresh)
        hprev = np.zeros((hidden_size, 1))

        # Sweep through corpus in chunks
        for pos in range(0, len(text) - seq_length - 1, seq_length):
            inputs  = [char2idx[c] for c in text[pos      : pos + seq_length]]
            targets = [char2idx[c] for c in text[pos + 1  : pos + seq_length + 1]]

            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_fun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # Adagrad update
            for param, dparam, mem in [
                (Wxh, dWxh, mWxh), (Whh, dWhh, mWhh), (Why, dWhy, mWhy),
                (bh,  dbh,  mbh),  (by,  dby,  mby),
            ]:
                mem  += dparam ** 2
                param -= learning_rate * dparam / (np.sqrt(mem) + 1e-8)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1:4d} | Smooth loss: {smooth_loss:.4f}")

# ---------------------------------------------------------------------------
# 6. Text generation
# ---------------------------------------------------------------------------
def generate(seed_char, length=80):
    """Generate `length` characters starting from `seed_char`."""
    h = np.zeros((hidden_size, 1))
    x = np.zeros((vocab_size, 1))
    x[char2idx[seed_char]] = 1
    result = [seed_char]

    for _ in range(length):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        exp_y = np.exp(y - y.max())
        p = exp_y / exp_y.sum()
        # Sample from the probability distribution
        idx = np.random.choice(vocab_size, p=p.ravel())
        x = np.zeros((vocab_size, 1)); x[idx] = 1
        result.append(idx2char[idx])

    return "".join(result)

# ---------------------------------------------------------------------------
# 7. Run training and show samples
# ---------------------------------------------------------------------------
print("Training RNN ...\n")
train()

print("\n=== Generated Text Samples ===\n")
np.random.seed(0)
for seed in ["t", "t", "t"]:
    print(" ", generate(seed, length=60))
