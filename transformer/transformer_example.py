"""
Transformer – Scaled Dot-Product Self-Attention
================================================
The Transformer architecture (Vaswani et al., 2017) replaced recurrent
layers with self-attention, which lets every position in a sequence
directly attend to every other position.

This script demonstrates the key building blocks step by step:
  1. Token embeddings and positional encoding
  2. Scaled dot-product self-attention (with attention weight visualisation)
  3. Multi-head attention
  4. A complete Transformer encoder block (attention + residual + FFN)
  5. A tiny next-token prediction model trained with SGD

Requires: numpy  (install with `pip install numpy`)
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True)
    return (x - mean) / (std + eps)

# ---------------------------------------------------------------------------
# 1. Vocabulary and token embeddings
# ---------------------------------------------------------------------------
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "cat", "ate", "the", "rat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["the", "dog", "ate", "the", "cat"],
]

vocab    = ["<PAD>"] + sorted({w for s in sentences for w in s})
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V        = len(vocab)
d_model  = 8       # small embedding size for readability

E = np.random.randn(V, d_model) * 0.1   # (V, d_model)

print("=== 1. Vocabulary and Embeddings ===")
print(f"Vocabulary ({V} words): {vocab}\n")

# ---------------------------------------------------------------------------
# 2. Positional encoding (sinusoidal)
# ---------------------------------------------------------------------------
def positional_encoding(seq_len, d_model):
    """Return a (seq_len, d_model) matrix of sinusoidal positional encodings."""
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i]   = np.sin(angle)
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(angle)
    return PE

# Encode the sentence "the cat sat on"
demo_sentence = ["the", "cat", "sat", "on"]
demo_ids = [word2idx[w] for w in demo_sentence]
seq_len  = len(demo_ids)
PE       = positional_encoding(seq_len, d_model)
X        = E[demo_ids] + PE          # shape: (seq_len, d_model)

print("=== 2. Input Embeddings + Positional Encoding ===")
print(f"Sentence : {demo_sentence}")
print(f"Shape    : {X.shape}  (seq_len={seq_len}, d_model={d_model})")
print(f"First token embedding+PE :\n  {X[0].round(3)}\n")

# ---------------------------------------------------------------------------
# 3. Scaled dot-product self-attention (single head)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V_mat):
    """
    Compute attention for a single head.
    Q, K : (seq_len, d_k)
    V_mat: (seq_len, d_k)
    Returns: output (seq_len, d_k), attention weights (seq_len, seq_len)
    """
    d_k     = Q.shape[-1]
    scores  = Q @ K.T / np.sqrt(d_k)      # similarity scores
    weights = softmax(scores, axis=-1)     # attention weights (sum to 1 per row)
    output  = weights @ V_mat             # weighted sum of values
    return output, weights

# Small Q, K, V projection matrices for the demo
d_k  = 4
Wq_s = np.random.randn(d_model, d_k) * 0.1
Wk_s = np.random.randn(d_model, d_k) * 0.1
Wv_s = np.random.randn(d_model, d_k) * 0.1

Q = X @ Wq_s   # (seq_len, d_k)
K = X @ Wk_s
Vm = X @ Wv_s

attn_out, attn_weights = scaled_dot_product_attention(Q, K, Vm)

print("=== 3. Scaled Dot-Product Self-Attention ===")
print(f"Attention weight matrix (how much each word attends to every other):")
header = "         " + "  ".join(f"{w:>5}" for w in demo_sentence)
print(header)
for i, row in enumerate(attn_weights):
    row_str = "  ".join(f"{v:5.3f}" for v in row)
    print(f"  {demo_sentence[i]:<6}  {row_str}")
print()

# ---------------------------------------------------------------------------
# 4. Multi-head attention  (2 heads)
# ---------------------------------------------------------------------------
n_heads = 2
assert d_model % n_heads == 0
dk = d_model // n_heads   # dimension per head

Wq_mh = np.random.randn(d_model, d_model) * 0.1
Wk_mh = np.random.randn(d_model, d_model) * 0.1
Wv_mh = np.random.randn(d_model, d_model) * 0.1
Wo_mh = np.random.randn(d_model, d_model) * 0.1

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    L = X.shape[0]
    Q_full = X @ Wq   # (L, d_model)
    K_full = X @ Wk
    V_full = X @ Wv

    Q_h = Q_full.reshape(L, n_heads, dk).transpose(1, 0, 2)  # (h, L, dk)
    K_h = K_full.reshape(L, n_heads, dk).transpose(1, 0, 2)
    V_h = V_full.reshape(L, n_heads, dk).transpose(1, 0, 2)

    head_outputs = []
    for h in range(n_heads):
        out, _ = scaled_dot_product_attention(Q_h[h], K_h[h], V_h[h])
        head_outputs.append(out)

    concat = np.concatenate(head_outputs, axis=-1)   # (L, d_model)
    return concat @ Wo

mha_out = multi_head_attention(X, Wq_mh, Wk_mh, Wv_mh, Wo_mh, n_heads)
print("=== 4. Multi-Head Attention ===")
print(f"Input shape : {X.shape}")
print(f"Output shape: {mha_out.shape}  (same as input)\n")

# ---------------------------------------------------------------------------
# 5. Full Transformer encoder block
# ---------------------------------------------------------------------------
d_ff = 16
W1_ff = np.random.randn(d_model, d_ff) * 0.1
b1_ff = np.zeros(d_ff)
W2_ff = np.random.randn(d_ff, d_model) * 0.1
b2_ff = np.zeros(d_model)

def feed_forward(X):
    return np.maximum(0, X @ W1_ff + b1_ff) @ W2_ff + b2_ff  # ReLU + linear

def transformer_block(X):
    attn = multi_head_attention(X, Wq_mh, Wk_mh, Wv_mh, Wo_mh, n_heads)
    X    = layer_norm(X + attn)    # Add & Norm
    ff   = feed_forward(X)
    X    = layer_norm(X + ff)      # Add & Norm
    return X

enc_out = transformer_block(X)
print("=== 5. Transformer Encoder Block ===")
print(f"Input  shape : {X.shape}")
print(f"Output shape : {enc_out.shape}  (context-aware representations)\n")

# ---------------------------------------------------------------------------
# 6. Train a tiny next-token predictor (SGD with analytic gradients)
# ---------------------------------------------------------------------------
print("=== 6. Training a Tiny Next-Token Predictor ===\n")

# Build training pairs: (input_ids, target_id)
# For each sentence we take every (prefix → next word) pair
train_pairs = []
for sentence in sentences:
    ids = [word2idx[w] for w in sentence]
    for i in range(1, len(ids)):
        train_pairs.append((ids[:i], ids[i]))

# Represent input as average of word embeddings (simple but fast)
# We keep only the embedding matrix E and a classification head W_cls.
W_cls = np.random.randn(d_model, V) * 0.1
b_cls = np.zeros(V)

def forward_simple(input_ids):
    """Average of embeddings → linear head → softmax."""
    emb    = E[input_ids].mean(axis=0)    # (d_model,)
    logits = emb @ W_cls + b_cls          # (V,)
    probs  = softmax(logits)
    return probs, emb

lr_train = 0.05
for epoch in range(1, 401):
    total_loss = 0.0
    for input_ids, target_id in train_pairs:
        probs, emb = forward_simple(input_ids)
        loss       = -np.log(probs[target_id] + 1e-9)
        total_loss += loss

        # Gradient of cross-entropy loss w.r.t. logits
        dlogits          = probs.copy()
        dlogits[target_id] -= 1.0

        # Update W_cls and b_cls
        W_cls -= lr_train * np.outer(emb, dlogits)
        b_cls -= lr_train * dlogits

        # Update embeddings for the input tokens
        demb = dlogits @ W_cls.T
        for idx in input_ids:
            E[idx] -= lr_train * demb / len(input_ids)

    avg_loss = total_loss / len(train_pairs)
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.4f}")

# ---------------------------------------------------------------------------
# 7. Predict the next word
# ---------------------------------------------------------------------------
def predict_next(context_words):
    ids   = [word2idx.get(w, 0) for w in context_words]
    probs, _ = forward_simple(ids)
    pred  = idx2word[int(np.argmax(probs))]
    return pred

print("\n=== 7. Next-Word Predictions ===\n")
test_contexts = [
    ["the", "cat", "sat", "on", "the"],
    ["the", "dog", "sat", "on", "the"],
    ["the", "cat", "ate", "the"],
]
for ctx in test_contexts:
    pred = predict_next(ctx)
    print(f"  Context: {' '.join(ctx)!r:35s} → Predicted next: '{pred}'")
