# NLP_examples

Simple, self-contained Python examples of core NLP concepts, written for first learners.
Each folder focuses on one concept and can be run independently.

## Folders

| Folder | Concept | Description |
|---|---|---|
| [`preprocessing/`](preprocessing/) | Text Preprocessing | Lowercasing, punctuation removal, tokenisation, stop-word filtering, and stemming using only the Python standard library. |
| [`ngram/`](ngram/) | N-Gram Language Model | Build bigram and trigram counts from a small corpus, compute next-word probabilities, and generate new sentences by sampling. |
| [`classification/`](classification/) | Text Classification | Multinomial Naive Bayes sentiment classifier (positive / negative) implemented from scratch. |
| [`FFNN/`](FFNN/) | Feed-Forward Neural Network | One-hidden-layer MLP (bag-of-words input) trained with backpropagation and NumPy for sentiment classification. |
| [`RNN/`](RNN/) | Recurrent Neural Network | Character-level Vanilla RNN trained with BPTT to generate text, implemented in NumPy. |
| [`transformer/`](transformer/) | Transformer (Self-Attention) | Scaled dot-product self-attention, multi-head attention, positional encoding, and a Transformer encoder block for next-word prediction, implemented in NumPy. |

## Requirements

- **preprocessing**, **ngram**, and **classification** require only the Python standard library (Python 3.7+).
- **FFNN**, **RNN**, and **transformer** require [NumPy](https://numpy.org/):

```bash
pip install numpy
```

## Running an example

```bash
python preprocessing/preprocessing_example.py
python ngram/ngram_example.py
python classification/classification_example.py
python FFNN/ffnn_example.py
python RNN/rnn_example.py
python transformer/transformer_example.py
```
