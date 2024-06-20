# Sentiment Analysis using Word Embeddings and Continuous Bag of Words (CBOW) Model

## Introduction

This project aims to compute word embeddings and use them for sentiment analysis. Instead of counting the number of positive and negative words, the approach focuses on representing each word numerically using vectors that capture syntactic (parts of speech) and semantic (meaning) structures. The CBOW model, a popular word embedding method, is implemented to achieve this goal.

## Continuous Bag of Words (CBOW) Model

The CBOW model predicts the center word in a given context of surrounding words. For instance, in the sentence:

`I am happy because I am learning`

With a context window size of \( C = 2 \), the model predicts the word "happy" using the context words `[I, am, because, I]`.

### Model Architecture

The model architecture consists of the following steps:

1. **Input Layer**: The input is the average of the one-hot vectors of the context words.
2. **Hidden Layer**: 
   \[
   h = W_1 \cdot X + b_1
   \]
   Where \( X \) is the input vector, \( W_1 \) is the weight matrix, and \( b_1 \) is the bias vector.
3. **Activation Function**: 
   \[
   h = \text{ReLU}(h)
   \]
   The Rectified Linear Unit (ReLU) activation function is applied element-wise: \( f(h) = \max(0, h) \).
4. **Output Layer**: 
   \[
   z = W_2 \cdot h + b_2
   \]
   The output layer produces a probability distribution over the vocabulary using the softmax function:
   \[
   \hat{y} = \text{softmax}(z)
   \]

### Forward Propagation

The forward propagation involves computing the following:

1. Compute the hidden layer:
   \[
   h = \text{ReLU}(W_1 \cdot X + b_1)
   \]
2. Compute the output layer:
   \[
   z = W_2 \cdot h + b_2
   \]

### Backpropagation and Training

The model is trained using backpropagation and gradient descent. The loss is computed using cross-entropy, and the weights are updated accordingly.