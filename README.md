# Transformers from Scratch

This repository is dedicated to building Transformers from scratch using PyTorch and Numpy. Our goal is to demystify the components of the Transformer architecture, as introduced by Vaswani et al. in the seminal paper "Attention is All You Need". This project serves as an educational resource for those looking to understand and implement Transformer models without relying on high-level libraries.

## Table of Contents

- [Introduction](#introduction)
- [Components of the Transformer](#components-of-the-transformer)
  - [Positional Encoding](#positional-encoding)
  - [Multi-Head Attention](#multi-head-attention)
  - [Feed-Forward Networks](#feed-forward-networks)
  - [Normalization and Residual Connections](#normalization-and-residual-connections)
  - [Encoder and Decoder](#encoder-and-decoder)
- [References](#references)

## Introduction

Transformers have revolutionized the field of natural language processing (NLP) and beyond, enabling breakthroughs in machine translation, text generation, and even in image and audio processing tasks. This repository provides step-by-step tutorials and implementations of each component of the Transformer architecture, fully coded in PyTorch and Numpy.

# Components of the Transformer
## Positional Encoding
Positional encodings add information about the position of each token in the sequence. We implement sinusoidal positional encoding as described in the original paper, which allows the model to utilize the order of the tokens.

## Multi-Head Attention
Key to the Transformer architecture, multi-head attention allows the model to focus on different parts of the sequence simultaneously. We dissect the implementation of scaled dot-product attention and extend it to multi-head attention for parallel processing.

## Feed-Forward Networks
Each layer of the Transformer contains a fully connected feed-forward network, which we implement using PyTorch's linear layers. These networks apply two linear transformations with a ReLU activation in between.

## Normalization and Residual Connections
Layer normalization and residual connections are crucial for stabilizing and accelerating training. We explain their roles within the Transformer and demonstrate how to implement them effectively.

## Encoder and Decoder
The Transformer model consists of an encoder to process the input and a decoder to generate the output. We build these components from the ground up, integrating attention, feed-forward networks, and normalization.

# References
1. ![Vaswani, A., et al. (2017). Attention is All You Need. arXiv:1706.03762.](https://arxiv.org/abs/1706.03762)
2. ![Transformers from scratch - CodeEmporium](https://youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&si=LuRzNb0mTis-miL8)
