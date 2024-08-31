# MNIST Digit Classification and Odd/Even Classification using Transfer Learning

This project demonstrates how to use transfer learning to classify MNIST digits and then classify them as odd or even numbers using a pre-trained model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Transfer Learning](#transfer-learning)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)

## Introduction

The MNIST dataset is a collection of 70,000 handwritten digits commonly used for training various image processing systems. In this project, we first train a Convolutional Neural Network (CNN) to classify these digits. We then use transfer learning to classify the digits as odd or even numbers.

## Dataset

The MNIST dataset is loaded using TensorFlow's Keras API. It consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

## Model Architecture

### Digit Classification Model

The model for digit classification consists of the following layers:

1. Convolutional Layer with 32 filters, kernel size of 3x3, and ReLU activation
2. MaxPooling Layer with pool size of 2x2
3. Convolutional Layer with 64 filters, kernel size of 3x3, and ReLU activation
4. MaxPooling Layer with pool size of 2x2
5. Convolutional Layer with 64 filters, kernel size of 3x3, and ReLU activation
6. Flatten Layer
7. Dense Layer with 64 units and ReLU activation
8. Dense Layer with 10 units and Softmax activation (for 10 digit classes)

### Odd/Even Classification Model

The model for odd/even classification reuses the convolutional layers from the digit classification model and adds a new dense layer for binary classification:

1. Convolutional Layers from the pre-trained digit classification model (frozen)
2. Dense Layer with 2 units and Softmax activation (for odd/even classification)

## Training

### Digit Classification

The digit classification model is trained for 5 epochs using the Adam optimizer and categorical cross-entropy loss.

### Odd/Even Classification

The odd/even classification model is trained for 5 epochs using the Adam optimizer and categorical cross-entropy loss. The convolutional layers from the pre-trained model are frozen to retain their learned features.

## Transfer Learning

Transfer learning is used to leverage the pre-trained digit classification model for the odd/even classification task. The convolutional layers from the pre-trained model are reused, and only the new dense layer is trained.

## Evaluation

The odd/even classification model is evaluated on the test set, and the test accuracy is printed.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/irtaza302/mnist-transfer-learning.git
    cd mnist-transfer-learning
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python mnist_transfer_learning.py
    ```

## Requirements

- TensorFlow
- NumPy

You can install the required packages using the following command:
```bash
pip install tensorflow numpy
