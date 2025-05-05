# Neural Network Library from Scratch

## Description

This repository provides a Python implementation of a feedforward neural network built from the ground up using only the NumPy library. It enables users to define network architectures, initialize weights appropriately, train the network using backpropagation with the Adam optimizer, and evaluate its performance on provided datasets [2].

## Features

*   **Flexible Architecture:** Define networks with custom numbers of layers and nodes per layer [2].
*   **Activation Functions:** Supports ReLU, LeakyReLU, Sigmoid, and SoftMax activation functions [3].
*   **Cost Functions:** Includes Mean Squared Error (MSE), Mean Absolute Error (MAE), and Cross-Entropy loss [3]. *Note: MSE and MAE in the current implementation seem designed for single-value outputs [3].*
*   **Weight Initialization:** Implements He initialization (for ReLU/LeakyReLU) and Xavier/Glorot initialization (for Sigmoid) to promote stable training [2].
*   **Backpropagation:** Core algorithm for gradient computation is implemented, handling derivatives for different activation and cost function combinations [2] [3].
*   **Adam Optimizer:** Utilizes the Adam optimization algorithm for efficient gradient-based weight updates during training [2].
*   **Mini-Batch Training:** Supports training with mini-batches and shuffling of data each epoch [2].
*   **L2 Regularization:** Option to include L2 weight regularization during training to prevent overfitting [2].
*   **Evaluation:** Provides methods to calculate the total cost (loss) and accuracy (using HardMax for classification with SoftMax output) on datasets [2].
*   **Modular Design:** Code is organized into distinct modules:
    *   `Layers.py`: Defines the `Layer` class, holding weights, biases, and activation for each layer [1].
    *   `Functions.py`: Defines the `function` class, providing activation and cost functions along with their derivatives [3].
    *   `Network.py`: Contains the main `network` class orchestrating initialization, forward/backward passes, training, and evaluation [2].

## Dependencies

*   Python 3.x
*   NumPy

## Core Components

*   **`Network.py`**: The central script defining the `network` class. Manages network layers, initialization, forward propagation (`compute_network`, `forward_pass`), backpropagation (`back_prop`, `batch_prop`), the training loop (`train`), and evaluation methods (`cost`, `total_cost`, `hardmax`) [2].
*   **`Layers.py`**: Defines the `Layer` class. Each layer object stores its weights (`w`), biases (`b`), node count (`nodes`), and the associated activation function (`function`). It includes a `compute` method for calculating the layer's output [1].
*   **`Functions.py`**: Defines the `function` class, which acts as a provider for various activation functions (`normal` method) and cost functions (`cost_normal` method), along with their respective derivatives (`derivative`, `cost_der`) needed for backpropagation [3].

## Basic Usage

