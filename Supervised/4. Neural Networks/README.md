# Neural Network

This document describes the implementation of a basic, feedforward neural network.

## 1. Introduction
The code implements a simple, feedforward neural network.  This type of network is characterized by a series of layers where information flows in one direction: from the input layer, through one or more hidden layers, to the output layer.  Each neuron in one layer is connected to all neurons in the next layer.  The network can be used for both classification and regression tasks.

## 2. Network Architecture
The architecture of the neural network is defined by the following:

-   **Input Layer:** The number of neurons in the input layer is determined by the number of features in the input data.
-   **Hidden Layers:** The network can have one or more hidden layers. The number of hidden layers and the number of neurons in each hidden layer are specified by the user.
-   **Output Layer:** The number of neurons in the output layer depends on the task. For example, in a binary classification problem, the output layer typically has one neuron, while in a multi-class classification problem, it may have one neuron per class.

## 3. Activation Functions
Each neuron applies an activation function to its weighted sum of inputs. The following activation functions are supported:

-   **Sigmoid:**
    
    $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
-   **ReLU (Rectified Linear Unit):**
    
    $$f(x) = \max(0, x)$$
-   **Tanh (Hyperbolic Tangent):**
    
    $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The choice of activation function can significantly impact the network's performance.  The sigmoid function is commonly used for binary classification, while ReLU is popular for many other tasks due to its simplicity and efficiency. Tanh is another option that can be used

## 4. Learning Algorithm
The network is trained using the backpropagation algorithm, which is a supervised learning method that calculates the gradient of the loss function with respect to the network's weights and biases.  These gradients are then used to update the weights and biases to minimize the loss function.

### 4.1 Forward Pass
During the forward pass, the input data is propagated through the network layer by layer.  At each layer, the neurons compute a weighted sum of their inputs, apply the activation function, and pass the result to the next layer.

### 4.2 Backward Pass
During the backward pass, the error between the predicted output and the actual target values is calculated.  This error is then propagated backward through the network, layer by layer, to compute the gradients of the loss function with respect to the weights and biases.  The chain rule of calculus is used to calculate these gradients.

### 4.3 Parameter Update
The weights and biases are updated using the gradient descent algorithm:

$$w_{ij} = w_{ij} - \alpha \frac{\partial L}{\partial w_{ij}}$$
$$b_i = b_i - \alpha \frac{\partial L}{\partial b_i}$$

where:
-  $w_{ij}$ is the weight connecting neuron j in the previous layer to neuron i in the current layer.
-  $b_i$ is the bias of neuron i in the current layer.
-  $\alpha$ is the learning rate, which controls the step size of the updates.
-  $L$ is the loss function (e.g., mean squared error for regression, cross-entropy for classification).

## 5. Implementation Details

The `NeuralNetwork` class implements the following:

-   `__init__(self, n_input, hidden_dims, n_output, learning_rate=0.01, n_iters=1000, activation_function='sigmoid')`:
    * Initializes the neural network with the number of input features, the number of neurons in each hidden layer, the number of output units, the learning rate, the number of iterations, and the activation function.
-    `_initialize_weights(self)`:
        * Initializes the weights and biases of the network with small random values.
-   `fit(self, X, y)`:
    * Trains the neural network on the given data using backpropagation.
-   `predict(self, X)`:
    * Predicts the output for the given data by performing a forward pass through the network.
-   `_sigmoid(self, x)`:
    * Computes the sigmoid function.
-   `_relu(self, x)`:
        * Computes the ReLU function
-   `_tanh(self, x)`:
        * Computes the tanh function
-   `_sigmoid_derivative(self, x)`:
    * Computes the derivative of the sigmoid function.
-   `_relu_derivative(self, x)`:
        * Computes the derivative of the ReLU function.
-   `_tanh_derivative(self, x)`:
        * Computes the derivative of the tanh function.
-   `_forward_pass(self, X)`:
    * Performs a forward pass through the network.
-   `_backward_pass(self, X, y, activations)`:
    * Performs backpropagation to compute the gradients.
-   `_update_parameters(self, gradients)`:
    * Updates the weights and biases based on the computed gradients.

## 6. Example Usage

Here's an example of how to use the `NeuralNetwork` class:

```python
import numpy as np

# Create sample data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])  # Example: XOR function

# Create a neural network
nn = NeuralNetwork(n_input=2, hidden_dims=[4, 4], n_output=1, learning_rate=0.1, n_iters=10000, activation_function='sigmoid')

# Train the network
nn.fit(X_train, y_train)

# Make predictions
predictions = nn.predict(X_train)
print("Predictions:", predictions)
```