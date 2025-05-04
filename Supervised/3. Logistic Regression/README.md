# Logistic Regression Neuron

This document describes the implementation of a single neuron for logistic regression.

## 1. Introduction
The code implements a basic logistic regression model using a single neuron.  This neuron learns to predict the probability of a binary outcome (0 or 1) based on a linear combination of input features, followed by a sigmoid activation function.

## 2. Mathematical Formulation
The model calculates a weighted sum of the input features, applies the sigmoid function to this sum to obtain a probability, and then classifies the input based on whether this probability exceeds a given threshold.

For a given input vector $\mathbf{x} = [x_1, x_2, ..., x_n]^T$, the neuron calculates the linear combination:

$$z = \mathbf{w}^T \mathbf{x} + b = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

where:
-   $\mathbf{x}$ is the input feature vector.
-   $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ is the weight vector.
-   $b$ is the bias term.

The sigmoid function, $\sigma(z)$, is then applied to this linear combination:

$$P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

This produces a probability between 0 and 1, representing the likelihood that the input $\mathbf{x}$ belongs to class 1.

## 3. Learning Algorithm
The neuron is trained using the Gradient Descent algorithm. The objective is to minimize the Binary Cross-Entropy loss function.

### 3.1 Binary Cross-Entropy Loss
The Binary Cross-Entropy loss (also known as log loss) is defined as:

$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

where:
-   $N$ is the number of samples.
-   $y_i$ is the true label (0 or 1) for the $i$-th sample.
-   $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$ is the predicted probability that the $i$-th sample belongs to class 1.

### 3.2 Gradient Descent
Gradient Descent is an iterative optimization algorithm that finds the minimum of a function by repeatedly moving in the direction of the steepest descent, defined by the negative of the gradient.

The update rules for the weights and bias are:

$$w_j = w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

where:
-   $\alpha$ is the learning rate.

The partial derivatives of the Binary Cross-Entropy loss with respect to the weights and bias are calculated as follows:

$$\frac{\partial J}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_{ij}$$
$$\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

These update rules are applied iteratively until the model converges to a minimum or a maximum number of iterations is reached.

## 4. Implementation Details
The `LogisticRegressionNeuron` class implements the following methods:

-   `__init__(self, learning_rate=0.01, n_iters=1000)`: Initializes the neuron with a learning rate and the number of iterations.
-   `fit(self, X, y)`: Trains the neuron on the given data using gradient descent.
-   `predict(self, X, threshold=0.5)`: Predicts the class labels for the given input data, using an optional probability threshold.
-   `predict_proba(self, X)`: Predicts the probabilities of the class labels for the given input data.
-   `score(self, X, y)`: Calculates the accuracy of the model on the given data.
-   `_sigmoid(self, x)`: Computes the sigmoid function.

## 5. Example Usage
The following code demonstrates how to use the `LogisticRegressionNeuron` class:

```python
import numpy as np

# Create sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 0, 1])

# Create and train the logistic regression neuron
neuron = LogisticRegressionNeuron(learning_rate=0.1, n_iters=1000)
neuron.fit(X, y)

# Make predictions
predictions = neuron.predict(X)
print("Predictions:", predictions)

# Evaluate the model
accuracy = neuron.score(X, y)
print(f"Accuracy: {accuracy:.2f}")
```