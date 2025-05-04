# Simple Perceptron

This document explains the mathematical foundations of the simple Perceptron class implemented in the provided Python code.

## 1. Introduction to the Perceptron

The Perceptron is a fundamental building block in neural networks. It's a simple algorithm used for binary classification, meaning it can learn to categorize data into two distinct classes (typically represented as 0 and 1, or -1 and 1 internally).

## 2. The Perceptron Model

A perceptron takes a vector of real-valued inputs, calculates a weighted sum of these inputs, adds a bias term, and then applies an activation function to produce a binary output.

Mathematically, for a given input vector $\mathbf{x} = [x_1, x_2, ..., x_n]^T$, the perceptron calculates a linear combination:

$$z = \mathbf{w}^T \mathbf{x} + b = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

where:
- $\mathbf{x}$ is the input feature vector.
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ is the weight vector. Each weight $w_i$ corresponds to an input feature $x_i$ and determines its importance.
- $b$ is the bias term. The bias allows the decision boundary to be shifted.

The output of the perceptron is then determined by applying a step activation function to this linear combination:

$$\hat{y} = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

Internally, during the training phase, the target labels are often converted to -1 and 1 to simplify the update rule. The prediction in the `predict` method then converts the internal representation back to 0 and 1 for consistency with the original input labels.

## 3. The Learning Process

The perceptron learns by adjusting its weights and bias in response to errors in its predictions. The training process involves iterating through the training data multiple times (epochs or `n_iters`). For each training example $(\mathbf{x}_i, y_i)$, the perceptron makes a prediction $\hat{y}_i$. If the prediction is incorrect, the weights and bias are updated according to the perceptron learning rule.

### 3.1 The Update Rule

The weight and bias update rule is as follows:

$$\mathbf{w} = \mathbf{w} + \alpha (y_i - \hat{y}_i') \mathbf{x}_i$$
$$b = b + \alpha (y_i - \hat{y}_i')$$

where:
- $\alpha$ is the learning rate (`learning_rate` in the code), a positive constant that controls the step size of the updates.
- $y_i$ is the true label for the input $\mathbf{x}_i$ (represented as -1 or 1 during the update).
- $\hat{y}_i'$ is the predicted label for the input $\mathbf{x}_i$ (also in the -1 or 1 representation during the update, based on the sign of the linear output $z$).

In the provided code, the update rule is implemented as:

```python
if y_[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
    self.weights += self.learning_rate * y_[idx] * x_i
    self.bias += self.learning_rate * y_[idx]
```