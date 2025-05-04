# Linear Regression Neuron

This document describes the implementation of a single neuron for linear regression.

## 1. Introduction
The code implements a basic linear regression model using a single neuron. This can be considered a simplified neural network with one input layer and one output neuron, without any hidden layers. It learns to predict a continuous output variable based on a linear combination of input features.

## 2. Mathematical Formulation
The model predicts the output $\hat{y}$ for a given input vector $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ using the following equation:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

where:
-  $\mathbf{x}$ is the input feature vector.
-  $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ is the weight vector.
-  $b$ is the bias term.

## 3. Learning Algorithm
The neuron is trained using the Gradient Descent algorithm. The objective is to minimize the Mean Squared Error (MSE) between the predicted values and the actual target values.

### 3.1 Mean Squared Error (MSE)
The MSE is defined as:

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

where:
-  $N$ is the number of samples.
-  $y_i$ is the actual target value for the $i$-th sample.
-  $\hat{y}_i$ is the predicted value for the $i$-th sample.

### 3.2 Gradient Descent
Gradient Descent is an iterative optimization algorithm that finds the minimum of a function by repeatedly moving in the direction of the steepest descent, defined by the negative of the gradient.

The update rules for the weights and bias are:

$$w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}$$
$$b = b - \alpha \frac{\partial MSE}{\partial b}$$

where:
-  $\alpha$ is the learning rate.

The partial derivatives of the MSE with respect to the weights and bias are calculated as follows:

$$\frac{\partial MSE}{\partial w_j} = \frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)(-x_{ij})$$
$$\frac{\partial MSE}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)(-1)$$

These update rules are applied iteratively until the model converges to a minimum or a maximum number of iterations is reached.

## 4. Implementation Details
The `LinearRegressionNeuron` class implements the following methods:
-   `__init__(self, learning_rate=0.01, n_iters=1000)`: Initializes the neuron with a learning rate and the number of iterations.
-   `fit(self, X, y)`: Trains the neuron on the given data using gradient descent.
-   `predict(self, X)`: Predicts the output for the given input data.
-   `score(self, X, y)`: Calculates the coefficient of determination ($R^2$) to evaluate the model's performance.

## 5. Example Usage
The following code demonstrates how to use the `LinearRegressionNeuron` class:

```python
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and train the neuron
neuron = LinearRegressionNeuron(learning_rate=0.01, n_iters=1000)
neuron.fit(X, y)

# Make predictions
X_test = np.array([[6], [7], [8]])
predictions = neuron.predict(X_test)
print("Predictions for X_test:", predictions)

# Evaluate the model
r_squared = neuron.score(X, y)
print(f"R^2 Score: {r_squared:.2f}")
6.  R^2 ScoreThe coefficient of determination (R2) measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 1 indicates a perfect fit.The R2 score is calculated as:R2=1−SStot​SSres​​Where:SSres​ is the sum of squared residuals:  SSres​=∑(yi​−y^​i​)2SStot​ is the total sum of squares: SStot​=∑(yi​−yˉ​)2yˉ​ is the mean of the observed values.
```