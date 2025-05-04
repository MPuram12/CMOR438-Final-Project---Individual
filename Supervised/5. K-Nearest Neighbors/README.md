# K-Nearest Neighbors (KNN) Classifier

This document describes the implementation of the K-Nearest Neighbors (KNN) classifier.

## 1. Introduction
The code implements a basic KNN classifier, a simple yet effective algorithm for classification. KNN is a non-parametric and lazy learning method. It classifies a new data point based on the majority class of its k nearest neighbors in the training set.

## 2. Algorithm

### 2.1 Training Phase
KNN is a lazy learner, meaning it doesn't learn a discriminative function from the training data. Instead, it memorizes the training dataset. The `fit` method in the code simply stores the training data (features `X_train` and labels `y_train`).

### 2.2 Prediction Phase
The prediction phase involves the following steps:
1.  **Distance Calculation:** Calculate the distance between the query point and each training data point. The Euclidean distance is commonly used, as implemented in the `_euclidean_distance` method:
    
    $$d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}$$
    
    where:
    -   $x$ is the query point.
    -   $x_i$ is a training data point.
    -   $n$ is the number of features.
2.  **Nearest Neighbors Selection:** Find the k nearest training data points to the query point based on the calculated distances.
3.  **Majority Voting:** Assign the class label to the query point based on the majority class among its k nearest neighbors.  The `_predict` method implements this voting process using the `Counter` class.

## 3. Implementation Details

The `KNearestNeighbors` class implements the following methods:

-   `__init__(self, k=3)`: Initializes the KNN classifier with the number of neighbors `k`.
-   `fit(self, X, y)`: Stores the training data.
-   `predict(self, X)`: Predicts the class labels for the given data.
-   `_predict(self, x)`: Predicts the class label for a single data point.
-   `_euclidean_distance(self, x1, x2)`: Computes the Euclidean distance between two data points.
-   `score(self, X, y)`: Calculates the accuracy of the model on the given data.

## 4. Example Usage

Here's how to use the `KNearestNeighbors` class:

```python
import numpy as np
from collections import Counter

# Sample training data
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[2.5, 2], [4, 2]])

# 1. Create and train the KNN classifier
knn = KNearestNeighbors(k=3)  # Initialize with k=3
knn.fit(X_train, y_train)    # Train the model

# 2. Make predictions
predictions = knn.predict(X_test)
print("Predictions:", predictions)  # Output: [0 1]

# 3. Calculate accuracy
accuracy = knn.score(X_test, np.array([0, 1]))
print(f"Accuracy: {accuracy:.2f}")  # Output: 1.00
```