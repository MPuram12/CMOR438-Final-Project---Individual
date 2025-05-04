# Decision Tree Regressor

This document describes the implementation of a decision tree regressor.

## 1. Introduction
The code implements a basic decision tree regressor, a non-parametric model used for regression tasks. It partitions the input space into a set of rectangular regions and fits a simple constant (mean) within each one.

## 2. Algorithm

The decision tree building process involves recursively splitting the data based on the feature that provides the best separation of the target variable. The splitting process continues until a stopping criterion is met.

### 2.1 Tree Building
The `_build_tree` method recursively builds the decision tree.  Here's a breakdown:

1.  **Base Cases:**
    * If there are no more samples, return a leaf node with the mean of the target values.
    * If all target values are the same, return a leaf node with that value.
    * If the maximum depth is reached, return a leaf node with the mean of the target values.
    * If there are no more features to split on, return a leaf node with the mean of the target values
2.  **Find Best Split:** The `_get_best_split` method finds the best feature and threshold to split the data.  This is done by iterating through each feature and each possible threshold, calculating the variance reduction for each split, and choosing the split that maximizes the variance reduction.
3.  **Recursive Splitting:** The data is split into left and right subsets based on the best split.  The `_build_tree` method is then recursively called on the left and right subsets to build the subtrees.
4.  **Node Representation:** Each node in the tree is represented as a dictionary.  Leaf nodes contain a `value` key, which is the mean of the target values in that region.  Internal nodes contain:
    * `feature_index`: The index of the feature used for splitting.
    * `threshold`: The threshold value used for splitting.
    * `left_child`: The left child node.
    * `right_child`: The right child node.

### 2.2 Best Split Selection
The `_get_best_split` method finds the best split by minimizing the variance of the target variable in the resulting subsets.  The variance reduction is calculated as:

$$
\text{Variance Reduction} = \text{Initial Variance} - \frac{N_{left}}{N} \text{Variance}_{left} - \frac{N_{right}}{N} \text{Variance}_{right}
$$

Where:
-   $N$ is the total number of samples.
-   $N_{left}$ and $N_{right}$ are the number of samples in the left and right subsets, respectively.
-   $\text{Variance}_{left}$ and $\text{Variance}_{right}$ are the variances of the target variable in the left and right subsets, respectively.

### 2.3 Prediction
The `predict` method predicts the output for a given input by traversing the tree.  The `_predict_one` method recursively traverses the tree, starting from the root node.  At each internal node, it checks whether the feature value for the input is less than or equal to the node's threshold.  If it is, it goes to the left child; otherwise, it goes to the right child.  When it reaches a leaf node, it returns the node's value (the predicted output).

## 4. Implementation Details

The `DecisionTreeRegressor` class implements the following methods:

-   `__init__(self, max_depth=None)`: Initializes the regressor with the maximum tree depth.
-   `fit(self, X, y)`: Builds the decision tree from the training data.
-   `predict(self, X)`: Predicts the output for the given data.
-   `_build_tree(self, X, y, depth)`: Recursively builds the decision tree.
-   `_get_best_split(self, X, y)`: Finds the best split for the data.
-   `_predict_one(self, x, node)`: Predicts the output for a single data point using the tree.
-  `score(self, X, y)`: Calculates the coefficient of determination ($R^2$) to evaluate the model's performance.

## 5. Example Usage

The following code demonstrates how to use the `DecisionTreeRegressor` class:

```python
import numpy as np

# Create sample data
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
y_train = np.array([2.5, 3.5, 2, 5, 4.5, 6])
X_test = np.array([[2.5, 2], [4, 2]])

# Create and train the decision tree regressor
dt = DecisionTreeRegressor(max_depth=3)  # You can adjust the max_depth
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)
print("Predictions:", predictions)

# Evaluate
r_squared = dt.score(X_test, np.array([3,4]))
print(f"R^2: {r_squared:.2f}")
```