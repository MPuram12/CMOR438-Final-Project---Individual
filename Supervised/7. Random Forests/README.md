# Random Forest Regressor

This document describes the implementation of a random forest regressor.

## 1. Introduction
The code implements a Random Forest Regressor, an ensemble learning method that combines multiple decision trees to improve prediction accuracy and robustness.  Random forests are used for regression tasks.

## 2. Algorithm

The random forest algorithm builds multiple decision trees and averages their predictions. The key steps involved in the random forest algorithm are:

### 2.1 Training Phase
1.  **Bootstrapping:** For each tree, create a bootstrap sample by randomly sampling the original training data *with replacement*. This means some samples may appear multiple times in a single bootstrap sample, while others may be left out.
2.  **Random Feature Selection:** For each tree, at each node, randomly select a subset of the features.  The tree is then allowed to choose only from this subset of features when determining the best split.
3.  **Decision Tree Building:** Build a decision tree using the bootstrapped data and the random subset of features. Each tree is constructed using a regression algorithm (in this case, `DecisionTreeRegressor`). The tree is grown to a maximum depth or until a stopping criterion is met (e.g., minimum number of samples required to split a node or minimum number of samples in a leaf node).
4.  **Ensemble of Trees:** Repeat steps 1-3 to build an ensemble of decision trees (a forest).

### 2.2 Prediction Phase
To make a prediction for a new data point:
1.  **Individual Tree Predictions:** Pass the data point through each decision tree in the forest to obtain individual predictions.
2.  **Averaging Predictions:** Average the predictions from all the individual trees to get the final prediction.

## 3. Implementation Details

The `RandomForestRegressor` class implements the following methods:

-   `__init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None)`:
    * Initializes the Random Forest Regressor with the following parameters:
        -   `n_estimators`: The number of trees in the forest.
        -   `max_depth`: The maximum depth of the trees.
        -   `min_samples_split`: The minimum number of samples required to split an internal node.
        -   `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
        -   `random_state`: Controls the randomness of the bootstrapping and feature selection.
-   `fit(self, X, y)`: Builds the random forest from the training data.
-   `predict(self, X)`: Predicts the output for the given data by averaging the predictions of all trees.
-   `score(self, X, y)`: Calculates the coefficient of determination ($R^2$) to evaluate the model's performance.

## 4. Key Concepts

-   **Ensemble Learning:** Combining multiple individual models (decision trees) to obtain a better overall model.
-   **Bootstrapping:** Random sampling with replacement from the training data.
-   **Random Subspace (Feature Selection):** Randomly selecting a subset of features to consider at each node in each tree.
-   **Variance Reduction:** Random forests reduce the variance of the predictions compared to a single decision tree, leading to more robust and less overfit models.
-   **Out-of-Bag (OOB) Error:** Since each tree is trained on a different bootstrap sample, some data points are left out of the training set for some trees. These data points can be used to estimate the model's performance without the need for a separate validation set.

## 5. Example Usage

The following code demonstrates how to use the `RandomForestRegressor` class:

```python
import numpy as np
from .decision_trees import DecisionTreeRegressor  # Import your DecisionTreeRegressor

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
y_train = np.array([2.5, 3.5, 2, 5, 4.5, 6])
X_test = np.array([[2.5, 2], [4, 2], [7, 7]])

# Create and train the random forest regressor
rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)  # You can adjust the hyperparameters
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)
print("Predictions:", predictions)

# Evaluate
r_squared = rf.score(X_test, np.array([3, 4, 8]))
print(f"R^2: {r_squared:.2f}")
```