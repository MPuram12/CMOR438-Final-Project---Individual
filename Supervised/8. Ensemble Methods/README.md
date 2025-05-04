# Boosting Regressor (Gradient Boosting for Regression)

This document describes the implementation of a gradient boosting regressor.

## 1. Introduction
The code implements a Gradient Boosting Regressor, an ensemble learning method that combines multiple weak learners (decision trees) in a sequential manner.  Boosting is used for regression tasks.  Each tree is trained to correct the errors of the previous ones, leading to improved prediction accuracy.

## 2. Algorithm

The Gradient Boosting algorithm builds an ensemble of decision trees in a stage-wise fashion. Here's how it works:

### 2.1 Training Phase
1.  **Initialization:** Initialize the predictions with the mean of the target variable.
2.  **Iterative Tree Building:** For each boosting iteration:
    * **Compute Residuals:** Calculate the residuals (the difference between the true target values and the current predictions).
    * **Fit a Tree to Residuals:** Train a decision tree to predict the residuals.  This tree is typically shallow.
    * **Update Predictions:** Update the predictions by adding a fraction of the tree's predictions to the previous predictions.  The fraction is controlled by the learning rate.

### 2.2 Prediction Phase
To make a prediction for a new data point:
1.  **Initialize with Initial Prediction:** Start with the mean of the target variable from the training data.
2.  **Iterate Through Trees:** For each tree in the ensemble, add the tree's weighted prediction (weighted by the learning rate) to the accumulated prediction.
3.  **Final Prediction:** The final prediction is the sum of the initial prediction and the weighted predictions of all the trees.

## 3. Implementation Details

The `BoostingRegressor` class implements the following methods:

-   `__init__(self, n_estimators=100, learning_rate=0.1, max_depth=3)`:
    * Initializes the Boosting Regressor with the following parameters:
        -   `n_estimators`: The number of trees in the ensemble.
        -   `learning_rate`: The learning rate, which controls the contribution of each tree to the final prediction.
        -   `max_depth`: The maximum depth of the trees.
-   `fit(self, X, y)`: Builds the boosting ensemble from the training data.
-   `predict(self, X)`: Predicts the output for the given data by summing the predictions of all trees.
-   `score(self, X, y)`: Calculates the coefficient of determination ($R^2$) to evaluate the model's performance.

## 4. Key Concepts

-   **Ensemble Learning:** Combining multiple individual models (decision trees) to obtain a better overall model.
-   **Boosting:** Sequentially building trees, where each tree tries to correct the errors of the previous ones.
-   **Residuals:** The difference between the true target values and the predictions.
-   **Learning Rate:** A parameter that controls the contribution of each tree to the final prediction.  A smaller learning rate requires more trees but can lead to better generalization.
-   **Tree Depth:** Controls the complexity of the individual trees.  Shallower trees are often preferred in boosting to reduce overfitting.

## 5. Example Usage

The following code demonstrates how to use the `BoostingRegressor` class:

```python
import numpy as np
from .decision_trees import DecisionTreeRegressor  # Import your DecisionTreeRegressor

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 2], [6, 4]])
y_train = np.array([2.5, 3.5, 2, 5, 4.5, 6])
X_test = np.array([[2.5, 2], [4, 2], [7, 7]])

# Create and train the boosting regressor
boosting = BoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=1)  # You can adjust the hyperparameters
boosting.fit(X_train, y_train)

# Make predictions
predictions = boosting.predict(X_test)
print("Predictions:", predictions)

# Evaluate
r_squared = boosting.score(X_test, np.array([3, 4, 8]))
print(f"R^2: {r_squared:.2f}")
```