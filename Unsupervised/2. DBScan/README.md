# DBSCAN Clustering

This document describes the implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

## 1. Introduction
The code implements the DBSCAN algorithm, a density-based clustering method that groups together data points that are close to each other (high density) while marking outliers as noise.

## 2. Algorithm

DBSCAN defines clusters as dense regions in the data, separated by sparser regions. The algorithm works based on two key parameters:

-   **Epsilon (eps):** The radius of the neighborhood around a data point.
-   **Minimum Samples (min_samples):** The minimum number of data points within the epsilon radius to define a dense region.

The algorithm works as follows:

1.  **Initialization:** Each point is initially considered unvisited.
2.  **Iteration:** For each unvisited point p:
    * **Neighborhood Retrieval:** Find all points within `eps` distance of p. These are the neighbors of p.
    * **Core Point Check:** If the number of neighbors is greater than or equal to `min_samples`, p is a core point.
        * If p is a core point, start a new cluster and add all of p's neighbors to the cluster.
        * **Cluster Expansion:** Recursively expand the cluster by finding the neighbors of each newly added point. If a neighbor is also a core point, add its neighbors to the cluster. This process continues until no more density-connected points can be found.
    * **Noise Check:** If the number of neighbors is less than `min_samples`, p is marked as noise.
3.  **Repeat:** Continue the iteration until all points have been visited.

## 3. Implementation Details

The `DBSCAN` class implements the following methods:

-   `__init__(self, eps=0.5, min_samples=5)`:
    * Initializes the DBSCAN object with the following parameters:
        -   `eps`: The maximum distance between two samples for them to be considered neighbors.
        -   `min_samples`: The number of samples in a neighborhood for a point to be considered a core point.
-   `fit(self, X)`: Performs DBSCAN clustering on the data.
-   `_get_neighbors(self, X, i)`: Finds the neighbors of a data point within a given radius (eps).
-   `_expand_cluster(self, X, neighbors, cluster_label)`: Expands the cluster starting from a core point.
-    `_euclidean_distance(self, x1, x2)`: Computes the Euclidean distance between two data points.
-   `fit_predict(self, X)`: Performs DBSCAN clustering on the data and returns cluster labels.

## 4. Key Concepts

-   **Core Point:** A data point with at least `min_samples` within its `eps` neighborhood.
-   **Border Point:** A data point that is within the `eps` neighborhood of a core point but has fewer than `min_samples` neighbors itself.
-   **Noise Point:** Any data point that is neither a core point nor a border point.
-   **Density-Connected:** Two points are density-connected if there is a chain of core points between them.
-   **Cluster:** A set of density-connected points.

## 5. Example Usage

The following code demonstrates how to use the `DBSCAN` class:

```python
import numpy as np

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10,2], [0, 2], [4, 10], [7, 9], [12,10]])


# Create and apply DBSCAN
dbscan = DBSCAN(eps=2, min_samples=3)
cluster_labels = dbscan.fit_predict(X)  # Use fit_predict()

print("Cluster Labels:", cluster_labels)  #  Output: [ 0  0 -1 -1  0 -1 -1 -1  0 -1 -1 -1]
```