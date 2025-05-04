# K-Means Clustering

This document describes the implementation of the K-Means clustering algorithm.

## 1. Introduction
The code implements the K-Means algorithm, a popular unsupervised learning algorithm used for partitioning a dataset into k clusters, where each data point belongs to the cluster with the nearest mean (centroid).

## 2. Algorithm

The K-Means algorithm aims to minimize the within-cluster sum of squares. It iteratively refines the clusters and centroids until convergence.  Here's a breakdown of the algorithm:

1.  **Initialization:** Randomly initialize k centroids, where k is the desired number of clusters.  The centroids can be initialized by randomly selecting k data points from the dataset.
2.  **Assignment Step:** Assign each data point to the cluster whose centroid it is nearest to, based on the Euclidean distance:
    
    $$d(x_i, \mu_j) = \sqrt{\sum_{l=1}^{n} (x_{il} - \mu_{jl})^2}$$
    
    where:
    -   $x_i$ is the i-th data point.
    -   $\mu_j$ is the j-th centroid.
    -   $n$ is the number of features.
3.  **Update Step:** Calculate the new centroids by taking the mean of all data points assigned to each cluster:
    
    $$\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$
    
    where:
    -   $C_j$ is the set of data points assigned to the j-th cluster.
    -   $|C_j|$ is the number of data points in the j-th cluster.
4.  **Convergence:** Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

## 3. Implementation Details

The `KMeans` class implements the following methods:

-   `__init__(self, n_clusters=3, max_iters=100, random_state=None)`:
    * Initializes the KMeans object with the following parameters:
        -   `n_clusters`: The number of clusters to form.
        -   `max_iters`: The maximum number of iterations of the k-means algorithm.
        -   `random_state`: Determines random number generation for centroid initialization.
-   `fit(self, X)`: Computes k-means clustering.
-   `predict(self, X)`: Predicts the closest cluster for each sample in X.
-   `_assign_clusters(self, X)`: Assigns each sample to the nearest centroid.
-   `_calculate_distances(self, X, centroids)`: Calculates the Euclidean distances between each sample in X and each centroid.
-   `_calculate_centroids(self, X)`: Calculates the new centroids by taking the mean of the samples in each cluster.
-   `fit_predict(self, X)`: Computes cluster centers and predicts cluster index for each sample.
-   `score(self, X)`: Calculates the Silhouette Coefficient for the given data.

## 4. Example Usage

The following code demonstrates how to use the `KMeans` class:

```python
import numpy as np

# Sample data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Create and apply KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(X)

print("Cluster Labels:", cluster_labels)
print("Centroids:", kmeans.centroids)

# Get Silhouette Coefficient
silhouette_avg = kmeans.score(X)
print(f"Silhouette Coefficient: {silhouette_avg:.2f}")
```