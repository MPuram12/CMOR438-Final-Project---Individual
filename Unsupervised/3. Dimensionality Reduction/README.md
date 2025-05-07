# Dimensionality Reduction Techniques: PCA and SVD

This document describes the implementation of two dimensionality reduction techniques: Principal Component Analysis (PCA) and Singular Value Decomposition (SVD).

## 1. Principal Component Analysis (PCA)

### 1.1 Introduction
Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms a high-dimensional dataset into a lower-dimensional one by identifying the principal components, which are the directions of maximum variance in the data.

### 1.2 Algorithm

1.  **Center the data:** Subtract the mean from each data point to center the data around the origin.
2.  **Compute the covariance matrix:** Calculate the covariance matrix of the centered data. The covariance matrix represents the relationships between different features in the data.
    
    $$Cov(X) = \frac{1}{n-1} X_{centered}^T X_{centered}$$
    
    where $X_{centered}$ is the centered data matrix.
3.  **Compute the eigenvectors and eigenvalues:** Find the eigenvectors and eigenvalues of the covariance matrix.  Eigenvectors represent the directions in which the data varies the most, and eigenvalues represent the magnitude of the variance along those directions.
4.  **Sort the eigenvalues and eigenvectors:** Sort the eigenvalues in descending order. The eigenvectors are sorted accordingly.
5.  **Select principal components:** Choose the top k eigenvectors corresponding to the k largest eigenvalues. These k eigenvectors are the principal components.
6.  **Transform the data:** Project the centered data onto the subspace spanned by the principal components.

### 1.3 Implementation Details

The `PCA` class implements the following methods:

-   `__init__(self, n_components=None)`:
    * Initializes the PCA object with the parameter `n_components`, the number of principal components to keep. If `n_components` is None, all components are kept.
-   `fit(self, X)`:
    * Fits the PCA model to the data by computing the principal components.
-   `transform(self, X)`:
    * Applies dimensionality reduction to the data using the fitted PCA model.
-   `fit_transform(self, X)`:
    * Fits the model to the data and then performs dimensionality reduction on it.
-   `inverse_transform(self, X_transformed)`:
    * Transforms the data back to its original space.
-  `explained_variance_ratio(self)`:
    * Returns the fraction of the total variance that is explained by each principal component.

### 1.4 Example Usage

```python
import numpy as np

# Sample data
X = np.array([[1, 2, 3], [2, 4, 1], [3, 8, 0], [4, 6, 5], [5, 10, 2]])

# Create and apply PCA
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print("Original Data:\n", X)
print("Transformed Data:\n", X_transformed)

# Demonstrate inverse transform
X_original = pca.inverse_transform(X_transformed)
print("Reconstructed Data:\n", X_original)

# Explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio())
```

# 2. Singular Value Decomposition (SVD) for Image Compression
## 2.1 Introduction
Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three other matrices. SVD can be used for various applications, including image compression. By keeping only the most significant singular values and vectors, we can represent an image with a significantly reduced amount of data.
## 2.2 Algorithm
Load the image: Read the image and convert it into a matrix representation. For a grayscale image, this will be a 2D matrix; for a color image, it can be treated as three 2D matrices (one for each color channel).Convert to grayscale (if color): If the image is a color image, convert it to grayscale to simplify the compression process.Perform SVD: Apply SVD to the image matrix (or each color channel matrix). The SVD decomposes the matrix A into three matrices: U, S, and V:A=UΣVTwhere:A is the original image matrix.U is a unitary matrix representing the left singular vectors.Σ is a diagonal matrix containing the singular values, sorted in descending order.V is a unitary matrix representing the right singular vectors.Keep top k components: Select the largest k singular values from Σ and the corresponding first k columns of U and first k rows of V.Reconstruct the compressed image: Reconstruct an approximation of the original image using the truncated SVD:Ak​=Uk​Σk​VkT​where:Ak​ is the compressed image matrix.Uk​, Σk​, and Vk​ are the truncated matrices.Save or display the compressed image: The reconstructed matrix Ak​ can be converted back into an image and saved or displayed.
## 2.3 Implementation Details
The SVDImageCompressor class implements the following methods :__init__(self, k=50): Initializes the SVDImageCompressor with the parameter k, the number of singular values/vectors to keep.compress(self, image_path, output_path=None):Compresses the image using SVD and either saves it to a file or displays it.get_compression_ratio(self, image_path):Calculates the compression ratio.
## 2.4 Example Usage

```python import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Import the PIL library

image_path = 'your_image.jpg'  # Replace with the path to your image file
k = 50  # Number of singular values to keep

# Create an instance of the SVDImageCompressor
compressor = SVDImageCompressor(k=k)

# Compress the image and save it
output_path = 'compressed_image.jpg'
compressor.compress(image_path, output_path)

# Optionally, display the compressed image
# compressor.compress(image_path)

# Print the compression ratio
compression_ratio = compressor.get_compression_ratio(image_path)
print(f"Compression Ratio: {compression_ratio:.2f}")
```