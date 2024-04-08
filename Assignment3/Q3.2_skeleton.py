# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import fetch_openml
import numpy as np
from matplotlib import pyplot as plt

# Load MNIST dataset.
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
k = 2
# Conduct PCA to reduce the dimensionality of X.
X_zero_mean = (X - X.mean(0)).transpose()  # subtract mean and transpose

C = np.dot(X_zero_mean, X_zero_mean.transpose())/len(X)  # covariance matrix

eigenvalues, eigenvectors = np.linalg.eig(C)  # eigen decomposition

# sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

U = eigenvectors[:, :k]  # first k eigenvectors
Y = np.dot(U.transpose(), X_zero_mean).transpose() # projection
X_reconstruct = np.dot(U, Y.transpose())  # reconstruction
error = ((X_zero_mean-X_reconstruct)**2).sum()/len(X)  # reconstruction error
# Visualize the data distribution of digits '0', '1' and '3' in a 2D scatter plot.
digits = ['0', '1', '3']
plt.figure(1)
plt.scatter(Y[y == '0', 0], Y[y == '0', 1], s=1)
plt.scatter(Y[y == '1', 0], Y[y == '1', 1], s=1)
plt.scatter(Y[y == '3', 0], Y[y == '3', 1], s=1)
plt.legend(digits)

# Generate an image of digit '3' using 2D representations of digits '0' and '1'.
mean_0 = np.mean(Y[y == '0', :], axis=0)
mean_1 = np.mean(Y[y == '1', :], axis=0)

mean_3 = (mean_0 + mean_1) / 2

rep_3 = np.dot(U, mean_3.transpose())
image_3 = 1 - rep_3.reshape(28,28) # inversing colour to make shape more clear

plt.figure(2)

plt.imshow(np.real(image_3), cmap='gray')
plt.show()