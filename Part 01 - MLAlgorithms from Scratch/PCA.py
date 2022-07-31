import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean Centering.
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # The covariance function needs samples as columns.
        cov = np.cov(X.T)

        # Eigenvalues and Eigenvectors.
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i] column vector, transpose for easier calculations.
        # Sort Eigenvectors.
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store the first N eigenvectors.
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # Project Data.
        X = X - self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto 2 primary principal components.
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()
