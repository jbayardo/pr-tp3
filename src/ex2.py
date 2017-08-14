import numpy as np
import scipy as sp


def multi_class_fisher(X: np.ndarray, Y: np.ndarray) -> (np.ndarray, np.ndarray):
    # X looks like a matrix
    assert len(X.shape) == 2
    # X has at one row
    assert X.shape[0] > 0
    # X has at least two features
    assert X.shape[1] >= 2
    # Sw is going to be invertible
    assert X.shape[1] < X.shape[0]
    # Y is an array
    assert len(Y.shape) == 1
    # Y has exactly one element per entry in the dataset
    assert Y.shape[0] == X.shape[0]

    # Fetch unique classes
    classes = np.unique(Y)
    # We have exactly two classes
    assert len(classes) > 2

    means = np.zeros(shape=(len(classes), X.shape[1]))
    St = np.cov(np.transpose(X))
    Sw = np.zeros(shape=(X.shape[1], X.shape[1]))
    # Process each class separately
    for index, c in enumerate(classes):
        # Indexes that have class `c`
        discriminator = np.where(Y == c)
        # Compute the mean for the class
        means[index] = np.mean(X[discriminator], axis=0)
        # Add up the local within-class covariance to the total within-class covariance matrix
        Sw += np.cov(np.transpose(X[discriminator] - means[index]))
    Sb = St - Sw

    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = sp.linalg.eigh(Sb, Sw)
    # Sort them according to largest eigenvalue
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    # Normalize the eigenvectors
    eigenvectors /= np.apply_along_axis(np.linalg.norm, 0, eigenvectors)

    W = np.dot(means, eigenvectors).dot(eigenvectors.T)
    w0 = -0.5 * np.diag(np.dot(means, W.T))
    return W, w0
