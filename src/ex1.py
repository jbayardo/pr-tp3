import numpy as np


def two_class_fisher(X: np.ndarray, Y: np.ndarray) -> (np.ndarray, np.ndarray):
    # X looks like a matrix
    assert len(X.shape) == 2
    # X has at one row
    assert X.shape[0] > 0
    # Sw is going to be invertible
    assert X.shape[1] < X.shape[0]
    # Y is an array
    assert len(Y.shape) == 1
    # Y has exactly one element per entry in the dataset
    assert Y.shape[0] == X.shape[0]

    # Fetch unique classes
    classes = np.unique(Y)
    # We have exactly two classes
    assert len(classes) == 2

    means = np.zeros(shape=(len(classes), X.shape[1]))
    Sw = np.zeros(shape=(X.shape[1], X.shape[1]))
    # Process each class separately
    for index, c in enumerate(classes):
        # Indexes that have class `c`
        discriminator = np.where(Y == c)
        # Compute the mean for the class
        means[index] = np.mean(X[discriminator], axis=0)
        # Add up the local within-class covariance to the total within-class covariance matrix
        Sw += np.cov(np.transpose(X[discriminator] - means[index]))

    # Compute `w` as seen in the above cell
    w = np.matmul(np.linalg.inv(Sw), means[1] - means[0])
    # The weight vector has exactly one weight per feature
    assert w.shape[0] == X.shape[1]
    mean = np.mean(X, axis=0)
    w0 = -np.matmul(w, mean)
    return w, w0
