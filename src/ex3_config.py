
def isotropic_normal(mean, variance):
    import numpy
    import scipy.stats
    assert len(mean) >= 1
    return scipy.stats.multivariate_normal(mean=mean, cov=variance * numpy.identity(len(mean)))

# These are configuration parameters
DATASET_SIZE = 1000
TRAINING_DATA_RELATIVE_SIZE = 0.1
FEATURE_SPACE_DIMENSIONS = 2
FEATURE_SPACE_DIMENSIONS_PLOT_RANGES = [[-10.0, -10.0], [10.0, 10.0]]
FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION = 0.125

LABEL_COLORS = ['blue', 'red', 'white', 'yellow']
CLASS_DISTRIBUTIONS = [
    isotropic_normal([5.0, -5.0], 5),
    #    isotropic_normal([5.0, 5.0], 0.5),
    isotropic_normal([-7.5, -7.5], 0.75),
    isotropic_normal([-5.0, 5.0], 1.0)]

assert DATASET_SIZE >= 1
assert FEATURE_SPACE_DIMENSIONS >= 2
assert len(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES) == 2 and all([len(x) == FEATURE_SPACE_DIMENSIONS for x in FEATURE_SPACE_DIMENSIONS_PLOT_RANGES])
assert 0.0 < TRAINING_DATA_RELATIVE_SIZE <= 1.0
assert len(CLASS_DISTRIBUTIONS) >= 2
assert all([d.dim == FEATURE_SPACE_DIMENSIONS for d in CLASS_DISTRIBUTIONS])
assert len(CLASS_DISTRIBUTIONS) <= len(LABEL_COLORS)