
def isotropic_normal(mean, stddev):
    import numpy
    import scipy.stats
    return scipy.stats.norm(loc=mean, scale=stddev)

# These are configuration parameters
DATASET_SIZE = 1000
TRAINING_DATA_RELATIVE_SIZE = 0.2
FEATURE_SPACE_DIMENSIONS_PLOT_RANGES = [0.0, 10.0]
FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION = 0.125

LABEL_COLORS = ['blue', 'red', 'white', 'yellow']
CLASS_DISTRIBUTIONS = [
    # Overlap completo entre las clases
    #isotropic_normal(2.5, 1),
    #isotropic_normal(2.5, 0.5),
    # Overlap parcial
    isotropic_normal(4.0, .5),
    isotropic_normal(6.0, .5),
    # Sin overlap
    #isotropic_normal(7.5, .5),
    #isotropic_normal(2.5, .5),
    ]

assert DATASET_SIZE >= 1
assert len(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES) == 2
assert 0.0 < TRAINING_DATA_RELATIVE_SIZE <= 1.0
assert len(CLASS_DISTRIBUTIONS) >= 2
assert len(CLASS_DISTRIBUTIONS) <= len(LABEL_COLORS)