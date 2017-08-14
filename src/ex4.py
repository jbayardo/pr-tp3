import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ex1 import two_class_fisher
from ex3 import generate_data, classify
from ex4_config import *

sns.set(color_codes=True)

# Train the classifier
X, Y = generate_data(DATASET_SIZE, CLASS_DISTRIBUTIONS)
selection_mask = np.random.rand(DATASET_SIZE) < TRAINING_DATA_RELATIVE_SIZE
w, w0 = two_class_fisher(X[selection_mask], Y[selection_mask])
labels = classify(w, w0, X)
predictions = np.vstack((labels, Y)).T

# Transform the data into a format more amenable for plotting
df = pd.concat([
    pd.DataFrame(X,
                 columns=['x']),
    pd.DataFrame(predictions,
                 columns=['predicted_label',
                          'label']),
    pd.DataFrame(selection_mask,
                 columns=['is_training_data'])],
    axis=1)


def plot_distribution(df, filename, label_axis='label'):
    plt.clf()
    labels = np.unique(df['label'])
    for index, label in enumerate(labels):
        plt.ylim(0.0, 1.0)
        plt.xlim(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1])
        intermediate = df[df[label_axis] == label]
        sns.distplot(intermediate['x'], color=LABEL_COLORS[index], rug=True, hist=False)
    plt.savefig(filename)
    plt.clf()


plot_distribution(df[df['is_training_data'] == True], 'ex4_training_distplot.png')
plot_distribution(df[df['is_training_data'] == False], 'ex4_testing_distplot.png')
plot_distribution(df[df['is_training_data'] == True], 'ex4_training_classified_distplot.png', 'predicted_label')
plot_distribution(df[df['is_training_data'] == False], 'ex4_testing_classified_distplot.png', 'predicted_label')