import sklearn as sk
import sklearn.metrics
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = sklearn.metrics.confusion_matrix(Y, labels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(df['label']),
                      title='Confusion matrix')
plt.savefig('ex4_confusion_matrix.png')

plot_distribution(df[df['is_training_data'] == True], 'ex4_training_distplot.png')
plot_distribution(df[df['is_training_data'] == False], 'ex4_testing_distplot.png')
plot_distribution(df[df['is_training_data'] == True], 'ex4_training_classified_distplot.png', 'predicted_label')
plot_distribution(df[df['is_training_data'] == False], 'ex4_testing_classified_distplot.png', 'predicted_label')