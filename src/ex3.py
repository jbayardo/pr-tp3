import numpy as np


# Sample from the process that generates the dataset
def generate_data(dataset_size, distributions):
    dataset = []
    labels = []
    for _ in range(dataset_size):
        # Fetch one of our chosen distributions, this will be the label
        y = np.random.randint(len(distributions))
        # Generate a random point sampled from it
        x = distributions[y].rvs(size=1)
        labels.append(y)
        dataset.append(x)

    # Convert into numpy format, this is what we are going to use for training
    return np.array(dataset), np.array(labels)


# Classify a dataset X using w and w0 as the linear discriminant parameters
def classify(w, w0, X):
    # Run all the data through the classifier
    predicted = np.dot(X, w.T) + w0
    try:
        if predicted.shape[1] >= 1:
            return np.argmax(predicted, axis=1)
    except (IndexError, TypeError):
        return np.array([1 if p >= 0 else 0 for p in predicted])


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pandas as pd

    from ex1 import two_class_fisher
    from ex2 import multi_class_fisher
    from ex3_config import *

    sns.set(color_codes=True)

    if FEATURE_SPACE_DIMENSIONS == 2:
        x = np.arange(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][0], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][0], FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION)
        y = np.arange(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][1], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][1], FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION)
        X, Y = np.meshgrid(x, y)
        zs = np.array([max([dist.pdf([x, y]) for dist in CLASS_DISTRIBUTIONS]) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        # Graph the surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Joint Probability Density Function')

        ax.plot_surface(X, Y, Z, color='b', vmin=0, vmax=0.1, cmap='RdYlGn_r')
        plt.savefig('ex3_probability_densities.png')

    # Train the classifier
    X, Y = generate_data(DATASET_SIZE, CLASS_DISTRIBUTIONS)
    selection_mask = np.random.rand(DATASET_SIZE) < TRAINING_DATA_RELATIVE_SIZE
    w, w0 = multi_class_fisher(X[selection_mask], Y[selection_mask])
    labels = classify(w, w0, X)
    predictions = np.vstack((labels, Y)).T

    # Transform the data into a format more amenable for plotting
    df = pd.concat([
        pd.DataFrame(X,
                     columns=['x' + str(i) for i in range(FEATURE_SPACE_DIMENSIONS)]),
        pd.DataFrame(predictions,
                     columns=['predicted_label',
                              'label']),
        pd.DataFrame(selection_mask,
                     columns=['is_training_data'])],
        axis=1)

    # Plot the input dataset and labelled samples
    if FEATURE_SPACE_DIMENSIONS == 2:
        plot_df = df[df['is_training_data'] == True]
        plot = sns.jointplot(x="x0", y="x1", data=plot_df, kind='kde',
                             color='m')
        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle('Training data distribution', fontsize=16)
        labels = np.unique(Y)
        for index, label in enumerate(labels):
            intermediate = plot_df[plot_df['label'] == label]
            plot.ax_joint.scatter(intermediate['x0'], intermediate['x1'],
                                  s=30, linewidth=1, marker='+',
                                  color=LABEL_COLORS[index])
        plt.savefig('ex3_training_data.png')

    # Plot the testing data and the classification output
    if FEATURE_SPACE_DIMENSIONS == 2:
        plot_df = df[df['is_training_data'] == False]
        plot = sns.jointplot(x="x0", y="x1", data=plot_df, kind='kde',
                             color='m')
        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle('Classified testing data', fontsize=16)
        labels = np.unique(Y)
        for index, label in enumerate(labels):
            intermediate = plot_df[plot_df['predicted_label'] == label]
            plot.ax_joint.scatter(intermediate['x0'], intermediate['x1'],
                                  s=30, linewidth=1, marker='+',
                                  color=LABEL_COLORS[index])
        plt.savefig('ex3_classified_testing_data.png')

    # Plot the classification boundaries
    if FEATURE_SPACE_DIMENSIONS == 2:
        import matplotlib.colors
        plt.clf()
        x = np.arange(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][0], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][0], FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION)
        y = np.arange(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][1], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][1], FEATURE_SPACE_DIMENSIONS_PLOT_PRECISION)
        X, Y = np.meshgrid(x, y)
        Z = classify(w, w0, np.c_[X.ravel(), Y.ravel()])
        Z = Z.reshape(X.shape)
        plt.xlim(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][0], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][0])
        plt.ylim(FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[0][1], FEATURE_SPACE_DIMENSIONS_PLOT_RANGES[1][1])
        plt.contourf(X, Y, Z, cmap=matplotlib.colors.ListedColormap(LABEL_COLORS[:len(CLASS_DISTRIBUTIONS)]))
        plt.savefig('ex3_decision_boundaries.png')
