from collections import namedtuple

import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt

Classifier = namedtuple("Classifier", "k, x_train, y_train")


def run_sample(n):
    list_err = []
    for i in range(1, 12):
        err = 0
        for _ in range(10):
            err += simple_test(200, i)
        list_err.append(err / 10)

    plot(list_err, n)


def plot(err_list, n):
    sample_sizes = [(i + 1) for i in range(len(err_list))]
    plt.plot(sample_sizes, err_list, marker='o', linestyle='-', color='b')
    plt.xlabel('k')
    plt.ylabel('Average Error')
    plt.title('Error vs k size')

    plt.show()


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """

    return Classifier(k, x_train, y_train)


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier.k
    x_train = classifier.x_train
    y_train = classifier.y_train

    # Initialize an array to store predictions
    predictions = np.zeros((x_test.shape[0], 1))

    for i, test_example in enumerate(x_test):
        # Calculate distances between the test example and all training examples
        distances = np.array([distance.euclidean(test_example, train_example) for train_example in x_train])

        # Get indices of the k-nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]

        # Get the corresponding labels of the k-nearest neighbors
        k_nearest_labels = y_train[k_nearest_indices]

        # Assign the most frequent label among the k-nearest neighbors to the test example
        predictions[i] = np.bincount(k_nearest_labels.astype(int)).argmax()

    return np.vstack(predictions.astype(int))


def simple_test():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    x_train, y_train = gensmallm([train2, train3, train5, train6], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test2, test3, test5, test6], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")
    # return np.mean(y_test.astype(int) != np.hstack(preds))


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # run_sample(20)
