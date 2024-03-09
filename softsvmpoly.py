import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import itertools

from matplotlib.colors import ListedColormap


def plot(trainX, trainy):
    # Plot points colored by their labels
    plt.scatter(trainX[:, 0], trainX[:, 1], c=trainy, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of Training Set')
    plt.show()


def part4(trainX, trainy):
    # Assuming trainX is a 2D array with shape (m, 2)
    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
    xgrid, ygrid = np.meshgrid(np.arange(x_min, x_max, 0.01),
                               np.arange(y_min, y_max, 0.01))

    for k in [3, 5, 8]:
        alpha = softsvmpoly(100, k, trainX, trainy)

        # Generate Mesh Grid
        h = 0.02

        # Predict Labels for Grid Points
        mesh_points = np.c_[xgrid.ravel(), ygrid.ravel()]
        predictions = np.sign(np.dot(np.multiply(trainy.flatten(), alpha.flatten()),
                                     polynomial_kernel(trainX, mesh_points, k)))

        # Reshape the predictions to the mesh grid shape
        predictions = predictions.reshape(xgrid.shape)

        # Create a color map for the plot
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_points = ListedColormap(['#FF0000', '#0000FF'])

        # Plot the decision boundary
        plt.contourf(xgrid, ygrid, predictions, cmap=cmap_background, alpha=0.8)

        # Plot the training points
        plt.scatter(trainX[:, 0], trainX[:, 1], c=trainy, cmap=cmap_points, edgecolors='k', marker='o')

        plt.title(f"Polynomial Soft SVM Decision Boundary (λ={100}, k={k})")
        plt.show()


def polynomial_kernel(x, y, degree):
    """
    Polynomial kernel function.

    Parameters:
    - x, y: Input vectors.
    - degree: Degree of the polynomial.

    Returns:
    - (dot product between x and y + 1) raised to the power of the degree.
    """
    if x.ndim == 2 and y.ndim == 2:
        # Both x and y are matrices
        return (np.dot(x, y.T) + 1) ** degree
    else:
        return (np.dot(x, y) + 1) ** degree


def cross_validation(trainX, trainy, lambdas, degrees):
    m = len(trainy)
    folds = 5
    fold_size = m // folds

    best__lambda_degree = None
    best_avg_error = float('inf')

    # Loop over lamdas X degress
    for lambda_v, degree in itertools.product(lambdas, degrees):
        avg_error = 0.0

        for i in range(folds):
            # Create folds for cross-validation
            validation_indices = np.arange(i * fold_size, min((i + 1) * fold_size, m))
            training_indices = np.setdiff1d(np.arange(m), validation_indices)

            # Split the data into training and validation sets
            _trainX = trainX[training_indices]
            _trainy = trainy[training_indices]
            validation_X = trainX[validation_indices]
            validation_y = trainy[validation_indices]

            # Train Soft SVM using softsvmpoly
            alpha = softsvmpoly(lambda_v, degree, _trainX, _trainy)

            # Calculate validation error
            predictions = np.sign(np.dot(np.multiply(_trainy.flatten(), alpha.flatten()),
                                         polynomial_kernel(_trainX, validation_X, degree)))
            error = np.sum(predictions != validation_y) / len(validation_y)
            avg_error += error

        avg_error /= folds

        # Update the best parameters if the current combination has a lower average error
        if avg_error < best_avg_error:
            best_avg_error = avg_error
            best__lambda_degree = (lambda_v, degree)

    return best__lambda_degree, best_avg_error


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param k: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m, d = trainX.shape

    # Create gram metrix
    gram = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            gram[i, j] = trainy[i] * trainy[j] * polynomial_kernel(trainX[i], trainX[j], k)

    # Gram matrix for the polynomial kernel
    Gram = matrix(gram, tc='d')

    q = matrix(-np.ones((m, 1)), tc='d')
    G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))), tc='d')
    # Lambda matrix
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * l)), tc='d')
    # Y liable
    Y = matrix(trainy.reshape(1, -1), tc='d')
    b = matrix(0.0, tc='d')

    # Solve the quadratic programming problem
    solution = solvers.qp(Gram, q, G, h, Y, b)

    # Extract alpha values from the solution
    alpha = np.array(solution['x'])

    return alpha


def simple_test():
    # load question 2 data
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = len(trainX)

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(100, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
