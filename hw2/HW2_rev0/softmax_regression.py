"""EECS545 HW2: Softmax Regression."""

import numpy as np
import math


def hello():
    print('Hello from softmax_regression.py')


def compute_softmax_probs(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_data, num_features).
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
    Returns:
      - probs: Numpy array of shape (num_data, num_class). The softmax
        probability with respect to W.
    """
    probs = None
    ###########################################################################
    # TODO: compute softmax probability of X with respect to W and store the  #
    # Softmax probability out to 'probs'.                                     #
    # If you are not careful here, it is easy to run into numeric instability #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/)   #
    # Hint: the pseudo code in the link will not be 100% matched with ours.   #
    # You may need to slightly edit the code script in the link.              #
    ###########################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return probs


def gradient_ascent_train(X_train: np.ndarray,
                          Y_train: np.ndarray,
                          num_class: int,
                          max_iters: int = 300) -> np.ndarray:
    """Computes w from the train set (X_train, Y_train).
    This implementation uses gradient ascent algorithm derived from the previous question.

    Inputs:
      - X_train: Numpy array of shape (num_data, num_features).
                 Please consider this input as \\phi(x) (feature vector).
      - Y_train: Numpy array of shape (num_data, 1) that has class labels in
                 [1 .. num_class].
      - num_class: Number of class labels
      - max_iters: Maximum number of iterations
    Returns:
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
           We will use the trained weights on the test set to measure the performance.
    """
    N, d = X_train.shape  # the number of samples in training dataset, dimension of feature
    W = np.zeros((num_class, d), dtype=X_train.dtype)
    class_matrix = np.eye(num_class, dtype=W.dtype)

    int_Y_train = Y_train.astype(np.int32)
    alpha = 0.0005
    count_c = 0
    for epoch in range(max_iters):
        # A single iteration over all training examples
        delta_W = np.zeros((num_class, d), dtype=W.dtype)
        ###################################################################
        # TODO: Compute the cumulated weight 'delta_W' for each point.    #
        # You are allowed to use compute_softmax_probs function.          #
        # Note that Y_train has class labels in [1 ~ num_class]           #
        ###################################################################
        raise NotImplementedError("TODO: Add your implementation here.")
        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################
        W_new = W + alpha * delta_W
        W[:num_class-1, :] = W_new[:num_class-1, :]

        # Stopping criteria
        count_c += 1 if epoch > 300 and np.sum(abs(alpha * delta_W)) < 0.05 else 0
        if count_c > 5:
            break

    return W


def compute_accuracy(X_test: np.ndarray,
                     Y_test: np.ndarray,
                     W: np.ndarray,
                     num_class: int) -> float:
    """Computes the accuracy of trained weight W on the test set.

    Inputs:
      - X_test: Numpy array of shape (num_data, num_features).
      - Y_test: Numpy array of shape (num_data, 1) consisting of class labels
                in the range [1 .. num_class].
      - W: Numpy array of shape (num_class, num_features).
      - num_class: Number of class labels
    Returns:
      - accuracy: accuracy value in 0 ~ 1.
    """
    count_correct = 0
    N_test = Y_test.shape[0]
    int_Y_test = Y_test.astype(np.int32)
    ###########################################################################
    # TODO: save the number of correct prediction to 'count_correct' variable.#
    # We are using this value at the end of this function by dividing it to   #
    # number of (X, Y) data pairs. Hint: check the equation in the homework.  #
    ###########################################################################
    raise NotImplementedError("TODO: Add your implementation here.")
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    accuracy = count_correct / (N_test * 1.0)
    return accuracy
