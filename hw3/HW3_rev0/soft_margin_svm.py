#-*- coding: utf-8 -*-
import numpy as np
import math


def hello():
    print('Hello from soft_margin_svm.py')


def svm_train_bgd(X: np.ndarray, y: np.ndarray, num_epochs: int=100, C: float=5.0, eta: float=0.001):
    """
    Computes probabilities for logit x being each class.
    Inputs:
      - X: Numpy array of shape (num_data, num_features).
           Please consider this input as \phi(x) (feature vector).
      - y: Numpy array of shape (num_data, 1) that store -1 or 1.
      - num_epochs: number of epochs during training.
      - C: Slack variables' coefficient hyperparameter when optimizing the SVM.
    Returns:
      - W: Numpy array of shape (1, num_features) which is the gradient of W.
      - b: Numpy array of shape (1) which is the gradient of b.
    """
    # Implement your algorithm and return state (e.g., learned model)
    num_data, num_features = X.shape
    
    np.random.seed(0)
    W = np.zeros((1, num_features), dtype=X.dtype)
    b = np.zeros((1), dtype=X.dtype)
    
    for j in range(1, num_epochs+1):
        #######################################################################
        # TODO: Implement the gradient and update it, with respect to W and b.# 
        # Your goal is to update W and b from each iteration (j). You should  #
        # first compute the gradient of W and b, and then update accordingly. #
        # Don't forget to implement this function in a vectorized form.       #
        #######################################################################
        raise NotImplementedError("TODO: Add your implementation here.")
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
    
    assert np.any(W != 0), "You are required to update the W value" # make sure to update b as well
    return W, b


def svm_test(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Computes probabilities for logit x being each class.
    Inputs:
      - W: Numpy array of shape (1, num_features).
      - b: Numpy array of shape (1)
      - X: Numpy array of shape (num_data, num_features).
           Please consider this input as \phi(x) (feature vector).
      - y: Numpy array of shape (num_data, 1) that store -1 or 1.
    Returns:
      - accuracy: accuracy value in 0 ~ 1.
    """
    
    pred = (X @ W.T + b[np.newaxis, :] > 0).astype(y.dtype)*2 - 1
    accuracy = np.mean((pred == y).astype(np.float32))
    return accuracy
