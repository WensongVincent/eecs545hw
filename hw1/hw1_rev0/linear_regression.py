"""EECS545 HW1: Linear Regression."""

from typing import Any, Dict, Tuple, Sequence
import numpy as np


def load_data():
    """Load the data required for Q2."""
    x_train = np.load('data/q2xTrain.npy').astype(np.float64)
    y_train = np.load('data/q2yTrain.npy').astype(np.float64)
    x_test = np.load('data/q2xTest.npy').astype(np.float64)
    y_test = np.load('data/q2yTest.npy').astype(np.float64)
    return x_train, y_train, x_test, y_test


def generate_polynomial_features(x: np.ndarray, M: int) -> np.ndarray:
    """Generate the polynomial features.

    Args:
        x: A numpy array with shape (N, ).
        M: the degree of the polynomial.
    Returns:
        phi: A feature vector represented by a numpy array with shape (N, M+1);
          each row being (x^{(i)})^j, for 0 <= j <= M.
    """
    N = len(x)
    phi = np.zeros((N, M + 1))
    for m in range(M + 1):
        phi[:, m] = np.power(x, m)
    return phi


def compute_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    r"""The least squares training objective for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The least square objective term with respect to the coefficient weight w, E(\mathbf{w}).
    """
    objective = 0.0
    ###################################################################
    # TODO: Implement the training objective.
    # In this block, you are required to implement the least squares 
    # training objective described in the homework. 
    # From X, y and the trained w, it should return a scalar E(\mathbf{w}).
    # Please check how the objetive is defined from HW1 problem set PDF file.
    # You need to pass the output to the 'objective' variable
    # (i.e., objective = the_final_value_from_your_end )
    ###################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    objective = 0.5 * np.sum((X @ w.reshape(-1, 1) - y.reshape(-1, 1))**2)
    # import pdb; pdb.set_trace()
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    assert objective != 0.0, "You need to update the objective variable"
    return objective


def batch_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    eta: float = 0.01,
    max_epochs: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by GD. Shape (M+1, ).
        info: A dict that contains additional information. It will include
              'train_objectives' and 'convergence_iter' (see the notebook
              and the implementation below).
    """
    N = X_train.shape[0]
    M = X_train.shape[1] - 1
    w = np.zeros(M + 1, dtype=y_train.dtype)
    
    train_objective_list = []
    convergence_iters = []
    for current_epoch_number in range(max_epochs):
        ###################################################################
        # TODO: Implement the Batch GD solver.
        # In this block, you have X_train and y_train. 
        # From those train set data, you are required to update the weight 'w' 
        ###################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        # dw = X_train.T @ np.ones((N, 1)).squeeze()
        # Why backprop is not working here?
        dw = (X_train.T @ X_train @ w.reshape(-1, 1) - X_train.T @ y_train.reshape(-1, 1)).squeeze()
        w -= eta * dw
        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################
        # check whether all w is 0.0
        assert np.any(w != 0.0), "You are asked to update w properly"
        
        objective = compute_objective(X_train, y_train, w)
        train_objective_list.append(objective)
    
    info = dict(
        train_objectives=train_objective_list,
    )
    return w, info


def stochastic_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eta=4e-2,
    max_epochs=200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stochastic gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by SGD. Shape (M+1, ).
        info: A dict that contains additional information. It will include
              'train_objectives' and 'convergence_iter' (see the notebook
              and the implementation below).
    """
    N = X_train.shape[0]
    M = X_train.shape[1] - 1
    w = np.zeros(M + 1, dtype=y_train.dtype)
    
    train_objective_list = []
    convergence_iters = []
    for current_epoch_number in range(max_epochs):
        for x_data_point, y_data_point in zip(X_train, y_train):
            ###################################################################
            # TODO: Implement the SGD solver.
            # In this block, you are required to update w from a single 
            # (x_data_point, y_data_point).
            ###################################################################
            # raise NotImplementedError("TODO: Add your implementation here.")
            # import pdb; pdb.set_trace()
            x_batch = x_data_point.reshape(1, -1)
            dw = (x_batch.T @ x_batch @ w.reshape(-1, 1) - x_batch.T * y_data_point.item()).squeeze()
            # import pdb; pdb.set_trace()
            w -= eta * dw
            
            ###################################################################
            #                        END OF YOUR CODE                         #
            ###################################################################
            # check whether all w is 0.0
            assert np.any(w != 0.0), "You are asked to update w properly"

        objective = compute_objective(X_train, y_train, w)
        train_objective_list.append(objective)
    
    info = dict(
        train_objectives=train_objective_list,
    )
    return w, info


def closed_form(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    reg: float = 0.0,
) -> np.ndarray:
    """Return the closed form solution of linear regression.

    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N).
        reg: The regularization coefficient lambda.

    Returns:
        The (optimal) coefficient w for the linear regression problem found,
        a numpy array of shape (M+1, ).
    """
    w = None
    ###################################################################
    # TODO: Implement the closed form solution.
    # Your final outcome needs to be passed to w.
    # Note that both 2.(c) and 2.(e) call this function, but with different 'reg'.
    ###################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    w = (np.linalg.inv(X_train.T @ X_train + np.identity(X_train.shape[1]) * reg) @ X_train.T @ y_train.reshape(-1, 1)).squeeze()
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    assert w is not None, "You are asked to update w properly"
    assert X_train.shape[1] == w.shape[0], "shape mismatched"
    return w


def compute_rms_for_m(x_train, y_train, x_test, y_test,
                      M: int, reg: float = 0.0,
) -> Tuple[float, float]:
    """Compute the RMS error for linear regression. Specifically, it uses closed_form to get the optimal coefficients 'w_m'.

    Args:
        x_train: A numpy array with shape (N_train, ).
        y_train: the target label for regression, with shape (N_train, ).
        x_test: A numpy array with shape (N_test, ).
        y_test: the target label for regression, with shape (N_test, ).
        M: the degree of the polynomial.
        reg: The regularization coefficient lambda.
    Returns: A tuple (train_rms_error, test_rms_error)
        train_rms_error: train set RMS error for the coefficient w_m obtained from closed_form
        test_rms_error: test set RMS error for the coefficient w_m obtained from closed_form
    """    
    
    w_m = None
    X_train_m = None
    X_test_m = None
    #########################################################################
    # TODO: Generate the train and test data points, with respect to M first.
    # It is fine to use 'generate_polynomial_features' for data generation.
    # Once the train and test data is ready 
    # (in X_train_m and X_test_m, accordingly), please compute w_m.
    # Among various optimization methods, we will use 'closed_form' approach.
    # You may want to see the previous cells.
    # Note that both 2.(d) and 2.(f) call this function, but with different 'reg'.
    #########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    X_train_m = generate_polynomial_features(x_train, M)
    X_test_m = generate_polynomial_features(x_test, M)
    w_m = closed_form(X_train_m, y_train, reg=reg)
    #########################################################################
    #                          END OF YOUR CODE                             #
    #########################################################################
    assert X_train_m is not None, "You are asked to generate polynomial train features with respect to M"
    assert X_test_m is not None, "You are asked to generate polynomial test features with respect to M"
    assert w_m is not None, "You are asked to update w_m properly"
    
    train_rms_error = np.sqrt(compute_objective(X_train_m, y_train, w_m) / X_train_m.shape[0] * 2)
    test_rms_error = np.sqrt(compute_objective(X_test_m, y_test, w_m) / X_test_m.shape[0] * 2)

    return (train_rms_error, test_rms_error)


def closed_form_locally_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
) -> np.ndarray:
    """Return the closed form solution of locally weighted linear regression.

    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N, ).
        r_train: The local weights for data point. Shape (N, ).

    Returns:
        The (optimal) coefficient w for the locally weighted linear regression
        problem found. A numpy array of shape (M+1, ).
    """
    N: int = X_train.shape[0]   # the number of data points.
    w = None
    ###################################################################
    # TODO: Implement the closed form solution.
    ###################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    R = np.diag(r_train)
    w = (np.linalg.inv(X_train.T @ R @ X_train) @ X_train.T @ R @ y_train).squeeze()
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    assert w is not None, "You are asked to update w properly"
    return w



def compute_y_space(
    X_train: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_space: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Return the y value for each matched x for plotting graph.
    You are first required to compute the local coefficients w from closed_form_locally_weighted function.
    Then, you will compute the matched y_space from the local w value.
    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        x_train: The x datapoint vector, shape (N, ).
        y_train: The y vector, shape (N, ).
        x_space: x point we would like to print out, shape (K, ).
        tau: bandwidth parameter. Please see the equation in the problemset.
    Returns:
        locally weighted linear regression y_space values of shape (K, ). Each item in this list are matched with x_space of the same index position.
    """
    y_space = np.zeros_like(x_space)
    for idx, x_point in enumerate(x_space):
        #########################################################################
        # TODO: Compute y_space value matched to each x_space item.             #
        # You first need to compute r for the x_point, and then compute w for r.#
        # And then, you can compute y_space (y_space[idx]) for the x_point with #
        # w value from the previous step.                                       #
        #########################################################################
        # raise NotImplementedError("TODO: Add your implementation here.")
        r_train = np.exp(- (x_point - x_train) ** 2 / (2 * tau ** 2))
        w = closed_form_locally_weighted(X_train, y_train, r_train)
        X_space = np.array([[1, x_point]])
        y_space[idx] = (X_space @ w.reshape(-1, 1)).squeeze()
        
        #########################################################################
        #                          END OF YOUR CODE                             #
        #########################################################################
    assert np.any(y_space != 0), "You are required to compute the y_space value for each matched x_space"
    return y_space
