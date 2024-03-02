"""
An implementation of SVMs using cvxopt.
raise NotImplementedError("TODO: Add your implementation here.")
"""
import warnings
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def kernel_dot(X1, X2, kernel_params):
    """
    Returns the elementwise kernel vector between X1 and X2.
    I.e. kernel_dot(X1, X2)_i = k(X1_i, X2_i)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples,)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return (X1 * X2).sum(1)
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * (X1 * X2).sum(1) + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        return np.exp(-kp['gamma'] * ((X1 - X2)**2).sum(1))
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * (X1 * X2).sum(1) + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def kernel_matrix(X1, X2, kernel_params):
    """
    Returns the pairwise kernel matrix between X1 and X2 (aka. gram matrix).
    I.e. kernel_dot(X1, X2)_{ij} = k(X1_i, X2_j)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (m_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples, m_samples)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return X1 @ X2.T
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * X1 @ X2.T + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        pw_norm = ((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2).sum(2)
        return np.exp(-kp['gamma'] * pw_norm)
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * X1 @ X2.T + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def get_qp_params(X, y, C, kernel_params):
    """
    Return the parameters to pass into cvxopt.solvers.qp for the SVM dual problem.
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    Arguments to be passed into cvxopt.solvers.qp
    P : ndarray of shape (n_samples, n_samples)
    q : ndarray of shape (n_samples,)
    G : ndarray of shape (n_1, n_samples)
    h : ndarray of shape (n_1,)
    A : ndarray of shape (n_2, n_samples)
    b : ndarray of shape (n_2,)
    """
    P, q, G, h, A, b = None, None, None, None, None, None

    ###################################################################################
    # TODO calculate the values for P, q, G, h, A, and b.
    # Make sure the shapes of these outputs are correct and pass the assertions below.
    # To calculate the kernel functions, refer to the instructions in the pdf or
    #   at https://scikit-learn.org/stable/modules/svm.html#svm-kernels
    # Note: you will not need all the parameters
    # (e.g. degree, gamma, coef0) for every kernel.
    # Hint: kernel_matrix may be useful here.
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    N, D = X.shape
    
    P = (y[:, np.newaxis] @ y[np.newaxis, :]) * kernel_matrix(X, X, kernel_params)
    q = -np.ones((N, ))
    G = np.concatenate((-np.identity(N), np.identity(N)), axis=0)
    h = np.concatenate((np.zeros((N, )), C * np.ones((N, )) ))
    A = np.copy(y).reshape(1, N).astype(np.float64)
    b = np.array([0.0])
    
    # print(P.dtype, q.dtype, G.dtype, h.dtype, A.dtype, b.dtype)
    # import pdb; pdb.set_trace()
    # print(P)
    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################

    # Check for shapes
    assert P.dtype == q.dtype == G.dtype == h.dtype \
            == A.dtype == b.dtype == np.float_, 'outputs must be numpy floats'
    assert len(P.shape) == len(G.shape) == len(A.shape) == 2, 'P, G, and A must be matrices'
    assert len(q.shape) == len(h.shape) == len(b.shape) == 1, 'q, h, and b must be vectors'
    assert P.shape[0] == P.shape[1] == X.shape[0], 'wrong shape for P'
    assert q.shape[0] == X.shape[0], 'wrong shape for q'
    assert G.shape == (h.shape[0], X.shape[0]), 'wrong shape for G or h'
    assert A.shape == (b.shape[0], X.shape[0]), 'wrong shape for A or b'
    return P, q, G, h, A, b


def fit_bias(X, y, alpha, kernel_params):
    """
    Return the calculated values the bias b using the given support vector values (alpha).
    Parameters
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples used to find alpha
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -----
    Scalar (float64)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    b = None
    ###################################################################################
    # TODO calculate the bias given X, y, and alpha
    # Hint: kernel_matrix or kernel_dot may be useful here
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    X_support = X[is_support]
    y_support = y[is_support]
    alpha_support = alpha[is_support]
    N_support = X_support.shape[0]
    # import pdb; pdb.set_trace()
    
    b = (y_support - kernel_matrix(X_support, X_support, kernel_params) @ (y_support * alpha_support)).sum() / N_support
    # print(b)
    # import pdb; pdb.set_trace()
    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################
    assert b is not None, "You are asked to update b properly"
    return b


def decision_function(X, X_train, y_train, b, alpha, kernel_params):
    """
    Return the calculated values for (w^T X + b) using the given support vector values (alpha).
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X_train : np.ndarray (float64) of shape (n_train_samples, n_features)
        The input samples used to find alpha
    y_train : np.ndarray (int64) of shape (n_train_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    b : scalar (float64)
        Bias value computed after training (along with alpha)
    alpha : np.ndarray (float64) of shape (n_train_samples,)
        Support vector values (solution of the cvxopt qp method)
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    h = np.zeros(X.shape[0])
    ###################################################################################
    # TODO calculate the values for h(x) = w^T X + b using the dual representation
    # Since this is kernelized, you may not be able to calculate w directly.
    # Hint: kernel_matrix or kernel_dot may be useful here
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    N_sample = X.shape[0]
    for i in range(N_sample):
        h[i] = kernel_matrix(X[i].reshape(1, -1), X_train, kernel_params) @ (alpha * y_train) + b
    # print(h)
    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################
    assert h.shape == (X.shape[0],)
    assert np.any(h != 0), "You are required to update the h value"
    return h


class CVXOPTSVC:
    """C-Support Vector Classification.

    Sklearn-style SVM implementation using the cvxopt solver.
    
    Parameters and interface are adapted from
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.
    gamma : float, default=1.0
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - must be non-negative.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    """
    _estimator_type = "classifier"
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=1.0,
        coef0=0.0
    ):
        self.C = C
        self.kernel_params = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0
        }

    @staticmethod
    def _H_linear(X, y):
        Xy = X * np.expand_dims(y, 1)
        return Xy @ Xy.T

    def fit(self, X, y):
        # Initialize and computing H. Note the 1. to force to float type
        y = y * 2 - 1  # transform to [-1, 1]
        self.classes_ = [-1, 1]
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Convert into cvxopt format
        _P, _q, _G, _h, _A, _b = get_qp_params(X, y, self.C, self.kernel_params)
        P = cvxopt_matrix(_P)
        q = cvxopt_matrix(np.expand_dims(_q, 1))
        G = cvxopt_matrix(_G)
        h = cvxopt_matrix(_h)
        A = cvxopt_matrix(_A)
        b = cvxopt_matrix(_b)

        # Run solver
        cvxopt_solvers.options['show_progress'] = False
        try:
            sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        except ValueError:
            # if fail to solve, try different solver
            # default solver always fails on windows cvxopt build
            sol = cvxopt_solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
        self.alpha = np.array(sol['x']).squeeze(1)

        self.support_ = np.where(self.alpha > 1e-4)
        self.b = fit_bias(X, y, self.alpha, self.kernel_params)

        return self

    def decision_function(self, X):
        return decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)

    def predict(self, X):
        h = decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)
        return (h >= 0).astype(np.int_)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
