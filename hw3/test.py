import numpy as np

# y = np.array([1, 2, 3])
# print(np.cross(y, y.T))
# print(y[:, np.newaxis] @ y[np.newaxis, :])
# N = 3
# C = 2
# # print(np.concatenate((-np.identity(N), np.identity(N)), axis=0))
# # print(np.concatenate((np.zeros((N, )), C * np.ones((N, )) ))  )

# rng = np.random.default_rng(545)
# X = rng.normal(size=(5, 4))
# y = rng.normal(size=(5,))
# alpha = rng.binomial(5, 0.2, size=(5,))

# is_support = alpha > 1e-4
# X_support = X[is_support]
# y_support = y[is_support]
# alpha_support = alpha[is_support]
# N_support = X_support.shape[0]

# print(y_support * alpha_support)
# print(y_support[:, np.newaxis] * alpha_support[:, np.newaxis])

x = np.arange(6).reshape((2, -1))
print(np.max(x, axis=1, keepdims=False))