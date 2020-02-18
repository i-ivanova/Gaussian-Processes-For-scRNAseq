import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
# TODO: IMplement more Kernel Functions and extend the Product and Sum over
# kernelw when derivative observations is on

class Kernel(object):
    """ Abstract class for all Kernel functions."""

    def __add__(self, other):
        if isinstance(other, Kernel):
            return Sum(self, other)
        raise ValueError("Summing over non-kernels objects.")

    def __radd__(self, other):
        if isinstance(b, Kernel):
            return Sum(self, other)
        raise ValueError("Summing over non-kernels objects.")


    def __mul__(self, other):
        if isinstance(b, Kernel):
            return Prod(self, other)
        raise ValueError("Summing over non-kernels objects.")

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Prod(self, other)
        raise ValueError("Summing over non-kernels objects.")

class Sum(object):
    """Wrapper class to support sum between kernels."""
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        try:
            return self.k1(X, Y) + self.k2(X, Y)
        except TypeError:
            raise NotImplementedError("Summing over kernels when derivative \
            observations are included is not yet implemented.")

class Prod(object):
    """Wrapper class to support product between kernels."""
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        try:
            return self.k1(X, Y) * self.k2(X, Y)
        except TypeError:
            raise NotImplementedError("Product over kernels when derivative \
            observations are included is not yet implemented.")


class RBFKernel(Kernel):
    """Implementation of the Radial-Base Function Kernel.

       Supports derivative_observations for the kernel.
    """
    def __init__(self, alpha=1.0, gamma=1.0):
        """ Build-in function. Essencially the constructor.

        Args:
            alpha: a constant representing the spread (horizontal)
            gamma: a constant representing the lenght_scale attribute (vertical)
        """
        self.alpha = alpha
        self.gamma = np.squeeze(gamma).astype(float)

    def __call__(self, X, Y=None, derivative_observations=False):
        """Build-in function

        Args:
            X: A matrix-like object with dimsensions (n_observations_x, n_dimensions)
            Y: A matrix-like object with deimnsions (n_observations_y, n_dimensions)
            derivative_observations: A boolean to describ whether derivative_observations
            to be included

        Returns:
            K: A matrix-like object with dimensions (n_observations_x, n_observations_y)
            if derivative_observations is set to False. Otherwise,

        Raises:
            ValueError: When the shape of the hyperparameters alpha and gamma does not
            fit the inputs X and Y dimensions

        """
        X = np.atleast_2d(X)
        gamma_ = self.gamma

        if np.ndim(self.gamma) > 1:
            raise ValueError("""'gamma' param dimensions exceed dimensions of input data 'X'.
                                Check dimensions of the 'length' parameter.""")
        elif np.ndim(self.gamma) == 1 and (X.shape[1] != self.gamma.shape[0]):
            raise ValueError("""Input data 'X' and 'length' param dimension mismatch.
                                Check if the 'gamma' parameter is set properly and its
                                shape fits the dimensions of 'X'.""")
        #TODO: Fix lenght_scale parameter for multiple dimensions
#         elif np.ndim(self.gamma) == 0:
#             gamma_ = np.repeat(self.gamma, X.shape[1])

        if Y is None:
            # compute pointwise distance in the space and normalize
            dists = pdist(X, metric='sqeuclidean')
            # compute the kernel
            Cov = self.alpha * np.exp(-.5 * gamma_* dists)
            # convert from upper-triangular matrix to square matrix
            Cov = squareform(Cov)
            np.fill_diagonal(Cov, 1) # = cov(x, x)

        else:
            dists = cdist(X, Y, metric='sqeuclidean')
            # compute the kernel
            Cov = self.alpha * np.exp(-0.5 * gamma_ * dists) # = cov(x*, x)

        if derivative_observations:
            # compute the covariance matrix of the deriavtive observations
            # for each dimension in our dimension space

            Cov_w_y = []
            Cov_w_w = []
            dists = []
            for i in range(X.shape[1]):
                if Y is None:
                    dist = np.tile(X[:, i], (X.shape[0], 1))
                    dists.append(dist.T - dist)

                else:
                    dist = np.tile(X[:, i].reshape(-1, 1), (1,  Y[:, i].shape[0])) \
                    - np.tile(Y[:, i].reshape(-1, 1), (1, X.shape[0])).T
                    dists.append(dist)
                Cov_w_y.append(-self.alpha * gamma_ * Cov * dists[i])


            if Y is None:
                for i in range(X.shape[1]):
                    Cov_wi_w = []
                    for j in range(X.shape[1]):
                        delta = 1 if j==i else 0
                        Cov_wi_w.append(
                        gamma_* (delta - gamma_ * dists[i] * dists[j]) * Cov)
                    if X.shape[1] == 1:
                        Cov_w_w.append(Cov_wi_w[0])
                    else:
                        Cov_w_w.append(np.column_stack(np.array(Cov_wi_w)))

                # return Cov(X_train, X_train)
                return  np.hstack((
                            np.vstack(
                                (Cov,
                                 np.vstack(tuple(Cov_w_y)))),
                            np.vstack(
                                (np.vstack(tuple(Cov_w_y)).T,
                                 np.vstack(tuple(Cov_w_w))))))

            else:
                # return Cov(X_test, X_train)
                return np.hstack((Cov, -1 * np.hstack(tuple(Cov_w_y))))


        else:
            return Cov
