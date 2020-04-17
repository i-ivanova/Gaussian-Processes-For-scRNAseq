import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


class Kernel(object):
    """ Abstract class for all Kernel functions."""

    
    def __add__(self, other):
        if isinstance(other, Kernel):
            return Sum(self, other)
        raise ValueError("Summing over non-kernels objects.")

        
    def __radd__(self, other):
        if isinstance(other, Kernel):
            return Sum(self, other)
        raise ValueError("Summing over non-kernels objects.")


    def __mul__(self, other):
        if isinstance(other, Kernel):
            return Prod(self, other)
        raise ValueError("Product over non-kernels objects.")

        
    def __rmul__(self, other):
        if not isinstance(other, Kernel):
            return Prod(self, other)
        raise ValueError("Product over non-kernels objects.")

        
class Sum(Kernel):
    """Wrapper class to support sum between kernels."""
    
    
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None, derivative_observations=False):
        return self.k1(X, Y, derivative_observations) + self.k2(X, Y, derivative_observations)
    
    
class Prod(Kernel):
    """Wrapper class to support product between kernels."""
    
    
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None, derivative_observations=False):
        return self.k1(X, Y, derivative_observations) * self.k2(X, Y, derivative_observations)

    
class RBFKernel(Kernel):
    """Implementation of the Radial-Base Function Kernel.

       Supports derivative_observations for the kernel.
    """
    
    
    def __init__(self, alpha=1.0, gamma=1.0):
        """ Build-in function. Essencially the constructor.

        Args:
            alpha: float a constant representing the spread (horizontal)
            gamma: float a constant representing the lenght_scale attribute
                (vertical)
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
            if derivative_observations is set to False. Otherwise, a matrix-like object
            with dimensions (n_observations_x * (1 + dimensions of X),
                             n_observations_y * (1 + dimensions of Y))
            containing derivative observations of the kernel function.
            NOTE: X and Y should be of the same dimensions

        Raises:
            ValueError: When the shape of the hyperparameters alpha and gamma does not
            fit the inputs X and Y dimensions

        """
        X = np.atleast_2d(X)

        if np.ndim(self.gamma) > 1:
            raise ValueError("""'gamma' param dimensions exceed dimensions of input data 'X'.
                                Check dimensions of the 'length' parameter.""")
        elif np.ndim(self.gamma) == 1 and (X.shape[1] != self.gamma.shape[0]):
            raise ValueError("""Input data 'X' and 'length' param dimension mismatch.
                                Check if the 'gamma' parameter is set properly and its
                                shape fits the dimensions of 'X'.""")

        elif np.ndim(self.gamma) == 0:
            self.gamma = np.repeat(self.gamma, X.shape[1])

        if Y is None:
            # compute pointwise distance in the space and normalize
            dists = pdist(X * self.gamma, metric='sqeuclidean')
            # compute the kernel
            Cov = self.alpha * np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            Cov = squareform(Cov)
            np.fill_diagonal(Cov, 1) # = cov(x, x)

        else:
            dists = cdist(X * self.gamma, Y * self.gamma, metric='sqeuclidean')
            # compute the kernel
            Cov = self.alpha * np.exp(-0.5 * dists) # = cov(x*, x)

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
                Cov_w_y.append(-self.alpha * self.gamma[i] * Cov * dists[i])


            if Y is None:
                for i in range(X.shape[1]):
                    Cov_wi_w = []
                    for j in range(X.shape[1]):
                        delta = 1 if j==i else 0
                        Cov_wi_w.append(
                        self.gamma[j]* (delta - self.gamma[i] * dists[i] * dists[j]) * Cov)
                    if X.shape[1] == 1:
                        Cov_w_w.append(Cov_wi_w[0])
                    else:
                        Cov_w_w.append(np.column_stack(np.array(Cov_wi_w)))

                # return Cov(X_train, X_train)
                return  np.hstack((
                            np.vstack(
                                (Cov,
                                 np.vstack(Cov_w_y))),
                            np.vstack(
                                (np.vstack(Cov_w_y).T,
                                 np.vstack(Cov_w_w)))))

            else:
                # return Cov(X_test, X_train)
                return np.hstack((Cov, -1 * np.hstack(Cov_w_y)))


        else:
            return Cov


    def gradient(self, X):
        """Compute the gradient of the kernel with respect to gamma."""
        if np.ndim(self.gamma) == 0:
            self.gamma = np.repeat(self.gamma, X.shape[1])

        dists = pdist(X * self.gamma, metric='sqeuclidean')
        Cov = self.alpha * np.exp(-.5 * dists)
        Cov = squareform(Cov)
        np.fill_diagonal(Cov, 1) # = cov(x, x)

        if X.shape[1] ==  1:
            Cov_gradient = - 0.5 * Cov * squareform(dists)
            return Cov_gradient[..., np.newaxis]
        
        else:
            Cov_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
            Cov_gradient *= Cov[..., np.newaxis]
            return Cov_gradient

        
class Matern52Kernel(Kernel):
    """Implementation of the Matern Kernel at v=5/2 so it is
    double differentiable.

       Supports derivative_observations for the kernel.
    """
    
    
    def __init__(self, alpha=1.0, gamma=1.0):
        """ Build-in function. Essencially the constructor.

        Args:
            alpha: float a constant representing the spread (horizontal)
            gamma: float a constant representing the lenght_scale attribute
                (vertical)
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
            if derivative_observations is set to False. Otherwise, a matrix-like object
            with dimensions (n_observations_x * (1 + dimensions of X),
                             n_observations_y * (1 + dimensions of Y))
            containing derivative observations of the kernel function.
            NOTE: X and Y should be of the same dimensions

        Raises:
            ValueError: When the shape of the hyperparameters alpha and gamma does not
            fit the inputs X and Y dimensions

        """
        X = np.atleast_2d(X)
        all_dists = None
        if np.ndim(self.gamma) > 1:
            raise ValueError("""'gamma' param dimensions exceed dimensions of input data 'X'.
                                Check dimensions of the 'length' parameter.""")
        elif np.ndim(self.gamma) == 1 and (X.shape[1] != self.gamma.shape[0]):
            raise ValueError("""Input data 'X' and 'length' param dimension mismatch.
                                Check if the 'gamma' parameter is set properly and its
                                shape fits the dimensions of 'X'.""")

        elif np.ndim(self.gamma) == 0:
            self.gamma = np.repeat(self.gamma, X.shape[1])

        if Y is None:
            # compute pointwise distance in the space and normalize
            all_dists = pdist(X * self.gamma, metric='euclidean')
            # compute the kernel
            Cov = all_dists * np.sqrt(5)
            Cov = self.alpha * (1 + Cov + Cov ** 2 / 3.0) * np.exp(-Cov)
            # convert from upper-triangular matrix to square matrix
            Cov = squareform(Cov)
            np.fill_diagonal(Cov, 1) # = cov(x, x)
            all_dists = squareform(all_dists)
            np.fill_diagonal(all_dists, 1) # = cov(x, x)            

        else:
            all_dists = cdist(X * self.gamma, Y * self.gamma, metric='euclidean')
            # compute the kernel
            Cov = all_dists * np.sqrt(5)
            Cov = self.alpha * (1 + Cov + Cov ** 2 / 3) * np.exp(-Cov)

        if derivative_observations:
            # compute the covariance matrix of the deriavtive observations
            # for each dimension in our dimension space
            
            sqrt_5_d = np.sqrt(5) * all_dists
            dCov_dd = -5/3 * (1 - sqrt_5_d) * np.exp(-sqrt_5_d)

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
                    
                dd_dxi = self.gamma[i] * dists[i]
                Cov_wi_y = self.alpha * dCov_dd * dd_dxi                
                Cov_w_y.append(Cov_wi_y)

            if Y is None:
                for i in range(X.shape[1]):
                    Cov_wi_w = []
                    for j in range(X.shape[1]):
                        delta = 1 if j==i else 0
                        
                        d2Cov_d2d = - 5 / 3 * np.exp(-sqrt_5_d) * (1 - sqrt_5_d - sqrt_5_d ** 2)                        
                        d2rdxixj = self.gamma[i] / all_dists ** 2 * (
                            delta - self.gamma[j] * dist[i] * dist[j] / all_dists) 
                        
                        Cov_wi_wj = self.alpha * d2Cov_d2d * d2rdxixj
                        Cov_wi_w.append(Cov_wi_wj)
                        
                    if X.shape[1] == 1:
                        Cov_w_w.append(Cov_wi_w[0])
                    else:
                        Cov_w_w.append(np.column_stack(np.array(Cov_wi_w)))

                # return Cov(X_train, X_train)
                return  np.hstack((
                            np.vstack(
                                (Cov,
                                 np.vstack(Cov_w_y))),
                            np.vstack(
                                (np.vstack(Cov_w_y).T,
                                 np.vstack(Cov_w_w)))))

            else:
                # return Cov(X_test, X_train)
                return np.hstack((Cov, np.hstack(Cov_w_y)))


        else:
            return Cov


    def gradient(self, X):
        """Compute the gradient of the kernel with respect to gamma."""
        if np.ndim(self.gamma) == 0:
            self.gamma = np.repeat(self.gamma, X.shape[1])

        dists = pdist(X * self.gamma, metric='euclidean')
        dists = squareform(dists)
        np.fill_diagonal(dists, 1)
        
        Cov_g = np.sqrt(5 * dists.sum(-1))[..., np.newaxis]
        Cov_gradient = 5.0 / 3.0 * dists * (Cov_g + 1) * np.exp(-Cov_g)
        return Cov_gradient
    
    
class LinearKernel(Kernel):
    """Implementation of the Linear Kernel.

       Supports derivative_observations for the kernel.
    """
    
    
    def __init__(self, sigma=0.0, scale=1.0):
        """ Build-in function. Essencially the constructor.

        Args:
            sigma: float a constant representing the homogenity
            scale: float a constant representing the scaling
        """
        self.sigma = sigma
        self.scale = scale

        
    def __call__(self, X, Y=None, derivative_observations=False):
        """Build-in function

        Args:
            X: A matrix-like object with dimsensions (n_observations_x, n_dimensions)
            Y: A matrix-like object with deimnsions (n_observations_y, n_dimensions)
            derivative_observations: A boolean to describ whether derivative_observations
            to be included

        Returns:
            K: A matrix-like object with dimensions (n_observations_x, n_observations_y)
            if derivative_observations is set to False. Otherwise, a matrix-like object
            with dimensions (n_observations_x * (1 + dimensions of X),
                             n_observations_y * (1 + dimensions of Y))
            containing derivative observations of the kernel function.
            NOTE: X and Y should be of the same dimensions
        """
        X = np.atleast_2d(X)

        if Y is None:
            Cov = self.sigma**2 + self.scale * X @ X.T

        else:
            Cov = self.sigma**2 + self.scale * X @ Y.T

        if derivative_observations:
            # compute the covariance matrix of the deriavtive observations
            # for each dimension in our dimension space

            Cov_w_w = np.zeros((X.shape[0] * X.shape[1], X.shape[0] * X.shape[1]))
            Cov_w_y = np.zeros((X.shape[1], X.shape[0], X.shape[1]))
            for i in range(X.shape[1]):
                Cov_w_y[i, :, i] = 1
                            
            if Y is None:  
                Cov_w_y = self.scale * (Cov_w_y @ X.T + X @ np.transpose(Cov_w_y, axes=(0, 2, 1)))

                # return Cov(X_train, X_train)
                return  np.hstack((
                            np.vstack(
                                (Cov,
                                 np.vstack(Cov_w_y))),
                            np.vstack(
                                (np.vstack(Cov_w_y).T,
                                 Cov_w_w))))

            else:
                # return Cov(X_test, X_train)
                Cov_w_y_transpose= np.zeros((Y.shape[1], Y.shape[1], Y.shape[0]))
                for i in range(Y.shape[1]):
                    Cov_w_y_transpose[i, i, :] = 1
                    
                Cov_w_y = Cov_w_y @ Y.T + X @ Cov_w_y_transpose
                return np.hstack((Cov, np.hstack(Cov_w_y)))


        else:
            return Cov


    def gradient(self, X):
        """Compute the gradient of the kernel with respect to sigma."""
        Cov_gradient = np.empty((X.shape[0], X.shape[0], 1))
        Cov_gradient[..., 0] = 2 * self.sigma ** 2
        return Cov_gradient
