import numpy as np
from scipy.linalg import cho_solve

class GaussianProcess(object):
    """ Gaussian Process Regressor Class adapted for scRNA seq datasets. """

    def __init__(self, kernel=None, alpha=1e-5, normalize=False,
                derivative_observations=False):
        """
        Args:
            kernel: Kernel object that computes the covariances.
            alpha: float number representing the noise of the signal for the
                input train data
            normalize: boolean, whether to normalize input train data
            derivative_observations: boolean, whether to include derivatives for
                the input dataset
        """
        self.kernel = kernel
        self.alpha = alpha
        self._K_train = None
        self.normalize = normalize
        self._log_marginal_likelihoods = {}
        self.derivative_observations = derivative_observations

    def _init_prior(self):
        """A helper function to compute the covariance matrix for the train data.

           Constructs the K_train matrix which is essencially the
           Cov(X_train, X_train). Computes it exactly once for a single
           dataset as the final mapping from high-dimensional space to 2d/3d
           does not change the coordinated of the cells.
        """
        # We compute the initial prior only once for all the
        # cells in a single dataset
        if self.derivative_observations:
            self._K_train = self.kernel(self._x_train,
            derivative_observations=True)

        else:
            self._K_train = self.kernel(self._x_train)

        # add noise to use cholesky decomposition
        self._K_train[np.diag_indices_from(self._K_train)] += self.alpha


    def _fit(self):
        """A helper private function for the fit function.


        Returns:
            self: the object itself

        Raises:
            np.linalg.LinAlgError: when the kernel matrix is not semi-positive
            definite and cholesky decomposition is not possible
        """
        if self.normalize:
            self._y_train_mean = np.mean(_y_train, axis=0)
            self._y_train = self._y_train - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if self._K_train is None: # compute the prior only once
            self._init_prior()

        try:
            self.L_ = np.linalg.cholesky(self._K_train)

        except np.linalg.LinAlgError as exc:
            if self._K_train.shape[0] == self._x_train.shape[0]:
                exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter"
                        % self.kernel,) + exc.args
                raise

        self.alpha_ = cho_solve((self.L_, True), self._y_train)
        return self


    def fit(self, X_train, Y_train, sample_ratio=0.5):
        """Fitting the model to the train dataset.

        Created a mask for the sampling. It saves the mask as GP attribute so
        that the same cells are used for each gene.

        #TODO: Add new setter function for the mask array

        Args:
            X_train: A matrix
            Y_train:
            sample_ratio: float between 0.0 and 1.0 to specify what portion of
                the train dataset to use for the training of the model
        """
        # 0. Create mask for the sampling
        if not hasattr(self, "_mask"):
            if sample_ratio >= 1.0 or sample_ratio <= 0.0:
                self._mask = np.arange(X_train.shape[0])
            else:
                self._mask = np.random.choice(X_train.shape[0],
                                              int(X_train.shape[0]*sample_ratio),
                                              replace=False)
        # 1. Sample
        self._x_train = X_train[self._mask]
        if self.derivative_observations:
            self._mask_der = np.ravel(
                np.add(
                    # initial mask
                    np.tile(self._mask.reshape(-1, 1), (1, X_train.shape[1] + 1)).T,
                    # step = # of observations
                    np.arange(0, X_train.shape[0]*X_train.shape[1] + 1, X_train.shape[0]).reshape(-1, 1)))
            self._y_train = Y_train[self._mask_der]
        else:
            self._y_train = Y_train[self._mask]

        # 2. Fit on sampled train data
        return self._fit()


    def _predict(self, X_test, cov=False):
        """Helper function to run analysis over all genes.

        Args:
            X_test: a matrix like object that presents the test points
            cov: a boolean whether to return the covariance matrix or not

        Returns:
            y_mean: a matrix like object that is the posterior mean for the
                input values of X_test (essencially the predicted values)
            if cov set to True:
            cov: a matrix like object that is the covariance matrix over the
                 functions for the given input values
        """
        K_test_train = self.kernel(X_test, self._x_train,
         derivative_observations=self.derivative_observations)


        y_mean = K_test_train.dot(self.alpha_)
        y_mean += self._y_train_mean

        if cov:
            K_test = self.kernel(X_test)
            # K_test_train.T @ K^-1/2
            v = cho_solve((self.L_, True), K_test_train.T)
            y_cov = K_test - K_test_train.dot(v)
            return y_mean, y_cov

        return y_mean


    def predict(self, X_test, cov=False):
        """"""
        return self._predict(X_test, cov=cov)

    def sample(self, X_test, n_samples=3):
        """Sampling over the function space.

        when conditioned on the train dataset with respect to the test data.

        Args:
            X_test: a matrix like object that presentsthe input test data
            n_samples: integer that specifies the number of samples to be
                returned
        Returns:
            a list of arrays where each array is a single sample over the
            function space for the given input X_test.
        """
        y_mean, y_cov = self.predict(X_test, cov=True)
        if y_mean.ndim == 1:
            return np.random.multivariate_normal(y_mean, y_cov, n_samples).T
        return [np.random.multivariate_normal(y_mean[:, i], y_cov, n_samples).T
                for i in range(y_mean.shape[1])]


    def _compute_log_likelihood(self, gene_name):
        """Compute the log_likelihood and save result to corresponding gene.

            Argv:
                gene_name: string, the gene_name for which the model has fitted

            Note: Assuming the latest fitted covariance matrices are for the
            given gene
        """
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self._y_train, self.alpha_)
        log_likelihood_dims -= np.log(np.diag(self.L_)).sum()
        log_likelihood_dims -= self._K_train.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)
        self._log_marginal_likelihoods[gene_name] = log_likelihood


    def log_marginal_likelihood(self, gene_name):
        """Returns the log_likelihood for the given gene.

        Returns the log_likelihood value for the given gene in the specidic
        setup if it exists already. Otherwise, computes it for the

        #TODO: better design for the implementation that will handle erroneous
            input gene_name and current X_train, Y_train values. Also, split the
            report into separate funtion or parameter.

        """
        if gene_name not in self._log_marginal_likelihoods:
            self._compute_log_likelihood(gene_name)
        return "Gene: {gene} Marginal Likelihood: {likelihood}".format(
            gene=gene_name.ljust(12),
            likelihood=self._log_marginal_likelihoods[gene_name])


    def get_all_likelihoods(self):
        """A getter function to return the likelihoods for all genes.

        Returns:
            a dictionary with keys representing genenames and values
            representing log_likelihood for that gene
        """
        return self._log_marginal_likelihoods
