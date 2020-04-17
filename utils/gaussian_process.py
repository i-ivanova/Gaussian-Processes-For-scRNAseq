import numpy as np
from scipy.linalg import cho_solve
import scipy.optimize

class GaussianProcess(object):
    """ Gaussian Process Regressor Class adapted for scRNA seq datasets. """

    
    def __init__(self, kernel=None, alpha=1e-5, normalize=False,
                derivative_observations=False, optimize=False, restarts=2):
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
        self.optimize = optimize
        self.restarts = restarts

        
    def _init_prior(self):
        """A helper function to compute the covariance matrix for the train data.

           Constructs the K_train matrix which is essencially the
           Cov(X_train, X_train). Computes it exactly once for a single
           dataset as the final mapping from high-dimensional space to 2d/3d
           does not change the coordinated of the cells.
           
           Raises:
               np.linalg.LinAlgError: when the kernel matrix is not semi-positive
               definite and cholesky decomposition is not possible
        """
        # We compute the initial prior only once for all the
        # cells in a single dataset
        if self.derivative_observations:
            self._K_train = self.kernel(self._X_train,
            derivative_observations=True)

        else:
            self._K_train = self.kernel(self._X_train)

        # add noise to use cholesky decomposition and 
        self._K_train[np.diag_indices_from(self._K_train)[:self._X_train.shape[0]]] += self.alpha
        
        # compute the inverse using choldesky decomposition only once
        try:
            L = np.linalg.cholesky(self._K_train)
            I = np.eye(L.shape[0])
            self.K_inv = cho_solve((L, True), I)

        except np.linalg.LinAlgError as exc:
            self.K_inv = np.linalg.inv(self._K_train)
            if self._K_train.shape[0] == self._X_train.shape[0]:
                print("WARNING: The kernel, %s, is not returning a "
                        "positive definite matrix. Using np.linalg.inv "
                        "may result in numerically unstable solution. "
                        " Consider gradually "
                        "increasing the 'alpha' parameter. "
                        % self.kernel, + exc.args)

                
    def _fit(self):
        """A helper private function for the fit function.

        Returns:
            self: the object itself
        """
        if self.normalize:
            self._Y_train_mean = np.mean(_Y_train, axis=0)
            self._Y_train = self._Y_train - self._Y_train_mean
        else:
            self._Y_train_mean = np.zeros(1)

        if self._K_train is None: # compute the prior only once
            self._init_prior()

        # alpha_ = K^-1 @ y_train
        self._alpha = self.K_inv @ self._Y_train
        return self


    def fit(self, X_train, Y_train, sample_ratio=0.5):
        """Fitting the model to the train dataset.

        Created a mask for the sampling. It saves the mask as GP attribute so
        that the same cells are used for each gene.

        #TODO: Add new setter function for the mask array

        Args:
            X_train: A matrix like object containing the datapoints 
                (cells coordinates)
            Y_train: A matrix like object containing the gene expression 
                values for each cell
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
        if not hasattr(self, "_X_train"):
            self._X_train = X_train[self._mask]
            
        if self.derivative_observations:
            if not hasattr(self, "_mask_der"):
                self._mask_der = np.ravel(
                    np.add(
                        # initial mask
                        np.tile(self._mask.reshape(-1, 1), (1, X_train.shape[1] + 1)).T,
                        # step = # of observations
                        np.arange(0, X_train.shape[0]*X_train.shape[1] + 1,
                                  X_train.shape[0]).reshape(-1, 1)))
            self._Y_train = Y_train[self._mask_der]
            
        else:
            self._Y_train = Y_train[self._mask]

        # 2. Fit on sampled train data
        self._fit()
        
        # 3. Optimize kernel hyperparameters if sepcified 
        # only once for first gene
        if self.optimize:
            self.optimize_fn()
            self.optimize = False

            
    def optimize_fn(self):
        """Funtion to optimize the gamma parameter for the kernel."""
        initial_gamma = self.kernel.gamma
        bounds = [(1e-5, 1e3)] * self._X_train.shape[1]

        def obj_func(gamma):
            """The objective function that is being minimized (log marginal likelihood).
                
                Args:
                    gamma: numpy array that presents the gamma hyperparam for the kernel
                        that is the lenght-scale constant.
            """
            lml, llgrad = self._compute_log_likelihood(gamma=gamma, eval_grad=True)
            return -lml, -llgrad

        def _optimize():
            """Helper funtion."""
            minimize = scipy.optimize.minimize(
                obj_func, self.kernel.gamma, method="L-BFGS-B", jac=True, bounds=bounds)
            if minimize.success:
                return (minimize.x, minimize.fun)

        result = _optimize()
        optimal_results = []
        if result is not None:
            optimal_results.append(result)

        sampling = np.logspace(-3.5, 2.5, 100)
        for i in range(self.restarts):
            self.kernel.gamma = np.random.choice(sampling, self._X_train.shape[1])
            result = _optimize()
            if result is not None:
                optimal_results.append(result)
                
        if not optimal_results:
            self.kernel.gamma = initial_gamma
        else:
            self.kernel.gamma = optimal_results[np.argmin(np.array(optimal_results)[:, 1])][0]
        
        self._init_prior()
        self._fit()

        
    def predict(self, X_test, cov=False, same_cells=True):
        """Predict the gene expression for the given data points.

        Args:
            X_test: a matrix like object that presents the test points
            cov: a boolean whether to return the covariance matrix or not
            same_cells: boolean specifying if the prediction for every gene 
               is always done on the same subset of cells, ie X_test is 
               a constant. If true, K(X_test, X_train) will be cached
               and the computations will be more efficient.           

        Returns:
            y_mean: a matrix like object that is the posterior mean for the
                input values of X_test (essencially the predicted values)           
            cov: a matrix like object that is the covariance matrix over the
                 functions for the given input values
        """
        if same_cells:
            if not hasattr(self, "K_test_train_full"):
                if self.derivative_observations:
                    self.K_test_train_full = self.kernel(
                        X_test, self._X_train, derivative_observations=True)

                else:
                    self.K_test_train_full = self.kernel(X_test, self._X_train)

            
            y_mean = self.K_test_train_full @ self._alpha
            y_mean += self._Y_train_mean

            if cov:
                if not hasattr(self, "K_test"):
                    self.K_test = self.kernel(X_test)
                    
                # K^-1 @ K_test_train.T
                y_cov = self.K_inv @ self.K_test_train_full.T
                y_cov = self.K_test - (self.K_test_train_full @ y_cov)
                return y_mean, y_cov
            
            else:
                return y_mean
            
        K_test_train = None
        
        if self.derivative_observations:
            K_test_train = self.kernel(X_test, self._X_train,
                                       derivative_observations=True)

        else:
            K_test_train = self.kernel(X_test, self._X_train)    

        y_mean = K_test_train @ self._alpha
        y_mean += self._Y_train_mean

        if cov:
            K_test = self.kernel(X_test)
            # K^-1 @ K_test_train.T
            y_cov = self.K_inv @ K_test_train.T
            # Cov = K_test - K_test_train @ K^-1 @ K_test_train.T
            y_cov = K_test - (K_test_train @ y_cov)
            return y_mean, y_cov

        return y_mean


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
            return [np.random.multivariate_normal(y_mean, y_cov, n_samples).T]
        return [np.random.multivariate_normal(y_mean[:, i], y_cov, n_samples).T
                for i in range(y_mean.shape[1])]


    def _compute_log_likelihood(self, gene_name=None, eval_grad=False, gamma=None):
        """Compute the log_likelihood and save result to corresponding gene.

            Argv:
                gene_name: string, the gene_name for which the model has fitted
                eval_grad: a boolean, specifies whether the gradient of the 
                    marginal likelihood to be returned, used only for the
                    optimization part.
                gamma: a float or array of floats, the gamma parameter for 
                    the kernel, this is used only in the optimization part.                

            Note: Assuming the latest fitted covariance matrices are for the
            given gene name
        """
        if gamma is not None:
            self.kernel.gamma = gamma
            self._init_prior()
            self._fit()
            
        log_likelihood = -0.5 * (self._Y_train.T @ self._alpha)
        log_likelihood -= 0.5 * np.log(np.diag(self._K_train)).sum()
        log_likelihood -= self._X_train.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood.sum(-1)

        if gene_name is not None:
            self._log_marginal_likelihoods[gene_name] = log_likelihood

        # compute the gradient only for the non-derivative part
        if eval_grad:
            K_inv = self.K_inv
            cov_slice = self._X_train.shape[0]
            alpha = self._alpha

            # if derivative observations are on, take the part of the
            # covariance that does not contain derivative observations
            if self.derivative_observations:
                L = np.linalg.cholesky(self._K_train[:cov_slice, :cov_slice])
                I = np.eye(L.shape[0])
                K_inv = cho_solve((L, True), I)
                alpha = K_inv @ self._Y_train[:cov_slice]

            Cov_gradient = self.kernel.gradient(self._X_train)
            llgrad = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            llgrad -= K_inv[:, :, np.newaxis]

            llgrad = 0.5 * np.einsum("ijl,ijk->kl", llgrad, Cov_gradient)
            log_likelihood_grad = llgrad.sum(-1)

            return log_likelihood, log_likelihood_grad
        
        return log_likelihood


    def log_marginal_likelihood(self, gene_name, info=False):
        """Returns the log_likelihood for the given gene.

        Returns the log_likelihood value for the given gene in the specidic
        setup if it exists already. Otherwise, computes it with respect
        to the given gene_name. 
        Note: Assuming the fitted covariances matrices are for the given
            gene name.
        
        Args:
            gene_name: string, the gene_name for which the model has fitted
            info: boolean, whether to display the results
        """
        if gene_name not in self._log_marginal_likelihoods:
            self._compute_log_likelihood(gene_name)
            
        if info:
            print("Gene: {gene} Marginal Likelihood: {likelihood}".format(
            gene=gene_name.ljust(12),
            likelihood=self._log_marginal_likelihoods[gene_name]))


    def get_all_likelihoods(self):
        """A getter function to return the likelihoods for all genes.

        Returns:
            a dictionary with keys representing genenames and values
            representing log_likelihood for that gene
        """
        return self._log_marginal_likelihoods