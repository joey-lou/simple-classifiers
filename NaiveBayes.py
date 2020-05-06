# naive-bayes classifier
import numpy as np

# TODO: add other feature models such as bernoulli, multinomial


class NaiveBayes:
    """Gaussian Naive Bayes model for classification problems.
        Assume features of dimension d, labels of dimension k, and samples of dimension n.

        Attributes:
            model (string): model class for inputs, only support Gaussian at the moment.
            _class (numpy int array): labels of shape [k], holds output labels.
            _mean (numpy float array): means of each class [d x k].
            _var (numpy float array): variances of each class [d x k].
            _prior (numpy float array): priors of each class [k].
            __fit (boolean): flag for whether the fitting stage has been called.
            _epsilon (float): a small number for variance smoothing, avoid division by zero.
    """

    def __init__(self, model="gaussian", epsilon=1e-8):
        self.model = model
        self._class = None
        self._mean = None
        self._var = None
        self._prior = None
        self.__fit = False
        self._epsilon = epsilon
        assert model == "gaussian", "Only 'gaussian' models supported at the moment"

    def _check_Xy(self, X, y):
        """Check input data pair X and y and convert to numpy objects.

            Args:
                X (array_like object): sample feature X, shape [n x d].
                y (array_like object): sample label y, shape [n].

            Returns:
                X (numpy float array): shape [n x d].
                y (numpy int array): shape [n].
        """
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
        assert X.shape[0] == y.shape[0] and len(
            y.shape) == 1, "Accepcted input dimesions: X ~ [m, d], y ~ [m, ]"
        return X, y

    def fit(self, X, y):
        """Fit Naive Bayes model given training data pair, update corresponding 
            model variables: mean, variance, and prior.

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].
                y (numpy int array): output class label y, shape [n].
        """
        X, y = self._check_Xy(X, y)
        self._class = np.sort(np.unique(y))
        m, d = X.shape
        k = len(self._class)
        self._mean = np.zeros((d, k))
        self._var = np.zeros((d, k))
        self._prior = np.zeros((k,))

        for i, _class in enumerate(self._class):
            X_class = X[y == _class, :]
            self._mean[:, i] = np.mean(X_class, axis=0)
            self._var[:, i] = np.var(X_class, axis=0)
            self._prior[i] = X_class.shape[0] / m
        # add smoothing for stability
        self._var += np.max(self._var) * self._epsilon
        self.__fit = True

    def _get_log_prob(self, X):
        """Get log probability of each sample being assigned to each class, 
            log(prior x likelihood), given input feature X. Result should have shape [n x k].

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].

            Returns:
                log_prob (numpy float array): log probabilities of each sample assigned to each 
                                            class, has shape [n x k].
        """
        log_prior = np.log(self._prior)
        log_normal = - 0.5 * np.sum(np.log(2.0 * np.pi * self._var) +
                                    np.power(X[:, :, np.newaxis] - self._mean, 2) / self._var, axis=1)
        return log_prior + log_normal

    def _predict_prob(self, X):
        """Get probabilities of each sample being assigned to each class, 
            prior x likelihood, given input feature X. Result should also have shape [n x k].

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].

            Returns:
                probs (numpy float array): probabilities of each sample assigned to each 
                                            class, has shape [n x k].
        """
        log_prob = self._get_log_prob(X)
        probs = np.exp(log_prob)
        # do not normalize to better show gaussian distribution
        return probs  # / np.sum(probs, axis=1, keepdims=True)

    def predict(self, X):
        """Predict labels of each sample given sample dataset X, this is done by taking
            the label producing the maximum log probability.

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].

            Returns:
                y (numpy int array): class label of each sample, shape [n].
        """
        if not self.__fit:
            raise "Dataset not yet fitted"
        log_prob = self._get_log_prob(X)
        y_idx = np.argmax(log_prob, axis=1)
        y = self._class[y_idx]
        return y


if __name__ == "__main__":
    nb_model = NaiveBayes()
    np.random.seed(9)
    # create dummy dataset
    X = np.concatenate((np.random.rand(50, 1) + 10,
                        np.random.rand(50, 1) + 5), axis=0)
    y = np.concatenate((np.ones(50), np.zeros(50)), axis=0)
    nb_model.fit(X, y)
    y_pred = nb_model.predict(X)
    print("Accuracy = %f" % (np.sum(y_pred == y) / len(y)))
