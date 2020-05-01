# naive-bayes classifier

import numpy as np


class naivebayes:

    def __init__(self, model="Gaussian", epsilon=1e-8):
        self.model = model
        self._class = None
        self._mean = None
        self._var = None
        self._prior = None
        self.__fit = False
        self._log_prob = None
        self._epsilon = epsilon

    def _check_Xy(self, X, y):
        """ Some input checking logic
        """
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
        assert X.shape[0] == y.shape[0] and len(y.shape) == 1, "Accepcted input dimesions: X ~ [m, d], y ~ [m, ]"
        return X, y

    def fit(self, X, y):
        """ Fit X, y dataset pair
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
        """get log(prior x likelihood) given sample data X
        """
        log_prior = np.log(self._prior)
        log_normal = - 0.5 * np.sum(np.log(2.0 * np.pi * self._var) +
                                    np.power(X[:, :, np.newaxis] - self._mean, 2) / self._var, axis=1)
        return log_prior + log_normal

    def _predict_prob(self, X):
        """ return probabilities of each class given dataset X
        """
        self._log_prob = self._get_log_prob(X)
        probs = np.exp(self._log_prob)
        return probs

    def predict(self, X):
        """predict labels given sample dataset X
        """
        if not self.__fit:
            raise "Dataset not yet fitted"
        self._log_prob = self._get_log_prob(X)
        y_idx = np.argmax(self._log_prob, axis=1)
        y = self._class[y_idx]
        return y


if __name__ == "__main__":
    nb_model = naivebayes()
    # X = np.vstack((np.random.rand(2, 1) + 10, np.random.rand(2, 1) * 3))
    # y = np.array([1, 1, 2, 2])
    np.random.seed(9)
    # create dummy dataset
    X = np.concatenate((np.random.rand(50, 1) + 10,
                        np.random.rand(50, 1) + 5), axis=0)
    y = np.concatenate((np.ones(50), np.zeros(50)), axis=0)
    nb_model.fit(X, y)
    y_pred = nb_model.predict(X)
    print("Accuracy = %f" % (np.sum(y_pred == y) / len(y)))

