import numpy as np
from abc import ABC, abstractmethod


class LogisticBase(ABC):
    """Logistic regression base class.
    """

    @abstractmethod
    def fit(self, X, y):
        """Fit model given X, y training data pair. 

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].
                y (numpy int array): output class label y, shape [n].
        """

    @abstractmethod
    def predict(self, X):
        """Predict model output given X. 

            Args:
                X (numpy float array): input feature matrix X, shape [n x d].

            Returns:
                y (numpy int array): class label of each sample, shape [n].
        """

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

    def _make_batch(self, X, y, batch_size):
        """Make samll batch to train model from sample, without replacement.

            Args:
                X (numpy float array): sample feature X, shape [n x d].
                y (numpy int array): sample label y, shape [n].
                batch_size (int): batch size should be an integer greater than 0.

            Returns:
                X_batch (numpy float array): sample feature X, shape [batch_size x d].
                y_batch (numpy int array): sample label y, shape [batch_size].
        """
        m, d = X.shape
        idx = np.arange(m)
        np.random.shuffle(idx)
        X, y = X[idx, ], y[idx, ]
        if m <= batch_size:
            return [X], [y]
        batches = (m + batch_size - 1) // batch_size
        X_batch, y_batch = [], []
        for i in range(batches):
            X_batch.append(X[i * batch_size:(i + 1) * batch_size, ])
            y_batch.append(y[i * batch_size:(i + 1) * batch_size, ])
        return X_batch, y_batch

    @staticmethod
    def one_hot_encode(y, classes):
        """One-hot-encode class labels. Convert categorical labels into binary columns.

            Args:
                y (numpy int array): sample label y, shape [n].
                classes (int array): possible class labels of the model, shape [k].

            Returns:
                y_one_hot (numpy int array): one-hot-encoded labels, shape [n x k].
        """
        one_hot = []
        for class_ in classes:
            one_hot.append((y == class_) * 1)
        return np.stack(one_hot, axis=-1)


class BinaryLogistic(LogisticBase):
    """Binary logistic model for classification problems
    """

    def __init__(self):
        self._w = None
        self._b = None
        self.__fit = False

    @staticmethod
    def _check_binary(y):
        assert set([0, 1]) == set(np.unique(y)), \
            "binary classification only takes in class labels of 0 and 1"

    def fit(self, X, y, *, Val_Xy=None, alpha=1e-1, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        X, y = self._check_Xy(X, y)

        if Val_Xy:
            X_val, y_val = self._check_Xy(*Val_Xy)
            assert X_val.shape[0] == y_val.shape[0], "Validation set dimension mismatch"
            self._check_binary(y_val)

        self._check_binary(y)
        m, d = X.shape
        # check literature for better intialization scheme, though
        # cross_entropy is convex
        w = np.random.normal(1e-3, 1e-6, (d,))
        b = np.zeros(1)
        prev_train = float("inf")
        for i in range(max_epoch):
            X_batch, y_batch = self._make_batch(X, y, batch_size)
            b_num = len(X_batch)
            for j, (Xb, yb) in enumerate(zip(X_batch, y_batch)):
                residue = yb - self._sigmoid(Xb @ w + b)
                w_grad = - Xb.T @ residue / Xb.shape[0]
                b_grad = - np.mean(residue)

                w -= alpha * w_grad
                b -= alpha * b_grad
                if flag:
                    print("Epoch %i [%i/%i]: Batch Loss: %f" %
                          (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b)))

            alpha *= decay
            train_loss = self._cross_entropy_loss(X, y, w, b)
            if Val_Xy:
                print("Epoch %i: Validation Loss: %f" %
                      (i + 1, self._cross_entropy_loss(X_val, y_val, w, b)))
            # early stop if loss stops decreasing
            if np.abs(prev_train - train_loss) < epsilon:
                break
            prev_train = train_loss

        self._w = w
        self._b = b
        self.__fit = True

    def _cross_entropy_loss(self, X, y, w, b):
        p = self._sigmoid(X @ w + b)
        return - (np.dot(y, np.log(p)) + np.dot(1 - y, np.log(1 - p))) / X.shape[0]

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _predict_prob(self, X):
        assert self.__fit, "No training data fitted yet"
        prob = self._sigmoid(X @ self._w + self._b)
        return prob

    def predict(self, X):
        prob = self._predict_prob(X)
        return (prob > 0.5) * 1


class MultiNomialLogsitic(LogisticBase):
    def __init__(self):
        self._w = None
        self._b = None
        self.__fit = False
        self._class = None

    def fit(self, X, y, *, Val_Xy=None, alpha=1e-1, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        X, y = self._check_Xy(X, y)
        self._class = np.sort(np.unique(y))
        y = self.one_hot_encode(y, self._class)

        if Val_Xy:
            X_val, y_val = Val_Xy
            y_val = self.one_hot_encode(y_val, self._class)
            assert X_val.shape[0] == y_val.shape[0], "Validation set dimension mismatch"

        m, d = X.shape
        k = len(self._class)
        # check literature for better intialization scheme, though
        # cross_entropy is convex
        w = np.random.normal(1e-3, 1e-6, (d, k))
        b = np.zeros(k)
        prev_train = float("inf")
        for i in range(max_epoch):
            X_batch, y_batch = self._make_batch(X, y, batch_size)
            b_num = len(X_batch)
            for j, (Xb, yb) in enumerate(zip(X_batch, y_batch)):
                residue = yb - self._soft_max(Xb @ w + b)
                w_grad = - Xb.T @ residue / Xb.shape[0]
                b_grad = - np.mean(residue, axis=0)

                w -= alpha * w_grad
                b -= alpha * b_grad
                if flag:
                    print("Epoch %i [%i/%i]: Batch Loss: %f" %
                          (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b)))
            alpha *= decay
            train_loss = self._cross_entropy_loss(X, y, w, b)
            if Val_Xy:
                print("Epoch %i: Validation Loss: %f" %
                      (i + 1, self._cross_entropy_loss(X_val, y_val, w, b)))
            # early stop if loss stops decreasing
            if np.abs(prev_train - train_loss) < epsilon:
                break
            prev_train = train_loss

        self._w = w
        self._b = b
        self.__fit = True

    def _predict_prob(self, X):
        assert self.__fit, "No training data fitted yet"
        prob = self._soft_max(X @ self._w + self._b)
        return prob

    def predict(self, X):
        probs = self._predict_prob(X)
        return self._class[np.argmax(probs, axis=1)]

    def _soft_max(self, Z):
        """ Z has dimension [m, k]
        """
        prob = np.exp(Z)
        return prob / np.sum(prob, axis=1, keepdims=True)

    def _cross_entropy_loss(self, X, y, w, b):
        p = self._soft_max(X @ w + b)
        return - np.sum(y * np.log(p)) / X.shape[0]


class OvrLogistic(LogisticBase):
    def __init__(self,):
        self._models = []
        self.__fit = False
        self._class = None

    def fit(self, X, y, *, Val_Xy=None, alpha=1e-1, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        # 'ovr' scheme
        X, y = self._check_Xy(X, y)
        self._class = np.sort(np.unique(y))
        Val_Xy_ = None
        for i, class_ in enumerate(self._class):
            model = BinaryLogistic()
            if flag:
                print("Training Model %i" % i)
            if Val_Xy:
                Val_Xy_ = (Val_Xy[0], (Val_Xy[1] == class_) * 1)
            y_ = (y == class_) * 1
            model.fit(X, y_, Val_Xy=Val_Xy_, alpha=alpha, decay=decay, max_epoch=max_epoch,
                      batch_size=batch_size, epsilon=epsilon, flag=flag)
            self._models.append(model)
        self.__fit = True

    def _predict_prob(self, X):
        """ Generate 
        """
        assert self.__fit, "No training data fitted yet"
        y_probs = []
        for i, class_ in enumerate(self._class):
            y_probs.append(self._models[i]._predict_prob(X))
        y_probs = np.stack(y_probs, axis=-1)
        return y_probs / np.sum(y_probs, axis=1, keepdims=True)

    def predict(self, X):
        probs = self._predict_prob(X)
        return self._class[np.argmax(probs, axis=1)]


class MulticlassLogistic:
    def __new__(self, scheme="multinomial"):
        if scheme == "multinomial":
            return MultiNomialLogsitic()
        elif scheme == "ovr":
            return OvrLogistic()
        elif scheme == 'binary':
            return BinaryLogistic()
        else:
            raise ValueError(
                "Scheme '%s' not supported, choose from 'binary', 'multinomial' or 'ovr'" % scheme)


if __name__ == "__main__":

    lg_model = MulticlassLogistic(scheme="binary")
    np.random.seed(9)
    # create dummy dataset
    X = np.concatenate((np.random.rand(50, 2) + 10,
                        np.random.rand(50, 2) + 5), axis=0)
    y = np.concatenate((np.ones(50) * 1, np.ones(50) * 0), axis=0)
    X_val = np.random.rand(20, 2) + 8
    y_val = np.ones(20)

    lg_model.fit(X, y, flag=0)

    y_pred = lg_model.predict(X)
    print("Accuracy = %f" % (np.sum(y_pred == y) / len(y)))
