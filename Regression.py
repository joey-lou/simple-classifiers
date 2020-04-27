import numpy as np
from abc import ABC, abstractmethod

# TODO: add ovr

class logisticbase(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _check_Xy(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
        assert X.shape[0] == y.shape[0] and len(y.shape) == 1, "Accepcted input dimesions: X ~ [m, d], y ~ [m, ]"
        return X, y

    def _make_batch(self, X, y, batch_size):
        """ Make batch from sample, without replacement
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
        one_hot = []
        for class_ in classes:
            one_hot.append((y == class_) * 1)
        return np.stack(one_hot, axis=-1)

class binarylogistic(logisticbase):
    def __init__(self):
        self._w = None
        self._b = None
        self.__fit = False

    def fit(self, X, y, *, Val_Xy=None, alpha=1e-1, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        X, y = self._check_Xy(X, y)
        
        if Val_Xy:
            flag = 2
            X_val, y_val = Val_Xy
            assert X_val.shape[0] == y_val.shape[0], "Validation set dimension mismatch"

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
                if flag == 1:
                    print("Epoch %i [%i/%i]: Batch Loss: %f" %
                        (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b)))
                elif flag == 2:
                    print("Epoch %i [%i/%i]: Batch Loss: %f, Validation Loss: %f" %
                        (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b), self._cross_entropy_loss(X_val, y_val, w, b)))
            alpha *= decay
            train_loss = self._cross_entropy_loss(X, y, w, b)
            if flag:
                print("Epoch %i: Validation Loss: %f" % (i + 1, self._cross_entropy_loss(X_val, y_val, w, b)))
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

    def predict(self, X):
        assert self.__fit, "No training data fitted yet"
        prob = self._sigmoid(X @ self._w + self._b)
        return (prob > 0.5) * 1
  

class multinormlogsitic(logisticbase):
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
            flag = 2
            X_val, y_val = Val_Xy
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
                if flag == 1:
                    print("Epoch %i [%i/%i]: Batch Loss: %f" %
                        (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b)))
                elif flag == 2:
                    print("Epoch %i [%i/%i]: Batch Loss: %f, Validation Loss: %f" %
                        (i + 1, j + 1, b_num, self._cross_entropy_loss(Xb, yb, w, b), self._cross_entropy_loss(X_val, y_val, w, b)))
            alpha *= decay
            train_loss = self._cross_entropy_loss(X, y, w, b)
            if flag:
                print("Epoch %i: Validation Loss: %f" % (i + 1, self._cross_entropy_loss(X_val, y_val, w, b)))
            # early stop if loss stops decreasing
            if np.abs(prev_train - train_loss) < epsilon:
                break
            prev_train = train_loss
        
        self._w = w
        self._b = b
        self.__fit = True

    def predict(self, X):
        assert self.__fit, "No training data fitted yet"
        prob = self._soft_max(X @ self._w + self._b)
        return self._class[np.argmax(prob, axis=1)]

    def _soft_max(self, Z):
        """ Z has dimension [m, k]
        """
        prob = np.exp(Z)
        return prob / np.sum(prob, axis=1, keepdims=True)

    def _cross_entropy_loss(self, X, y, w, b):
        p = self._soft_max(X @ w + b)
        return - np.sum(y * np.log(p)) / X.shape[0]



class multiclasslogistic(logisticbase):
    def __init__(self, scheme="multinorm"):
        self._models = []
        self._classes = []
        self.__fit = False
        self.scheme = scheme

        if self.scheme == "multinorm":
            self._models = multinormlogsitic()

    def predict(self, X):
        if self.scheme == "multinorm":
            return self._models.predict(X)
    
    def fit(self, X, y):
        if self.scheme == "multinorm":
            self._models.fit(X, y)

if __name__ == "__main__":
    lg_model = multiclasslogistic()
    np.random.seed(9)
    # create dummy dataset
    X = np.concatenate((np.random.rand(50, 1) + 10,
                        np.random.rand(50, 1) + 5), axis=0)
    y = np.concatenate((np.ones(50), np.zeros(50)), axis=0)

    lg_model.fit(X, y)
    y_pred = lg_model.predict(X)
    print("Accuracy = %f" % (np.sum(y_pred == y) / len(y)))