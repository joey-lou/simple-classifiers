import numpy as np

# TODO: add multi-class logreg
class binarylogsitic:
    def __init__(self, mode="primal", solver="sgd"):
        self.mode = mode
        self.solver = solver
        self._w = None
        self._b = None
        self.__fit = False

    def fit(self, X, y, *, Val_Xy=None, alpha=1e-3, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        X, y = self._check_Xy(X, y)
        self._class = np.sort(np.unique(y))
        if self.solver == "sgd":
            self._w, self._b = self._sgd_fit(X, y, Val_Xy=Val_Xy, 
                                alpha=alpha, decay=decay, max_epoch=max_epoch, 
                                batch_size=batch_size, epsilon=epsilon, flag=flag)
            self.__fit = True
            
    def _check_Xy(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)
        assert (np.sort(np.unique(y)) == [0, 1]).all(), "Binary logistic regression only takes in [0, 1] as predicted labels"
        assert X.shape[0] == y.shape[0] and len(y.shape) == 1, "Accepcted input dimesions: X ~ [m, d], y ~ [m, ]"
        return X, y

    def predict(self, X):
        assert self.__fit, "No training data fitted yet"
        prob = self._sigmoid(X @ self._w + self._b)
        return (prob > 0.5) * 1

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

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _cross_entropy_loss(self, X, y, w, b):
        p = self._sigmoid(X @ w + b)
        return (- np.dot(y, np.log(p)) - np.dot((1 - y), np.log(1 - p))) / X.shape[0]

    def _sgd_fit(self, X, y, *, Val_Xy=None, alpha=1e-3, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0):
        """ Vanilla SGD
        """
        if Val_Xy:
            flag = 2
            X_val, y_val = Val_Xy
            assert X_val.shape[0] == y_val.shape[0], "Validation set dimension mismatch"

        m, d = X.shape
        # check literature for better intialization scheme, though
        # cross_entropy is convex
        w = np.random.normal(1e-3, 1e-6, d)
        b = np.zeros(d)
        prev_train = float("inf")
        for i in range(max_epoch):
            X_batch, y_batch = self._make_batch(X, y, batch_size)
            b_num = len(X_batch)
            for j, (Xb, yb) in enumerate(zip(X_batch, y_batch)):
                residue = yb - self._sigmoid(Xb @ w + b)
                w_grad = - residue.T @ Xb / Xb.shape[0]
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
                print("Epoch %i: Training Loss: %f" % (i + 1, train_loss))
            # early stop if loss stops decreasing
            if np.abs(prev_train - train_loss) < epsilon:
                break
            prev_train = train_loss
        return w, b


if __name__ == "__main__":
    lg_model = binarylogsitic()
    np.random.seed(9)
    # create dummy dataset
    X = np.concatenate((np.random.rand(50, 1) + 10,
                        np.random.rand(50, 1) + 5), axis=0)
    y = np.concatenate((np.ones(50), np.zeros(50)), axis=0)
    lg_model.fit(X, y, alpha=1e-1)
    y_pred = lg_model.predict(X)
    print("Accuracy = %f" % (np.sum(y_pred == y) / len(y)))