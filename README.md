# Simple Classifiers from Scratch

Another implementation exercise on writing common ML packages. Here I implmented two commonly used classifiers, namely naive bayes and logistic regression for multi-class classification problems.

## Set-up
To use this package, only numpy is required. If not installed run: `pip install numpy` in terminal.

To run comparisons with Scikit-learn in sample_test.ipynb, the corresponding package is required. If not yet installed, run `pip install -U scikit-learn` in terminal.

## How-to-use 

### Naive Bayes
The interfaces is shown as follows: 
```python
class naivebayes.NaiveBayes(model="gaussian", epsilon=1e-8):
```

#### Tunable parameters include:
- `model: string, optional (default=gaussian)`, the model used to describe feature, currently only `gaussian` is implemented.
- `epsilon: float, optional (default=1e-8)`, a small number for variance smoothing, avoid division by zero.

#### Methods:
- `fit(self, X, y)`, fits naive bayes model from training set `(X, y)`:
  - `X` is numpy array of shape [ n_samples x n_features ].
  - `y` is numpy array of shape [ n_samples ].
- `predict(self, X)`, predicts labels given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ].
- `_predict_prob(self, X)`, predicts probabilities associated with each label given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ].
*** 
### Logistic Regression
Binary logistic regression uses cross entroy loss and is solved iteratively via stochastic gradient descent.
Multi-class can use either `ovr` or `multinomial`.

The interfaces are shown as follows: 
```python
class regression.MulticlassLogisitc(scheme="multinomial"):
```

### Tunable parameters include:
- `scheme: string, optional (default=multinomial)`, scheme used for logistic regression, possible choices include `multinomial`, `ovr` or `binary`, note `binary` is only suited for class number of 2.

### Methods:
- `fit(self, X, y, *, Val_Xy=None, alpha=1e-1, decay=0.99, max_epoch=100, batch_size=32, epsilon=1e-8, flag=0)`, fits logistic model from training set `(X, y)`:
  - `X` is numpy array of shape [ n_samples x n_features ].
  - `y` is numpy array of shape [ n_samples ].
  - `Val_Xy, optional, (default=None)` is a tuple containing validation set `X` and `y`, automatically triggers loss display during fitting.
  - `alpha, optional, (default=1e-1)` is learning rate for gradient update. 
  - `decay, optional, (default=0.99)` is the decay rate for gradient update.
  - `max_epoch, optional, (default=100)` is the epoch limit for running SGD.
  - `batch_size, optional, (default=32)` is the batch size of batch gradient descent.
  - `epsilon, optional, (default=1e-8)` is the smallest admissable delta in loss, anything smaller will trigger early stop.
  - `flag, optional, (default=0)` set to `1` for training loss display during fitting.
- `predict(self, X)`, predicts labels given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ].
- `_predict_prob(self, X)`, predicts probabilities associated with each label given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ].


## Sample Use
Run sample_test.ipynb for a quick demo and comparing the results from naive bayes and multi-class logistic regression.  
An example use code snippet is shown below.
```python
...
bayes_model = NaiveBayes()
bayes_model.fit(X_train, y_train)
y_pred = bayes_model.predict(X_test)
print("Naive Bayes Accuracy = %f" % sum(y_test != y_pred)/len(y_test))

logistic_model = MulticlassLogistic()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic Regression Accuracy = %f" % sum(y_test != y_pred)/len(y_test))
```

Example code also includes image generations to visualize class distributions as shown below:
<div>
<img src=data/gaussian_log_multi.png width="500"/>
</div>