# Simple Classifiers from Scratch

Another implementation exercise on writing common ML packages. Here I implmented two commonly used classifiers, namely naive bayes and logistic regression for multi-class classification problems.

## Set-up
To use this package, only numpy is required. If not installed run: `pip install numpy` in terminal.

To run comparisons with Scikit-learn in sample_test.ipynb, the corresponding package is required. If not yet installed, run `pip install -U scikit-learn` in terminal.

## How-to-use
### Naive Bayes
The interfaces is shown as follows: 
```python
class naivebayes.NaiveBayes(model="gaussian", epsilon=1e-8)
```

#### Tunable parameters include:
- `model: string, optional (default=gaussian)`, the model used to describe feature, currently only `gaussian` is implemented.
- `epsilon: float, optional (default=1e-8)`, a small number for variance smoothing, avoid division by zero.

#### Methods:
- `fit(self, X, y)`, fits naive bayes model from training set `(X, y)`:
  - `X` is numpy array of shape [ n_samples x n_features ].
  - `y` is numpy array of shape [ n_samples ].
- `predict(self, X)`, predicts labels given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ]
- `_predict_prob(self, X)`, predicts probabilities associated with each label given dataset `X`.
  - `X` is numpy array of shape [ n_samples x n_features ]

### Logistic Regression
Binary logistic regression uses cross entroy loss and is solved iteratively via stochastic gradient descent.
Multi-class can use either `ovr` or `multinomial`.

#### Tunable parameters include:
- `feature_count: integer, optional (default=1)`, number of subset features selected.
- `feature_count: integer, optional (default=1)`, number of subset features selected.


The interfaces are shown as follows: 
```python
class naivebayes.NaiveBayes(model="gaussian", epsilon=1e-8)

class regression.MulticlassLogisitc(scheme="multinomial")

class regression.BinaryLogistic()
```

### Tunable parameters include:
- `feature_count: integer, optional (default=1)`, number of subset features selected.
- `tree_num: integer, optional (default=10)`, number of decision trees to use in forest.
- `depth: integer, optional (default=10)`, maximum depth of each decision tree.
- `min_improv: float, optional (default=1e-8)`, minimum improvement in gini impurity/entropy required to split a node further.
- `eval_func: string, optional (default="gini_impurity")`, evaluation criteria, either gini impurity `"gini_impurity"` or entropy `"entropy"`.

### Methods:
- `fit(self, X, y)`, fits random forest of trees from training set `(X, y)`:
  - `X` is numpy array of shape [ n_samples x n_features ].
  - `y` is numpy array of shape [ n_samples ].
- `predict(self, X, rule="prob")`, predicts labels given dataset `X` and prediction rule.
  - `X` is numpy array of shape [ n_samples x n_features ]
  - `rule, optional (default="prob")` can also be `"majority"` whereby prediction is based on majority ruling from each decision trees.

## Sample Use
Run sample_test.ipynb for a quick demo and comparing the results from naive bayes and multi-class logistic regression.  
An example use code snippet is shown below.
```python
...
forest = RandomForest()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("Accuracy = %f" % sum(y_test != y_pred)/len(y_test))
```
