# Machine Learning, K Nearest Neighbors

Most models squeeze a training set down into a handful of parameters and then
discard the data. K nearest neighbors refuses. It keeps every example it was
shown and, when a new point arrives, it finds the k closest things it has
seen before and lets them vote. Nothing is fitted, which is why it is called
a lazy learner, and all of the cost lands at prediction time instead of
training time.

That simplicity is the appeal. There is no assumption about the shape of the
decision boundary, no optimizer to babysit, and no way for the model to be
wrong about the functional form because it never proposes one.

## The pieces

* `standardize(x)` rescales each feature to zero mean and unit spread, and
  hands back the mean and scale so the same transform can be applied to test
  data.
* `KNearestNeighbors(k, task, weights)` holds the training set. Set `task` to
  "classification" for a vote over labels or "regression" for an average of
  values.
* `fit(x, y)` stores the data. That really is the entire training step.
* `predict(x)` returns one prediction per row.
* `predict_proba(x)` returns the share of the neighbor vote each class won.
* `accuracy(y_true, y_pred)` is the usual hit rate helper.

## Why scaling is not optional

Distance is measured in raw feature units. Put a price in dollars next to a
return in decimals and the price column decides every neighbor by itself,
because a move of one dollar swamps a move of one percent no matter which one
actually carries the signal. Standardizing puts every feature on the same
footing before the distances are computed. This is the single most common
reason a neighbors model performs badly for reasons that have nothing to do
with neighbors.

## Choosing k

Small k follows the training data closely. With k of one the model reproduces
its training set perfectly and cheerfully memorises every mislabelled point
along the way. Large k averages over a wider neighborhood, smoothing the
boundary and eventually flattening toward the overall class balance. The
useful value sits between those failures and is found by cross validation
rather than by argument. Distance weighting softens the transition, since a
far away neighbor stops counting as much as a close one.

## What it costs

Every prediction compares the new point against the whole training set, so
the work grows with the amount of data you have kept. That is fine for a few
thousand rows and painful for a few million. The other cost is subtler. In
high dimensions distances between points all drift toward the same value, so
the idea of a nearest neighbor stops meaning much. Neighbors models want few
features and plenty of rows, which is the opposite of what a wide factor
panel usually offers.

## Example

```python
from knn import KNearestNeighbors, accuracy, standardize

x = [[0.1, 1.0], [0.2, 0.9], [1.9, 0.1], [2.0, 0.2], [0.15, 1.1], [2.1, 0.15]]
y = [0, 0, 1, 1, 0, 1]

x_scaled, mean, scale = standardize(x)
model = KNearestNeighbors(k=3).fit(x_scaled, y)

print(model.predict([[0.0, 0.0]]))
print(model.predict_proba([[0.0, 0.0]]))
```

## Where to go next

* For a model that does compress its training data into parameters see
  [`Machine Learning - Logistic Regression`](../Machine%20Learning%20-%20Logistic%20Regression/).
* For picking k honestly see
  [`Machine Learning - Cross Validation`](../Machine%20Learning%20-%20Cross%20Validation/).
* For the unsupervised cousin that also lives on distances see
  [`Machine Learning - K-Means Clustering`](../Machine%20Learning%20-%20K-Means%20Clustering/).
