# Machine Learning, Support Vector Machine

<p class="lq-meta">Advanced <span>/</span> AI & Machine Learning <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Machine%20Learning%20-%20Support%20Vector%20Machine">source</a></p>


Most classifiers stop as soon as they find a line that separates the two
classes. A support vector machine keeps going, and looks for the line with the
widest empty corridor around it. The argument is simple. A boundary that sits
far away from every training point is the one most likely to still be right on
data it has never seen, while a boundary that shaves past the nearest points
is one bad observation away from being wrong.

The points that end up touching the edges of that corridor are the support
vectors, and they are the only ones that matter. Take a point sitting deep
inside its own class and move it further in, and the fitted boundary does not
move at all. That is a genuinely different property from a model where every
observation tugs on the answer.

## The pieces

* `standardize(X)` puts every feature on the same scale and returns the mean
  and deviation it used, which must be reused on later data.
* `LinearSVM(learning_rate, lam, epochs)` is the model, trained by subgradient
  descent on the soft margin objective.
* `fit(X, y)` accepts labels as either zero and one or minus one and one.
* `decision_function(X)` returns the signed distance from the boundary, where
  the sign is the class and the size is the confidence.
* `predict(X)` returns plus or minus one per row.
* `support_vectors(X, y)` returns the indices of the points the boundary is
  actually resting on.
* `margin_width()` returns how wide the corridor ended up.
* `hinge_loss(scores, labels)` and `accuracy(predicted, actual)` are the two
  scores worth watching while you tune.

## Standardize or do not bother

The margin is a distance, so the model is measuring the geometry of your
feature space directly. Give it one feature in dollars and another in decimal
percent, and the dollar feature will decide almost the entire boundary purely
because its numbers are larger. There is no version of this model that is
robust to that. Standardizing first is part of the method rather than a
finishing touch, and the statistics used to standardize the training set are
what must be applied to any new data. Recomputing them from a test set leaks
information backwards and inflates whatever score you get.

## The soft margin and the one knob

Real market data is never cleanly separable, so points are allowed to sit
inside the corridor and even on the wrong side of it. Each one pays a penalty
equal to how far it strayed, which is the hinge loss. A point that clears the
margin comfortably pays nothing at all, and this is what makes the model
ignore the easy cases and concentrate entirely on the boundary region.

The regularization strength `lam` sets the trade. Raise it and the model
prefers a wide corridor even at the cost of misclassifying more points. Lower
it and the model will bend itself into whatever shape gets the training set
right, which usually means memorising noise. Watch the number of support
vectors as you vary it. When almost every point has become a support vector,
the regularizer rather than the data is setting your boundary.

Cross validation is the honest way to pick it. Picking the value with the best
training accuracy will always hand you the lowest one available.

## Why the fit can diverge

Training is gradient descent, and gradient descent on this objective becomes
unstable when the learning rate multiplied by the regularization strength
approaches two. Past that point the weights flip sign and grow every epoch
until they overflow, and what comes back is not a bad fit but arithmetic
garbage. The constructor refuses that combination rather than let it happen
quietly. If the loss history is not falling, the learning rate is the first
thing to halve.

## What is left out

This is the linear version. The kernel trick, which is what made these models
famous, replaces every dot product with a kernel function and lets the same
machinery draw curved boundaries in the original feature space without ever
computing the coordinates it would need to do so. It is a beautiful idea and
it belongs in its own lesson. Real work also usually reaches for a solver that
finds the exact optimum rather than a descent that approaches it, and for more
than a few thousand points that difference starts to matter.

## Example

```python
from support_vector_machine import LinearSVM, accuracy, standardize

features = [[1.2, -0.4], [0.9, -0.7], [-1.1, 0.8], [-0.8, 1.2]]
labels = [1, 1, -1, -1]

scaled, mean, std = standardize(features)
model = LinearSVM(learning_rate=0.05, lam=0.01, epochs=1000).fit(scaled, labels)

print(model.predict(scaled))
print(model.margin_width())
print(accuracy(model.predict(scaled), labels))
```

## Where to go next

* For the probability outputs this model does not give you see
  [`Machine Learning - Logistic Regression`](Machine Learning - Logistic Regression.md).
* For the optimizer underneath the fit see
  [`Machine Learning - Gradient Descent`](Machine Learning - Gradient Descent.md).
* For picking the regularization strength honestly see
  [`Machine Learning - Cross Validation`](Machine Learning - Cross Validation.md).
* For preparing the inputs before any of this see
  [`Machine Learning - Feature Engineering`](Machine Learning - Feature Engineering.md).

---

## Running it

```bash
cd "Machine Learning - Support Vector Machine"
python "support_vector_machine.py"
```

## More in AI & Machine Learning

- [AI Development](AI Development.md): Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js.
- [Learning Platform](Learning Platform.md): An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface.
- [Machine Learning - Cross Validation](Machine Learning - Cross Validation.md): A model scored on the data it was trained on is a student grading their own exam.
- [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md): The dirty secret of quant machine learning: the model is rarely the bottleneck.
- [Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md): Gradient descent is the engine inside almost every model that learns.
- [Machine Learning - K Nearest Neighbors](Machine Learning - K Nearest Neighbors.md): Most models squeeze a training set down into a handful of parameters and then discard the data.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
