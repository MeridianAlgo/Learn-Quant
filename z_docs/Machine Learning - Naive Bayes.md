# Machine Learning, Naive Bayes

<p class="lq-meta">Advanced <span>/</span> AI & Machine Learning <span>/</span> Python <span>/</span> <a href="https://github.com/MeridianAlgo/Learn-Quant/tree/main/Machine%20Learning%20-%20Naive%20Bayes">source</a></p>


Bayes' theorem tells you how to update a belief about a class label after
seeing evidence. The naive part is a bold simplification, assume every
feature is independent of every other feature once you know the class. That
assumption is almost always false in practice, yet the resulting classifier
is fast, needs remarkably little data, and is famously hard to beat as a
baseline.

This lesson implements the Gaussian flavour, which models each feature
within each class as a normal distribution. Training is nothing more than
computing a mean and a variance per feature per class plus the class
frequencies. There is no iteration, no learning rate, and no convergence to
worry about.

## The pieces

* `fit_gaussian_nb(features, labels)` learns the class priors and the per
  class feature means and variances, returned as a plain dictionary. A small
  variance floor protects against constant features.
* `log_posteriors(model, features)` returns the unnormalized log posterior
  of each class for each sample. Everything happens in log space so hundreds
  of tiny likelihoods do not underflow to zero.
* `predict(model, features)` picks the class with the highest posterior for
  each sample.
* `predict_proba(model, features)` normalizes the posteriors into
  probabilities using the stable shifted softmax.
* `accuracy(model, features, labels)` scores predictions against known
  labels.

## Why the wrong assumption still works

Independence between features is rarely true, correlated features get
counted twice and the posterior probabilities come out overconfident. But
classification only needs the ranking of the classes to be right, not the
probabilities themselves, and the ranking survives a great deal of double
counting. That is why naive Bayes often matches far more elaborate models on
small and noisy datasets, the kind quant work is full of.

## A market flavoured use

The demo classifies trading days into calm and stressed regimes from two
features, the size of the day's move and the volume relative to normal. The
same pattern extends to any labelled classification problem, spam versus
ham, default versus repay, fill versus no fill.

## Example

```python
import numpy as np
from naive_bayes import fit_gaussian_nb, predict, predict_proba

X = np.array([[0.2, 1.0], [0.3, 0.9], [1.5, 2.0], [1.1, 1.8]])
y = np.array(["calm", "calm", "stress", "stress"])

model = fit_gaussian_nb(X, y)
print(predict(model, np.array([[0.4, 1.1]])))
print(predict_proba(model, np.array([[1.0, 1.6]])))
```

## Where to go next

* For the Bayesian machinery behind the update see
  [`Quantitative Methods - Bayesian Inference`](Quantitative Methods - Bayesian Inference.md).
* For a discriminative counterpart that learns boundaries directly see
  [`Machine Learning - Logistic Regression`](Machine Learning - Logistic Regression.md).
* For judging any classifier honestly see
  [`Machine Learning - Cross Validation`](Machine Learning - Cross Validation.md).

---

## Running it

```bash
cd "Machine Learning - Naive Bayes"
python "naive_bayes.py"
```

## More in AI & Machine Learning

- [AI Development](AI Development.md): Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js.
- [Learning Platform](Learning Platform.md): An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface.
- [Machine Learning - Cross Validation](Machine Learning - Cross Validation.md): A model scored on the data it was trained on is a student grading their own exam.
- [Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md): The dirty secret of quant machine learning: the model is rarely the bottleneck.
- [Machine Learning - Gradient Descent](Machine Learning - Gradient Descent.md): Gradient descent is the engine inside almost every model that learns.
- [Machine Learning - K Nearest Neighbors](Machine Learning - K Nearest Neighbors.md): Most models squeeze a training set down into a handful of parameters and then discard the data.

[Every module on one page](modules.md) or [suggested reading orders](learning-paths.md).
