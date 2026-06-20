<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Machine Learning - Logistic Regression"
    python "logistic_regression.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Machine%20Learning%20-%20Logistic%20Regression)

---
# Machine Learning — Logistic Regression

Linear regression predicts a number. **Logistic regression** predicts a
*probability* — the chance an example belongs to the positive class — which is
exactly the question behind *"will tomorrow be an up day?"*, *"will this loan
default?"*, or *"is this order toxic flow?"*. It is the simplest genuinely
useful classifier and the foundation the bigger models build on.

This module implements the whole thing from scratch with NumPy: the **sigmoid**,
the **log-loss**, **gradient descent**, **L2 regularisation**, and the standard
metrics — so none of it is a black box.

## Functions

| Function | Description |
|---|---|
| `fit(X, y, lr, epochs, l2)` | Train weights by batch gradient descent; returns `(w, b)` |
| `predict_proba(X, w, b)` | Positive-class probabilities |
| `predict(X, w, b, threshold)` | Hard 0/1 predictions |
| `sigmoid(z)` | Numerically-stable logistic squashing function |
| `log_loss(y, p)` | Binary cross-entropy — the training objective |
| `accuracy(y_true, y_pred)` | Fraction predicted correctly |
| `standardize(X)` | Z-score features; returns `(X_scaled, mean, std)` for reuse |

## How it works

1. Compute a linear score `z = X·w + b`.
2. Squash it into a probability with the **sigmoid** `p = 1 / (1 + e^-z)`, which
   maps any real number into `(0, 1)`.
3. Measure error with **log-loss** (cross-entropy), which punishes confident
   wrong answers harshly.
4. The gradient of log-loss is beautifully simple — `Xᵀ(p − y) / n` — so
   **gradient descent** nudges the weights downhill. The loss is **convex**, so
   there is a single global optimum and no local-minima traps.

## Example — predicting up days

```python
import numpy as np
from logistic_regression import fit, predict_proba, predict, accuracy, standardize

# X: features per day (e.g. momentum, volatility). y: 1 if next day is up.
X = np.array([[0.4, 1.1], [-0.2, 0.9], [1.3, 0.5], [-1.1, 1.4]])
y = np.array([1, 0, 1, 0])

Xs, mu, sd = standardize(X)           # always scale first
w, b = fit(Xs, y, lr=0.5, epochs=2000, l2=0.01)

print(predict_proba(Xs, w, b))        # probability of an up day
print(accuracy(y, predict(Xs, w, b)))
```

## Reading the output

- The **weights** are interpretable: a positive weight means the feature pushes
  the probability up; its size (on standardised features) is its importance.
- The **probability**, not just the 0/1 label, is the useful output for quant
  work — it sizes the bet. Pair it with the
  [`Finance - Kelly Criterion`](Finance - Kelly Criterion.md) to turn a
  probability into a position.

## Practical notes

- **Standardise your features.** Gradient descent converges far faster, and L2
  only makes sense when features share a scale. Keep the training `mean`/`std`
  and apply them to test data — never re-fit the scaler on test data.
- **Regularise.** A little `l2` shrinks weights, fights overfitting, and keeps
  things stable when features are correlated.
- **The decision boundary is linear.** If classes curve around each other,
  logistic regression underfits — add interaction/polynomial features (see
  [`Machine Learning - Feature Engineering`](Machine Learning - Feature Engineering.md))
  or move to a non-linear model like
  [`Machine Learning - Random Forest`](Machine Learning - Random Forest.md).
- **Watch class imbalance.** If 95% of days are "up", 95% accuracy is worthless.
  Look at log-loss and the confusion of the rare class, not raw accuracy.


---

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[AI Development](AI Development.md)__

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   :material-robot-outline: __[Learning Platform](Learning Platform.md)__

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   :material-robot-outline: __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   :material-robot-outline: __[Machine Learning - K-Means Clustering](Machine Learning - K-Means Clustering.md)__

    Given a few hundred stocks and their return characteristics, which ones behave

-   :material-robot-outline: __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

-   :material-robot-outline: __[Machine Learning Time Series](Machine Learning Time Series.md)__

    Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
