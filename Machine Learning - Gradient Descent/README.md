# Machine Learning, Gradient Descent

Gradient descent is the engine inside almost every model that learns. The idea
is small enough to hold in your head. You have a loss that measures how wrong
the model is, and the gradient of that loss points in the direction the error
grows fastest. So you step the opposite way, downhill, and repeat. Take enough
small steps with a sensible step size and the parameters slide into the values
that fit the data.

This lesson fits a plain linear model, predict y from X, by minimising the mean
squared error. It does the maths on NumPy with no learning framework hiding the
mechanics, so you can watch the loss fall epoch by epoch.

## Two ways to take a step

* `gradient_descent(X, y, lr, epochs)` is full batch descent. Every step uses
  the whole dataset to compute one exact gradient. It is steady and easy to
  reason about, and it is what the example uses to recover known coefficients.
* `sgd(X, y, lr, epochs)` is stochastic descent. Each step uses a single
  shuffled row, so the steps are noisy but cheap. The noise averages out over a
  full pass, and this is the version that scales to data too large to fit a full
  gradient over.

Both return the fitted weights, the bias, and the loss recorded at each epoch so
you can plot the descent and confirm it is heading down.

## The learning rate is the whole game

The learning rate is how big a step you take. Too small and the model crawls,
needing far more epochs than it should. Too large and each step overshoots the
bottom, the loss bounces and can blow up to infinity. The demo sweeps three
rates so you can see the slow one, the good one, and the unstable one side by
side. When in doubt start small and increase until the loss stops improving
faster.

## Why standardise the features first

Descent struggles when features live on very different scales, because the large
feature dominates the gradient and a single learning rate cannot suit both. The
`standardize` helper rescales every column to zero mean and unit spread so they
pull with equal weight. It also returns the mean and spread it used, which you
need to apply the same transform to new data and to map the learned weights back
to the original units.

## Example

```python
import numpy as np
from gradient_descent import gradient_descent, standardize, predict, mse

X = np.random.default_rng(0).normal(size=(200, 2))
y = X @ np.array([3.0, -2.0]) + 5.0

Xs, mean, std = standardize(X)
result = gradient_descent(Xs, y, lr=0.1, epochs=300)

print(result["history"][0], result["history"][-1])   # loss falls
print(mse(y, predict(Xs, result["weights"], result["bias"])))
```

## What gradient descent does and does not give you

It finds parameters that minimise the loss you handed it, nothing more. For the
straight line fit here a closed form solution exists and would be exact, so the
point of descent is not this problem but every harder one where no formula
exists, from logistic regression to deep networks. The same loop, gradient then
step, scales all the way up.

## Where to go next

* For a model trained the same way on a yes or no target see
  [`Machine Learning - Logistic Regression`](../Machine%20Learning%20-%20Logistic%20Regression/).
* For the optimisation theory behind descent and its cousins see
  [`Quantitative Methods - Optimization`](../Quantitative%20Methods%20-%20Optimization/).
* For preparing inputs before a model sees them see
  [`Machine Learning - Feature Engineering`](../Machine%20Learning%20-%20Feature%20Engineering/).
