<p class="lq-badges"><span class="lq-badge lq-advanced">Advanced</span><span class="lq-badge lq-cat">AI & Machine Learning</span><span class="lq-badge lq-lang">Python</span></p>

!!! tip "Run this module"
    ```bash
    cd "Machine Learning - K-Means Clustering"
    python "kmeans_clustering.py"
    ```

    [:octicons-mark-github-16: View source on GitHub](https://github.com/MeridianAlgo/Learn-Quant/tree/main/Machine%20Learning%20-%20K-Means%20Clustering)

---
# Machine Learning — K-Means Clustering

Given a few hundred stocks and their return characteristics, which ones behave
alike? **K-means** is the simplest answer: it partitions data into `k` groups so
that each point sits with the cluster whose centre (centroid) is nearest. In
quant work it groups assets into peers, compresses a correlation matrix into a
handful of regimes, or finds clusters of similar trading days.

This module implements the whole thing from scratch — including the **k-means++**
initialisation and a **silhouette score** for choosing `k` — so the algorithm is
no longer a black box.

## Functions

| Function | Description |
|---|---|
| `kmeans(X, k, n_init, random_state)` | Full k-means++ clustering; returns `(labels, centroids, inertia)` |
| `assign_clusters(X, centroids)` | Label each point with its nearest centroid |
| `update_centroids(X, labels, k)` | Recompute centroids as cluster means |
| `inertia(X, labels, centroids)` | Within-cluster sum of squares (the loss) |
| `silhouette_score(X, labels)` | Cohesion-vs-separation score in `[-1, 1]` |

## The algorithm (Lloyd's)

A two-step loop you can hold in your head:

1. **Assign** — put every point with its nearest centroid.
2. **Update** — move each centroid to the mean of its assigned points.

Repeat until nothing moves. It is guaranteed to converge, but only to a *local*
optimum — which is why initialisation matters so much.

## k-means++ — starting well

Random initial centroids can land two seeds inside the same blob and split it in
half. **k-means++** picks the first centroid at random, then each subsequent one
with probability proportional to its squared distance from the nearest existing
centroid — spreading the seeds out and dramatically improving the final result.

## Example

```python
import numpy as np
from kmeans_clustering import kmeans, silhouette_score

# Rows are assets; columns are (annual return, annual volatility), say.
X = np.array([[0.05, 0.10], [0.06, 0.12], [0.20, 0.35], [0.22, 0.40]])

labels, centroids, wcss = kmeans(X, k=2, random_state=0)
print(labels)                          # e.g. [0 0 1 1] — low-vol vs high-vol
print(silhouette_score(X, labels))     # closer to 1 = tighter, cleaner split
```

## Choosing k — elbow and silhouette

`k` is not learned; you choose it. Two standard tools:

- **Elbow method** — plot `inertia` against `k`. It always falls as `k` grows,
  but the *rate* of improvement flattens at the natural number of clusters — the
  "elbow".
- **Silhouette score** — averages, per point, how much closer it is to its own
  cluster than to the next nearest one. Unlike inertia it does **not** simply
  improve with more clusters, so its maximum is a genuine signal.

## Practical notes

- **Scale your features first.** K-means uses Euclidean distance, so a feature
  measured in thousands will dominate one measured in decimals. Standardise
  (z-score) returns and volatilities before clustering — see
  [`Quantitative Methods - Statistics`](Quantitative Methods - Statistics.md).
- **It assumes round, similar-sized clusters.** Elongated or nested shapes break
  it; that is a limitation of the distance metric, not a bug.
- Clustering the **correlation matrix** of returns is a powerful way to find
  asset groups; combine with
  [`Quantitative Methods - Principal Component Analysis`](Quantitative Methods - Principal Component Analysis.md)
  for dimensionality reduction first.
- For labelled prediction instead of unsupervised grouping, continue to
  [`Machine Learning - Random Forest`](Machine Learning - Random Forest.md).


---

## Continue in AI & Machine Learning

<div class="grid cards" markdown>

-   :material-robot-outline: __[AI Development](AI Development.md)__

    Command-line chatbots for Google's Gemini API, implemented in both Python and Node.js. This module demonstrates how to integrate a hosted large language model into a simple interactive application.

-   :material-robot-outline: __[Learning Platform](Learning Platform.md)__

    An all-in-one learning hub that delivers progressive Python lessons through both a guided CLI and a hostable Flask web interface. Lessons combine narrative walkthroughs, executable code examples, mini quizzes, and follow-up practice ideas geared toward aspiring quantitative developers.

-   :material-robot-outline: __[Machine Learning - Feature Engineering](Machine Learning - Feature Engineering.md)__

    The dirty secret of quant machine learning: the model is rarely the bottleneck.

-   :material-robot-outline: __[Machine Learning - Logistic Regression](Machine Learning - Logistic Regression.md)__

    Linear regression predicts a number. **Logistic regression** predicts a

-   :material-robot-outline: __[Machine Learning - Random Forest](Machine Learning - Random Forest.md)__

    This module provides a basic implementation of a Random Forest Predictor for quantitative finance. It uses scikit-learn's `RandomForestRegressor` to predict time series data or returns based on a set of features.

-   :material-robot-outline: __[Machine Learning Time Series](Machine Learning Time Series.md)__

    Applying incredibly sophisticated statistical and advanced computational matrix calculating algorithms to historical sequential asset prices explicitly enables quantitative researchers to discover heavily latent non linear correlation patterns. Standard basic linear techniques lack the internal theoretical mapping memory required to fully process continuous progression data natively. Therefore, explicit sequential data pattern prediction necessitates deeply specialized memory architectures uniquely capable of successfully retaining vast contextual numerical memory safely across thousands of chronologically independent market observations simultaneously.

</div>

[:material-view-grid-plus-outline: Browse all modules](modules.md){ .md-button } [:material-school-outline: Learning paths](learning-paths.md){ .md-button }
