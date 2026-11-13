"""Evaluation metrics."""
import numpy as np


def balanced_topk(P, y, k=1):
    """Class-balanced top-k accuracy.

    P : (n, n_classes) scores or probabilities; column j scores class j.
    y : (n,) integer labels.
    Returns the mean over classes present in y of the per-class top-k hit rate, so that
    frequent words do not dominate. Chance is k / n_classes under uniform scores.
    """
    P = np.asarray(P); y = np.asarray(y)
    k = min(int(k), P.shape[1])
    top = np.argpartition(-P, k - 1, axis=1)[:, :k]
    hit = (top == y[:, None]).any(1)
    return float(np.mean([hit[y == c].mean() for c in np.unique(y)]))
