"""Class balancing of a probability matrix."""
import numpy as np


def sinkhorn(P, n_iter=20, T=1.0):
    """Sinkhorn class balancing.

    P : (n, n_classes) row-stochastic probabilities (or non-negative scores).
    T : temperature; P is first sharpened/flattened as P**(1/T) and renormalised per row
        (equivalent to softmax(logits / T) when P = softmax(logits)).
    Alternately rescales columns so each class carries equal total mass n / n_classes, and
    rows so each sums to one. Returns the balanced (n, n_classes) probability matrix.
    Intended for a whole test session at once: it assumes the true class marginal is uniform.
    """
    P = np.asarray(P, float)
    P = np.maximum(P, 1e-12) ** (1.0 / T)
    P = P / P.sum(1, keepdims=True)
    n, c = P.shape
    tgt = n / c
    for _ in range(n_iter):
        P = P / np.maximum(P.sum(0, keepdims=True), 1e-12) * tgt
        P = P / P.sum(1, keepdims=True)
    return P
