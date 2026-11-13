import numpy as np
import pandas as pd
import pytest

from libribrain_floor import (balanced_topk, sinkhorn, timing_features, timing_floor,
                              onset_kernel, twin_signal, fit_trf, subtract_trf)


def test_balanced_topk():
    y = np.array([0, 0, 0, 1, 2])
    P = np.eye(3)[y]
    assert balanced_topk(P, y, 1) == 1.0
    # always predicting class 0: balanced accuracy is 1/3, not 3/5
    Q = np.tile([1.0, 0.0, 0.0], (5, 1))
    assert balanced_topk(Q, y, 1) == pytest.approx(1 / 3)
    assert balanced_topk(Q, y, 3) == 1.0


def test_sinkhorn_equalises_mass():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5) * 0.3, 400); P[:, 0] += 0.5; P /= P.sum(1, keepdims=True)
    B = sinkhorn(P, n_iter=100, T=1.0)
    np.testing.assert_allclose(B.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(B.sum(0), 400 / 5, rtol=1e-3)
    assert P.sum(0).std() > 10 * B.sum(0).std()


def _twin(seed=1, fs=100, n_words=60, T=3000, C=8, noise=0.5):
    rng = np.random.default_rng(seed)
    on = np.sort(rng.choice(np.arange(50, T - 100, 15), n_words, replace=False))
    K = onset_kernel(fs, C, amp=1.0)
    return on, K, twin_signal(on, T, K, noise=noise, rng=rng)


def test_trf_recovers_twin_kernel():
    fs = 100; on, K, Y = _twin(fs=fs)
    B = fit_trf([(Y, on)], fs, lags=(0.0, 0.6), ridge=1e-4)
    assert B.shape == (K.shape[1], K.shape[0])
    r = np.corrcoef(B.ravel(), K.T.ravel())[0, 1]
    assert r > 0.95


def test_subtract_removes_kernel():
    fs = 100; on, K, Y = _twin(fs=fs)
    B = fit_trf([(Y, on)], fs, lags=(-0.1, 0.8), ridge=1e-4)
    R = subtract_trf(Y, on, B, fs, lags=(-0.1, 0.8))
    assert R.var() < Y.var()
    assert R.std() == pytest.approx(0.5, rel=0.1)  # only the white noise is left
    ch = np.argmax(np.abs(K).max(1))
    at = lambda Z: np.mean(Z[ch, on + int(round(0.1 * fs))])
    assert abs(at(R)) < 0.3 * abs(at(Y))


def _events(rng, n, gap_dependent):
    on = np.cumsum(rng.uniform(0.1, 1.5, n))
    gap = np.append(np.diff(on), 1.0)
    y = np.minimum((gap / 1.6 * 5).astype(int), 4) if gap_dependent else rng.integers(0, 5, n)
    return pd.DataFrame({"onset": on, "label": y})


def test_timing_floor_chance_and_signal():
    rng = np.random.default_rng(0)
    tr, te = _events(rng, 3000, False), _events(rng, 1500, False)
    r = timing_floor(tr, te, W=1.0, n_classes=5, topk=(1,))
    assert abs(r["top1"] - 0.2) < 0.05 and r["packard_bits"] < 0.05
    tr, te = _events(rng, 3000, True), _events(rng, 1500, True)
    r = timing_floor(tr, te, W=1.0, n_classes=5, topk=(1,))
    assert r["top1"] > 0.5 and r["packard_bits"] > 0.3


def test_features_and_nonvocab_onsets():
    X = timing_features([0.0, 0.2, 0.5, 5.0], [0.0], W=1.0, K=3)
    np.testing.assert_allclose(X, [[2, 0.2, 0.5, 1.0]])
    ev = (np.array([0.0, 0.2, 0.5]), np.array([3, -1, 4]))
    r = timing_floor([ev] * 5, [ev], W=1.0, n_classes=5, topk=(1,))
    assert r["n_test"] == 2
