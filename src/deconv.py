"""Onset deconvolution (flagship method, 2026-11-03): regress a generic word-onset response out of the MEG.

A linear temporal response function (one 306-channel kernel shared by every word, lags LAG0..LAG1 s) is fitted by ridge
regression to the MEG of one recording (or one subject's holdout epochs) on the impulse train of ALL word onsets, and
its prediction is subtracted. What is left cannot carry word timing through a response common to every word, which is
the channel the synthetic-timing twin exploits; word-specific responses survive. Uses onsets only, never labels."""
import numpy as np
import scipy.sparse as sp

LAG0, LAG1, RIDGE = -0.1, 0.8, 1e-2


def _design(T, onsets, fs):
    """Sparse (T, L) design: column l is the onset impulse train shifted by lag l."""
    lags = np.arange(int(round(LAG0 * fs)), int(round(LAG1 * fs)))
    r = (onsets[:, None] + lags[None, :]).ravel(); c = np.tile(np.arange(len(lags)), len(onsets))
    ok = (r >= 0) & (r < T)
    return sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], c[ok])), shape=(T, len(lags)))


def fit_kernel(segments, fs):
    """segments: list of (Y (C,T) array, onset samples). Returns the kernel B (L, C) fitted jointly over all segments."""
    XtX = 0; XtY = 0
    for Y, on in segments:
        X = _design(Y.shape[1], np.asarray(on, int), fs); Yf = _clean(Y)
        c = 20 * np.median(np.abs(Yf[:, ::10])) + 1e-6; Yf = np.clip(Yf, -c, c)   # artifacts do not steer the kernel
        XtX = XtX + (X.T @ X).toarray(); XtY = XtY + X.T @ Yf.T
    L = XtX.shape[0]; lam = RIDGE * np.trace(XtX) / L
    return np.linalg.solve(XtX + lam * np.eye(L), XtY).astype(np.float32)


def subtract(Y, on, B, fs):
    X = _design(Y.shape[1], np.asarray(on, int), fs)
    return (_clean(Y) - (X @ B).T).astype(np.float16)


def _clean(Y):
    """Non-finite samples -> 0 (the decoder's normalise() also zeroes them)."""
    return np.nan_to_num(np.asarray(Y, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
