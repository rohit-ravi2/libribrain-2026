"""Onset deconvolution: fit a generic word-onset response (temporal response function) and subtract it."""
import numpy as np
import scipy.sparse as sp


def _lags(fs, lags):
    lo, hi = lags
    return np.arange(int(round(lo * fs)), int(round(hi * fs)))


def _design(T, onsets, fs, lags):
    """Sparse (T, L) design: column l is the onset impulse train shifted by lag l samples."""
    lg = _lags(fs, lags); on = np.asarray(onsets, int)
    r = (on[:, None] + lg[None, :]).ravel(); c = np.tile(np.arange(len(lg)), len(on))
    ok = (r >= 0) & (r < T)
    return sp.csr_matrix((np.ones(ok.sum(), np.float32), (r[ok], c[ok])), shape=(T, len(lg)))


def _clean(Y):
    return np.nan_to_num(np.asarray(Y, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_trf(segments, fs, lags=(-0.1, 0.8), ridge=1e-2, clip=20.0):
    """Ridge fit of one kernel shared by every word onset.

    segments : list of (Y, onsets) with Y (n_channels, T) and onsets in samples.
    lags     : (start, stop) in seconds relative to the onset; stop is exclusive.
    ridge    : penalty as a fraction of the mean diagonal of X'X.
    clip     : samples beyond clip * median|Y| are clipped so artefacts do not steer the fit.
    Returns the kernel B with shape (L, n_channels); lag index i corresponds to sample
    round(lags[0] * fs) + i after the onset.
    """
    XtX = XtY = 0
    for Y, on in segments:
        Yf = _clean(Y); X = _design(Yf.shape[1], on, fs, lags)
        c = clip * np.median(np.abs(Yf[:, ::10])) + 1e-6
        Yf = np.clip(Yf, -c, c)
        XtX = XtX + (X.T @ X).toarray(); XtY = XtY + X.T @ Yf.T
    L = XtX.shape[0]; lam = ridge * np.trace(XtX) / L
    return np.linalg.solve(XtX + lam * np.eye(L), XtY).astype(np.float32)


def subtract_trf(Y, onsets, B, fs, lags=(-0.1, 0.8)):
    """Return Y minus the kernel's prediction at the given onsets (float32, same shape as Y)."""
    Yf = _clean(Y)
    X = _design(Yf.shape[1], onsets, fs, lags)
    return (Yf - (X @ B).T).astype(np.float32)
