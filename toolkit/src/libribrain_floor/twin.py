"""Timing twin: synthetic MEG that carries word timing and nothing else.

A twin is noise plus one fixed onset-locked kernel placed at every word onset (vocabulary or not). A decoder
scored on a twin can only use timing, so its score on a twin estimates how much of its real score timing
could explain.

Caveat. With white, spatially independent noise and a per-channel amplitude matched to the real onset
response, the twin is an optimistic (easy) one: real MEG background is coloured, spatially correlated,
non-stationary and has large artefacts, all of which lower how well a single-channel-amplitude kernel can
be read out. A realistic twin must add the kernel to real background recordings (for example the same
recording with its onset response removed by subtract_trf, or a recording from a segment with no words) via
the `noise` argument.
"""
import numpy as np


def onset_kernel(fs, n_channels=306, amp=1.0, seed=777, duration=0.6):
    """Fixed kernel (n_channels, L): a random spatial pattern (max-normalised, scaled by amp) times a
    biphasic time course: a narrow positive peak at 0.1 s and a broader negative one at 0.4 s."""
    g = np.random.default_rng(seed)
    u = g.standard_normal(n_channels); u /= np.abs(u).max()
    t = np.arange(int(duration * fs)) / fs
    w = np.exp(-((t - 0.1) / 0.03) ** 2) - 0.8 * np.exp(-((t - 0.4) / 0.08) ** 2)
    return (amp * u[:, None] * w[None, :]).astype(np.float32)


def twin_signal(onsets, T, kernel, noise=1.0, rng=None, fs=None):
    """Synthetic recording (n_channels, T).

    onsets : onset times in samples if fs is None, else in seconds (converted with fs).
    noise  : float std of white Gaussian noise, or an array (n_channels, T) of real background to add to.
    """
    rng = np.random.default_rng(rng)
    on = np.asarray(onsets, float)
    on = np.round(on * fs).astype(int) if fs is not None else on.astype(int)
    C, L = kernel.shape
    if np.ndim(noise) == 0:
        out = rng.standard_normal((C, T)).astype(np.float32) * np.float32(noise)
    else:
        out = np.array(noise, np.float32)
        if out.shape != (C, T):
            raise ValueError("noise array must have shape (n_channels, T)")
    for o in on[(on > -L) & (on < T)]:
        k0, k1 = max(-o, 0), min(T - o, L)
        out[:, o + k0:o + k1] += kernel[:, k0:k1]
    return out
