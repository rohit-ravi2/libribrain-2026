"""Synthetic fixtures with a known residual law. These are not MEG recordings."""
from __future__ import annotations

import numpy as np

from .protocol import FEATURE_SET, NULL_STATE_SEED


def make_null_state(n: int, k: int, seed: int = NULL_STATE_SEED) -> np.ndarray:
    """Frozen covariate drawn without looking at y. Same shape as the real state."""
    return np.random.default_rng(seed).normal(size=(n, k))


def _smooth_state(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    state = np.zeros((n, k))
    state[0] = rng.normal(size=k)
    for t in range(1, n):
        state[t] = 0.85 * state[t - 1] + 0.45 * rng.normal(size=k)
    return state


def make_dataset(kind: str, seed: int = 20261106, n_days: int = 5, n_per: int = 240,
                 n_dim: int = 4, n_state: int = 2) -> dict:
    if kind not in ("hetero", "ar1", "homo", "day_shift"):
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    weight = np.array([
        [0.45, 0.25],
        [-0.20, 0.50],
        [0.15, -0.35],
        [0.30, 0.10],
    ], dtype=np.float64)
    if n_state != 2 or n_dim != 4:
        raise ValueError("the frozen synthetic fixture is D=4, K=2")
    intercept = np.array([0.2, -0.1, 0.05, 0.0])
    u_true = np.array([1.1, -1.0, 0.8, -0.7])
    v_true = np.array([0.85, -0.55])
    v_true = v_true / np.linalg.norm(v_true)
    chunks = []
    for day in range(n_days):
        state = _smooth_state(rng, n_per, n_state)
        mu = intercept + state @ weight.T
        if kind == "hetero":
            log_var = np.log(0.20) + (state @ v_true)[:, None] * u_true[None, :]
            noise = rng.normal(size=(n_per, n_dim)) * np.exp(0.5 * log_var)
        elif kind == "ar1":
            rho, sig = 0.65, 0.40
            noise = np.zeros((n_per, n_dim))
            noise[0] = rng.normal(scale=sig / np.sqrt(1.0 - rho ** 2), size=n_dim)
            for t in range(1, n_per):
                noise[t] = rho * noise[t - 1] + rng.normal(scale=sig, size=n_dim)
        elif kind == "homo":
            noise = rng.normal(scale=0.45, size=(n_per, n_dim))
        else:
            mu = np.zeros_like(mu)
            noise = rng.normal(scale=0.25, size=(n_per, n_dim)) + (day + 1) * 8.0
        y = mu + noise
        signal = state @ weight.T
        logits = np.stack([signal[:, 0], -signal[:, 0], signal[:, 1], -signal[:, 1]], axis=1)
        label = logits.argmax(axis=1).astype(np.int64)
        chunks.append((y, state, np.full(n_per, day, dtype=np.int64),
                       np.arange(n_per, dtype=np.int64), label))
    y, state, day, time_index, label = (np.concatenate(list(part), axis=0) for part in zip(*chunks))
    return {
        "y": y,
        "state": state,
        "state_names": ("s0", "s1"),
        "day": day,
        "time_index": time_index,
        "label": label,
        "feature_set": FEATURE_SET,
        "kind": kind,
        "evidence_class": "synthetic_dgp",
    }
