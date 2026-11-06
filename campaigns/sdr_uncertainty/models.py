"""Frozen residual models. Mean and noise parameters are fit on train rows only."""
from __future__ import annotations

import numpy as np

from .protocol import (
    MODEL_NAMES,
    NEWTON_INNER,
    NEWTON_OUTER,
    RIDGE,
    RHO_MAX,
    VAR_FLOOR,
    VAR_LOG_CLIP,
    VAR_PRIOR,
)


def parameter_count(name: str, n_dim: int, n_state: int, n_train_days: int = 0) -> int:
    if name not in MODEL_NAMES:
        raise ValueError(name)
    shared_mean = n_dim * (1 + n_state)
    if name == "M0_fixed_gaussian":
        return shared_mean + n_dim
    if name == "M1_temporal_corr":
        return shared_mean + n_dim + 1
    if name in ("M2_state_variance", "C_matched_random_state"):
        return shared_mean + n_dim + n_dim + n_state
    if name == "C_rich_mean":
        return n_dim * (1 + 2 * n_state) + n_dim
    if name == "C_day_intercept":
        return shared_mean + n_dim + n_dim * int(n_train_days)
    raise ValueError(name)


def design_mean(state: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((state.shape[0], 1)), state], axis=1)


def design_rich(state: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((state.shape[0], 1)), state, state ** 2], axis=1)


def ridge_multi(x: np.ndarray, y: np.ndarray, lam: float = RIDGE) -> np.ndarray:
    a = x.T @ x
    scale = float(x.shape[0])
    pen = lam * scale
    a.flat[:: a.shape[0] + 1] += pen
    a[0, 0] -= pen
    a[0, 0] += 1e-8
    return np.linalg.solve(a, x.T @ y)


def _constant_logvar(resid: np.ndarray) -> np.ndarray:
    return np.clip(np.log(np.mean(resid ** 2, axis=0) + 1e-8), *VAR_LOG_CLIP)


def _var_from_log(log_var: np.ndarray, n: int) -> np.ndarray:
    return np.exp(np.clip(log_var, *VAR_LOG_CLIP))[None, :] * np.ones((n, 1))


def fit_rank1_variance(resid: np.ndarray, state: np.ndarray, lam: float = VAR_PRIOR):
    """log sigma_d^2 = alpha_d + u_d * (v · s), with ||v|| = 1."""
    n, n_dim = resid.shape
    n_state = state.shape[1]
    alpha = _constant_logvar(resid)
    u = np.zeros(n_dim)
    centered = state - state.mean(axis=0, keepdims=True)
    q = np.log(resid ** 2 + 1e-6)
    q = q - q.mean(axis=0, keepdims=True)
    gram = centered.T @ q @ q.T @ centered
    gram.flat[:: n_state + 1] += 1e-8
    evals, evecs = np.linalg.eigh(gram)
    v = evecs[:, -1]
    v = v / (np.linalg.norm(v) + 1e-12)
    z = centered @ v
    den = float(np.dot(z, z) + lam)
    for d in range(n_dim):
        u[d] = float(np.dot(z, q[:, d]) / den)
    u = np.clip(u, -3.0, 3.0)

    for _ in range(NEWTON_OUTER):
        z = state @ v
        for _inner in range(NEWTON_INNER):
            for d in range(n_dim):
                eta = np.clip(alpha[d] + u[d] * z, *VAR_LOG_CLIP)
                w = resid[:, d] ** 2 * np.exp(-eta)
                g = np.array([
                    np.sum(1.0 - w),
                    np.sum(z * (1.0 - w)) + lam * u[d],
                ])
                h = np.array([
                    [np.sum(w) + 1e-3, np.sum(z * w)],
                    [np.sum(z * w), np.sum((z ** 2) * w) + lam + 1e-3],
                ])
                step = np.clip(np.linalg.solve(h, g), -0.5, 0.5)
                alpha[d] -= step[0]
                u[d] -= step[1]
            u = np.clip(u, -3.0, 3.0)
            alpha = np.clip(alpha, *VAR_LOG_CLIP)
        z = state @ v
        eta = np.clip(alpha[None, :] + z[:, None] * u[None, :], *VAR_LOG_CLIP)
        w = resid ** 2 * np.exp(-eta)
        g_z = (1.0 - w) @ u
        h_z = (w * (u ** 2)[None, :]).sum(axis=1)
        g_v = state.T @ g_z
        h_v = state.T @ (state * h_z[:, None])
        h_v.flat[:: n_state + 1] += 1e-3
        step = np.clip(np.linalg.solve(h_v, g_v), -0.5, 0.5)
        v = v - step
        scale = float(np.linalg.norm(v))
        if scale < 1e-8:
            v = np.zeros(n_state)
            v[0] = 1.0
            break
        u = np.clip(u * scale, -3.0, 3.0)
        v = v / scale
    return alpha, u, v


def _predict_rank1(alpha, u, v, state: np.ndarray) -> np.ndarray:
    z = state @ v
    log_var = np.clip(alpha[None, :] + z[:, None] * u[None, :], *VAR_LOG_CLIP)
    return np.exp(log_var)


def _rho_and_innov(resid: np.ndarray, day: np.ndarray):
    num = 0.0
    den = 0.0
    pieces = []
    for d in np.unique(day):
        e = resid[day == d]
        if len(e) < 3:
            continue
        num += float(np.sum(e[1:] * e[:-1]))
        den += float(np.sum(e[:-1] ** 2))
        pieces.append(e)
    rho = 0.0 if den <= 0 else float(np.clip(num / den, -RHO_MAX, RHO_MAX))
    etas = []
    for e in pieces:
        etas.append(e[1:] - rho * e[:-1])
    if not etas:
        innov = np.mean(resid ** 2, axis=0) + 1e-8
    else:
        innov = np.mean(np.concatenate(etas, axis=0) ** 2, axis=0) + 1e-8
    innov = np.clip(innov, np.exp(VAR_LOG_CLIP[0]), np.exp(VAR_LOG_CLIP[1]))
    return rho, innov


def _day_breaks(day: np.ndarray) -> np.ndarray:
    breaks = np.empty(len(day), dtype=bool)
    breaks[0] = True
    if len(day) > 1:
        breaks[1:] = day[1:] != day[:-1]
    return breaks


def fit_model(name: str, y: np.ndarray, state: np.ndarray, day: np.ndarray, null_state: np.ndarray):
    if name not in MODEL_NAMES:
        raise ValueError(name)
    y = np.asarray(y, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    day = np.asarray(day)
    if name == "C_rich_mean":
        beta = ridge_multi(design_rich(state), y)
        resid = y - design_rich(state) @ beta
        return {"name": name, "beta": beta, "log_var": _constant_logvar(resid)}
    beta = ridge_multi(design_mean(state), y)
    resid = y - design_mean(state) @ beta
    if name == "M0_fixed_gaussian":
        return {"name": name, "beta": beta, "log_var": _constant_logvar(resid)}
    if name == "M1_temporal_corr":
        rho, innov = _rho_and_innov(resid, day)
        return {"name": name, "beta": beta, "rho": rho, "innov_var": innov}
    if name == "C_day_intercept":
        deltas = {}
        for _ in range(6):
            adjusted = y.copy()
            for d, delta in deltas.items():
                adjusted[day == d] -= delta
            beta = ridge_multi(design_mean(state), adjusted)
            resid = y - design_mean(state) @ beta
            deltas = {int(d): resid[day == d].mean(axis=0) for d in np.unique(day)}
        centered = resid.copy()
        for d, delta in deltas.items():
            centered[day == d] -= delta
        return {
            "name": name,
            "beta": beta,
            "log_var": _constant_logvar(centered),
            "n_train_days": int(len(deltas)),
        }
    variance_state = null_state if name == "C_matched_random_state" else state
    alpha, u, v = fit_rank1_variance(resid, np.asarray(variance_state, dtype=np.float64))
    return {"name": name, "beta": beta, "alpha": alpha, "u": u, "v": v}


def predict_model(params: dict, y: np.ndarray, state: np.ndarray, day: np.ndarray, null_state: np.ndarray):
    """Conditional mean and variance. Test-day labels and test-day means are not parameters.

    M1's one-step mean uses the previous residual inside the same day. That is the
    temporal model, not a test-day intercept. The first row of a day has no lag.
    """
    name = params["name"]
    y = np.asarray(y, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    day = np.asarray(day)
    n = len(y)
    if name == "C_rich_mean":
        mu = design_rich(state) @ params["beta"]
        var = _var_from_log(params["log_var"], n)
        return mu, np.maximum(var, VAR_FLOOR)
    mu = design_mean(state) @ params["beta"]
    if name == "M1_temporal_corr":
        rho = float(params["rho"])
        innov = np.asarray(params["innov_var"], dtype=np.float64)
        marginal = innov / max(1.0 - rho ** 2, 1e-3)
        out_mu = np.empty_like(y)
        out_var = np.empty_like(y)
        prev = None
        prev_day = None
        for t in range(n):
            if prev_day is None or day[t] != prev_day:
                out_mu[t] = mu[t]
                out_var[t] = marginal
            else:
                out_mu[t] = mu[t] + rho * prev
                out_var[t] = innov
            prev = y[t] - mu[t]
            prev_day = day[t]
        return out_mu, np.maximum(out_var, VAR_FLOOR)
    if name in ("M2_state_variance", "C_matched_random_state"):
        variance_state = null_state if name == "C_matched_random_state" else state
        var = _predict_rank1(params["alpha"], params["u"], params["v"], np.asarray(variance_state, dtype=np.float64))
        return mu, np.maximum(var, VAR_FLOOR)
    var = _var_from_log(params["log_var"], n)
    return mu, np.maximum(var, VAR_FLOOR)


def breaks_from_day(day: np.ndarray) -> np.ndarray:
    return _day_breaks(np.asarray(day))
