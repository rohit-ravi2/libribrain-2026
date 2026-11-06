"""Budgeted 306-channel linear probe. Not the competition encoder.

Preprocessing matches the 1-second word window: sensor scale, then the
per-window mean removal, magnetometer/gradiometer median-std, and ±10 clip
used by src/data.py. Torch's std divides by N-1, and its median on an even
count is the lower of the two central values. Time is then mean-pooled by 5,
which is the cached 50 Hz window. The probe keeps that (306, 50) vector.
A full 15300-dimensional covariance is not formed. Each channel has its own
50×50 covariance, shrunk toward its diagonal on train rows only.
"""
from __future__ import annotations

import time

import numpy as np

from .metrics import day_group_uncertainty
from .protocol import (
    DECODE306_CHUNK_CHANNELS,
    DECODE306_COV_SHRINK,
    DECODE306_MAX_DAYS,
    DECODE306_MAX_READ_BYTES,
    DECODE306_MIN_PER_CLASS,
    DECODE306_N_CLASSES,
    DECODE306_PLANNED_ITEMSIZE,
    DECODE306_POOL,
    DECODE306_TIME,
    DECODE306_TOP_K,
    DECODE306_WEIGHT_CLIP,
    DECODE306_WEIGHT_SHRINK,
    FS_HZ,
    N_SENSORS,
    NULL_STATE_SEED,
    OOD_SELECTION_RULE,
    OOD_SOFT_DEADLINE_SEC,
    SCALE_GRAD,
    SCALE_MAG,
    WINDOW_SAMPLES,
)

STAGED_EVALUATION = (
    "Stage A is this command: at most 512 MiB of hyperslabs, four earliest "
    "subject-0 dates, one run each, 2024-10-04 excluded, in-vocabulary words in "
    "onset order. It does not establish a 50-class balanced accuracy unless every "
    "held-out day and its training complement each have at least 3 observations of "
    "all 50 words. Stage B is not launched. A person may later read the next "
    "chronological run on those same dates, still inside 4 GiB, no swap, one CPU, "
    "600 seconds, and a fresh 512 MiB cap, until coverage holds or the date's runs "
    "are exhausted. Partial-class accuracy is not the competition metric."
)
AR_NOT_RUN = (
    "Causal AR is not run. Word onsets are irregular, so an AR on window index "
    "would treat unequal gaps as one lag."
)


def planned_budget(itemsize: int = DECODE306_PLANNED_ITEMSIZE) -> dict:
    if itemsize not in (4, 8):
        raise RuntimeError(f"refusing dtype itemsize {itemsize}; expected 4 or 8")
    per_window = N_SENSORS * WINDOW_SAMPLES * int(itemsize)
    if per_window <= 0 or per_window > DECODE306_MAX_READ_BYTES:
        raise RuntimeError("one window does not fit in the read budget")
    total = DECODE306_MAX_READ_BYTES // per_window
    per_day = total // DECODE306_MAX_DAYS
    if per_day < 1:
        raise RuntimeError("read budget holds no complete window")
    return {
        "max_read_bytes": DECODE306_MAX_READ_BYTES,
        "bytes_per_window": per_window,
        "itemsize": int(itemsize),
        "max_windows_total": int(total),
        "max_windows_per_day": int(per_day),
        "max_days": DECODE306_MAX_DAYS,
        "channels": N_SENSORS,
        "window_samples": WINDOW_SAMPLES,
        "chunk_channels": DECODE306_CHUNK_CHANNELS,
        "pool": DECODE306_POOL,
        "time_bins": DECODE306_TIME,
        "soft_deadline_sec": OOD_SOFT_DEADLINE_SEC,
        "selection_rule": OOD_SELECTION_RULE,
    }


def _lower_median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[(len(ordered) - 1) // 2])


def normalize_raw_window(raw_tesla: np.ndarray) -> np.ndarray:
    """(306, 250) tesla -> (306, 50) normalized, mean-pooled by 5."""
    raw = np.asarray(raw_tesla, dtype=np.float64)
    if raw.shape != (N_SENSORS, WINDOW_SAMPLES):
        raise RuntimeError(f"expected {(N_SENSORS, WINDOW_SAMPLES)}, got {raw.shape}")
    mag = (np.arange(N_SENSORS) % 3) == 0
    scale = np.where(mag, SCALE_MAG, SCALE_GRAD)
    x = np.nan_to_num(raw * scale[:, None], nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1)
    s_mag = max(_lower_median(sd[mag]), 1e-6)
    s_grad = max(_lower_median(sd[~mag]), 1e-6)
    div = np.where(mag, s_mag, s_grad)
    x = np.clip(x / div[:, None], -10.0, 10.0)
    return x.reshape(N_SENSORS, DECODE306_TIME, DECODE306_POOL).mean(axis=2)


def assert_time_chunk(start: int, stop: int, n_times: int) -> None:
    if stop <= start:
        raise RuntimeError("empty time chunk")
    if start == 0 and stop == n_times:
        raise RuntimeError("refusing a full-duration MEG read")
    if stop - start > WINDOW_SAMPLES:
        raise RuntimeError("time chunk longer than one word window")


def read_window(dataset, start: int, stop: int, used: int, max_bytes: int):
    n_channels, n_times = dataset.shape
    if n_channels != N_SENSORS:
        raise RuntimeError(f"expected {N_SENSORS} channels, found {n_channels}")
    assert_time_chunk(start, stop, n_times)
    itemsize = int(np.dtype(dataset.dtype).itemsize)
    need = n_channels * (stop - start) * itemsize
    if used + need > max_bytes:
        return None, used
    block = np.empty((n_channels, stop - start), dtype=np.float64)
    step = DECODE306_CHUNK_CHANNELS
    for c0 in range(0, n_channels, step):
        c1 = min(n_channels, c0 + step)
        block[c0:c1] = np.asarray(dataset[c0:c1, start:stop], dtype=np.float64)
    return block, used + need


def collect_day(dataset, rows: list[dict], vocab: tuple[str, ...], *, per_day: int,
                used: int, max_bytes: int, deadline: float | None = None) -> dict:
    from .stream import vocab_index
    indexed = []
    for row in rows:
        label = vocab_index(row["segment"], vocab)
        if label < 0:
            continue
        indexed.append((float(row["timemeg"]), label))
    indexed.sort(key=lambda item: item[0])
    features, labels, onsets = [], [], []
    bytes_before = used
    n_times = int(dataset.shape[1])
    for onset_s, label in indexed:
        if deadline is not None and time.monotonic() > deadline:
            break
        if len(features) >= per_day:
            break
        start = int(round(onset_s * FS_HZ))
        stop = start + WINDOW_SAMPLES
        if start < 0 or stop > n_times:
            continue
        block, used = read_window(dataset, start, stop, used, max_bytes)
        if block is None:
            break
        features.append(normalize_raw_window(block))
        labels.append(label)
        onsets.append(onset_s)
        del block
    if not features:
        return {
            "features": np.zeros((0, N_SENSORS, DECODE306_TIME)),
            "label": np.zeros(0, dtype=np.int64),
            "onset_s": np.zeros(0),
            "bytes_read": used - bytes_before,
            "n_windows": 0,
            "bytes_cursor": used,
        }
    return {
        "features": np.stack(features, axis=0),
        "label": np.asarray(labels, dtype=np.int64),
        "onset_s": np.asarray(onsets, dtype=np.float64),
        "bytes_read": used - bytes_before,
        "n_windows": len(features),
        "bytes_cursor": used,
    }


def _counts(labels: np.ndarray) -> np.ndarray:
    keep = labels[(labels >= 0) & (labels < DECODE306_N_CLASSES)]
    return np.bincount(keep, minlength=DECODE306_N_CLASSES)


def _covered(counts: np.ndarray) -> bool:
    return bool(counts.shape == (DECODE306_N_CLASSES,) and np.all(counts >= DECODE306_MIN_PER_CLASS))


def _center(x: np.ndarray, weights: np.ndarray | None = None):
    if weights is None:
        mu = x.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        mu = w @ x
    return x - mu, mu


def _ridge_dual(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Centered ridge. Uses the n×n Gram when there are fewer rows than columns."""
    n, p = x.shape
    pen = lam * n
    if n <= p:
        gram = x @ x.T
        gram.flat[:: n + 1] += pen
        return x.T @ np.linalg.solve(gram, y)
    gram = x.T @ x
    gram.flat[:: p + 1] += pen
    return np.linalg.solve(gram, x.T @ y)


def _shrunk_invsqrt(block: np.ndarray) -> np.ndarray:
    cov = (block.T @ block) / max(len(block), 1)
    alpha = DECODE306_COV_SHRINK
    shrunk = (1.0 - alpha) * cov
    shrunk.flat[:: shrunk.shape[0] + 1] += alpha * np.diag(cov)
    shrunk.flat[:: shrunk.shape[0] + 1] += 1e-6
    evals, evecs = np.linalg.eigh(shrunk)
    evals = np.clip(evals, 1e-6, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


def _apply_whiten(features: np.ndarray, transforms: list[np.ndarray], mean: np.ndarray) -> np.ndarray:
    out = np.empty_like(features)
    centered = features - mean
    for channel, transform in enumerate(transforms):
        out[:, channel, :] = centered[:, channel, :] @ transform
    return out


def _fit_whiten(train: np.ndarray):
    mean = train.mean(axis=0)
    centered = train - mean
    transforms = [_shrunk_invsqrt(centered[:, channel, :]) for channel in range(train.shape[1])]
    return mean, transforms


def _sample_weights(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    energy = np.mean(train ** 2, axis=(1, 2))
    log_e = np.log(energy + 1e-8)
    design = np.column_stack([np.ones(len(train)), energy])
    # Constant-plus-energy slope fit on train only. Query energy is known at scoring time
    # but the slope is not refit there. Weights below are for training rows only.
    beta = _ridge_dual(design - design.mean(axis=0), (log_e - log_e.mean())[:, None]).ravel()
    pred = (np.column_stack([np.ones(len(query)), np.mean(query ** 2, axis=(1, 2))]) - design.mean(axis=0)) @ beta
    pred = pred + log_e.mean()
    precision = np.exp(-pred)
    raw = precision / np.median(precision)
    mixed = DECODE306_WEIGHT_SHRINK + (1.0 - DECODE306_WEIGHT_SHRINK) * raw
    low, high = DECODE306_WEIGHT_CLIP
    return np.clip(mixed, low, high)


def _bacc10(scores: np.ndarray, labels: np.ndarray):
    recalls = []
    for cls in range(DECODE306_N_CLASSES):
        mask = labels == cls
        if not np.any(mask):
            return None
        chosen = np.argpartition(scores[mask], -DECODE306_TOP_K, axis=1)[:, -DECODE306_TOP_K:]
        recalls.append(float(np.mean(np.any(chosen == cls, axis=1))))
    return float(np.mean(recalls))


def _one_hot(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((len(labels), DECODE306_N_CLASSES), dtype=np.float64)
    keep = (labels >= 0) & (labels < DECODE306_N_CLASSES)
    out[np.arange(len(labels))[keep], labels[keep]] = 1.0
    return out


def _rank1_remove(train: np.ndarray, full: np.ndarray) -> np.ndarray:
    flat_train = train.reshape(len(train), -1)
    flat_full = full.reshape(len(full), -1)
    centered = flat_train - flat_train.mean(axis=0)
    vector = np.ones(centered.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(6):
        coeff = centered @ vector
        norm = np.linalg.norm(coeff) + 1e-12
        coeff /= norm
        vector = centered.T @ coeff
        vector /= np.linalg.norm(vector) + 1e-12
    removed = flat_full - (flat_full @ vector)[:, None] * vector[None, :]
    return removed.reshape(full.shape)


def _head(train_x, train_y, test_x, weights) -> np.ndarray:
    centered, mu = _center(train_x, weights)
    y_mu = (weights / np.sum(weights)) @ train_y
    scale = np.sqrt(np.clip(weights, 1e-8, None))
    beta = _ridge_dual(centered * scale[:, None], (train_y - y_mu) * scale[:, None])
    return (test_x - mu) @ beta + y_mu


def _arm_scores(features, labels, train, test, arm: str, seed: int) -> float | None:
    y_train = _one_hot(labels[train])
    flat = features.reshape(len(features), -1)
    if arm == "uniform":
        scores = _head(flat[train], y_train, flat[test], np.ones(int(train.sum())))
    elif arm == "shrunk_whiten":
        mean, transforms = _fit_whiten(features[train])
        whitened = _apply_whiten(features, transforms, mean).reshape(len(features), -1)
        scores = _head(whitened[train], y_train, whitened[test], np.ones(int(train.sum())))
    elif arm == "shuffled_weight":
        weights = _sample_weights(features[train], features[train])
        weights = np.random.default_rng(seed).permutation(weights)
        scores = _head(flat[train], y_train, flat[test], weights)
    elif arm == "stronger_mean":
        removed = _rank1_remove(features[train], features).reshape(len(features), -1)
        scores = _head(removed[train], y_train, removed[test], np.ones(int(train.sum())))
    else:
        raise ValueError(arm)
    return _bacc10(scores, labels[test])


def evaluate_decode306(features: np.ndarray, labels: np.ndarray, day: np.ndarray, dates: list[str]) -> dict:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    day = np.asarray(day)
    days = np.unique(day)
    coverage = []
    defined = []
    for held in days:
        test = _counts(labels[day == held])
        train = _counts(labels[day != held])
        ok = _covered(test) and _covered(train)
        defined.append(ok)
        coverage.append({
            "day": int(held),
            "test_per_class": [int(v) for v in test],
            "train_min": int(train.min()) if train.size else 0,
            "test_min": int(test.min()) if test.size else 0,
            "test_classes_present": int(np.sum(test > 0)),
            "fifty_class_metric_defined": ok,
        })
    base = {
        "real_meg_scientific_validation": False,
        "effect_claimed": False,
        "competition_encoder": False,
        "evidence_scope": "linear_306ch_50hz_probe",
        "tenfold_labeled_minutes_claimed": False,
        "label_fractions_are_minutes": False,
        "causal_ar": "not_run",
        "causal_ar_reason": AR_NOT_RUN,
        "dates": list(dates),
        "coverage": coverage,
        "n_days": int(len(days)),
        "selection_rule": OOD_SELECTION_RULE,
        "normalization": "scale_then_torch_matching_data_py_normalize_then_pool5",
    }
    if not days.size or not all(defined):
        base.update({
            "claim_status": "engineering_cap_metric_undefined",
            "fifty_class_metric_defined": False,
            "frozen_eval": None,
            "reason": (
                "BAcc@10 over all 50 words is undefined unless every held-out day and its "
                f"training days each have at least {DECODE306_MIN_PER_CLASS} observations of every class. "
                "No partial-class accuracy is reported as the result."
            ),
            "staged_evaluation": STAGED_EVALUATION,
            "engineering_cap_is_not_scientific_evidence": True,
        })
        return base
    arms = ("uniform", "shrunk_whiten", "shuffled_weight", "stronger_mean")
    per_day = {arm: [] for arm in arms}
    for held in days:
        test = day == held
        train = ~test
        for arm in arms:
            score = _arm_scores(features, labels, train, test, arm, NULL_STATE_SEED + int(held))
            if score is None:
                base.update({
                    "claim_status": "engineering_cap_metric_undefined",
                    "fifty_class_metric_defined": False,
                    "frozen_eval": None,
                    "reason": "The 50-class top-10 recall became undefined while scoring.",
                    "staged_evaluation": STAGED_EVALUATION,
                    "engineering_cap_is_not_scientific_evidence": True,
                })
                return base
            per_day[arm].append(score)
    contrast = [per_day["shrunk_whiten"][i] - per_day["uniform"][i] for i in range(len(days))]
    summary = day_group_uncertainty(contrast)
    base.update({
        "claim_status": "decode306_probe_estimated",
        "fifty_class_metric_defined": True,
        "frozen_eval": {
            "metric": "bacc_at_10_over_all_50_classes",
            "arms": {arm: day_group_uncertainty(per_day[arm]) for arm in arms},
            "shrunk_whiten_minus_uniform": summary,
            "effect_claimed": False,
            "distinguishable_from_zero": bool(
                summary["se"] is not None and summary["se"] > 0 and abs(summary["mean"]) > 2.0 * summary["se"]
            ),
        },
        "reason": (
            "Paired day differences for a linear probe on normalized 306-by-50 windows. "
            "This is not the competition encoder, and a single contrast is not a claim that whitening helps."
        ),
        "staged_evaluation": None,
    })
    return base
