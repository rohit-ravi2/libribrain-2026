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

import os
import time

import numpy as np

from .metrics import day_group_uncertainty
from .protocol import (
    DECODE306_CHUNK_CHANNELS,
    DECODE306_COV_SHRINK,
    DECODE306_MAX_DAYS,
    DECODE306_MAX_READ_BYTES,
    DECODE306_MIN_HELDOUT_DAYS,
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


def _weights_from_energies(train_energy: np.ndarray, query_energy: np.ndarray) -> np.ndarray:
    train_energy = np.asarray(train_energy, dtype=np.float64)
    query_energy = np.asarray(query_energy, dtype=np.float64)
    log_e = np.log(train_energy + 1e-8)
    design = np.column_stack([np.ones(len(train_energy)), train_energy])
    design_mean = design.mean(axis=0)
    beta = _ridge_dual(design - design_mean, (log_e - log_e.mean())[:, None]).ravel()
    query_design = np.column_stack([np.ones(len(query_energy)), query_energy])
    pred = (query_design - design_mean) @ beta + log_e.mean()
    precision = np.exp(-pred)
    raw = precision / np.median(precision)
    mixed = DECODE306_WEIGHT_SHRINK + (1.0 - DECODE306_WEIGHT_SHRINK) * raw
    low, high = DECODE306_WEIGHT_CLIP
    return np.clip(mixed, low, high)


def _sample_weights(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Constant-plus-energy slope fit on train only. Query energy is known at scoring time
    # but the slope is not refit there. Weights below are for training rows only.
    train_energy = np.mean(np.asarray(train, dtype=np.float64) ** 2, axis=(1, 2))
    query_energy = np.mean(np.asarray(query, dtype=np.float64) ** 2, axis=(1, 2))
    return _weights_from_energies(train_energy, query_energy)


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


def score_memory_ok(flat_dim: int, headroom: int | None) -> bool:
    """Gram plus solver workspace has to fit in the bytes still free under the cap."""
    gram = flat_dim * flat_dim * 8
    if gram < 32 * 1024 * 1024:
        return True
    if headroom is None:
        return False
    return int(headroom) >= gram * 2


def memory_headroom_bytes() -> int | None:
    """Bytes left under the nearest numeric cgroup memory.max. None if it cannot be read."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    rel = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2]
            break
    if not rel:
        return None
    path = "/sys/fs/cgroup" + rel
    for _ in range(8):
        limit_path = os.path.join(path, "memory.max")
        current_path = os.path.join(path, "memory.current")
        if os.path.exists(limit_path) and os.path.exists(current_path):
            try:
                with open(limit_path, encoding="utf-8") as handle:
                    limit_text = handle.read().strip()
                if limit_text != "max":
                    with open(current_path, encoding="utf-8") as handle:
                        current = int(handle.read().strip())
                    return max(0, int(limit_text) - current)
            except (OSError, ValueError):
                return None
        parent = os.path.dirname(path.rstrip("/"))
        if parent == path:
            return None
        path = parent
    return None


def _chunks(factory):
    for chunk in factory():
        features = np.asarray(chunk["features"], dtype=np.float64)
        labels = np.asarray(chunk["label"], dtype=np.int64)
        day = np.asarray(chunk["day"], dtype=np.int64)
        if len(features) == 0:
            continue
        if features.ndim != 3 or len(labels) != len(features) or len(day) != len(features):
            raise RuntimeError("score chunk shape is inconsistent")
        yield features, labels, day


def _fold_counts(factory, train_days: set[int], test_day: int):
    train = np.zeros(DECODE306_N_CLASSES, dtype=np.int64)
    test = np.zeros(DECODE306_N_CLASSES, dtype=np.int64)
    for _features, labels, day in _chunks(factory):
        train_labels = labels[np.isin(day, list(train_days))]
        test_labels = labels[day == test_day]
        if len(train_labels):
            train += _counts(train_labels)
        if len(test_labels):
            test += _counts(test_labels)
    return train, test


class ScoreMemoryCap(RuntimeError):
    """The 50-class counts are met and the Gram matrix does not fit in the memory cap."""


def _prepared(features, whitened, transforms, channel_mean):
    if not whitened:
        return features
    centered = features - channel_mean
    out = np.empty_like(centered)
    for channel, transform in enumerate(transforms):
        out[:, channel, :] = centered[:, channel, :] @ transform
    return out


def _channel_mean_and_whiten(factory, train_days: set[int]):
    total = None
    count = 0
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        block = features[mask]
        total = block.sum(axis=0) if total is None else total + block.sum(axis=0)
        count += len(block)
    if count == 0 or total is None:
        raise RuntimeError("training fold is empty")
    mean = total / count
    grams = None
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        centered = features[mask] - mean
        if grams is None:
            grams = np.zeros((centered.shape[1], centered.shape[2], centered.shape[2]), dtype=np.float64)
        for channel in range(centered.shape[1]):
            block = centered[:, channel, :]
            grams[channel] += block.T @ block
    transforms = []
    for channel in range(grams.shape[0]):
        cov = grams[channel] / count
        alpha = DECODE306_COV_SHRINK
        shrunk = (1.0 - alpha) * cov
        shrunk.flat[:: shrunk.shape[0] + 1] += alpha * np.diag(cov)
        shrunk.flat[:: shrunk.shape[0] + 1] += 1e-6
        evals, evecs = np.linalg.eigh(shrunk)
        evals = np.clip(evals, 1e-6, None)
        transforms.append((evecs * (1.0 / np.sqrt(evals))) @ evecs.T)
    return mean, transforms


def _train_energies(factory, train_days: set[int]) -> np.ndarray:
    parts = []
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        block = features[mask]
        parts.append(np.mean(block ** 2, axis=(1, 2)))
    if not parts:
        raise RuntimeError("training fold is empty")
    return np.concatenate(parts)


def _rank1_vector(factory, train_days: set[int]) -> np.ndarray:
    total = None
    count = 0
    width = None
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        flat = features[mask].reshape(int(mask.sum()), -1)
        width = flat.shape[1]
        total = flat.sum(axis=0) if total is None else total + flat.sum(axis=0)
        count += len(flat)
    if count == 0 or total is None or width is None:
        raise RuntimeError("training fold is empty")
    mean = total / count
    vector = np.ones(width, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    for _ in range(6):
        coeffs = []
        for features, _labels, day in _chunks(factory):
            mask = np.isin(day, list(train_days))
            if not np.any(mask):
                continue
            centered = features[mask].reshape(int(mask.sum()), -1) - mean
            coeffs.append(centered @ vector)
        coeff = np.concatenate(coeffs)
        coeff /= np.linalg.norm(coeff) + 1e-12
        acc = np.zeros(width, dtype=np.float64)
        offset = 0
        for features, _labels, day in _chunks(factory):
            mask = np.isin(day, list(train_days))
            if not np.any(mask):
                continue
            centered = features[mask].reshape(int(mask.sum()), -1) - mean
            acc += centered.T @ coeff[offset:offset + len(centered)]
            offset += len(centered)
        vector = acc
        vector /= np.linalg.norm(vector) + 1e-12
    return vector


def _ridge_passes(factory, train_days, test_day, *, weights, whitened, transforms, channel_mean, remove):
    """Primal ridge, one chunk at a time. Same penalty as _ridge_dual: lam * n on the diagonal."""
    y_parts = []
    w_parts = []
    cursor = 0
    n_train = 0
    for _features, labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        taken = int(np.sum(mask))
        if taken == 0:
            continue
        y_parts.append(_one_hot(labels[mask]))
        if weights is None:
            w_parts.append(np.ones(taken, dtype=np.float64))
        else:
            w_parts.append(np.asarray(weights[cursor:cursor + taken], dtype=np.float64))
            cursor += taken
        n_train += taken
    if n_train == 0:
        raise RuntimeError("training fold is empty")
    y_all = np.concatenate(y_parts)
    w_all = np.concatenate(w_parts)
    # Moments of the prepared matrix. A second streaming pass fills the Gram.
    sum_x = None
    sum_y = (w_all / np.sum(w_all)) @ y_all
    seen = 0
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        block = _prepared(features[mask], whitened, transforms, channel_mean).reshape(int(mask.sum()), -1)
        if remove is not None:
            block = block - (block @ remove)[:, None] * remove[None, :]
        weight = w_all[seen:seen + len(block)]
        seen += len(block)
        weighted = (weight / np.sum(w_all))[:, None] * block
        sum_x = weighted.sum(axis=0) if sum_x is None else sum_x + weighted.sum(axis=0)
    mu = sum_x
    scale = np.sqrt(np.clip(w_all, 1e-8, None))
    flat_dim = sum_x.shape[0]
    if not score_memory_ok(flat_dim, memory_headroom_bytes()):
        raise ScoreMemoryCap()
    gram = np.zeros((flat_dim, flat_dim), dtype=np.float64)
    rhs = np.zeros((flat_dim, DECODE306_N_CLASSES), dtype=np.float64)
    seen = 0
    for features, _labels, day in _chunks(factory):
        mask = np.isin(day, list(train_days))
        if not np.any(mask):
            continue
        block = _prepared(features[mask], whitened, transforms, channel_mean).reshape(int(mask.sum()), -1)
        if remove is not None:
            block = block - (block @ remove)[:, None] * remove[None, :]
        weight = scale[seen:seen + len(block)]
        y_block = y_all[seen:seen + len(block)]
        seen += len(block)
        centered = (block - mu) * weight[:, None]
        y_centered = (y_block - sum_y) * weight[:, None]
        gram += centered.T @ centered
        rhs += centered.T @ y_centered
    gram.flat[:: flat_dim + 1] += 1e-2 * n_train
    beta = np.linalg.solve(gram, rhs)
    del gram, rhs
    scores = []
    for features, _labels, day in _chunks(factory):
        mask = day == test_day
        if not np.any(mask):
            continue
        block = _prepared(features[mask], whitened, transforms, channel_mean).reshape(int(mask.sum()), -1)
        if remove is not None:
            block = block - (block @ remove)[:, None] * remove[None, :]
        scores.append((block - mu) @ beta + sum_y)
    if not scores:
        return None
    return np.concatenate(scores)


def _arm_scores_streaming(factory, train_days, test_day, arm: str, seed: int):
    whitened = arm == "shrunk_whiten"
    transforms = None
    channel_mean = None
    weights = None
    remove = None
    if whitened:
        channel_mean, transforms = _channel_mean_and_whiten(factory, train_days)
    elif arm == "shuffled_weight":
        energy = _train_energies(factory, train_days)
        weights = _weights_from_energies(energy, energy)
        weights = np.random.default_rng(seed).permutation(weights)
    elif arm == "stronger_mean":
        remove = _rank1_vector(factory, train_days)
    elif arm != "uniform":
        raise ValueError(arm)
    scores = _ridge_passes(
        factory, train_days, test_day,
        weights=weights, whitened=whitened, transforms=transforms,
        channel_mean=channel_mean, remove=remove,
    )
    if scores is None:
        return None
    return _bacc10(scores, _test_labels(factory, test_day))


def _test_labels(factory, test_day: int) -> np.ndarray:
    parts = []
    for _features, labels, day in _chunks(factory):
        mask = day == test_day
        if np.any(mask):
            parts.append(labels[mask])
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def score_consumed_days(factory, dates: list[str], eligible_days: list[int]) -> dict:
    """50-class BAcc@10 on every eligible held-out day. No partial-class average.

    `factory` is called again for every pass and yields chunks with features, label, and day.
    Days that are not eligible may appear in those chunks; they are training rows only.
    """
    eligible = [int(day) for day in eligible_days]
    undefined = {
        "claim_status": "engineering_cap_metric_undefined",
        "fifty_class_metric_defined": False,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "competition_encoder": False,
        "frozen_eval": None,
        "partial_class_accuracy_reported": False,
        "dates": [dates[day] for day in eligible],
        "reason": (
            "BAcc@10 over all 50 words is undefined unless every held-out day and its "
            f"training days each have at least {DECODE306_MIN_PER_CLASS} observations of every class."
        ),
    }
    if len(eligible) < DECODE306_MIN_HELDOUT_DAYS:
        undefined["reason"] = (
            f"Fewer than {DECODE306_MIN_HELDOUT_DAYS} dates have 3 observations of every class. "
            "No single-day or partial-class accuracy is reported."
        )
        return undefined
    blocked = {
        "claim_status": "label_gate_met_score_blocked_by_memory_cap",
        "fifty_class_metric_defined": False,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "competition_encoder": False,
        "frozen_eval": None,
        "partial_class_accuracy_reported": False,
        "dates": [dates[day] for day in eligible],
        "reason": (
            "Every held-out day and its train fold has 3 observations of all 50 words, "
            "but the ridge normal equations do not fit under the 4 GiB cap. No accuracy was computed."
        ),
    }
    present = set()
    for _features, _labels, day in _chunks(factory):
        present.update(int(value) for value in np.unique(day))
    if any(day not in present for day in eligible):
        return undefined
    for held in eligible:
        train_days = {day for day in present if day != held}
        train, test = _fold_counts(factory, train_days, held)
        if not _covered(test) or not _covered(train):
            return undefined
    probe = _chunks(factory)
    try:
        first = next(probe)
    except StopIteration:
        return undefined
    finally:
        probe.close()
    flat_dim = int(np.prod(first[0].shape[1:]))
    if not score_memory_ok(flat_dim, memory_headroom_bytes()):
        return blocked
    arms = ("uniform", "shrunk_whiten", "shuffled_weight", "stronger_mean")
    per_day = {arm: [] for arm in arms}
    try:
        for held in eligible:
            train_days = {day for day in present if day != held}
            for arm in arms:
                score = _arm_scores_streaming(factory, train_days, held, arm, NULL_STATE_SEED + int(held))
                if score is None:
                    return undefined
                per_day[arm].append(score)
    except ScoreMemoryCap:
        return blocked
    contrast = [per_day["shrunk_whiten"][i] - per_day["uniform"][i] for i in range(len(eligible))]
    summary = day_group_uncertainty(contrast)
    return {
        "claim_status": "decode306_probe_estimated",
        "fifty_class_metric_defined": True,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "competition_encoder": False,
        "tenfold_labeled_minutes_claimed": False,
        "partial_class_accuracy_reported": False,
        "dates": [dates[day] for day in eligible],
        "frozen_eval": {
            "metric": "bacc_at_10_over_all_50_classes",
            "heldout_dates": [dates[day] for day in eligible],
            "arms": {arm: day_group_uncertainty(per_day[arm]) for arm in arms},
            "shrunk_whiten_minus_uniform": summary,
            "effect_claimed": False,
            "distinguishable_from_zero": bool(
                summary["se"] is not None and summary["se"] > 0 and abs(summary["mean"]) > 2.0 * summary["se"]
            ),
        },
        "reason": (
            "Paired differences across held-out days that each have 3 observations of all 50 words, "
            "with the same count in the train fold. This is not an effect claim."
        ),
    }
