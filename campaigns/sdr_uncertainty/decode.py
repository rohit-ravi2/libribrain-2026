"""Matched-budget linear probe. Not the competition encoder, and not a claim that weights help."""
from __future__ import annotations

import numpy as np

from .metrics import day_group_uncertainty
from .models import design_mean, design_rich, ridge_multi
from .protocol import (
    DECODE_BINS,
    DECODE_FRACTIONS,
    DECODE_MIN_CLASSES,
    DECODE_MIN_DAYS,
    DECODE_MIN_LABELED_PER_DAY,
    DECODE_MIN_TRAIN_LABELED,
    DECODE_SHRINK_TOWARD_UNIFORM,
    DECODE_TOP_K,
    DECODE_WEIGHT_CLIP,
    FS_HZ,
    NULL_STATE_SEED,
    RHO_MAX,
    SCALE_GRAD,
    SCALE_MAG,
    STATE_CHANNELS,
    TARGET_CHANNELS,
    WINDOW_SAMPLES,
)


def _scale(channel: int) -> float:
    return SCALE_MAG if channel % 3 == 0 else SCALE_GRAD


def summarize_demeaned(samples: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
    """Bin means after removing each channel's window mean. A constant window becomes zero."""
    scaled = np.vstack([np.asarray(samples[i], dtype=np.float64) * _scale(channels[i]) for i in range(len(channels))])
    scaled = scaled - scaled.mean(axis=1, keepdims=True)
    edges = np.linspace(0, scaled.shape[1], DECODE_BINS + 1, dtype=int)
    return np.concatenate([scaled[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(DECODE_BINS)])


def keep_nonoverlapping(onsets_s: np.ndarray) -> np.ndarray:
    """Keep a window only when it starts at or after the previous kept window ends."""
    order = np.argsort(np.asarray(onsets_s, dtype=np.float64), kind="mergesort")
    keep = []
    next_free = -1e18
    span = WINDOW_SAMPLES / FS_HZ
    for index in order:
        onset = float(onsets_s[index])
        if onset + 1e-9 < next_free:
            continue
        keep.append(int(index))
        next_free = onset + span
    return np.asarray(keep, dtype=np.int64)


def _wls(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(weights, 1e-8, None))[:, None]
    return ridge_multi(x * scale, y * scale)


def _pack(labels: np.ndarray):
    classes = np.unique(labels[labels >= 0])
    packed = np.full(len(labels), -1, dtype=np.int64)
    for column, cls in enumerate(classes):
        packed[labels == cls] = column
    return packed, classes


def _one_hot(packed: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(packed), n_classes), dtype=np.float64)
    keep = packed >= 0
    out[np.arange(len(packed))[keep], packed[keep]] = 1.0
    return out


def bacc_at_k(scores: np.ndarray, labels: np.ndarray, k: int = DECODE_TOP_K):
    n_classes = int(scores.shape[1])
    if n_classes < max(k, DECODE_MIN_CLASSES):
        return None
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    recalls = []
    for cls in np.unique(labels):
        if cls < 0:
            continue
        mask = labels == cls
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(np.any(top[mask] == cls, axis=1))))
    if not recalls:
        return None
    return float(np.mean(recalls))


def shrunk_weights(energy_train: np.ndarray, resid_train: np.ndarray, energy_query: np.ndarray) -> np.ndarray:
    """Train-only slope of log residual energy, then shrink and clip toward uniform."""
    target = np.log(np.mean(resid_train ** 2, axis=1) + 1e-8)
    design = np.column_stack([np.ones(len(energy_train)), energy_train])
    beta = ridge_multi(design, target[:, None]).ravel()
    pred = np.column_stack([np.ones(len(energy_query)), energy_query]) @ beta
    precision = np.exp(-pred)
    raw = precision / np.median(precision)
    mixed = DECODE_SHRINK_TOWARD_UNIFORM * np.ones_like(raw) + (1.0 - DECODE_SHRINK_TOWARD_UNIFORM) * raw
    low, high = DECODE_WEIGHT_CLIP
    return np.clip(mixed, low, high)


def fit_rho(y: np.ndarray, day: np.ndarray, train: np.ndarray) -> float:
    num = 0.0
    den = 0.0
    for t in range(1, len(y)):
        if day[t] != day[t - 1] or not train[t] or not train[t - 1]:
            continue
        num += float(np.sum(y[t] * y[t - 1]))
        den += float(np.sum(y[t - 1] ** 2))
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, -RHO_MAX, RHO_MAX))


def causal_whiten(y: np.ndarray, day: np.ndarray, rho: float) -> np.ndarray:
    out = np.array(y, dtype=np.float64, copy=True)
    for t in range(1, len(out)):
        if day[t] == day[t - 1]:
            out[t] = y[t] - rho * y[t - 1]
    return out


def _support(labels: np.ndarray, day: np.ndarray) -> list[int]:
    return [int(np.sum(labels[day == d] >= 0)) for d in np.unique(day)]


def _inadequate(reason: str, labels, day, dates) -> dict:
    return {
        "claim_status": "decoding_inadequate",
        "real_meg_scientific_validation": False,
        "endpoint_is_decoding": False,
        "effect_claimed": False,
        "adequate_for_competition_claim": False,
        "reason": reason,
        "labeled_per_day": _support(labels, day) if labels is not None else [],
        "dates": list(dates or []),
        "frozen_eval": None,
        "exploratory": None,
    }


def evaluate_decoding(batch: dict) -> dict:
    """Cross-fit a linear word probe. Weights are learned on train days only and may not help."""
    y = np.asarray(batch["y"], dtype=np.float64)
    state = np.asarray(batch["state"], dtype=np.float64)
    labels = np.asarray(batch["label"])
    day = np.asarray(batch["day"])
    time_index = np.asarray(batch["time_index"])
    order = np.lexsort((time_index, day))
    y, state, labels, day = y[order], state[order], labels[order], day[order]
    dates = list(batch.get("dates") or [])
    per_day = _support(labels, day)
    classes = np.unique(labels[labels >= 0])
    if len(per_day) < DECODE_MIN_DAYS or any(count < DECODE_MIN_LABELED_PER_DAY for count in per_day):
        return _inadequate(
            "Too few in-vocabulary labels on a held-out day for BAcc@10. No decoding score is reported.",
            labels, day, dates,
        )
    if len(classes) < DECODE_MIN_CLASSES:
        return _inadequate(
            f"Fewer than {DECODE_MIN_CLASSES} vocabulary classes, so top-10 balanced accuracy is not interpretable.",
            labels, day, dates,
        )
    energy = state[:, -1]
    days = np.unique(day)
    arms = ("uniform", "shrunk_weight", "shuffled_weight", "quadratic_features", "causal_ar")
    per_day_scores = {arm: [] for arm in arms}
    rhos = []
    for held in days:
        test = day == held
        train = ~test
        mu = y[train].mean(axis=0)
        sd = np.maximum(y[train].std(axis=0), 1e-6)
        y_s = (y - mu) / sd
        resid = y_s[train] - design_mean(state[train]) @ ridge_multi(design_mean(state[train]), y_s[train])
        weights = shrunk_weights(energy[train], resid, energy)
        weights = np.where(train, weights, 1.0)
        shuffled = weights.copy()
        rng = np.random.default_rng(NULL_STATE_SEED + int(held))
        shuffled[train] = rng.permutation(weights[train])
        rho = fit_rho(y_s, day, train)
        rhos.append(rho)
        whitened = causal_whiten(y_s, day, rho)
        specs = {
            "uniform": (y_s, np.ones(len(y_s)), design_mean),
            "shrunk_weight": (y_s, weights, design_mean),
            "shuffled_weight": (y_s, shuffled, design_mean),
            "quadratic_features": (y_s, np.ones(len(y_s)), design_rich),
            "causal_ar": (whitened, np.ones(len(y_s)), design_mean),
        }
        for arm, (features, arm_weights, design) in specs.items():
            # Fit on train rows only by zeroing test labels inside a copy.
            fit_labels = labels.copy()
            fit_labels[test] = -1
            score = _fit_score_split(features, fit_labels, labels, arm_weights, design, test)
            per_day_scores[arm].append(score)
    if any(score is None for scores in per_day_scores.values() for score in scores):
        return _inadequate("A day could not define BAcc@10 after the train-class restriction.", labels, day, dates)
    contrast = [
        per_day_scores["shrunk_weight"][i] - per_day_scores["uniform"][i] for i in range(len(days))
    ]
    frozen = {
        "metric": "bacc_at_10_over_classes_present",
        "competition_bacc_at_10_over_all_50": False,
        "chance_if_classes_equal_minimum": DECODE_TOP_K / DECODE_MIN_CLASSES,
        "arms": {arm: day_group_uncertainty(per_day_scores[arm]) for arm in arms},
        "weighted_minus_uniform": day_group_uncertainty(contrast),
        "effect_claimed": False,
        "weighting_assumed_helpful": False,
    }
    frozen["distinguishable_from_zero"] = _distinguishable(frozen["weighted_minus_uniform"])
    exploratory = {
        "not_a_frozen_result": True,
        "label_permutation": _permutation_control(y, labels, day),
        "independent_noise_contrast": _noise_control(y, labels, day, state),
        "labeled_word_fractions": _fraction_curve(y, labels, day, state),
        "tenfold_fraction_is_aspirational": True,
        "fractions_are_recording_minutes": False,
    }
    interpretable = exploratory["label_permutation"]["diagnose_required"] is False
    return {
        "claim_status": "decode_probe_completed" if interpretable else "decode_probe_diagnose",
        "real_meg_scientific_validation": False,
        "endpoint_is_decoding": True,
        "effect_claimed": False,
        "adequate_for_competition_claim": False,
        "sensor_count": len(TARGET_CHANNELS),
        "reason": (
            "Linear probe on four demeaned magnetometer channels. "
            "A difference from uniform weights is an estimate, not a claim that weighting helps. "
            "This is not the 306-channel competition encoder."
        ),
        "labeled_per_day": per_day,
        "dates": dates,
        "train_only_rhos": rhos,
        "frozen_eval": frozen,
        "exploratory": exploratory,
        "frozen_eval_interpretable": interpretable,
    }


def _distinguishable(summary: dict) -> bool:
    estimate, se = summary.get("mean"), summary.get("se")
    if estimate is None or se is None or se <= 0:
        return False
    return abs(float(estimate)) > 2.0 * float(se)


def _fit_score_split(features, fit_labels, eval_labels, weights, design, test) -> float | None:
    keep = fit_labels >= 0
    packed, classes = _pack(fit_labels)
    if len(classes) < DECODE_MIN_CLASSES:
        return None
    beta = _wls(design(features[keep]), _one_hot(packed[keep], len(classes)), weights[keep])
    scores = design(features[test]) @ beta
    eval_packed = np.full(int(test.sum()), -1, dtype=np.int64)
    class_index = {int(cls): column for column, cls in enumerate(classes)}
    raw = eval_labels[test]
    for row, label in enumerate(raw):
        if int(label) in class_index:
            eval_packed[row] = class_index[int(label)]
    return bacc_at_k(scores, eval_packed)


def _permutation_control(y, labels, day) -> dict:
    rng = np.random.default_rng(NULL_STATE_SEED)
    scores = []
    for held in np.unique(day):
        test = day == held
        train = ~test
        fit_labels = np.full(len(labels), -1, dtype=np.int64)
        train_idx = np.flatnonzero(train & (labels >= 0))
        permuted = labels[train_idx].copy()
        rng.shuffle(permuted)
        fit_labels[train_idx] = permuted
        weights = np.ones(len(y))
        score = _fit_score_split(y, fit_labels, labels, weights, design_mean, test)
        scores.append(score)
    finite = [score for score in scores if score is not None]
    return {
        "per_day": scores,
        "mean": float(np.mean(finite)) if finite else None,
        "diagnose_required": False if not finite else float(np.mean(finite)) > (DECODE_TOP_K / DECODE_MIN_CLASSES) + 0.15,
    }


def _noise_control(y, labels, day, state) -> dict:
    rng = np.random.default_rng(NULL_STATE_SEED + 1)
    noise = rng.normal(size=y.shape)
    uniform, weighted = [], []
    for held in np.unique(day):
        test = day == held
        train = ~test
        resid = noise[train] - noise[train].mean(axis=0)
        weights = shrunk_weights(state[train, -1], resid, state[:, -1])
        weights = np.where(train, weights, 1.0)
        fit_labels = labels.copy()
        fit_labels[test] = -1
        uniform.append(_fit_score_split(noise, fit_labels, labels, np.ones(len(noise)), design_mean, test))
        weighted.append(_fit_score_split(noise, fit_labels, labels, weights, design_mean, test))
    if any(score is None for score in uniform + weighted):
        return {"mean_weighted_minus_uniform": None}
    contrast = [weighted[i] - uniform[i] for i in range(len(uniform))]
    return {"mean_weighted_minus_uniform": float(np.mean(contrast)), "not_a_result": True}


def _fraction_curve(y, labels, day, state) -> list[dict]:
    rows = []
    for fraction in DECODE_FRACTIONS:
        kept = np.zeros(len(y), dtype=bool)
        for held in np.unique(day):
            # Prefix in time within every day; the held-out day is removed again at fit time.
            idx = np.flatnonzero((day == held) & (labels >= 0))
            n_keep = max(1, int(np.floor(fraction * len(idx))))
            kept[idx[:n_keep]] = True
        usable = int(kept.sum())
        if usable < DECODE_MIN_TRAIN_LABELED and fraction < 1.0:
            rows.append({
                "fraction_of_labeled_words": fraction,
                "train_labeled_used_if_one_day_held_out": None,
                "weighted_minus_uniform_mean": None,
                "status": "underpowered",
                "aspirational": fraction <= 0.10,
            })
            continue
        contrast = []
        for held in np.unique(day):
            test = day == held
            train = kept & ~test & (labels >= 0)
            if int(train.sum()) < DECODE_MIN_CLASSES:
                contrast.append(None)
                continue
            fit_labels = np.full(len(labels), -1, dtype=np.int64)
            fit_labels[train] = labels[train]
            mu = y[train].mean(axis=0)
            sd = np.maximum(y[train].std(axis=0), 1e-6)
            y_s = (y - mu) / sd
            resid = y_s[train] - y_s[train].mean(axis=0)
            weights = shrunk_weights(state[train, -1], resid, state[:, -1])
            weights = np.where(train, weights, 1.0)
            uniform = _fit_score_split(y_s, fit_labels, labels, np.ones(len(y)), design_mean, test)
            weighted = _fit_score_split(y_s, fit_labels, labels, weights, design_mean, test)
            contrast.append(None if uniform is None or weighted is None else weighted - uniform)
        if any(value is None for value in contrast):
            summary = None
            status = "underpowered"
        else:
            summary = float(np.mean(contrast))
            status = "estimated_not_a_claim"
        rows.append({
            "fraction_of_labeled_words": fraction,
            "weighted_minus_uniform_mean": summary,
            "status": status,
            "aspirational": fraction <= 0.10,
            "recording_minutes": False,
        })
    return rows


def collect_decode_windows(dataset, rows: list[dict], vocab: tuple[str, ...], *, max_windows: int, max_bytes: int) -> dict:
    """Non-overlapping demeaned bins. Onset is used only to place and separate windows."""
    from .stream import vocab_index
    onsets = np.array([float(row["timemeg"]) for row in rows], dtype=np.float64)
    chosen = keep_nonoverlapping(onsets)
    n_times = int(dataset.shape[1])
    features, energy, labels, ranges = [], [], [], []
    bytes_read = 0
    channels = tuple(TARGET_CHANNELS) + tuple(STATE_CHANNELS)
    for index in chosen:
        if len(features) >= max_windows:
            break
        onset = int(round(onsets[index] * FS_HZ))
        start, stop = onset, onset + WINDOW_SAMPLES
        if start < 0 or stop > n_times:
            continue
        need = (stop - start) * 8 * len(channels)
        if bytes_read + need > max_bytes:
            break
        samples = []
        for channel in channels:
            piece = np.asarray(dataset[int(channel), start:stop], dtype=np.float64)
            bytes_read += int(piece.nbytes)
            samples.append(piece)
        block = np.vstack(samples)
        target = summarize_demeaned(block[:len(TARGET_CHANNELS)], tuple(TARGET_CHANNELS))
        state_block = block[len(TARGET_CHANNELS):]
        state_scaled = np.vstack([
            state_block[i] * _scale(STATE_CHANNELS[i]) for i in range(len(STATE_CHANNELS))
        ])
        state_scaled = state_scaled - state_scaled.mean(axis=1, keepdims=True)
        features.append(target)
        energy.append(float(np.log(np.mean(state_scaled ** 2) + 1e-12)))
        labels.append(vocab_index(rows[index]["segment"], vocab))
        ranges.append((start, stop))
    if not features:
        raise RuntimeError("no non-overlapping decode windows inside the caps")
    n = len(features)
    return {
        "y": np.asarray(features, dtype=np.float64),
        "state": np.column_stack([
            np.arange(n, dtype=np.float64) / max(n - 1, 1),
            np.asarray(energy, dtype=np.float64),
        ]),
        "state_names": ("run_fraction", "state_energy"),
        "label": np.asarray(labels, dtype=np.int64),
        "day": np.zeros(n, dtype=np.int64),
        "time_index": np.arange(n, dtype=np.int64),
        "sample_ranges": ranges,
        "bytes_read": int(bytes_read),
        "n_windows": n,
        "nonoverlapping": True,
        "feature_set": "frozen_v1_disjoint_channels",
    }
