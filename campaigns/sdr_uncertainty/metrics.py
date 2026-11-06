"""Frozen metrics. Task scores below the sample floor are not interpreted."""
from __future__ import annotations

import numpy as np

from .models import design_mean, ridge_multi
from .protocol import CHANCE_MARGIN, MIN_CLASSES_FOR_GATE, MIN_TASK_N, RIDGE


def residual_logscore(y, mean, var) -> float:
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    quad = (np.asarray(y, dtype=np.float64) - mean) ** 2 / var
    per = -0.5 * (np.log(2.0 * np.pi) + np.log(var) + quad)
    return float(np.mean(per))


def calibration_error(y, mean, var) -> float:
    pred = np.maximum(np.mean(var, axis=1), 1e-12)
    emp = np.mean((y - mean) ** 2, axis=1)
    if np.allclose(pred, pred[0]):
        return float(abs(np.log(np.mean(emp) + 1e-12) - np.log(np.mean(pred))))
    edges = np.quantile(pred, np.linspace(0.0, 1.0, 6))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    gaps = []
    for i in range(5):
        mask = (pred > edges[i]) & (pred <= edges[i + 1])
        if int(mask.sum()) < 5:
            continue
        gaps.append(abs(np.log(np.mean(emp[mask]) + 1e-12) - np.log(np.mean(pred[mask]))))
    if not gaps:
        return float(abs(np.log(np.mean(emp) + 1e-12) - np.log(np.mean(pred))))
    return float(np.mean(gaps))


def lag1_autocorr(series: np.ndarray, day: np.ndarray) -> float:
    num = 0.0
    den = 0.0
    for d in np.unique(day):
        block = series[day == d]
        if len(block) < 3:
            continue
        num += float(np.sum(block[1:] * block[:-1]))
        den += float(np.sum(block[:-1] ** 2))
    if den <= 0:
        return 0.0
    return float(num / den)


def innovation_autocorrelations(y, mean, var, day) -> dict:
    std = (y - mean) / np.sqrt(np.maximum(var, 1e-12))
    centered_sq = std ** 2 - 1.0
    return {
        "innovation_acf_lag1": lag1_autocorr(std, day),
        "squared_innovation_acf_lag1": lag1_autocorr(centered_sq, day),
    }


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=np.float64)
    keep = (labels >= 0) & (labels < n_classes)
    out[np.arange(len(labels))[keep], labels[keep]] = 1.0
    return out


def fit_task_head(y, labels: np.ndarray):
    keep = labels >= 0
    if int(keep.sum()) < 2:
        return None
    n_classes = int(labels[keep].max()) + 1
    beta = ridge_multi(design_mean(y[keep]), _one_hot(labels[keep], n_classes), lam=RIDGE)
    return {"beta": beta, "n_classes": n_classes}


def predict_task(head, y) -> np.ndarray:
    if head is None:
        return np.full(len(y), -1, dtype=np.int64)
    scores = design_mean(y) @ head["beta"]
    return scores.argmax(axis=1).astype(np.int64)


def balanced_accuracy(labels: np.ndarray, pred: np.ndarray) -> float | None:
    keep = labels >= 0
    labels = labels[keep]
    pred = pred[keep]
    classes = np.unique(labels)
    if len(classes) == 0:
        return None
    recalls = []
    for c in classes:
        mask = labels == c
        recalls.append(float(np.mean(pred[mask] == c)))
    return float(np.mean(recalls))


def assess_task(y_train, y_test, labels_train, labels_test, null_seed: int) -> dict:
    head = fit_task_head(y_train, labels_train)
    if head is None:
        return {"status": "not_computed", "neural_inference_allowed": False, "diagnose_required": False}
    pred = predict_task(head, y_test)
    bacc = balanced_accuracy(labels_test, pred)
    rng = np.random.default_rng(null_seed)
    permuted = labels_test.copy()
    keep = permuted >= 0
    permuted[keep] = rng.permutation(permuted[keep])
    null_bacc = balanced_accuracy(permuted, pred)
    shuffled_y = y_test.copy()
    shuffled_y = shuffled_y[rng.permutation(len(shuffled_y))]
    shuffle_bacc = balanced_accuracy(labels_test, predict_task(head, shuffled_y))
    n_eval = int((labels_test >= 0).sum())
    n_classes = int(head["n_classes"])
    present = int(len(np.unique(labels_test[labels_test >= 0]))) if n_eval else 0
    interpretable = n_eval >= MIN_TASK_N and present >= MIN_CLASSES_FOR_GATE and n_classes >= MIN_CLASSES_FOR_GATE
    chance = 1.0 / n_classes
    nulls = [x for x in (null_bacc, shuffle_bacc) if x is not None]
    above = bool(interpretable and any(x > chance + CHANCE_MARGIN for x in nulls))
    return {
        "status": "interpreted" if interpretable else "underpowered_not_interpreted",
        "balanced_accuracy": bacc,
        "label_permutation_balanced_accuracy": null_bacc,
        "shuffled_feature_balanced_accuracy": shuffle_bacc,
        "chance": chance,
        "n_eval": n_eval,
        "n_classes_present": present,
        "diagnose_required": above,
        "neural_inference_allowed": False,
        "neural_inference_reason": (
            "Above-chance noise control; diagnose before any neural inference."
            if above else
            "This rung does not run neural inference. A later decoder intervention stays blocked."
        ),
    }


def aggregate_noise_controls(per_day: list[dict]) -> dict:
    """Gate on the mean of per-day null scores. One noisy day is not a second look."""
    usable = [row for row in per_day if row.get("status") == "interpreted"]
    if len(usable) != len(per_day) or not usable:
        return {
            "status": "underpowered_not_interpreted",
            "diagnose_required": False,
            "neural_inference_allowed": False,
            "neural_inference_reason": (
                "Task noise controls are not interpreted. Neural inference is not run."
            ),
        }
    chance = float(usable[0]["chance"])
    perm = float(np.mean([row["label_permutation_balanced_accuracy"] for row in usable]))
    shuffled = float(np.mean([row["shuffled_feature_balanced_accuracy"] for row in usable]))
    above = perm > chance + CHANCE_MARGIN or shuffled > chance + CHANCE_MARGIN
    return {
        "status": "interpreted",
        "mean_label_permutation_balanced_accuracy": perm,
        "mean_shuffled_feature_balanced_accuracy": shuffled,
        "chance": chance,
        "margin": CHANCE_MARGIN,
        "diagnose_required": above,
        "neural_inference_allowed": False,
        "neural_inference_reason": (
            "Above-chance noise control; diagnose before any neural inference."
            if above else
            "Noise controls are within the pre-registered band. Neural inference is still not run."
        ),
    }


def day_group_uncertainty(per_day_scores: list[float]) -> dict:
    scores = np.asarray(per_day_scores, dtype=np.float64)
    if len(scores) == 0:
        return {"n_days": 0, "mean": None, "std": None, "se": None, "per_day": []}
    std = float(scores.std(ddof=1)) if len(scores) > 1 else None
    se = float(std / np.sqrt(len(scores))) if std is not None else None
    return {
        "n_days": int(len(scores)),
        "mean": float(scores.mean()),
        "std": std,
        "se": se,
        "per_day": [float(x) for x in scores],
    }
