"""Train-only noise equations for the frozen v3 decoder. None is assumed valid for MEG.

The cited forecasting papers motivate candidates. Coefficients are fit on training
sessions and chosen on validation sessions. The locked test block is scored once.
Ground-truth word identity, class, and word frequency are not inference features.
"""
from __future__ import annotations

import numpy as np

PROTOCOL_ID = "strong-decoder-noise-1"
LPP_NOTE = (
    "Le Petit Prince listen cache is a separate PDDL transfer stage. "
    "It is not pooled with LibriBrain. Historical mixing of 20 LPP listeners cost about 0.15. "
    "This command does not run that stage."
)
ESTIMAND = (
    "Paired session-level difference in 50-word top-10 balanced accuracy between the "
    "validation-chosen train-only noise weight and the uniform deep_c readout. "
    "This is not the linear 306-channel probe and not a claim that a forecasting equation holds."
)

# Split v2, copied from the literals in src/split2.py. Importing that module reads local MEG.
EVAL_SUBJECTS = (3, 6, 9, 12)
VAL_SUBJECTS = (1, 5, 10)
SEEN_SUBJECTS = (2, 4, 7, 8, 11)
# Historically scored in October Packard and split-v2 Q3/Q4. Not a pristine confirmatory holdout.
HISTORICALLY_INSPECTED_SUBJECTS = EVAL_SUBJECTS + VAL_SUBJECTS + SEEN_SUBJECTS
LINEAR_PROBE_EXHAUSTED_DATES = (
    "2024-07-31",
    "2024-08-07",
    "2024-08-12",
    "2024-08-14",
)
EXCLUDED_SHARED_DATE = "2024-10-04"
CHECKPOINT_BASENAME = "deep_c_ep6.pt"
CHECKPOINT_BYTES = 24690963
CHANCE_BACC10_50 = 0.2

FORBIDDEN_INFERENCE_FEATURES = frozenset({
    "label", "word", "class", "class_id", "word_freq", "word_frequency",
    "zipf", "onset", "onset_s", "timemeg", "timechapter", "shuffle_seed",
    "word_onsets_s", "segment",
})
ALLOWED_INFERENCE_FEATURES = (
    "day_rank",
    "run_fraction",
    "artifact_rms",
    "bandpower",
    "train_window_count",
    "day_distance",
)
CANDIDATES = (
    "uniform",
    "constant",
    "random_covariate",
    "learned_state",
    "error_scaling",
)
# True-label word frequency was listed in the handoff and is rejected here.
REJECTED_CANDIDATES = ("true_label_word_frequency",)


class ProtocolError(RuntimeError):
    pass


def assert_inference_features(names) -> None:
    for name in names:
        key = str(name).strip().lower()
        if key in FORBIDDEN_INFERENCE_FEATURES or key not in ALLOWED_INFERENCE_FEATURES:
            raise ProtocolError(f"inference feature {name!r} is not in the frozen auditory set")


def assign_split(sessions: list[dict]) -> dict:
    """Chronological subject-0 sessions. Test scores are not an argument.

    sessions: {date, base} already in acquisition order. Shared Sherlock day and the
    four exhausted linear-probe dates are removed from the confirmatory test slice.
    They may remain in train. Validation chooses the noise candidate. Test is last.
    """
    rows = [dict(row) for row in sessions]
    if [row["date"] for row in rows] != sorted(row["date"] for row in rows):
        raise ProtocolError("sessions are not in acquisition order")
    usable = [row for row in rows if row["date"] != EXCLUDED_SHARED_DATE]
    if len(usable) < 5:
        raise ProtocolError("fewer than 5 subject-0 sessions after dropping 2024-10-04")
    n_train = max(1, int(round(0.6 * len(usable))))
    n_val = max(1, int(round(0.2 * len(usable))))
    if n_train + n_val >= len(usable):
        n_val = 1
        n_train = len(usable) - 2
    train = usable[:n_train]
    val = usable[n_train:n_train + n_val]
    test = usable[n_train + n_val:]
    confirmatory = [row for row in test if row["date"] not in LINEAR_PROBE_EXHAUSTED_DATES]
    return {
        "train": train,
        "val": val,
        "test": test,
        "confirmatory_test": confirmatory,
        "exhausted_linear_probe_dates_not_confirmatory": list(LINEAR_PROBE_EXHAUSTED_DATES),
        "eval_subjects_are_pristine_holdout": False,
        "historically_inspected_subjects": list(HISTORICALLY_INSPECTED_SUBJECTS),
        "transfer_label": "historically_inspected_transfer",
        "broad_pooling": "forbidden",
        "lpp_pooling": "forbidden",
        "q3q4_is_not_confirmation": True,
    }


def _design(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    return np.column_stack([np.ones(len(z)), z])


def fit_noise(z_train: np.ndarray, energy_train: np.ndarray, candidate: str, *, seed: int = 0) -> dict:
    """Fit log variance on training rows only. energy_train may use train labels. z must not."""
    if candidate not in CANDIDATES:
        raise ProtocolError(f"unknown candidate {candidate}")
    energy = np.asarray(energy_train, dtype=np.float64)
    z = np.asarray(z_train, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if len(energy) != len(z):
        raise ProtocolError("train energy and covariates differ in length")
    target = np.log(np.clip(energy, 1e-8, None))
    if candidate == "uniform":
        return {"candidate": candidate, "intercept": 0.0, "slope": np.zeros(z.shape[1])}
    if candidate == "constant":
        return {"candidate": candidate, "intercept": float(target.mean()), "slope": np.zeros(z.shape[1])}
    if candidate == "random_covariate":
        rng = np.random.default_rng(seed)
        z = rng.normal(size=z.shape)
    design = _design(z)
    pen = 1e-2 * len(z)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += pen
    gram[0, 0] -= pen
    beta = np.linalg.solve(gram, design.T @ target)
    return {"candidate": candidate, "intercept": float(beta[0]), "slope": beta[1:], "train_rows": int(len(z))}


def log_variance(z: np.ndarray, params: dict) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if params["candidate"] == "uniform":
        return np.zeros(len(z))
    return params["intercept"] + z @ np.asarray(params["slope"], dtype=np.float64)


def weighted_nll(energy: np.ndarray, z: np.ndarray, params: dict) -> float:
    """Gaussian energy score. Uniform is the unweighted mean energy, not a fitted law."""
    energy = np.asarray(energy, dtype=np.float64)
    if params["candidate"] == "uniform":
        return float(np.mean(energy))
    lv = np.clip(log_variance(z, params), -8.0, 8.0)
    return float(np.mean(0.5 * (lv + energy * np.exp(-lv))))


def select_on_validation(val_nll: dict) -> str:
    unknown = set(val_nll) - set(CANDIDATES)
    if unknown or set(val_nll) != set(CANDIDATES):
        raise ProtocolError("validation selection must score every frozen candidate and no others")
    return min(CANDIDATES, key=lambda name: (float(val_nll[name]), name))


def session_bootstrap(session_ids, values, *, seed: int = 0, draws: int = 200) -> dict:
    """Resample recordings, not windows. One session makes the interval undefined."""
    session_ids = np.asarray(session_ids)
    values = np.asarray(values, dtype=np.float64)
    sessions = np.unique(session_ids)
    if len(sessions) < 2:
        return {"mean": float(values.mean()) if len(values) else None, "low": None, "high": None, "defined": False}
    means = []
    for session in sessions:
        means.append(float(values[session_ids == session].mean()))
    means = np.asarray(means)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        take = rng.integers(0, len(means), len(means))
        boots.append(float(means[take].mean()))
    boots = np.sort(boots)
    return {
        "mean": float(means.mean()),
        "low": float(boots[int(0.025 * (draws - 1))]),
        "high": float(boots[int(0.975 * (draws - 1))]),
        "defined": True,
        "n_sessions": int(len(sessions)),
    }


def timing_inputs(rel: np.ndarray, meg: np.ndarray, mode: str, *, seed: int = 0):
    """Primary decoder timing is zero. Timing-only is a shortcut control, not a result."""
    rel = np.asarray(rel, dtype=np.float64).copy()
    meg = np.asarray(meg, dtype=np.float64).copy()
    if mode == "zero":
        return np.zeros_like(rel), meg
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        flat = rel.ravel().copy()
        rng.shuffle(flat)
        return flat.reshape(rel.shape), meg
    if mode == "timing_only":
        return rel, np.zeros_like(meg)
    raise ProtocolError(mode)


def bacc10_50(top_hit_by_class: dict) -> float | None:
    """Mean over all 50 classes. Missing a class makes the competition-style score undefined."""
    if set(top_hit_by_class) != set(range(50)):
        return None
    return float(np.mean([top_hit_by_class[c] for c in range(50)]))


def next_sessions(all_ids, done) -> list:
    done = set(done)
    return [item for item in all_ids if item not in done]
