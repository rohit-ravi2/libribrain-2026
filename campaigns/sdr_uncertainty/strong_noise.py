"""Train-only noise equations for the v3 decoder. Amendment 1.4.

Forecasting papers motivate candidates. Their equations are not assumed to hold
for MEG. Every candidate is scored with the same Gaussian negative log-likelihood
on a defined decoder residual: the frozen network's per-word cross-entropy.
A scalar per-row temperature is not an accuracy result. Subject-0 days are not
a frozen-trunk holdout.
"""
from __future__ import annotations

from datetime import date as Date

import numpy as np

AMENDMENT = "1.4"
PROTOCOL_ID = "strong-decoder-noise-1.4"
LPP_NOTE = (
    "Le Petit Prince is a separate PDDL transfer stage. It is not pooled with LibriBrain. "
    "Historical mixing of about 20 LPP listeners cost about 0.15. "
    "This stage writes its own file and does not enter selection."
)
ESTIMAND = (
    "Validation-chosen train-only noise model versus uniform and constant baselines, "
    "all scored by Gaussian NLL of the frozen decoder's per-word cross-entropy. "
    "Top-10 accuracy is reported from logits. A per-row temperature is not applied to those logits. "
    "A matched-budget head finetune is a separate comparison and is not a full retrain."
)
FROZEN_TRUNK_TRAIN_FACT = (
    "deep_c train_mask keeps every non-EVAL, non-VAL window in quarter 'other', 'Q1', or 'Q2'. "
    "Subject-0 sessions that are not the shared Sherlock stimulus are all quarter 'other', "
    "so those recording days were in the frozen trunk. They are not a held-out set for deep_c_ep6. "
    "A partial finetune that starts from that checkpoint still contains them."
)
FULL_RETRAIN = {
    "launched": False,
    "practical_under_cpu_4g_one_thread": False,
    "reason": (
        "deep_c was 8 epochs of 1500 steps on a GPU. One CPU, 4 GiB, and no swap cannot host that "
        "as a silent substitution. The runnable substitute is a checkpointed matched-budget finetune "
        "of the classification head and the noise equation on cached trunk embeddings. "
        "The trunk stays frozen. This is not the strongest full fit, and it does not make "
        "subject-0 days a clean holdout."
    ),
}

EVAL_SUBJECTS = (3, 6, 9, 12)
VAL_SUBJECTS = (1, 5, 10)
SEEN_SUBJECTS = (2, 4, 7, 8, 11)
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
N_CLASSES = 50
TOP_K = 10
# Subject-0 non-shared runs used only for the head finetune. Not the noise-equation fit,
# and not the full 162-run corpus. The noise equation is fit on trunk-unseen Q3 instead.
HEAD_TRAIN_RUNS = 16
TRAIN_FORWARD_RUNS = HEAD_TRAIN_RUNS
PARTIAL_LR = 1e-3
# Smoke is a plumbing stage. It is not what `next` runs, and it is not a result.
SMOKE_SCHEDULE = {
    "name": "smoke",
    "max_steps": 8,
    "steps_per_launch": 8,
    "min_steps": 8,
    "patience_checks": None,
    "substantive": False,
}
# Head-only finetune on cached embeddings. Feasible on one CPU at 4 GiB.
# Stop is the validation curve, not a fixed 32-step cutoff and not the eval score.
SUBSTANTIVE_SCHEDULE = {
    "name": "substantive_head_finetune",
    "max_steps": 2000,
    "steps_per_launch": 100,
    "min_steps": 200,
    "patience_checks": 3,
    "substantive": True,
}
PARTIAL_STEPS_PER_LAUNCH = SUBSTANTIVE_SCHEDULE["steps_per_launch"]
PARTIAL_TARGET_STEPS = SUBSTANTIVE_SCHEDULE["max_steps"]
LOCK_TTL_SECONDS = 14400
LOCK_TTL_MIN_SECONDS = 3600
LOCK_TTL_MAX_SECONDS = 86400
# The verified scheduler stores this number in holder.json and does not expire the lock.
LOCK_TTL_ENFORCED = False
HEARTBEAT_SECONDS = None
QUARTER_PRE = 1.0
QUARTER_POST = 2.0
QUARTER_GUARD = 1.0

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
    "log_train_window_count",
    "day_distance",
)
CANDIDATES = (
    "uniform",
    "constant",
    "random_covariate",
    "learned_state",
    "error_scaling",
)
PARTIAL_ARMS = ("uniform", "random_covariate", "learned_state")
ERROR_SCALING_FEATURES = ("log_train_window_count", "day_distance")
REJECTED_CANDIDATES = ("true_label_word_frequency",)
LOG_VAR_CLIP = (-8.0, 8.0)


class ProtocolError(RuntimeError):
    pass


def assert_inference_features(names) -> None:
    for name in names:
        key = str(name).strip().lower()
        if key in FORBIDDEN_INFERENCE_FEATURES or key not in ALLOWED_INFERENCE_FEATURES:
            raise ProtocolError(f"inference feature {name!r} is not in the frozen auditory set")


def assign_split(sessions: list[dict]) -> dict:
    """Chronological subject-0 plan for a retrain that would exclude the test days.

    Not valid with the frozen deep_c trunk. Test scores are not an argument.
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
        "broad_pooling": "forbidden_with_lpp",
        "lpp_pooling": "forbidden",
        "q3q4_is_not_confirmation": True,
        "valid_with_frozen_deep_c": False,
        "frozen_trunk_fact": FROZEN_TRUNK_TRAIN_FACT,
        "requires": "from_scratch_or_unseen_init_retrain_excluding_these_test_days",
        "pristine_confirmatory_subjects": [],
    }


def frozen_trunk_role(subject: int, shared: bool) -> dict:
    """What a frozen deep_c trunk may claim about this recording."""
    subject = int(subject)
    if subject in EVAL_SUBJECTS:
        role = "exploratory_untrained"
        phase = "eval"
        holdout = True
    elif subject in VAL_SUBJECTS:
        role = "selection_untrained"
        phase = "selection"
        holdout = True
    elif subject == 0 and not shared:
        role = "train_subject0_not_a_frozen_holdout"
        phase = "train"
        holdout = False
    elif subject == 0 and shared:
        role = "train_shared_quarters_only"
        phase = "train_shared"
        holdout = False
    elif subject in SEEN_SUBJECTS or 13 <= subject <= 32:
        role = "not_untrained"
        phase = "seen_or_late"
        holdout = False
    else:
        role = "unused"
        phase = "unused"
        holdout = False
    return {
        "subject": subject,
        "role": role,
        "phase": phase,
        "frozen_holdout": holdout,
        "pristine_confirmatory": False,
        "subject0_day_holdout_with_frozen_trunk": False,
    }


class NumericClip:
    """Train-only z-score, then clip to the train range. Unseen values cannot form a new category."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.lo = None
        self.hi = None

    def fit(self, values: np.ndarray) -> "NumericClip":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) < 2:
            raise ProtocolError("standardizer needs at least two training rows")
        self.mean = x.mean(axis=0)
        std = x.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        z = (x - self.mean) / self.std
        self.lo = z.min(axis=0)
        self.hi = z.max(axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ProtocolError("standardizer was not fit on training rows")
        x = np.asarray(values, dtype=np.float64)
        width = int(len(self.mean))
        if x.ndim == 1:
            x = x.reshape(-1, 1) if width == 1 else x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != width:
            raise ProtocolError(f"feature width {getattr(x, 'shape', None)} != {width}")
        z = (x - self.mean) / self.std
        return np.clip(z, self.lo, self.hi)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NumericClip":
        clip = cls()
        clip.mean = np.asarray(payload["mean"], dtype=np.float64)
        clip.std = np.asarray(payload["std"], dtype=np.float64)
        clip.lo = np.asarray(payload["lo"], dtype=np.float64)
        clip.hi = np.asarray(payload["hi"], dtype=np.float64)
        return clip


def _to_date(text: str) -> Date:
    try:
        year, month, day = (int(part) for part in str(text).split("-"))
        return Date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"acquisition date {text!r} is not YYYY-MM-DD") from exc


def _rank_in_train(moment: Date, train_days: list[Date]) -> float:
    ords = [day.toordinal() for day in train_days]
    raw = int(np.searchsorted(np.asarray(ords), moment.toordinal()))
    return float(np.clip(raw, 0, len(train_days) - 1))


def _distance_days(moment: Date, others: list[Date]) -> float:
    if not others:
        raise ProtocolError("day distance needs another training date")
    return float(min(abs(moment.toordinal() - day.toordinal()) for day in others))


def assemble_features(rows: list[dict], *, train_split: str = "train") -> dict:
    """Build the clipped feature matrix. Statistics are fit on training rows only.

    rows need split, subject, date, run_fraction, artifact_rms, bandpower, group_windows, base.
    """
    if not rows:
        raise ProtocolError("no rows to assemble")
    for row in rows:
        if row["split"] == train_split and int(row["subject"]) in EVAL_SUBJECTS + VAL_SUBJECTS:
            raise ProtocolError(
                f"subject {row['subject']} is untrained under split v2 and leaked into the noise fit"
            )
        for name in ("run_fraction", "artifact_rms", "bandpower", "date", "group_windows", "base"):
            if name not in row:
                raise ProtocolError(f"row is missing {name}")
    train_rows = [row for row in rows if row["split"] == train_split]
    if len(train_rows) < 2:
        raise ProtocolError("noise fit needs at least two training rows")
    train_days = sorted({_to_date(row["date"]) for row in train_rows})
    if len(train_days) < 2:
        raise ProtocolError("day_distance needs at least two distinct training acquisition dates")
    # group_windows is repeated on every word of a run. Count a base once.
    windows_on = {}
    seen_base = set()
    for row in train_rows:
        if row["base"] in seen_base:
            continue
        seen_base.add(row["base"])
        day = _to_date(row["date"])
        windows_on[day] = windows_on.get(day, 0) + int(row["group_windows"])
    total_train_windows = int(sum(windows_on.values()))
    raw = np.zeros((len(rows), len(ALLOWED_INFERENCE_FEATURES)), dtype=np.float64)
    for i, row in enumerate(rows):
        moment = _to_date(row["date"])
        if row["split"] == train_split:
            others = [day for day in train_days if day != moment]
            earlier = [day for day in train_days if day < moment]
            count = sum(windows_on[day] for day in earlier)
        else:
            others = list(train_days)
            count = total_train_windows
        raw[i, 0] = _rank_in_train(moment, train_days)
        raw[i, 1] = float(row["run_fraction"])
        raw[i, 2] = float(row["artifact_rms"])
        raw[i, 3] = float(row["bandpower"])
        raw[i, 4] = float(np.log1p(count))
        raw[i, 5] = _distance_days(moment, others)
    train_index = np.array([row["split"] == train_split for row in rows])
    clip = NumericClip().fit(raw[train_index])
    state = {
        "names": list(ALLOWED_INFERENCE_FEATURES),
        "train_days": [day.isoformat() for day in train_days],
        "windows_on": {day.isoformat(): int(count) for day, count in windows_on.items()},
        "total_train_windows": int(total_train_windows),
        "clip": clip.to_dict(),
        "train_bases": sorted(seen_base),
        "n_train_days": len(train_days),
    }
    return {
        "names": state["names"],
        "z": clip.transform(raw),
        "train_index": train_index,
        "clip": clip,
        "state": state,
        "n_train_days": len(train_days),
        "train_bases": state["train_bases"],
    }


def transform_with_state(rows: list[dict], state: dict) -> np.ndarray:
    """Apply a frozen training feature state. Does not refit on these rows."""
    train_days = sorted(_to_date(day) for day in state["train_days"])
    windows_on = {_to_date(day): int(count) for day, count in state["windows_on"].items()}
    total = int(state["total_train_windows"])
    train_bases = set(state["train_bases"])
    raw = np.zeros((len(rows), len(ALLOWED_INFERENCE_FEATURES)), dtype=np.float64)
    for i, row in enumerate(rows):
        moment = _to_date(row["date"])
        if row["base"] in train_bases:
            others = [day for day in train_days if day != moment]
            earlier = [day for day in train_days if day < moment]
            count = sum(windows_on[day] for day in earlier)
        else:
            others = list(train_days)
            count = total
        raw[i, 0] = _rank_in_train(moment, train_days)
        raw[i, 1] = float(row["run_fraction"])
        raw[i, 2] = float(row["artifact_rms"])
        raw[i, 3] = float(row["bandpower"])
        raw[i, 4] = float(np.log1p(count))
        raw[i, 5] = _distance_days(moment, others)
    return NumericClip.from_dict(state["clip"]).transform(raw)


def _design(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    return np.column_stack([np.ones(len(z)), z])


def _ridge(z: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = _design(z)
    pen = 1e-2 * len(z)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += pen
    gram[0, 0] -= pen
    return np.linalg.solve(gram, design.T @ target)


def fit_noise(z_train: np.ndarray, residual_train: np.ndarray, candidate: str, *, seed: int = 0, feature_names=None) -> dict:
    """Fit one candidate on training residuals only. Residual is per-word decoder cross-entropy."""
    if candidate not in CANDIDATES:
        raise ProtocolError(f"unknown candidate {candidate}")
    if feature_names is not None:
        assert_inference_features(feature_names)
    residual = np.asarray(residual_train, dtype=np.float64)
    z = np.asarray(z_train, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if len(residual) != len(z):
        raise ProtocolError("train residual and covariates differ in length")
    if len(residual) < 2:
        raise ProtocolError("noise fit needs at least two training rows")
    mu = float(residual.mean())
    centered = residual - mu
    energy = np.clip(centered ** 2, 1e-8, None)
    clip = NumericClip().fit(z)
    z_s = clip.transform(z)
    slope = np.zeros(z_s.shape[1])
    projection = None
    if candidate == "uniform":
        intercept = 0.0
    elif candidate == "constant":
        intercept = float(np.log(np.mean(energy)))
    else:
        design_z = z_s
        if candidate == "random_covariate":
            rng = np.random.default_rng(seed)
            projection = rng.normal(size=(z_s.shape[1], z_s.shape[1]))
            design_z = z_s @ projection
        beta = _ridge(design_z, np.log(energy))
        intercept = float(beta[0])
        slope = np.asarray(beta[1:], dtype=np.float64)
    return {
        "candidate": candidate,
        "intercept": float(np.clip(intercept, *LOG_VAR_CLIP)) if candidate != "uniform" else 0.0,
        "slope": slope,
        "mu": mu,
        "scaler": clip.to_dict(),
        "projection": None if projection is None else projection.tolist(),
        "train_rows": int(len(z)),
        "objective": "gaussian_nll",
        "feature_names": None if feature_names is None else list(feature_names),
    }


def _restore_scaler(params: dict) -> NumericClip:
    return NumericClip.from_dict(params["scaler"])


def log_variance(z: np.ndarray, params: dict) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if params["candidate"] == "uniform":
        return np.zeros(len(z))
    if params["candidate"] == "constant":
        return np.full(len(z), float(params["intercept"]))
    scaled = _restore_scaler(params).transform(z)
    if params["candidate"] == "random_covariate":
        scaled = scaled @ np.asarray(params["projection"], dtype=np.float64)
    lv = float(params["intercept"]) + scaled @ np.asarray(params["slope"], dtype=np.float64)
    return np.clip(lv, *LOG_VAR_CLIP)


def gaussian_nll(residual: np.ndarray, mu: float, log_var: np.ndarray) -> float:
    """One formula for every candidate, including uniform (log variance fixed at 0) and constant."""
    residual = np.asarray(residual, dtype=np.float64)
    lv = np.clip(np.asarray(log_var, dtype=np.float64), *LOG_VAR_CLIP)
    centered = residual - float(mu)
    return float(np.mean(0.5 * (np.log(2.0 * np.pi) + lv + centered ** 2 * np.exp(-lv))))


def weighted_nll(residual: np.ndarray, z: np.ndarray, params: dict) -> float:
    """Held-out Gaussian NLL. `mu` stays the training mean stored in params."""
    if "mu" not in params or params.get("objective") != "gaussian_nll":
        raise ProtocolError("candidate is missing a training residual mean or the shared objective")
    return gaussian_nll(residual, params["mu"], log_variance(z, params))


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
    means = np.asarray([float(values[session_ids == session].mean()) for session in sessions])
    rng = np.random.default_rng(seed)
    boots = np.sort([float(means[rng.integers(0, len(means), len(means))].mean()) for _ in range(draws)])
    return {
        "mean": float(means.mean()),
        "low": float(boots[int(0.025 * (draws - 1))]),
        "high": float(boots[int(0.975 * (draws - 1))]),
        "defined": True,
        "n_sessions": int(len(sessions)),
    }


def timing_inputs(rel: np.ndarray, meg: np.ndarray, mode: str, *, seed: int = 0):
    """Primary decoder timing is zero. Timing-only is a shortcut control, not a result.

    Zeros keep the incoming floating dtype. The model casts that dtype to its
    parameter dtype; MEG is not upcast here.
    """
    rel = np.array(rel, dtype=np.result_type(np.asarray(rel).dtype, np.float32), copy=True)
    meg = np.array(meg, copy=True)
    if mode == "zero":
        return np.zeros(rel.shape, dtype=rel.dtype), meg
    if mode == "shuffle":
        rng = np.random.default_rng(seed)
        flat = rel.ravel().copy()
        rng.shuffle(flat)
        return flat.reshape(rel.shape), meg
    if mode == "timing_only":
        return rel, np.zeros(meg.shape, dtype=meg.dtype)
    raise ProtocolError(mode)


def bacc10_50(top_hit_by_class: dict) -> float | None:
    """Mean over all 50 classes. Missing a class makes the competition-style score undefined."""
    if set(top_hit_by_class) != set(range(50)):
        return None
    return float(np.mean([top_hit_by_class[c] for c in range(50)]))


def bacc10_from_logits(logits: np.ndarray, labels: np.ndarray, *, n_classes: int = N_CLASSES, k: int = TOP_K):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[1] < k:
        raise ProtocolError("logits and labels do not form a top-10 score")
    present = set(int(c) for c in np.unique(labels))
    if present != set(range(n_classes)):
        return None
    chosen = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = []
    for cls in range(n_classes):
        rows = labels == cls
        hits.append(float((chosen[rows] == cls).any(axis=1).mean()))
    return float(np.mean(hits))


def row_temperature(logits: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """Scalar per-row temperature. This cannot change argmax or top-10 membership when there are no ties."""
    logits = np.asarray(logits, dtype=np.float64)
    scale = np.exp(0.5 * np.clip(np.asarray(log_var, dtype=np.float64), *LOG_VAR_CLIP))
    return logits / scale[:, None]


def topk_sets(logits: np.ndarray, k: int = TOP_K) -> list[set]:
    chosen = np.argpartition(-np.asarray(logits, dtype=np.float64), kth=k - 1, axis=1)[:, :k]
    return [set(int(c) for c in row) for row in chosen]


def temperature_changes_top10(logits: np.ndarray, log_var: np.ndarray) -> bool:
    return topk_sets(logits) != topk_sets(row_temperature(logits, log_var))


def quarter_labels(timechapter, t_max: float) -> np.ndarray:
    """Shared-stimulus quarters with the same window guard as src/cont.py Words.quarters.

    Q1 and Q2, and non-shared 'other', are what deep_c trained on. Q3 is the locked
    trunk-unseen block used to fit the noise equation. Empty string means the guard drop.
    """
    if t_max is None or float(t_max) <= 0:
        raise ProtocolError("stimulus duration is missing; refusing to invent quarters")
    t = np.asarray(timechapter, dtype=np.float64)
    q = np.clip((t / float(t_max) * 4.0).astype(int), 0, 3)
    out = np.array([f"Q{int(k) + 1}" for k in q], dtype=object)
    for boundary in (2, 3):
        edge = boundary * float(t_max) / 4.0
        bad = (t + QUARTER_POST > edge - QUARTER_GUARD) & (t - QUARTER_PRE < edge + QUARTER_GUARD)
        out[bad] = ""
    return out


def validation_stop(curve, schedule: dict) -> dict:
    """Pick a stop from the validation curve only. Eval and test keys are refused."""
    for row in curve:
        banned = set(row) & {"eval_ce", "test_ce", "eval", "test", "eval_bacc", "test_bacc"}
        if banned:
            raise ProtocolError(f"validation stop cannot see {sorted(banned)}")
    if not curve:
        return {"stop": False, "best_step": None, "reason": None, "checks_without_improvement": 0}
    best = min(curve, key=lambda row: (float(row["learned_ce"]), int(row["step"])))
    after = [row for row in curve if int(row["step"]) > int(best["step"])]
    last = int(curve[-1]["step"])
    stop = False
    reason = None
    patience = schedule.get("patience_checks")
    if last >= int(schedule["max_steps"]):
        stop = True
        reason = "max_steps"
    elif patience is not None and last >= int(schedule["min_steps"]) and len(after) >= int(patience):
        stop = True
        reason = "validation_plateau"
    return {
        "stop": stop,
        "best_step": int(best["step"]),
        "reason": reason,
        "checks_without_improvement": int(len(after)),
        "substantive": bool(schedule.get("substantive")),
    }


def coverage_report(logits, labels, subjects) -> dict:
    """Per-subject class counts. Balanced accuracy is eligible only when all 50 classes are present."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects)
    per_subject = {}
    for subject in sorted(set(int(s) for s in subjects)):
        mask = subjects == subject
        counts = {
            str(cls): int(np.sum(labels[mask] == cls))
            for cls in range(N_CLASSES)
            if np.any(labels[mask] == cls)
        }
        per_subject[str(subject)] = {
            "n": int(np.sum(mask)),
            "n_classes_present": len(counts),
            "missing_classes": [cls for cls in range(N_CLASSES) if str(cls) not in counts],
            "class_counts": counts,
            "bacc10_50": bacc10_from_logits(logits[mask], labels[mask]),
            "mean_cross_entropy": float(np.mean(decoder_residual(logits[mask], labels[mask]))),
        }
    present = sorted(set(int(c) for c in labels))
    eligible = ["mean_cross_entropy", "gaussian_nll_of_cross_entropy"]
    if present == list(range(N_CLASSES)):
        eligible.append("bacc10_50")
    return {
        "per_subject": per_subject,
        "pooled_n": int(len(labels)),
        "pooled_n_classes_present": len(present),
        "pooled_missing_classes": [cls for cls in range(N_CLASSES) if cls not in present],
        "eligible_metrics": eligible,
        "bacc10_50_eligible": "bacc10_50" in eligible,
        "gaussian_nll_note": (
            "Gaussian NLL of the frozen per-word cross-entropy compares calibration with the "
            "uniform and constant baselines. It is not evidence that a forecasting equation holds for MEG."
        ),
    }


def measurement_status(checks: dict) -> dict:
    """Engineering completeness is not a confirmatory claim.

    Historically inspected Broad subjects 3/6/9/12 stay exploratory. Fixtures that
    only satisfy row counts and a decoder name do not become a scientific claim.
    """
    try:
        candidates = tuple(checks.get("candidates") or ())
    except TypeError:
        candidates = ()
    complete = bool(
        checks.get("decoder") == "src.v3lib.Model"
        and checks.get("checkpoint_bytes") == CHECKPOINT_BYTES
        and checks.get("objective") == "gaussian_nll"
        and candidates == CANDIDATES
        and list(checks.get("eval_subjects_in_train") or []) == []
        and checks.get("subject0_frozen_holdout") is False
        and checks.get("temperature_applied_to_reported_logits") is False
        and checks.get("in_sample_trunk_trained_residual") is False
        and int(checks.get("eval_rows") or 0) > 0
        and int(checks.get("fit_rows") or 0) > 0
        and int(checks.get("selection_rows") or 0) > 0
        and checks.get("lock_held") is True
        and checks.get("fixture") is not True
    )
    return {
        "measurement_complete": complete,
        "valid_scientific_claim": False,
        "effect_claimed": False,
        "exploratory": True,
        "not_confirmatory": True,
        "independent_confirmation_required": True,
        "historically_inspected_broad_eval": list(EVAL_SUBJECTS),
    }


def measurement_claim(checks: dict) -> bool:
    """Retired name. Engineering checks must not read as a claim."""
    return False


def next_sessions(all_ids, done) -> list:
    done = set(done)
    return [item for item in all_ids if item not in done]


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def per_row_ce(logits: np.ndarray, labels: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    """Per-word cross-entropy. Smoothing 0 is the held-out residual. Training uses 0.1 to match train_v3."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ProtocolError("label is outside the decoder classes")
    logp = np.log(np.clip(softmax_rows(logits), 1e-12, 1.0))
    n_classes = logits.shape[1]
    trust = np.full_like(logits, smoothing / n_classes)
    trust[np.arange(len(labels)), labels] += 1.0 - smoothing
    return -(trust * logp).sum(axis=1)


def decoder_residual(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Frozen-decoder residual used by every noise candidate. Unsmoothed cross-entropy, in nats."""
    return per_row_ce(logits, labels, smoothing=0.0)


def dlogits_ce(logits: np.ndarray, labels: np.ndarray, *, smoothing: float = 0.1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = logits.shape[1]
    trust = np.full_like(logits, smoothing / n_classes)
    trust[np.arange(len(labels)), labels] += 1.0 - smoothing
    return softmax_rows(logits) - trust


def step_head(weight, bias, embeddings, labels, log_var, lr: float):
    """One heteroscedastic step. Precision exp(-log var) multiplies the logit gradient, so the head can move."""
    logits = embeddings @ weight + bias
    precision = np.exp(-np.clip(log_var, *LOG_VAR_CLIP))
    grad = precision[:, None] * dlogits_ce(logits, labels, smoothing=0.1)
    scale = 1.0 / max(len(labels), 1)
    return weight - lr * (embeddings.T @ grad) * scale, bias - lr * grad.mean(axis=0)


def step_logvar_linear(intercept: float, slope: np.ndarray, features: np.ndarray, ce: np.ndarray, lr: float):
    """Gradient of mean(exp(-s) * CE + s) for s = intercept + features @ slope."""
    score = np.clip(intercept + features @ slope, *LOG_VAR_CLIP)
    grad_s = 1.0 - ce * np.exp(-score)
    scale = 1.0 / max(len(ce), 1)
    return float(intercept - lr * grad_s.mean()), slope - lr * (features.T @ grad_s) * scale


def initial_partial_arms(weight: np.ndarray, bias: np.ndarray, n_features: int) -> dict:
    arms = {}
    for name in PARTIAL_ARMS:
        arms[name] = {
            "W": np.array(weight, dtype=np.float64, copy=True),
            "b": np.array(bias, dtype=np.float64, copy=True),
            "intercept": 0.0,
            "slope": np.zeros(n_features, dtype=np.float64),
        }
    return arms


def matched_partial_steps(arms: dict, embeddings, labels, features, steps: int, *, lr: float = PARTIAL_LR, seed: int = 0, batch: int = 32, frozen_log_var=None) -> dict:
    """Same minibatches and the same step count on every arm. Uniform is the unweighted head. Random matches capacity."""
    if set(arms) != set(PARTIAL_ARMS):
        raise ProtocolError("partial train must step uniform, random_covariate, and learned_state together")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    features = np.asarray(features, dtype=np.float64)
    if not (len(embeddings) == len(labels) == len(features)):
        raise ProtocolError("partial-train tensors differ in length")
    if steps < 1:
        raise ProtocolError("partial train needs at least one step")
    rng = np.random.default_rng(seed)
    projection = np.random.default_rng(seed + 1).normal(size=(features.shape[1], features.shape[1]))
    for _ in range(int(steps)):
        take = rng.integers(0, len(labels), size=min(int(batch), len(labels)))
        emb = embeddings[take]
        y = labels[take]
        feat = features[take]
        rand = feat @ projection
        for name in PARTIAL_ARMS:
            arm = arms[name]
            if frozen_log_var is not None:
                log_var = np.zeros(len(y)) if name == "uniform" else np.asarray(frozen_log_var[name], dtype=np.float64)[take]
                used = None
            elif name == "uniform":
                log_var = np.zeros(len(y))
                used = None
            elif name == "random_covariate":
                used = rand
                log_var = arm["intercept"] + used @ arm["slope"]
            else:
                used = feat
                log_var = arm["intercept"] + used @ arm["slope"]
            ce = per_row_ce(emb @ arm["W"] + arm["b"], y, smoothing=0.1)
            arm["W"], arm["b"] = step_head(arm["W"], arm["b"], emb, y, log_var, lr)
            if used is not None:
                arm["intercept"], arm["slope"] = step_logvar_linear(arm["intercept"], arm["slope"], used, ce, lr)
    return arms


def arm_logits(arm: dict, embeddings: np.ndarray) -> np.ndarray:
    return np.asarray(embeddings, dtype=np.float64) @ arm["W"] + arm["b"]
