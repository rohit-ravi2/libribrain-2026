"""Train-only noise equations for the v3 decoder. Amendment 1.4.

Forecasting papers motivate candidates. Their equations are not assumed to hold
for MEG. Every candidate is scored with the same Gaussian negative log-likelihood
on a defined decoder residual: the frozen network's per-word cross-entropy.
A scalar per-row temperature is not an accuracy result. Subject-0 days are not
a frozen-trunk holdout.

Amendment 1.4.2: a row with no BIDS acquisition date does not receive an invented
day_rank or day_distance. Those two terms are omitted. Subject 0 still requires
a scans.tsv date. See AMENDMENT_1.4.2.md.

Amendment 1.4.3: if the training rows contain fewer than two distinct acquisition
dates, day_rank and day_distance are inestimable and leave the equation. The
other features stay. Dates are not invented. See AMENDMENT_1.4.3.md.

Amendment 1.4.4 does not change this equation, the estimand, or the caps.
The runner merges the honesty mappings that score-frozen and later stages
pass together. A claim flag that disagrees keeps the value that does not
advance a claim. Any other disagreement raises ProtocolError.
See AMENDMENT_1.4.4.md.

Amendment 1.4.5 does not refit this equation. partial-train, score-partial,
and timing-control read the day-term decision recorded here. They do not
count acquisition dates again. A kept feature that is non-finite still fails.
See AMENDMENT_1.4.5.md.
"""
from __future__ import annotations

from datetime import date as Date

import numpy as np

AMENDMENT = "1.4"
AMENDMENT_142 = "1.4.2"
AMENDMENT_143 = "1.4.3"
AMENDMENT_144 = "1.4.4"
AMENDMENT_145 = "1.4.5"
PROTOCOL_ID = "strong-decoder-noise-1.4"
# Pre-specified before any 1.4.2 result. Undated rows omit these terms.
# They are not filled with a session ordinal or a calendar day.
UNDATED_DAY_TERMS = "dropped"
# Pre-specified before any 1.4.3 result. One training date cannot identify a day term.
DAY_TERMS_WHEN_INESTIMABLE = "dropped"
DATE_SOURCE = "bids_scans_tsv_only"
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
DAY_TERMS = ("day_rank", "day_distance")
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


def missing_acquisition_date(base: str) -> str:
    return f"no acquisition date for {base}; refusing to invent day_rank"


def _to_date(text: str) -> Date:
    try:
        year, month, day = (int(part) for part in str(text).split("-"))
        return Date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"acquisition date {text!r} is not YYYY-MM-DD") from exc


def _date_text(row: dict) -> str | None:
    if "date" not in row or row["date"] is None:
        return None
    text = str(row["date"]).strip()
    return text or None


def _day_term_index() -> list[int]:
    return [ALLOWED_INFERENCE_FEATURES.index(name) for name in DAY_TERMS]


def drop_undefined_day_terms(z: np.ndarray, defined: np.ndarray) -> np.ndarray:
    """Omit day_rank and day_distance where no acquisition date exists.

    The matrix is already centered on training rows, so a zero contribution is
    the omitted term. It is not a calendar day, a mean imputed date, or a
    BIDS session ordinal.
    """
    out = np.array(z, dtype=np.float64, copy=True)
    defined = np.asarray(defined, dtype=bool)
    if defined.shape != (len(out),):
        raise ProtocolError("day-term mask does not match the rows")
    missing = ~defined
    if np.any(missing):
        out[np.ix_(missing, _day_term_index())] = 0.0
    assert_finite_used_features(out, ALLOWED_INFERENCE_FEATURES)
    return out


def _use_or_drop_date(row: dict, *, coefficient: bool) -> str | None:
    """Return a YYYY-MM-DD, or None when day terms must be dropped.

    Coefficient rows and subject 0 always need a real scans.tsv date.
    """
    text = _date_text(row)
    if text is not None:
        return text
    subject = row.get("subject")
    if coefficient or subject is None or int(subject) == 0:
        raise ProtocolError(missing_acquisition_date(str(row["base"])))
    return None


def _log_window_count(moment: Date | None, *, in_train: bool, train_days, windows_on, total_train_windows) -> float:
    """Windows on earlier training days, or the training total for a non-training row.

    A single training date has no earlier day, so every training row gets log1p(0).
    That column stays in the equation. It is not a day rank and not a day distance.
    """
    if in_train:
        if moment is None:
            raise ProtocolError("training row is missing the date required to count earlier windows")
        count = sum(windows_on[day] for day in train_days if day < moment)
    else:
        count = total_train_windows
    return float(np.log1p(count))


def _day_feature_values(moment: Date, *, in_train: bool, train_days, windows_on, total_train_windows):
    if in_train:
        others = [day for day in train_days if day != moment]
    else:
        others = list(train_days)
    return (
        _rank_in_train(moment, train_days),
        _log_window_count(
            moment,
            in_train=in_train,
            train_days=train_days,
            windows_on=windows_on,
            total_train_windows=total_train_windows,
        ),
        _distance_days(moment, others),
    )


def _rank_in_train(moment: Date, train_days: list[Date]) -> float:
    ords = [day.toordinal() for day in train_days]
    raw = int(np.searchsorted(np.asarray(ords), moment.toordinal()))
    return float(np.clip(raw, 0, len(train_days) - 1))


def _distance_days(moment: Date, others: list[Date]) -> float:
    if not others:
        raise ProtocolError("day distance needs another training date")
    return float(min(abs(moment.toordinal() - day.toordinal()) for day in others))


def _fill_feature_row(raw, defined, index, row, *, in_train: bool, coefficient: bool, train_days, windows_on, total_train_windows) -> None:
    for name in ("run_fraction", "artifact_rms", "bandpower", "base"):
        if name not in row:
            raise ProtocolError(f"row is missing {name}")
    raw[index, 1] = float(row["run_fraction"])
    raw[index, 2] = float(row["artifact_rms"])
    raw[index, 3] = float(row["bandpower"])
    text = _use_or_drop_date(row, coefficient=coefficient or in_train)
    if text is None:
        raw[index, 0] = np.nan
        raw[index, 4] = float(np.log1p(total_train_windows))
        raw[index, 5] = np.nan
        defined[index] = False
        return
    defined[index] = True
    rank, log_count, distance = _day_feature_values(
        _to_date(text),
        in_train=in_train,
        train_days=train_days,
        windows_on=windows_on,
        total_train_windows=total_train_windows,
    )
    raw[index, 0] = rank
    raw[index, 4] = log_count
    raw[index, 5] = distance


def _active_feature_names(day_terms_inestimable: bool) -> list[str]:
    if not day_terms_inestimable:
        return list(ALLOWED_INFERENCE_FEATURES)
    return [name for name in ALLOWED_INFERENCE_FEATURES if name not in DAY_TERMS]


def _reduced_raw(rows, in_train_flags, *, train_days, windows_on, total_train_windows) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix with day_rank and day_distance omitted.

    The omission was recorded when the training rows had fewer than two
    acquisition dates. This does not read a date to rebuild those columns.
    A date string that is present must still be YYYY-MM-DD, so a session
    ordinal cannot sit in the date field. It is not used as a feature.
    One training date has no earlier day, so an in-training row's
    log_train_window_count is log1p(0) without consulting that row's date.
    """
    del train_days, windows_on
    names = _active_feature_names(True)
    columns = {name: index for index, name in enumerate(names)}
    raw = np.zeros((len(rows), len(names)), dtype=np.float64)
    defined = np.zeros(len(rows), dtype=bool)
    outside = float(np.log1p(total_train_windows))
    for index, row in enumerate(rows):
        for name in ("run_fraction", "artifact_rms", "bandpower", "base", "subject"):
            if name not in row:
                raise ProtocolError(f"row is missing {name}")
        text = _date_text(row)
        if text is not None:
            _to_date(text)
        defined[index] = text is not None
        raw[index, columns["run_fraction"]] = float(row["run_fraction"])
        raw[index, columns["artifact_rms"]] = float(row["artifact_rms"])
        raw[index, columns["bandpower"]] = float(row["bandpower"])
        raw[index, columns["log_train_window_count"]] = 0.0 if in_train_flags[index] else outside
    return raw, defined


def assemble_features(rows: list[dict], *, train_split: str = "train") -> dict:
    """Build the clipped feature matrix. Statistics are fit on training rows only.

    Training rows need a real acquisition date. A non-training row may omit it.
    Amendment 1.4.2 then drops day_rank and day_distance for that row.
    Amendment 1.4.3 drops those two terms from the whole equation when the
    training rows contain fewer than two distinct acquisition dates.
    """
    if not rows:
        raise ProtocolError("no rows to assemble")
    for row in rows:
        if row["split"] == train_split and int(row["subject"]) in EVAL_SUBJECTS + VAL_SUBJECTS:
            raise ProtocolError(
                f"subject {row['subject']} is untrained under split v2 and leaked into the noise fit"
            )
        for name in ("run_fraction", "artifact_rms", "bandpower", "group_windows", "base", "subject"):
            if name not in row:
                raise ProtocolError(f"row is missing {name}")
    train_rows = [row for row in rows if row["split"] == train_split]
    if len(train_rows) < 2:
        raise ProtocolError("noise fit needs at least two training rows")
    for row in train_rows:
        if _date_text(row) is None:
            raise ProtocolError(missing_acquisition_date(str(row["base"])))
    train_days = sorted({_to_date(_date_text(row)) for row in train_rows})
    # One calendar day cannot identify day_rank or day_distance. Drop the terms.
    # Do not split a day by clock time, and do not borrow a non-training date.
    day_terms_inestimable = len(train_days) < 2
    day_terms_dropped = list(DAY_TERMS) if day_terms_inestimable else []
    # group_windows is repeated on every word of a run. Count a base once.
    windows_on = {}
    seen_base = set()
    for row in train_rows:
        if row["base"] in seen_base:
            continue
        seen_base.add(row["base"])
        day = _to_date(_date_text(row))
        windows_on[day] = windows_on.get(day, 0) + int(row["group_windows"])
    total_train_windows = int(sum(windows_on.values()))
    names = _active_feature_names(day_terms_inestimable)
    if day_terms_inestimable:
        raw, defined = _reduced_raw(
            rows,
            [row["split"] == train_split for row in rows],
            train_days=train_days,
            windows_on=windows_on,
            total_train_windows=total_train_windows,
        )
    else:
        raw = np.zeros((len(rows), len(ALLOWED_INFERENCE_FEATURES)), dtype=np.float64)
        defined = np.zeros(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            _fill_feature_row(
                raw, defined, index, row,
                in_train=row["split"] == train_split,
                coefficient=row["split"] == train_split,
                train_days=train_days,
                windows_on=windows_on,
                total_train_windows=total_train_windows,
            )
    train_index = np.array([row["split"] == train_split for row in rows])
    clip = NumericClip().fit(raw[train_index])
    z = clip.transform(raw)
    if day_terms_inestimable:
        assert_finite_used_features(z, names)
    else:
        z = drop_undefined_day_terms(z, defined)
    state = {
        "names": names,
        "train_days": [day.isoformat() for day in train_days],
        "windows_on": {day.isoformat(): int(count) for day, count in windows_on.items()},
        "total_train_windows": int(total_train_windows),
        "clip": clip.to_dict(),
        "train_bases": sorted(seen_base),
        "n_train_days": len(train_days),
        "day_terms": list(DAY_TERMS),
        "day_terms_inestimable": bool(day_terms_inestimable),
        "day_terms_dropped": day_terms_dropped,
        "day_terms_when_inestimable": DAY_TERMS_WHEN_INESTIMABLE,
        "undated_day_terms": UNDATED_DAY_TERMS,
        "amendment_1_4_2": AMENDMENT_142,
        "amendment_1_4_3": AMENDMENT_143,
        "date_source": DATE_SOURCE,
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
    }
    return {
        "names": state["names"],
        "z": z,
        "day_defined": defined,
        "train_index": train_index,
        "clip": clip,
        "state": state,
        "n_train_days": len(train_days),
        "train_bases": state["train_bases"],
        "day_terms_inestimable": bool(day_terms_inestimable),
        "day_terms_dropped": list(day_terms_dropped),
    }


def _recorded_bool(value, label: str) -> bool:
    if type(value) is bool or isinstance(value, np.bool_):
        return bool(value)
    raise ProtocolError(f"{label} is not a boolean")


def assert_finite_used_features(values: np.ndarray, names) -> None:
    """Refuse a non-finite value in a column that is actually in the matrix.

    A dropped day term is not a column. A non-finite day column that is still
    present is a refused invention. Any other non-finite column is named.
    """
    matrix = np.asarray(values, dtype=np.float64)
    names = list(names)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(names):
        raise ProtocolError(f"feature width {matrix.shape} != {len(names)}")
    for index, name in enumerate(names):
        if np.isfinite(matrix[:, index]).all():
            continue
        if name in DAY_TERMS:
            raise ProtocolError("noise features are non-finite; refusing to invent a day")
        raise ProtocolError(f"used feature {name} is non-finite")


def frozen_day_term_decision(fit: dict) -> dict:
    """Return the day-term decision stored on the noise fit.

    Does not look at the rows being scored and does not count their dates.
    """
    if not isinstance(fit, dict):
        raise ProtocolError("noise fit did not record a day-term decision")
    nested = fit.get("state") if isinstance(fit.get("state"), dict) else None
    if nested is not None and "day_terms_inestimable" in fit and "day_terms_inestimable" in nested:
        if _recorded_bool(fit["day_terms_inestimable"], "day_terms_inestimable") != _recorded_bool(
            nested["day_terms_inestimable"], "day_terms_inestimable"
        ):
            raise ProtocolError("frozen day-term decision disagrees with the noise fit")
        if set(fit.get("day_terms_dropped") or []) != set(nested.get("day_terms_dropped") or []):
            raise ProtocolError("frozen day-term decision disagrees with the noise fit")
    source = nested if nested is not None and "day_terms_inestimable" in nested else fit
    if "day_terms_inestimable" not in source:
        raise ProtocolError("noise fit did not record a day-term decision")
    inestimable = _recorded_bool(source["day_terms_inestimable"], "day_terms_inestimable")
    dropped = list(source.get("day_terms_dropped") or [])
    names = list(source.get("names") or [])
    if not names:
        raise ProtocolError("frozen day-term decision has no feature names")
    if inestimable:
        if set(dropped) != set(DAY_TERMS):
            raise ProtocolError("frozen day-term drop does not match the recorded decision")
        if any(name in DAY_TERMS for name in names):
            raise ProtocolError("frozen day-term drop still lists a day column")
        recorded_days = source.get("train_days")
        if recorded_days is not None and len(recorded_days) >= 2:
            raise ProtocolError("frozen day-term drop does not match the recorded training dates")
    elif dropped:
        raise ProtocolError("frozen day-term decision keeps day terms and also drops them")
    return {
        "day_terms_inestimable": inestimable,
        "day_terms_dropped": list(DAY_TERMS) if inestimable else [],
        "active_features": names,
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
    }


def transform_with_state(rows: list[dict], state: dict) -> np.ndarray:
    """Apply a frozen training feature state. Does not refit on these rows.

    A non-subject-0 row with no date keeps the non-day features and drops
    day_rank and day_distance. Subject 0 and coefficient bases still refuse
    a missing date when those terms are in the equation. A state that recorded
    day_terms_inestimable keeps those terms out for every row. That recorded
    flag is not recomputed from the dates on these rows.
    """
    decision = frozen_day_term_decision(state)
    if decision["day_terms_inestimable"]:
        train_days = sorted(_to_date(day) for day in state["train_days"])
        windows_on = {_to_date(day): int(count) for day, count in state["windows_on"].items()}
        total = int(state["total_train_windows"])
        train_bases = set(state["train_bases"])
        raw, _defined = _reduced_raw(
            rows,
            [row["base"] in train_bases for row in rows],
            train_days=train_days,
            windows_on=windows_on,
            total_train_windows=total,
        )
        if raw.shape[1] != len(state["names"]) or list(state["names"]) != list(decision["active_features"]):
            raise ProtocolError("frozen feature width does not match the noise fit")
        transformed = NumericClip.from_dict(state["clip"]).transform(raw)
        assert_finite_used_features(transformed, state["names"])
        return transformed
    train_days = sorted(_to_date(day) for day in state["train_days"])
    windows_on = {_to_date(day): int(count) for day, count in state["windows_on"].items()}
    total = int(state["total_train_windows"])
    train_bases = set(state["train_bases"])
    raw = np.zeros((len(rows), len(ALLOWED_INFERENCE_FEATURES)), dtype=np.float64)
    defined = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        _fill_feature_row(
            raw, defined, index, row,
            in_train=row["base"] in train_bases,
            coefficient=row["base"] in train_bases,
            train_days=train_days,
            windows_on=windows_on,
            total_train_windows=total,
        )
    return drop_undefined_day_terms(NumericClip.from_dict(state["clip"]).transform(raw), defined)


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
    """Unweighted mean of per-class top-k recall. Undefined unless every class is present.

    A row-weighted hit rate tracks word frequency and is not this metric.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[1] < k:
        raise ProtocolError("logits and labels do not form a top-10 score")
    if len(labels) and (
        np.any(labels < 0) or np.any(labels >= n_classes) or np.any(labels >= logits.shape[1])
    ):
        raise ProtocolError("label is outside the decoder classes")
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


def in_vocab_mask(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """True where the label is a decoder class. Out-of-range ids are not clipped into range."""
    labels = np.asarray(labels, dtype=np.int64)
    return (labels >= 0) & (labels < int(n_classes))


def oov_record(base: str, labels: np.ndarray, *, used: int | None = None, n_classes: int = N_CLASSES) -> dict:
    """How many rows of one recording sit outside the decoder vocabulary.

    excluded_oov counts the whole file. used is how many rows a stage kept after
    that filter and any quarter filter. The two are equal when no quarter filter runs.
    """
    labels = np.asarray(labels, dtype=np.int64)
    mask = in_vocab_mask(labels, n_classes)
    record = {
        "base": str(base),
        "rows": int(labels.shape[0]),
        "excluded_oov": int(np.sum(~mask)),
        "in_vocab": int(np.sum(mask)),
    }
    if used is not None:
        record["used"] = int(used)
    return record


def vocab_view(base: str, labels, arrays: dict | None = None, *, extra_keep=None, n_classes: int = N_CLASSES) -> dict:
    """Slice every array with the same in-vocabulary mask.

    extra_keep is a further boolean mask of the same length, used for the Q3
    noise-fit rows. Labels outside 0..n_classes-1 are dropped, never clipped.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ProtocolError(f"{base} labels are not one id per row")
    mask = in_vocab_mask(labels, n_classes)
    if extra_keep is not None:
        extra = np.asarray(extra_keep, dtype=bool)
        if extra.shape != mask.shape:
            raise ProtocolError(f"{base} keep mask does not match the rows")
        mask = mask & extra
    sliced = {}
    for key, value in (arrays or {}).items():
        array = np.asarray(value)
        if len(array) != len(labels):
            raise ProtocolError(f"{base} {key} does not match the rows")
        sliced[key] = array[mask]
    return {
        "labels": labels[mask],
        "arrays": sliced,
        "mask": mask,
        "oov": oov_record(base, labels, used=int(np.sum(mask)), n_classes=n_classes),
    }


def per_row_ce(logits: np.ndarray, labels: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    """Per-word cross-entropy. Smoothing 0 is the held-out residual. Training uses 0.1 to match train_v3.

    Does not drop or clip labels. Callers that read a forward cache apply vocab_view first.
    A label that still arrives out of range is a protocol error.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) and (np.any(labels < 0) or np.any(labels >= logits.shape[1])):
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
    if len(labels) and (np.any(labels < 0) or np.any(labels >= logits.shape[1])):
        raise ProtocolError("label is outside the decoder classes")
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
