"""Amendment 1.5. A powered noise-equation test. Protocol parent stays 1.4.

The 1.4 candidate list, the 1.4 output tree, and the 1.4 fit population are
unchanged. This module declares a larger trunk-unseen coefficient pool, a
primary balanced top-10 that stays defined when some classes are absent, a
paired bootstrap against the matched controls, and one regularised
one-component arm. Dates are not invented. Session tokens are not days.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

from .strong_noise import (
    CANDIDATES,
    ERROR_SCALING_FEATURES,
    EVAL_SUBJECTS,
    LOG_VAR_CLIP,
    N_CLASSES,
    PARTIAL_LR,
    SEEN_SUBJECTS,
    TOP_K,
    VAL_SUBJECTS,
    ProtocolError,
    _design,
    assemble_features,
    bacc10_from_logits,
    fit_noise,
    gaussian_nll,
    log_variance,
    step_head,
)

AMENDMENT_15 = "1.5"
# Launcher and inner loop share this. None means the scope has no RuntimeMaxSec
# and the chunk loop has no monotonic deadline. A finite scope runtime disagrees.
RUNTIME_CEILING_SEC = None
INNER_CHUNK_DEADLINE_SEC = None
TIMER_SENTINEL_LAUNCHER = "runtime_ceiling_sec=none"
TIMER_SENTINEL_INNER = "inner_chunk_deadline_sec=none"

# Normal quantiles. Phi^{-1}(0.975) and Phi^{-1}(0.80). No scipy dependency.
Z_ALPHA_975 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
# Smallest standardized log-energy slope the fit is sized to detect.
SLOPE_BETA = 0.15
# learned_state after the day-term drop: intercept plus the four remaining covariates.
LEARNED_STATE_PARAMETERS = 5
REDUCED_STATE_PARAMETERS = 2
# 1.4 ridge is 0.01 * n. The reduced arm is a hundred times stronger, on one component.
LEARNED_RIDGE_PER_ROW = 1e-2
REDUCED_RIDGE_PER_ROW = 1.0
DEAD_COLUMN_STD = 1e-8

# Equivalence margins, declared before any 1.5 score. A contrast that sits
# inside the margin is a resolved null. A contrast that excludes zero is a
# resolved difference. An interval that does neither is inconclusive.
BACC_EQUIVALENCE_MARGIN = 0.02
NLL_EQUIVALENCE_MARGIN = 0.05
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 1505
SUPPORT_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MIN_CLASSES_FOR_PRIMARY = 10
PRIMARY_METRIC = "bacc10_present"
FIFTY_CLASS_METRIC = "bacc10_50"
RECORDINGS_PER_LAUNCH = 1

# Coefficient rows: trunk-unseen shared Q3. Subject 0 and the seen subjects were
# trained on Q1-Q2 only. Subjects 13-24 are the same split. Validation and the
# original eval subjects stay out of this list.
FIT_SUBJECTS = (0,) + tuple(SEEN_SUBJECTS) + tuple(range(13, 25))
# Subjects 25-32 were not in the 1.4 eval set. Their Q1-Q2 windows were in
# train_mask, so only Q3 and Q4 are scored. They are not coefficient rows.
EVAL_EXTENSION_SUBJECTS = tuple(range(25, 33))
EVAL_EXTENSION_QUARTERS = ("Q3", "Q4")
if set(FIT_SUBJECTS) & (set(EVAL_SUBJECTS) | set(VAL_SUBJECTS) | set(EVAL_EXTENSION_SUBJECTS)):
    raise ProtocolError("fit pool overlaps a held-out subject list")
if set(EVAL_SUBJECTS) & set(EVAL_EXTENSION_SUBJECTS):
    raise ProtocolError("eval lists overlap")
FIT_QUARTER = "Q3"
HELD_OUT_QUARTER = "Q4"
CONTROLS = ("uniform", "constant", "random_covariate")
STRUCTURED_ARMS = ("learned_state", "reduced_state")
CANDIDATES_15 = tuple(CANDIDATES) + ("reduced_state",)
PARTIAL_ARMS_15 = ("uniform", "constant", "random_covariate", "learned_state", "reduced_state")
PARTIAL_SCHEDULE_15 = {
    "name": "substantive_head_finetune_1_5",
    "max_steps": 2000,
    "steps_per_launch": 100,
    "min_steps": 200,
    "patience_checks": 3,
    "substantive": True,
    "stop_on": "mean_validation_ce_of_uniform_constant_random_covariate",
}
HEAD_TRAIN_RUNS = 16

# Planning description of the 1.4 eval panel. Not a threshold fitted to scores.
# Four exploratory subjects, identical stimulus counts, 48 of 50 classes.
PLANNING_EVAL_SUBJECTS = 4
PLANNING_EXTENSION_SUBJECTS = len(EVAL_EXTENSION_SUBJECTS)
PLANNING_EXTENSION_ROW_FRACTION = 0.5


def _z_sum() -> float:
    return float(Z_ALPHA_975 + Z_POWER_80)


def rows_for_one_slope(beta: float = SLOPE_BETA) -> int:
    """Rows for 80% power at two-sided 5% to detect one standardized slope.

    The covariate is standardized and the residual SD of log energy is taken
    to be 1, so beta is on the correlation scale. This is a design assumption,
    not an estimate from the 1.4 residuals.
    """
    if beta <= 0:
        raise ProtocolError("slope beta must be positive")
    return int(math.ceil((_z_sum() ** 2) / float(beta) ** 2))


def fit_row_floor(parameters: int = LEARNED_STATE_PARAMETERS, beta: float = SLOPE_BETA) -> int:
    """Give each declared parameter its own one-slope sample."""
    if int(parameters) < 1:
        raise ProtocolError("parameter count must be positive")
    return int(rows_for_one_slope(beta) * int(parameters))


FIT_ROW_FLOOR = fit_row_floor()
REDUCED_ROW_FLOOR = fit_row_floor(REDUCED_STATE_PARAMETERS)


def power_report() -> dict:
    one = rows_for_one_slope()
    return {
        "alpha_two_sided": 0.05,
        "power": 0.80,
        "z_alpha_975": Z_ALPHA_975,
        "z_power_80": Z_POWER_80,
        "standardized_slope": SLOPE_BETA,
        "residual_sd_assumption": 1.0,
        "rows_for_one_slope": one,
        "learned_state_parameters": LEARNED_STATE_PARAMETERS,
        "learned_state_row_floor": FIT_ROW_FLOOR,
        "reduced_state_parameters": REDUCED_STATE_PARAMETERS,
        "reduced_state_row_floor": REDUCED_ROW_FLOOR,
        "rows_observed_in_1_4": 680,
        "rows_per_parameter_observed_in_1_4": 680 / LEARNED_STATE_PARAMETERS,
        "rows_per_parameter_required": one,
        "observed_1_4_fit_meets_learned_floor": 680 >= FIT_ROW_FLOOR,
        "observed_1_4_fit_meets_reduced_floor": 680 >= REDUCED_ROW_FLOOR,
        "bacc_equivalence_margin": BACC_EQUIVALENCE_MARGIN,
        "nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
        "planning_note": (
            "The 1.4 coefficient sample was 680 subject-0 Q3 rows, 136 rows for each of "
            "5 learned_state parameters. The one-slope calculation asks for "
            f"{one} rows per parameter, so the learned arm needs {FIT_ROW_FLOOR} rows and "
            f"the one-component arm needs {REDUCED_ROW_FLOOR}. Both floors are above 680. "
            "The fit list below is the pre-registered sample. It is not stopped early, "
            "and it is not refit on the same 680 rows."
        ),
    }


def minimum_detectable_bacc_difference(class_counts: dict, min_count: int) -> dict:
    """Conservative paired difference of a balanced top-10.

    Each class recall has variance at most 1/(4 n). Arms are treated as
    independent, which widens the difference. The live interval is the paired
    bootstrap; this bound only chooses the support floor and states power.
    """
    counts = [int(n) for n in class_counts.values() if int(n) >= int(min_count) and int(n) > 0]
    if not counts:
        return {"min_count": int(min_count), "n_classes": 0, "mdd": None, "variance_bound": None}
    variance = (1.0 / (len(counts) ** 2)) * sum(0.25 / float(n) for n in counts)
    difference_variance = 2.0 * variance
    mdd = float(math.sqrt(difference_variance) * _z_sum())
    return {
        "min_count": int(min_count),
        "n_classes": int(len(counts)),
        "variance_bound": float(difference_variance),
        "mdd": mdd,
        "powered_for_margin": bool(mdd <= BACC_EQUIVALENCE_MARGIN),
    }


def choose_support(class_counts: dict) -> dict:
    """Smallest grid point whose conservative MDD is within the margin.

    Uses label counts only. Hit rates are not an input. If no grid point with
    at least 10 classes meets the margin, keep the smallest MDD on that grid
    and record that the accuracy contrast is not powered.
    """
    best = None
    for min_count in SUPPORT_GRID:
        info = minimum_detectable_bacc_difference(class_counts, min_count)
        if info["n_classes"] < MIN_CLASSES_FOR_PRIMARY:
            continue
        if info["powered_for_margin"]:
            info["powered"] = True
            info["rule"] = "smallest_grid_point_with_mdd_at_most_margin_and_at_least_10_classes"
            info["grid"] = list(SUPPORT_GRID)
            return info
        if best is None or float(info["mdd"]) < float(best["mdd"]):
            best = info
    if best is None:
        raise ProtocolError("eval labels do not contain 10 classes at any support grid point")
    best["powered"] = False
    best["rule"] = "smallest_mdd_on_the_grid_among_points_with_at_least_10_classes"
    best["grid"] = list(SUPPORT_GRID)
    return best


def class_counts(labels: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    return {int(cls): int(np.sum(labels == cls)) for cls in range(n_classes) if np.any(labels == cls)}


def metric_declaration(labels: np.ndarray) -> dict:
    """Freeze the primary metric from labels before any hit rate is computed."""
    counts = class_counts(labels)
    support = choose_support(counts)
    present = sorted(counts)
    return {
        "primary_metric": PRIMARY_METRIC,
        "k": TOP_K,
        "aggregation": "unweighted_mean_of_per_class_top_k_recall",
        "min_class_count": int(support["min_count"]),
        "support_rule": support["rule"],
        "support_powered": bool(support["powered"]),
        "support_mdd": support["mdd"],
        "n_classes_in_primary": int(support["n_classes"]),
        "fifty_class_metric": FIFTY_CLASS_METRIC,
        "fifty_class_reported_when_defined": True,
        "classes_present": present,
        "classes_absent": [cls for cls in range(N_CLASSES) if cls not in counts],
        "declared_before_scoring": True,
        "hit_rates_used_to_choose_support": False,
        "equivalence_margin": BACC_EQUIVALENCE_MARGIN,
    }


def bacc10_present(logits: np.ndarray, labels: np.ndarray, *, min_count: int, n_classes: int = N_CLASSES, k: int = TOP_K):
    """Balanced top-k over classes with at least min_count rows. None if none qualify.

    The 50-class metric is bacc10_from_logits. This function does not replace it.
    A missing class has count 0 and stays out of the mean, so the score is defined
    when the 50-class score is not.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[1] < k:
        raise ProtocolError("logits and labels do not form a top-10 score")
    if len(labels) and (
        np.any(labels < 0) or np.any(labels >= n_classes) or np.any(labels >= logits.shape[1])
    ):
        raise ProtocolError("label is outside the decoder classes")
    if int(min_count) < 1:
        raise ProtocolError("min_class_count must be at least 1")
    chosen = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = []
    used = []
    for cls in range(n_classes):
        rows = labels == cls
        count = int(np.sum(rows))
        if count < int(min_count):
            continue
        hits.append(float((chosen[rows] == cls).any(axis=1).mean()))
        used.append(cls)
    if not hits:
        return None
    return float(np.mean(hits))


def parse_base(base: str) -> dict:
    """Split a BIDS recording id. The session token is a label, not a day."""
    parts = str(base).split("_")
    if len(parts) < 2 or not parts[0].startswith("sub-"):
        raise ProtocolError(f"recording id {base!r} is not a BIDS base")
    try:
        subject = int(parts[0].split("-", 1)[1])
    except ValueError as exc:
        raise ProtocolError(f"recording id {base!r} has no subject") from exc
    session = ""
    run = ""
    task = ""
    for part in parts[1:]:
        if part.startswith("ses-"):
            session = part[4:]
        elif part.startswith("run-"):
            run = part[4:]
        elif part.startswith("task-"):
            task = part[5:]
    return {
        "base": str(base),
        "subject": subject,
        "session_label": session,
        "run_label": run,
        "task_label": task,
        "session_ordinal_used_as_day": False,
    }


def _shared(base: str) -> bool:
    return ("ses-11_task-Sherlock1" in base) or ("ses-12_task-Sherlock1" in base)


def _catalog(row: dict) -> tuple:
    return (int(row["subject"]), str(row["session_label"]), str(row["run_label"]), str(row["base"]))


def _chronological_key(row: dict) -> tuple:
    """Real acquisition dates sort first. A missing date sorts last and is not filled in."""
    date = row.get("date")
    if date:
        return (0, str(date), _catalog(row))
    return (1, "", _catalog(row))


def plan_recordings(bases, dates=None) -> dict:
    """Pre-registered order. One phase list, chronological within a phase.

    Dated recordings use the scans.tsv date. Undated recordings follow in
    catalog order. Catalog order is subject, session token, run token, and the
    full id. The session token is not written as a date.
    """
    dates = dates or {}
    scheduled = {name: [] for name in ("noise_pool", "selection", "eval", "eval_extension", "train")}
    refused = []
    for base in sorted(set(bases)):
        if "holdout" in str(base).lower():
            refused.append({"base": str(base), "reason": "holdout path"})
            continue
        ident = parse_base(base)
        ident["date"] = dates.get(base) or None
        if ident["date"] is not None:
            text = str(ident["date"])
            if not (len(text) == 10 and text[4] == "-" and text[7] == "-"):
                raise ProtocolError(f"acquisition date {text!r} is not YYYY-MM-DD")
        subject = ident["subject"]
        shared = _shared(base)
        if subject in FIT_SUBJECTS and shared:
            phase = "noise_pool"
        elif subject in VAL_SUBJECTS and shared:
            phase = "selection"
        elif subject in EVAL_SUBJECTS and shared:
            phase = "eval"
        elif subject in EVAL_EXTENSION_SUBJECTS and shared:
            phase = "eval_extension"
        elif subject == 0 and not shared:
            phase = "train"
        else:
            continue
        ident["phase"] = phase
        ident["shared"] = shared
        ident["order_uses_acquisition_date"] = ident["date"] is not None
        scheduled[phase].append(ident)
    for phase, rows in scheduled.items():
        rows.sort(key=_chronological_key)
        if phase == "train":
            kept = rows[:HEAD_TRAIN_RUNS]
            for extra in rows[HEAD_TRAIN_RUNS:]:
                extra["in_budget"] = False
            rows[:] = kept
        for index, row in enumerate(rows):
            row["order_index"] = index
            row["in_budget"] = True
            row["date_sentinal_used_as_date"] = False
    return {
        "phases": scheduled,
        "refused_holdout": refused,
        "fit_subjects": list(FIT_SUBJECTS),
        "eval_subjects": list(EVAL_SUBJECTS),
        "eval_extension_subjects": list(EVAL_EXTENSION_SUBJECTS),
        "selection_subjects": list(VAL_SUBJECTS),
        "fit_quarter": FIT_QUARTER,
        "held_out_quarter": HELD_OUT_QUARTER,
        "eval_extension_quarters": list(EVAL_EXTENSION_QUARTERS),
        "recordings_per_launch": RECORDINGS_PER_LAUNCH,
        "session_ordinal_used_as_day": False,
        "dates_invented": False,
    }


def iter_budgeted(plan: dict):
    for phase in ("noise_pool", "selection", "eval", "eval_extension", "train"):
        for row in plan["phases"][phase]:
            if row.get("in_budget", True):
                yield row


def _below_floor(fit_rows: int, schedule: str) -> dict:
    return {
        "stage": "stop",
        "base": None,
        "phase": "noise_pool",
        "next_action": "stop",
        "reason": "fit_pool_exhausted_below_power_floor",
        "decisive": False,
        "measurement_complete": False,
        "fit_rows": int(fit_rows),
        "fit_row_floor": FIT_ROW_FLOOR,
        "schedule": schedule,
    }


def next_action_15(plan: dict, *, done_bases, noise_fit: bool, fit_rows: int, partial_steps: int, partial_stopped: bool, score_frozen: bool, score_partial: bool, score_heldout: bool, pool_exhausted_below_floor: bool = False) -> dict:
    """One recording, then the fit, then scores. Eval files are not the fit."""
    done = set(done_bases)
    schedule = PARTIAL_SCHEDULE_15["name"]
    remaining_pool = [
        row for row in plan["phases"]["noise_pool"]
        if row.get("in_budget", True) and row["base"] not in done
    ]
    if not noise_fit and (pool_exhausted_below_floor or (not remaining_pool and int(fit_rows) < FIT_ROW_FLOOR)):
        return _below_floor(fit_rows, schedule)
    for row in iter_budgeted(plan):
        if row["base"] in done:
            continue
        if row["phase"] == "noise_pool" and noise_fit:
            continue
        if row["phase"] in ("eval", "eval_extension") and not noise_fit:
            continue
        if row["phase"] == "train" and not score_frozen:
            continue
        return {
            "stage": "forward",
            "base": row["base"],
            "phase": row["phase"],
            "next_action": "rerun_this_command",
            "schedule": schedule,
            "recordings_this_launch": RECORDINGS_PER_LAUNCH,
        }
    if not noise_fit:
        if int(fit_rows) < FIT_ROW_FLOOR:
            return {
                "stage": "stop",
                "base": None,
                "phase": "noise_pool",
                "next_action": "stop",
                "reason": "fit_pool_exhausted_below_power_floor",
                "decisive": False,
                "measurement_complete": False,
                "fit_rows": int(fit_rows),
                "fit_row_floor": FIT_ROW_FLOOR,
                "schedule": schedule,
            }
        return {"stage": "noise-fit", "base": None, "phase": "noise_pool", "next_action": "rerun_this_command", "schedule": schedule}
    if not score_frozen:
        return {"stage": "score-frozen", "base": None, "phase": "eval", "next_action": "rerun_this_command", "schedule": schedule}
    steps = int(partial_steps)
    if not partial_stopped and steps < int(PARTIAL_SCHEDULE_15["max_steps"]):
        return {
            "stage": "partial-train",
            "base": None,
            "phase": "train",
            "steps": min(int(PARTIAL_SCHEDULE_15["steps_per_launch"]), int(PARTIAL_SCHEDULE_15["max_steps"]) - steps),
            "next_action": "rerun_this_command",
            "schedule": schedule,
        }
    if not score_partial:
        return {"stage": "score-partial", "base": None, "phase": "eval", "next_action": "rerun_this_command", "schedule": schedule}
    if not score_heldout:
        return {"stage": "score-heldout-q4", "base": None, "phase": "noise_pool", "next_action": "rerun_this_command", "schedule": schedule}
    return {"stage": "stop", "base": None, "phase": None, "next_action": "stop", "reason": "stages_complete", "schedule": schedule}


def _ridge(z: np.ndarray, target: np.ndarray, penalty_per_row: float) -> np.ndarray:
    design = _design(z)
    penalty = float(penalty_per_row) * len(z)
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += penalty
    gram[0, 0] -= penalty
    return np.linalg.solve(gram, design.T @ target)


def fit_reduced_state(z_train: np.ndarray, residual_train: np.ndarray, feature_names) -> dict:
    """One principal component and a stronger ridge. Declared, not searched.

    Columns with no training variation are dropped before the decomposition.
    On one acquisition date, log_train_window_count is that kind of column.
    The component is fit on these training rows only. Its sign is fixed by the
    first kept column so a rerun does not flip it.
    """
    names = list(feature_names)
    z = np.asarray(z_train, dtype=np.float64)
    residual = np.asarray(residual_train, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if len(z) != len(residual) or len(z) < 2:
        raise ProtocolError("reduced_state needs at least two aligned training rows")
    if z.shape[1] != len(names):
        raise ProtocolError("reduced_state feature names do not match the matrix")
    mu = float(residual.mean())
    energy = np.clip((residual - mu) ** 2, 1e-8, None)
    deviation = z.std(axis=0)
    keep = [index for index, scale in enumerate(deviation) if float(scale) >= DEAD_COLUMN_STD]
    base = {
        "candidate": "reduced_state",
        "mu": mu,
        "objective": "gaussian_nll",
        "feature_names": names,
        "ridge_penalty_per_row": REDUCED_RIDGE_PER_ROW,
        "learned_ridge_penalty_per_row": LEARNED_RIDGE_PER_ROW,
        "train_rows": int(len(z)),
        "dead_column_std": DEAD_COLUMN_STD,
    }
    if not keep:
        constant = fit_noise(z, residual, "constant", feature_names=names)
        base.update({
            "intercept": constant["intercept"],
            "slope": np.zeros(0, dtype=np.float64),
            "column_index": [],
            "kept_feature_names": [],
            "loading": [],
            "collapsed_to_constant": True,
            "projection": None,
            "scaler": constant["scaler"],
        })
        return base
    cols = z[:, keep]
    centered = cols - cols.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    loading = np.asarray(vt[0], dtype=np.float64)
    if loading[0] < 0:
        loading = -loading
    score = centered @ loading
    beta = _ridge(score.reshape(-1, 1), np.log(energy), REDUCED_RIDGE_PER_ROW)
    base.update({
        "intercept": float(np.clip(beta[0], *LOG_VAR_CLIP)),
        "slope": np.asarray(beta[1:], dtype=np.float64),
        "column_index": [int(index) for index in keep],
        "kept_feature_names": [names[index] for index in keep],
        "loading": loading,
        "column_mean": cols.mean(axis=0),
        "collapsed_to_constant": False,
        "projection": None,
        "scaler": None,
    })
    return base


def log_variance_15(z: np.ndarray, params: dict) -> np.ndarray:
    if params.get("candidate") != "reduced_state":
        return log_variance(z, params)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    if params.get("collapsed_to_constant"):
        return np.full(len(z), float(params["intercept"]))
    index = list(params["column_index"])
    loading = np.asarray(params["loading"], dtype=np.float64)
    mean = np.asarray(params["column_mean"], dtype=np.float64)
    score = (z[:, index] - mean) @ loading
    slope = np.asarray(params["slope"], dtype=np.float64)
    lv = float(params["intercept"]) + score * float(slope[0])
    return np.clip(lv, *LOG_VAR_CLIP)


def nll_rows(residual: np.ndarray, mu: float, log_var: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    lv = np.clip(np.asarray(log_var, dtype=np.float64), *LOG_VAR_CLIP)
    centered = residual - float(mu)
    return 0.5 * (np.log(2.0 * np.pi) + lv + centered ** 2 * np.exp(-lv))


def _percentile_interval(samples: list[float]) -> dict:
    if len(samples) < 2:
        return {"low": None, "high": None, "mean": None, "defined": False, "n_draws": len(samples)}
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    draws = len(ordered)
    return {
        "mean": float(ordered.mean()),
        "low": float(ordered[int(0.025 * (draws - 1))]),
        "high": float(ordered[int(0.975 * (draws - 1))]),
        "defined": True,
        "n_draws": int(draws),
    }


def _session_groups(session_ids):
    session_ids = np.asarray(session_ids)
    sessions = []
    groups = []
    for session in session_ids.tolist():
        if session not in sessions:
            sessions.append(session)
    if len(sessions) < 2:
        return sessions, None
    for session in sessions:
        groups.append(np.flatnonzero(session_ids == session))
    return sessions, groups


def paired_mean_contrasts(session_ids, values_by_arm: dict, *, focals=STRUCTURED_ARMS, controls=CONTROLS, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Paired session bootstrap of mean differences. The same sessions are drawn for every arm."""
    _sessions, groups = _session_groups(session_ids)
    point = {}
    if groups is None:
        for focal in focals:
            for control in controls:
                point[f"{focal}_minus_{control}"] = {
                    "estimate": None, "low": None, "high": None, "defined": False, "n_sessions": len(_sessions),
                }
        return {"contrasts": point, "defined": False, "n_sessions": len(_sessions), "unit": "session_mean"}
    arms = {}
    for name, values in values_by_arm.items():
        arms[name] = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    buckets = {(focal, control): [] for focal in focals for control in controls}
    estimates = {}
    session_means = {name: np.asarray([float(arms[name][group].mean()) for group in groups]) for name in arms}
    for focal in focals:
        for control in controls:
            estimates[(focal, control)] = float(session_means[focal].mean() - session_means[control].mean())
    for _ in range(int(draws)):
        take = rng.integers(0, len(groups), size=len(groups))
        for focal in focals:
            for control in controls:
                diff = float(session_means[focal][take].mean() - session_means[control][take].mean())
                buckets[(focal, control)].append(diff)
    contrasts = {}
    for key, samples in buckets.items():
        interval = _percentile_interval(samples)
        interval["estimate"] = estimates[key]
        interval["n_sessions"] = len(groups)
        contrasts[f"{key[0]}_minus_{key[1]}"] = interval
    return {"contrasts": contrasts, "defined": True, "n_sessions": len(groups), "unit": "session_mean", "seed": int(seed), "draws": int(draws)}


def paired_bacc_contrasts(session_ids, labels, logits_by_arm: dict, *, min_count: int, focals=STRUCTURED_ARMS, controls=CONTROLS, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Paired bootstrap. Each draw pools the resampled sessions, then computes balanced top-10."""
    _sessions, groups = _session_groups(session_ids)
    labels = np.asarray(labels, dtype=np.int64)
    if groups is None:
        return {"contrasts": {}, "defined": False, "n_sessions": len(_sessions), "unit": "pooled_resampled_sessions"}
    logits = {name: np.asarray(value, dtype=np.float64) for name, value in logits_by_arm.items()}
    rng = np.random.default_rng(seed)
    buckets = {(focal, control): [] for focal in focals for control in controls}
    full = np.arange(len(labels))

    def difference(index, focal, control):
        left = bacc10_present(logits[focal][index], labels[index], min_count=min_count)
        right = bacc10_present(logits[control][index], labels[index], min_count=min_count)
        if left is None or right is None:
            return None
        return float(left - right)

    estimates = {}
    for focal in focals:
        for control in controls:
            estimates[(focal, control)] = difference(full, focal, control)
    for _ in range(int(draws)):
        take = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in take])
        for focal in focals:
            for control in controls:
                value = difference(index, focal, control)
                if value is not None:
                    buckets[(focal, control)].append(value)
    contrasts = {}
    defined = True
    for key, samples in buckets.items():
        interval = _percentile_interval(samples)
        interval["estimate"] = estimates[key]
        interval["n_sessions"] = len(groups)
        if not interval["defined"] or estimates[key] is None:
            defined = False
        contrasts[f"{key[0]}_minus_{key[1]}"] = interval
    return {
        "contrasts": contrasts,
        "defined": defined,
        "n_sessions": len(groups),
        "unit": "pooled_resampled_sessions",
        "min_class_count": int(min_count),
        "seed": int(seed),
        "draws": int(draws),
    }


def measurement_complete_15(*, checkpoint_bytes: int, fit_rows: int, primary_defined: bool, lock_held: bool, fixture: bool = False) -> bool:
    """Engineering completeness. A decisive interval is a separate flag and is not a claim."""
    from .strong_noise import CHECKPOINT_BYTES
    return bool(
        fixture is not True
        and int(checkpoint_bytes) == CHECKPOINT_BYTES
        and int(fit_rows) >= FIT_ROW_FLOOR
        and primary_defined is True
        and lock_held is True
    )


def contrast_resolved(interval: dict, margin: float) -> bool:
    """True when the interval excludes 0 or lies inside the equivalence margin."""
    if not interval or not interval.get("defined"):
        return False
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return False
    if high < 0 or low > 0:
        return True
    return bool(low >= -float(margin) and high <= float(margin))


def decision_from_contrasts(bacc_contrasts: dict, nll_contrasts: dict) -> dict:
    unresolved = []
    resolved = []
    for name, interval in bacc_contrasts.get("contrasts", {}).items():
        key = f"bacc:{name}"
        if contrast_resolved(interval, BACC_EQUIVALENCE_MARGIN):
            resolved.append(key)
        else:
            unresolved.append(key)
    for name, interval in nll_contrasts.get("contrasts", {}).items():
        key = f"nll:{name}"
        if contrast_resolved(interval, NLL_EQUIVALENCE_MARGIN):
            resolved.append(key)
        else:
            unresolved.append(key)
    return {
        "decisive": not unresolved and bool(resolved),
        "resolved": resolved,
        "unresolved": unresolved,
        "bacc_margin": BACC_EQUIVALENCE_MARGIN,
        "nll_margin": NLL_EQUIVALENCE_MARGIN,
        "effect_claimed": False,
        "valid_scientific_claim": False,
    }


def validation_stop_15(curve, schedule: dict) -> dict:
    """Stop on the mean validation CE of the three controls. Structured arms do not pick the step."""
    for row in curve:
        banned = set(row) & {"eval_ce", "test_ce", "eval", "test", "eval_bacc", "test_bacc"}
        if banned:
            raise ProtocolError(f"validation stop cannot see {sorted(banned)}")
        if "control_mean_ce" not in row:
            raise ProtocolError("validation stop requires control_mean_ce")
    if not curve:
        return {"stop": False, "best_step": None, "reason": None, "checks_without_improvement": 0}
    best = min(curve, key=lambda row: (float(row["control_mean_ce"]), int(row["step"])))
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
        "stop_on": schedule.get("stop_on"),
        "substantive": bool(schedule.get("substantive")),
    }


def matched_partial_steps_15(arms: dict, embeddings, labels, frozen_log_var: dict, steps: int, *, lr: float = PARTIAL_LR, seed: int = 0, batch: int = 32) -> dict:
    """Same minibatches and the same step count. Precision is frozen. Only the head moves."""
    if set(arms) != set(PARTIAL_ARMS_15):
        raise ProtocolError("partial train must step the declared 1.5 arms together")
    for name in PARTIAL_ARMS_15:
        if name == "uniform":
            continue
        if name not in frozen_log_var:
            raise ProtocolError(f"missing frozen log variance for {name}")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(embeddings) != len(labels):
        raise ProtocolError("partial-train tensors differ in length")
    if steps < 1:
        raise ProtocolError("partial train needs at least one step")
    length = len(labels)
    for name, values in frozen_log_var.items():
        if len(values) != length:
            raise ProtocolError(f"frozen log variance for {name} does not match the rows")
    rng = np.random.default_rng(seed)
    for _ in range(int(steps)):
        take = rng.integers(0, length, size=min(int(batch), length))
        emb = embeddings[take]
        y = labels[take]
        for name in PARTIAL_ARMS_15:
            arm = arms[name]
            if name == "uniform":
                log_var = np.zeros(len(y))
            else:
                log_var = np.asarray(frozen_log_var[name], dtype=np.float64)[take]
            arm["W"], arm["b"] = step_head(arm["W"], arm["b"], emb, y, log_var, lr)
    return arms


def initial_partial_arms_15(weight: np.ndarray, bias: np.ndarray) -> dict:
    arms = {}
    for name in PARTIAL_ARMS_15:
        arms[name] = {
            "W": np.array(weight, dtype=np.float64, copy=True),
            "b": np.array(bias, dtype=np.float64, copy=True),
        }
    return arms


def _candidate_matrix_15(z: np.ndarray, names, candidate: str):
    if candidate == "error_scaling":
        features = [feature for feature in ERROR_SCALING_FEATURES if feature in list(names)]
        if not features:
            raise ProtocolError("error_scaling has no remaining features")
        columns = [list(names).index(feature) for feature in features]
        return z[:, columns], features
    return z, list(names)


def fit_equations(rows: list[dict]) -> dict:
    """Fit every declared 1.5 candidate. Selection rows are not coefficient rows."""
    if any(row["split"] == "eval" for row in rows):
        raise ProtocolError("eval rows were passed to the noise fit")
    if any(row.get("quarter") in ("other", "Q1", "Q2") for row in rows if row["split"] == "train"):
        raise ProtocolError("refusing in-sample trunk-trained residuals for the noise fit")
    if any(row.get("quarter") == HELD_OUT_QUARTER for row in rows if row["split"] == "train"):
        raise ProtocolError("Q4 rows were passed to the coefficient fit")
    train_subjects = {int(row["subject"]) for row in rows if row["split"] == "train"}
    illegal = train_subjects & (set(EVAL_SUBJECTS) | set(VAL_SUBJECTS) | set(EVAL_EXTENSION_SUBJECTS))
    if illegal:
        raise ProtocolError(f"held-out subjects {sorted(illegal)} leaked into the coefficient fit")
    unknown = train_subjects - set(FIT_SUBJECTS)
    if unknown:
        raise ProtocolError(f"noise fit subjects {sorted(unknown)} are outside the pre-registered pool")
    bundle = assemble_features(rows, allow_undated_train_when_days_inestimable=True)
    residual = np.asarray([row["residual"] for row in rows], dtype=np.float64)
    train = bundle["train_index"]
    n_fit = int(np.sum(train))
    if n_fit < FIT_ROW_FLOOR:
        raise ProtocolError(
            f"noise fit has {n_fit} rows, below the pre-registered floor {FIT_ROW_FLOOR}"
        )
    if int(np.sum(~train)) < 1:
        raise ProtocolError("validation selection has no in-vocabulary rows")
    if n_fit != sum(1 for row in rows if row["split"] == "train"):
        raise ProtocolError("selection rows leaked into the coefficient fit")

    params = {}
    val_nll = {}
    names = bundle["names"]
    matrix = bundle["z"]
    for name in CANDIDATES:
        used, feat_names = _candidate_matrix_15(matrix, names, name)
        params[name] = fit_noise(used[train], residual[train], name, feature_names=feat_names, seed=0)
        val_nll[name] = float(gaussian_nll(
            residual[~train], params[name]["mu"], log_variance(used[~train], params[name]),
        ))
    params["reduced_state"] = fit_reduced_state(matrix[train], residual[train], names)
    val_nll["reduced_state"] = float(gaussian_nll(
        residual[~train],
        params["reduced_state"]["mu"],
        log_variance_15(matrix[~train], params["reduced_state"]),
    ))
    if set(val_nll) != set(CANDIDATES_15):
        raise ProtocolError("validation selection must score every 1.5 candidate")
    for value in val_nll.values():
        if not np.isfinite(value):
            raise ProtocolError("validation NLL is non-finite")
    chosen = min(CANDIDATES_15, key=lambda name: (float(val_nll[name]), name))
    return {
        "bundle": bundle,
        "params": params,
        "val_nll": val_nll,
        "chosen_on_validation": chosen,
        "fit_rows": n_fit,
        "selection_rows": int(len(rows) - n_fit),
        "train_subjects": sorted(train_subjects),
        "power": power_report(),
    }


def timers_coherent(observed_scope_runtime_sec) -> dict:
    """The launcher ceiling and the inner deadline are the same policy."""
    if RUNTIME_CEILING_SEC != INNER_CHUNK_DEADLINE_SEC:
        raise ProtocolError("launcher ceiling and inner deadline disagree")
    if observed_scope_runtime_sec not in (None, "infinity"):
        raise ProtocolError(
            f"scope runtime {observed_scope_runtime_sec} disagrees with declared ceiling none"
        )
    return {
        "launcher_runtime_ceiling_sec": RUNTIME_CEILING_SEC,
        "inner_chunk_deadline_sec": INNER_CHUNK_DEADLINE_SEC,
        "observed_scope_runtime_sec": None if observed_scope_runtime_sec in (None, "infinity") else observed_scope_runtime_sec,
        "coherent": True,
        "runtime_ceiling_required": False,
    }


def assert_timer_contract(launcher_text: str) -> None:
    if TIMER_SENTINEL_LAUNCHER not in launcher_text or TIMER_SENTINEL_INNER not in launcher_text:
        raise ProtocolError("launcher is missing the timer sentinels")
    if "RuntimeMaxSec" in launcher_text:
        raise ProtocolError("launcher declares a runtime ceiling")
    if RUNTIME_CEILING_SEC is not None or INNER_CHUNK_DEADLINE_SEC is not None:
        raise ProtocolError("inner timer constants disagree with the launcher")


def append_exposure(directory: str, record: dict) -> None:
    path = os.path.join(directory, "exposure_log.jsonl")
    os.makedirs(directory, exist_ok=True)
    payload = dict(record)
    payload.setdefault("amendment", AMENDMENT_15)
    payload.setdefault("dates_invented", False)
    payload.setdefault("session_ordinal_used_as_day", False)
    payload.setdefault("competition_holdout_read", False)
    line = json.dumps(payload, sort_keys=True, default=_json_default) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def read_exposure(directory: str) -> list[dict]:
    path = os.path.join(directory, "exposure_log.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value).__name__)


def chunk_marker_path(directory: str) -> str:
    return os.path.join(directory, "chunks", "current.json")


def begin_chunk(directory: str, base: str, phase: str) -> dict:
    """Detect a chunk that started and did not publish. Do not delete a finished npz."""
    os.makedirs(os.path.join(directory, "chunks"), exist_ok=True)
    marker = chunk_marker_path(directory)
    failure = None
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as handle:
            previous = json.load(handle)
        published = os.path.isfile(os.path.join(directory, "forward", previous["phase"], previous["base"] + ".npz"))
        temporary = os.path.join(directory, "forward", previous["phase"], "." + previous["base"] + ".writing.npz")
        if published:
            os.remove(marker)
            append_exposure(directory, {
                "role": "marker_cleared_after_complete",
                "base": previous["base"],
                "phase": previous["phase"],
                "entered_coefficients": False,
                "meg_read": False,
            })
        else:
            if os.path.exists(temporary):
                os.remove(temporary)
            failure = {
                "role": "failure_detected",
                "base": previous["base"],
                "phase": previous["phase"],
                "reason": "chunk marker without a published forward",
                "entered_coefficients": False,
                "meg_read": False,
                "retry": True,
            }
            append_exposure(directory, failure)
            os.remove(marker)
    payload = {"base": base, "phase": phase, "started": True}
    temporary_marker = marker + ".tmp"
    with open(temporary_marker, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary_marker, marker)
    return {"failure": failure, "base": base, "phase": phase}


def finish_chunk(directory: str, base: str, phase: str) -> None:
    marker = chunk_marker_path(directory)
    if not os.path.isfile(marker):
        return
    with open(marker, encoding="utf-8") as handle:
        current = json.load(handle)
    if current.get("base") == base and current.get("phase") == phase:
        os.remove(marker)


def params_to_json(params: dict) -> dict:
    payload = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
        elif isinstance(value, np.floating):
            payload[key] = float(value)
        elif isinstance(value, np.integer):
            payload[key] = int(value)
        else:
            payload[key] = value
    return payload


def params_from_json(payload: dict) -> dict:
    params = dict(payload)
    for key in ("slope", "loading", "column_mean"):
        if key in params and params[key] is not None:
            params[key] = np.asarray(params[key], dtype=np.float64)
    return params


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
