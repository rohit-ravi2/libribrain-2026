"""Day cross-fit. Scalers and parameters for a held-out day see only the other days."""
from __future__ import annotations

import numpy as np

from .decode_link import encoding_to_decoder_note
from .leakage import LeakageError, assert_batch_allowed, null_adaptive_search
from .metrics import (
    aggregate_noise_controls,
    assess_task,
    calibration_error,
    day_group_uncertainty,
    innovation_autocorrelations,
    residual_logscore,
)
from .models import MODEL_NAMES, fit_model, parameter_count, predict_model
from .protocol import FEATURE_SET, GUARD_WINDOWS, LADDER, NULL_STATE_SEED, PRIMARY_CONTRAST, PRIMARY_ENDPOINT, next_rung
from .stream import drop_overlapping_train, time_split
from .synthetic import make_null_state


def fit_scaler(values: np.ndarray, train_idx: np.ndarray):
    block = np.asarray(values, dtype=np.float64)[train_idx]
    mu = block.mean(axis=0)
    sd = block.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def apply_scaler(values: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mu) / sd


def _as_result_base(batch, claim_status: str) -> dict:
    return {
        "claim_status": claim_status,
        "real_meg_scientific_validation": False,
        "evidence_class": batch.get("evidence_class", "unspecified"),
        "feature_set": FEATURE_SET,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_contrast": PRIMARY_CONTRAST,
        "model_selection": "none_frozen_report_all",
        "decoder_intervention_executed": False,
        "decoder_link": encoding_to_decoder_note(),
        "neural_inference_allowed": False,
    }


def run_crossfit(batch: dict, *, mined_features: bool = False, feature_candidates=None) -> dict:
    if mined_features or (
        feature_candidates is not None and list(feature_candidates) != [FEATURE_SET]
    ):
        return null_adaptive_search(
            "Feature mining nulls the entire search. No score from this call is a result.",
            feature_candidates if feature_candidates is not None else ["mined_features"],
        )
    assert_batch_allowed(batch)
    y = np.asarray(batch["y"], dtype=np.float64)
    state = np.asarray(batch["state"], dtype=np.float64)
    day = np.asarray(batch["day"])
    label = np.asarray(batch["label"])
    time_index = np.asarray(batch["time_index"])
    order = np.lexsort((time_index, day))
    y, state, day, label, time_index = y[order], state[order], day[order], label[order], time_index[order]
    # Drawn from n and a frozen seed only. Sorting above used time and day, not y.
    null_state = make_null_state(len(y), state.shape[1], NULL_STATE_SEED)
    days = np.unique(day)
    if len(days) < 2:
        raise LeakageError("cross-fit by day needs at least two days")

    folds = []
    trials = []
    n_dim, n_state = y.shape[1], state.shape[1]
    for held in days:
        test_idx = np.flatnonzero(day == held)
        train_idx = np.flatnonzero(day != held)
        y_mu, y_sd = fit_scaler(y, train_idx)
        s_mu, s_sd = fit_scaler(state, train_idx)
        y_s = apply_scaler(y, y_mu, y_sd)
        state_s = apply_scaler(state, s_mu, s_sd)
        # Null covariates stay in the frozen draw's scale. They are not re-fit to y.
        fold_models = {}
        for name in MODEL_NAMES:
            params = fit_model(name, y_s[train_idx], state_s[train_idx], day[train_idx], null_state[train_idx])
            mu, var = predict_model(params, y_s[test_idx], state_s[test_idx], day[test_idx], null_state[test_idx])
            score = residual_logscore(y_s[test_idx], mu, var)
            acf = innovation_autocorrelations(y_s[test_idx], mu, var, day[test_idx])
            record = {
                "logscore": score,
                "calibration_error": calibration_error(y_s[test_idx], mu, var),
                **acf,
                "n_params": parameter_count(name, n_dim, n_state, n_train_days=len(np.unique(day[train_idx]))),
                "predicted_mean": float(np.mean(mu)),
                "test_mean": float(np.mean(y_s[test_idx])),
            }
            fold_models[name] = record
            trials.append({
                "hypothesis_id": f"crossfit/{name}",
                "held_out_day": int(held),
                "pre_registered": True,
                "model": name,
                "logscore": score,
                "retrospective_confirmation": False,
            })
        task = assess_task(
            y_s[train_idx], y_s[test_idx], label[train_idx], label[test_idx], null_seed=NULL_STATE_SEED + int(held)
        )
        folds.append({
            "held_out_day": int(held),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "scaler_mean_y": [float(v) for v in y_mu],
            "models": fold_models,
            "task": task,
        })

    by_model = {}
    for name in MODEL_NAMES:
        scores = [fold["models"][name]["logscore"] for fold in folds]
        summary = day_group_uncertainty(scores)
        summary["calibration_error_mean"] = float(np.mean([fold["models"][name]["calibration_error"] for fold in folds]))
        summary["innovation_acf_lag1_mean"] = float(np.mean([fold["models"][name]["innovation_acf_lag1"] for fold in folds]))
        summary["squared_innovation_acf_lag1_mean"] = float(
            np.mean([fold["models"][name]["squared_innovation_acf_lag1"] for fold in folds])
        )
        summary["n_params"] = folds[0]["models"][name]["n_params"]
        by_model[name] = summary
    noise = aggregate_noise_controls([fold["task"] for fold in folds])
    result = _as_result_base(
        batch,
        "synthetic_evaluation_only" if batch.get("evidence_class") == "synthetic_dgp" else "not_a_scientific_result",
    )
    result.update({
        "models": by_model,
        "folds": folds,
        "trials": trials,
        "noise_controls": noise,
        "diagnose_required": bool(noise["diagnose_required"]),
        "neural_inference_allowed": False,
        "primary_contrast_value": float(
            by_model["M2_state_variance"]["mean"] - by_model["M0_fixed_gaussian"]["mean"]
        ),
    })
    return result


def smoke_one_series(batch: dict) -> dict:
    """Fit on the early part of one day and score the later part. Not a cross-day result."""
    assert_batch_allowed(batch)
    if len(np.unique(batch["day"])) != 1:
        raise LeakageError("the first smoke accepts one day group only")
    y = np.asarray(batch["y"], dtype=np.float64)
    state = np.asarray(batch["state"], dtype=np.float64)
    day = np.asarray(batch["day"])
    label = np.asarray(batch["label"])
    time_index = np.asarray(batch["time_index"])
    ranges = batch.get("sample_ranges")
    order = np.argsort(time_index, kind="mergesort")
    y, state, day, label = y[order], state[order], day[order], label[order]
    if ranges is not None:
        ranges = [ranges[int(i)] for i in order]
    train_idx, test_idx = time_split(len(y), GUARD_WINDOWS)
    train_idx = drop_overlapping_train(train_idx, test_idx, ranges)
    if len(train_idx) < 8 or len(test_idx) < 8:
        raise LeakageError("the gap between train and test windows left too few rows")
    if ranges is not None:
        test_ranges = [ranges[int(i)] for i in test_idx]
        for i in train_idx:
            a, b = ranges[int(i)]
            if any(a < tb and b > ta for ta, tb in test_ranges):
                raise LeakageError("train and test windows still share samples")
    null_state = make_null_state(len(y), state.shape[1], NULL_STATE_SEED)
    y_mu, y_sd = fit_scaler(y, train_idx)
    s_mu, s_sd = fit_scaler(state, train_idx)
    y_s = apply_scaler(y, y_mu, y_sd)
    state_s = apply_scaler(state, s_mu, s_sd)
    models = {}
    trials = []
    for name in MODEL_NAMES:
        params = fit_model(name, y_s[train_idx], state_s[train_idx], day[train_idx], null_state[train_idx])
        mu, var = predict_model(params, y_s[test_idx], state_s[test_idx], day[test_idx], null_state[test_idx])
        score = residual_logscore(y_s[test_idx], mu, var)
        acf = innovation_autocorrelations(y_s[test_idx], mu, var, day[test_idx])
        models[name] = {
            "logscore": score,
            "calibration_error": calibration_error(y_s[test_idx], mu, var),
            **acf,
            "n_params": parameter_count(name, y.shape[1], state.shape[1], n_train_days=1),
        }
        trials.append({
            "hypothesis_id": f"smoke/{name}",
            "model": name,
            "logscore": score,
            "pre_registered": True,
            "retrospective_confirmation": False,
            "scientific_claim": False,
        })
    task = assess_task(
        y_s[train_idx], y_s[test_idx], label[train_idx], label[test_idx], null_seed=NULL_STATE_SEED
    )
    # A single short split is under the task-sample floor on purpose.
    task["campaign_gate"] = aggregate_noise_controls([task])
    result = _as_result_base(batch, "not_a_scientific_result")
    result.update({
        "rung": LADDER[0],
        "rung_completed_as_science": False,
        "cross_day": False,
        "reason": "One training-day group cannot support out-of-day replication.",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_guard_dropped": int(len(y) - len(train_idx) - len(test_idx)),
        "models": models,
        "task": task,
        "trials": trials,
        "diagnose_required": bool(task["campaign_gate"]["diagnose_required"]),
        "next_rung": next_rung(LADDER[0]),
    })
    return result
