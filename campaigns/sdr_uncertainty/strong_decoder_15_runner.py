"""Amendment 1.5 stages. One launch, one chunk, a new output tree.

Does not open strong_decoder_1_4 and does not rewrite its files. The parent
protocol id stays strong-decoder-noise-1.4. effect_claimed stays false.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

from .amendment_15 import (
    AMENDMENT_15,
    BACC_EQUIVALENCE_MARGIN,
    BOOTSTRAP_DRAWS,
    CANDIDATES_15,
    CONTROLS,
    EVAL_EXTENSION_QUARTERS,
    FIT_QUARTER,
    FIT_ROW_FLOOR,
    HELD_OUT_QUARTER,
    NLL_EQUIVALENCE_MARGIN,
    PARTIAL_ARMS_15,
    PARTIAL_SCHEDULE_15,
    PRIMARY_METRIC,
    STRUCTURED_ARMS,
    append_exposure,
    bacc10_present,
    begin_chunk,
    decision_from_contrasts,
    finish_chunk,
    fit_equations,
    fit_reduced_state,
    initial_partial_arms_15,
    log_variance_15,
    matched_partial_steps_15,
    measurement_complete_15,
    metric_declaration,
    next_action_15,
    nll_rows,
    paired_bacc_contrasts,
    paired_mean_contrasts,
    params_from_json,
    params_to_json,
    plan_recordings,
    power_report,
    timers_coherent,
    utc_now,
    validation_stop_15,
)
from .cgroup import CgroupError, verify_live_hardware
from .strong_decoder import is_strong_decoder, load_strong_decoder
from .strong_decoder_runner import (
    _require_checkpoint,
    _rows_from_forward,
    _stimulus_tmax,
    execute_forward,
    honesty,
    list_cache_bases,
    lock_held,
    read_dates,
    repo_root,
)
from .strong_noise import (
    N_CLASSES,
    PARTIAL_LR,
    PROTOCOL_ID,
    ProtocolError,
    arm_logits,
    bacc10_from_logits,
    decoder_residual,
    frozen_day_term_decision,
    gaussian_nll,
    transform_with_state,
)

LOCK_NAME = "libribrain-strong-decoder-noise"
OUT_DIRNAME = "strong_decoder_1_5"
PRIOR_OUT_DIRNAME = "strong_decoder_1_4"


def assert_output_dir(path: str, root: str) -> str:
    real = os.path.realpath(path)
    repo = os.path.realpath(root)
    if real == repo or real.startswith(repo + os.sep):
        raise ProtocolError("refusing to write inside the git repository")
    lowered = real.lower()
    if "holdout" in lowered or "competition_holdout" in lowered:
        raise ProtocolError("refusing a holdout path")
    if os.path.basename(real) != OUT_DIRNAME:
        raise ProtocolError("amendment 1.5 outputs go only to strong_decoder_1_5")
    if PRIOR_OUT_DIRNAME in real.split(os.sep):
        raise ProtocolError("refusing to write the 1.4 output tree")
    return real


def out_dir(root: str) -> str:
    base = os.environ.get("LIBRIBRAIN_SDR_OUT", "/mnt/ssd4tb/libribrain_sdr_out")
    return assert_output_dir(os.path.join(base, OUT_DIRNAME), root)


def _cache_paths(root: str):
    del root
    cache_root = os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache")
    data_root = os.environ.get("LIBRIBRAIN_DATA", "/mnt/ssd4tb/data/LibriBrain100")
    return os.path.join(cache_root, "cont"), data_root


def honesty_15(*sources, **extra) -> dict:
    payload = honesty(*sources, amendment=AMENDMENT_15, **extra)
    payload["amendment"] = AMENDMENT_15
    payload["amendment_1_5"] = AMENDMENT_15
    payload["protocol_id"] = PROTOCOL_ID
    payload["parent_protocol_id"] = PROTOCOL_ID
    payload["output_tree"] = OUT_DIRNAME
    payload["prior_output_tree_not_opened"] = PRIOR_OUT_DIRNAME
    payload["old_outputs_not_rewritten"] = True
    payload["dates_invented"] = False
    payload["session_ordinal_used_as_day"] = False
    payload["competition_holdout_read"] = False
    payload["effect_claimed"] = False
    payload["valid_scientific_claim"] = False
    payload["exploratory"] = True
    payload["not_confirmatory"] = True
    payload["independent_confirmation_required"] = True
    payload["primary_metric"] = PRIMARY_METRIC
    payload["fit_row_floor"] = FIT_ROW_FLOOR
    payload["runtime_ceiling_sec"] = None
    payload["inner_chunk_deadline_sec"] = None
    return payload


def _save_json(path: str, payload: dict) -> None:
    if os.path.exists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _phase_paths(directory: str, phase: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "forward", phase, "*.npz")))


def _load_phase(paths, split: str, dates: dict, *, quarter: str | None = None, keep_quarters=None, keep_arrays: bool = True) -> dict:
    """Stream one forward file at a time. Embeddings are dropped when the caller does not need them."""
    rows = []
    oov = {}
    logits = []
    labels = []
    embeddings = []
    subjects = []
    sessions = []
    tmax = {}
    if quarter is not None or keep_quarters is not None:
        tmax = _stimulus_tmax(paths)
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            base = str(handle["base"])
            session = "ses-11" if "ses-11_task-Sherlock1" in base else "ses-12" if "ses-12_task-Sherlock1" in base else None
        t_max = None
        if quarter is not None or keep_quarters is not None:
            if session is None or session not in tmax:
                raise ProtocolError(f"{os.path.basename(path)} is not a shared Sherlock run")
            t_max = tmax[session]
        loaded = _rows_from_forward(
            path, split, dates,
            quarter=quarter, keep_quarters=keep_quarters, t_max=t_max,
            require_acquisition_date=False,
        )
        oov[loaded["oov"]["base"]] = loaded["oov"]
        rows.extend(loaded["rows"])
        if keep_arrays and len(loaded["labels"]):
            logits.append(loaded["logits"])
            labels.append(loaded["labels"])
            embeddings.append(loaded["embedding"])
            subjects.append(loaded["subjects"])
            sessions.extend([loaded["rows"][0]["base"]] * len(loaded["labels"]))
        del loaded
    return {
        "rows": rows,
        "oov": oov,
        "logits": np.concatenate(logits, axis=0) if logits else np.zeros((0, N_CLASSES)),
        "labels": np.concatenate(labels, axis=0) if labels else np.zeros(0, dtype=np.int64),
        "embeddings": np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1)),
        "subjects": np.concatenate(subjects, axis=0) if subjects else np.zeros(0, dtype=np.int64),
        "sessions": sessions,
    }


def q3_row_count(directory: str, dates: dict) -> int:
    paths = _phase_paths(directory, "noise_pool")
    if not paths:
        return 0
    loaded = _load_phase(paths, "train", dates, quarter=FIT_QUARTER, keep_arrays=False)
    return int(len(loaded["rows"]))


def _disk_state(directory: str, dates: dict) -> dict:
    done = []
    for phase in ("noise_pool", "selection", "eval", "eval_extension", "train"):
        for path in _phase_paths(directory, phase):
            done.append(os.path.basename(path)[:-4])
    partial = 0
    stopped = False
    state_path = os.path.join(directory, "partial", "state.npz")
    if os.path.isfile(state_path):
        with np.load(state_path, allow_pickle=False) as handle:
            partial = int(handle["steps"])
            stopped = bool(int(handle["stopped"])) if "stopped" in handle.files else False
    return {
        "done_bases": done,
        "noise_fit": os.path.isfile(os.path.join(directory, "noise", "fit.json")),
        "fit_rows": q3_row_count(directory, dates) if not os.path.isfile(os.path.join(directory, "noise", "fit.json")) else 0,
        "partial_steps": partial,
        "partial_stopped": stopped,
        "score_frozen": os.path.isfile(os.path.join(directory, "score_frozen.json")),
        "score_partial": os.path.isfile(os.path.join(directory, "score_partial.json")),
        "score_heldout": os.path.isfile(os.path.join(directory, "score_heldout_q4.json")),
    }


def _log_files(directory: str, paths, role: str, *, entered_coefficients: bool, entered_selection: bool, scored: bool) -> None:
    for path in paths:
        base = os.path.basename(path)[:-4]
        append_exposure(directory, {
            "role": role,
            "base": base,
            "when": utc_now(),
            "entered_coefficients": entered_coefficients,
            "entered_selection": entered_selection,
            "scored": scored,
            "meg_read": False,
            "source": "checkpointed_forward",
        })


def fit_directory(directory: str, dates: dict) -> dict:
    """Fit on the pre-registered Q3 pool. Do not open eval, the extension, or Q4 labels as coefficients."""
    fit_paths = _phase_paths(directory, "noise_pool")
    val_paths = _phase_paths(directory, "selection")
    if glob.glob(os.path.join(directory, "forward", "eval", "*.npz")):
        raise ProtocolError("eval forwards are present; the noise fit refuses to run after eval was opened")
    if not fit_paths or not val_paths:
        raise ProtocolError("noise fit needs the trunk-unseen pool and every validation forward")
    pool = _load_phase(fit_paths, "train", dates, quarter=FIT_QUARTER, keep_arrays=False)
    selection = _load_phase(val_paths, "selection", dates, keep_arrays=False)
    _log_files(directory, fit_paths, "coefficient_fit", entered_coefficients=True, entered_selection=False, scored=False)
    _log_files(directory, val_paths, "validation_selection", entered_coefficients=False, entered_selection=True, scored=False)
    fitted = fit_equations(pool["rows"] + selection["rows"])
    decision = frozen_day_term_decision(fitted["bundle"]["state"])
    candidates = {name: params_to_json(value) for name, value in fitted["params"].items()}
    return {
        "objective": "gaussian_nll",
        "objective_note": (
            "Gaussian NLL of decoder cross-entropy compares calibration with uniform, constant, "
            "and the random-covariate control. It is not evidence that a forecasting equation holds for MEG."
        ),
        "chosen_on_validation": fitted["chosen_on_validation"],
        "val_nll": fitted["val_nll"],
        "candidates": candidates,
        "state": fitted["bundle"]["state"],
        "fit_rows": fitted["fit_rows"],
        "train_rows": fitted["fit_rows"],
        "selection_rows": fitted["selection_rows"],
        "train_subjects": fitted["train_subjects"],
        "eval_subjects_in_train": [],
        "residual_fit_population": "trunk_unseen_shared_sherlock_Q3_pre_registered_subjects",
        "trunk_unseen_quarter": FIT_QUARTER,
        "held_out_quarter_not_used_for_fit_or_selection": HELD_OUT_QUARTER,
        "in_sample_trunk_trained_residual": False,
        "selection_population": "broad_val_subjects_1_5_10",
        "selection_rows_used_for_coefficients": False,
        "broad_eval_rows_used": False,
        "eval_extension_rows_used": False,
        "q4_not_used_for_fit_or_selection": True,
        "eval_files_read": False,
        "temperature_applied_to_reported_logits": False,
        "day_terms_inestimable": decision["day_terms_inestimable"],
        "day_terms_dropped": list(decision["day_terms_dropped"]),
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
        "date_source": "bids_scans_tsv_only",
        "power": fitted["power"],
        "fit_row_floor": FIT_ROW_FLOOR,
        "candidates_scored": list(CANDIDATES_15),
        "reduced_state": {
            "kept_feature_names": candidates["reduced_state"].get("kept_feature_names"),
            "collapsed_to_constant": candidates["reduced_state"].get("collapsed_to_constant"),
            "ridge_penalty_per_row": candidates["reduced_state"].get("ridge_penalty_per_row"),
        },
        "oov_excluded": {**pool["oov"], **selection["oov"]},
        "amendment_1_5": AMENDMENT_15,
        "old_outputs_not_rewritten": True,
    }


def _candidate_log_variance(matrix: np.ndarray, names, candidate: str, params: dict) -> np.ndarray:
    if candidate == "reduced_state":
        return log_variance_15(matrix, params)
    if candidate == "error_scaling":
        from .amendment_15 import _candidate_matrix_15
        used, _feat = _candidate_matrix_15(matrix, names, candidate)
        return log_variance_15(used, params)
    return log_variance_15(matrix, params)


def _declare_metric(directory: str, labels: np.ndarray) -> dict:
    path = os.path.join(directory, "metric_declaration.json")
    declaration = metric_declaration(labels)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if int(existing["min_class_count"]) != int(declaration["min_class_count"]):
            raise ProtocolError("metric declaration already froze a different support floor")
        if existing.get("primary_metric") != PRIMARY_METRIC:
            raise ProtocolError("metric declaration does not match the pre-registered primary")
        return existing
    _save_json(path, declaration)
    return declaration


def score_frozen_directory(directory: str, dates: dict) -> dict:
    fit_path = os.path.join(directory, "noise", "fit.json")
    with open(fit_path, encoding="utf-8") as handle:
        fit = json.load(handle)
    paths = _phase_paths(directory, "eval") + _phase_paths(directory, "eval_extension")
    if not _phase_paths(directory, "eval"):
        raise ProtocolError("frozen score needs the exploratory eval forwards")
    extension_quarters = tuple(EVAL_EXTENSION_QUARTERS)
    original = _load_phase(_phase_paths(directory, "eval"), "eval", dates, keep_arrays=True)
    extension_paths = _phase_paths(directory, "eval_extension")
    extension = _load_phase(
        extension_paths, "eval", dates, keep_quarters=extension_quarters, keep_arrays=True,
    ) if extension_paths else {"rows": [], "oov": {}, "logits": np.zeros((0, N_CLASSES)), "labels": np.zeros(0, dtype=np.int64), "sessions": [], "subjects": np.zeros(0, dtype=np.int64)}
    _log_files(directory, _phase_paths(directory, "eval"), "score_frozen", entered_coefficients=False, entered_selection=False, scored=True)
    _log_files(directory, extension_paths, "score_frozen_extension", entered_coefficients=False, entered_selection=False, scored=True)
    rows = original["rows"] + extension["rows"]
    if not rows:
        raise ProtocolError("exploratory eval forwards contained no rows")
    logits = np.concatenate([original["logits"], extension["logits"]], axis=0) if len(extension["labels"]) else original["logits"]
    labels = np.concatenate([original["labels"], extension["labels"]], axis=0) if len(extension["labels"]) else original["labels"]
    sessions = original["sessions"] + extension["sessions"]
    declaration = _declare_metric(directory, labels)
    min_count = int(declaration["min_class_count"])
    matrix = transform_with_state(rows, fit["state"])
    if not np.isfinite(matrix).all():
        raise ProtocolError("eval features are non-finite; refusing to invent a day")
    residual = decoder_residual(logits, labels)
    per_arm = {}
    point = {}
    for name in CANDIDATES_15:
        params = params_from_json(fit["candidates"][name])
        log_var = _candidate_log_variance(matrix, fit["state"]["names"], name, params)
        per_arm[name] = nll_rows(residual, params["mu"], log_var)
        point[name] = float(gaussian_nll(residual, params["mu"], log_var))
    contrasts = paired_mean_contrasts(sessions, per_arm, focals=STRUCTURED_ARMS, controls=CONTROLS, draws=BOOTSTRAP_DRAWS)
    primary = bacc10_present(logits, labels, min_count=min_count)
    fifty = bacc10_from_logits(logits, labels)
    return {
        "objective": "gaussian_nll",
        "chosen_on_validation": fit["chosen_on_validation"],
        "eval_nll": point,
        "nll_contrasts": contrasts,
        "nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
        "primary_metric": PRIMARY_METRIC,
        "primary_bacc10_present": primary,
        "primary_defined": primary is not None,
        "min_class_count": min_count,
        "support_powered": declaration.get("support_powered"),
        "primary_bacc10_50": fifty,
        "fifty_class_metric_defined": fifty is not None,
        "frozen_primary_does_not_differ_by_arm": True,
        "frozen_primary_reason": "A scalar temperature is not applied to the reported logits, so top-10 does not differ by noise arm. The arm contrast is the matched head finetune.",
        "temperature_applied_to_reported_logits": False,
        "eval_rows": int(len(labels)),
        "eval_sessions": sorted(set(sessions)),
        "fit_rows": int(fit["fit_rows"]),
        "selection_rows": int(fit["selection_rows"]),
        "candidates_scored": list(CANDIDATES_15),
        "exploratory_not_pristine": True,
        "not_confirmatory": True,
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
        "day_terms_inestimable": bool(fit.get("day_terms_inestimable")),
        "day_terms_dropped": list(fit.get("day_terms_dropped") or []),
        "oov_excluded": {**original["oov"], **extension["oov"]},
        "amendment_1_5": AMENDMENT_15,
    }


def _frozen_log_variance(features: np.ndarray, fit: dict) -> dict:
    names = fit["state"]["names"]
    frozen = {}
    for name in PARTIAL_ARMS_15:
        if name == "uniform":
            continue
        params = params_from_json(fit["candidates"][name])
        frozen[name] = _candidate_log_variance(features, names, name, params)
    return frozen


def partial_train_directory(directory: str, root: str, steps: int, dates: dict) -> dict:
    fit_path = os.path.join(directory, "noise", "fit.json")
    with open(fit_path, encoding="utf-8") as handle:
        fit = json.load(handle)
    paths = _phase_paths(directory, "train")
    if not paths:
        raise ProtocolError("partial train needs cached train embeddings from the real forward")
    loaded = _load_phase(paths, "train", dates, keep_arrays=True)
    if len(loaded["labels"]) == 0:
        raise ProtocolError("partial train has no in-vocabulary rows")
    features = transform_with_state(loaded["rows"], fit["state"])
    frozen = _frozen_log_variance(features, fit)
    schedule = dict(PARTIAL_SCHEDULE_15)
    state_path = os.path.join(directory, "partial", "state.npz")
    if os.path.isfile(state_path):
        with np.load(state_path, allow_pickle=False) as handle:
            if "schedule" in handle.files and str(handle["schedule"]) != schedule["name"]:
                raise ProtocolError("partial state belongs to a different schedule")
            arms = {}
            for name in PARTIAL_ARMS_15:
                arms[name] = {
                    "W": np.asarray(handle[f"W_{name}"], dtype=np.float64),
                    "b": np.asarray(handle[f"b_{name}"], dtype=np.float64),
                }
            done = int(handle["steps"])
    else:
        ckpt, _size, _sha = _require_checkpoint(root)
        model = load_strong_decoder(ckpt)
        if not is_strong_decoder(model):
            raise ProtocolError("loaded object is not src.v3lib.Model")
        weight = model.head_ce.weight.detach().cpu().numpy().astype(np.float64).T
        bias = model.head_ce.bias.detach().cpu().numpy().astype(np.float64)
        del model
        arms = initial_partial_arms_15(weight, bias)
        done = 0
    if done >= int(schedule["max_steps"]):
        return {"steps": done, "advanced": 0, "stopped": True, "schedule": schedule["name"], "trunk_frozen": True}
    advancing = min(int(steps), int(schedule["max_steps"]) - done)
    matched_partial_steps_15(
        arms, loaded["embeddings"], loaded["labels"], frozen, advancing,
        lr=PARTIAL_LR, seed=done,
    )
    step_now = done + advancing
    val_paths = _phase_paths(directory, "selection")
    if not val_paths:
        raise ProtocolError("head finetune needs validation forwards for the stop curve")
    validation = _load_phase(val_paths, "selection", dates, keep_arrays=True)
    curve_path = os.path.join(directory, "partial", "val_curve.json")
    curve = []
    if os.path.isfile(curve_path):
        with open(curve_path, encoding="utf-8") as handle:
            curve = json.load(handle)
    arm_ce = {}
    for name in PARTIAL_ARMS_15:
        arm_ce[name] = float(np.mean(decoder_residual(arm_logits(arms[name], validation["embeddings"]), validation["labels"])))
    control_mean = float(np.mean([arm_ce[name] for name in CONTROLS]))
    curve.append({"step": int(step_now), "control_mean_ce": control_mean, **{f"{name}_ce": arm_ce[name] for name in PARTIAL_ARMS_15}})
    decision = validation_stop_15(curve, schedule)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump(curve, handle)
    if decision["best_step"] == step_now:
        best_payload = {"steps": np.int64(step_now), "schedule": np.array(schedule["name"])}
        for name, arm in arms.items():
            best_payload[f"W_{name}"] = arm["W"]
            best_payload[f"b_{name}"] = arm["b"]
        best_path = os.path.join(directory, "partial", "best_val.npz")
        temporary_best = best_path + ".tmp.npz"
        np.savez_compressed(temporary_best, **best_payload)
        os.replace(temporary_best, best_path)
    payload = {
        "steps": np.int64(step_now),
        "stopped": np.int64(1 if decision["stop"] else 0),
        "stop_reason": np.array(decision["reason"] or ""),
        "best_step": np.int64(decision["best_step"] or 0),
        "schedule": np.array(schedule["name"]),
    }
    for name, arm in arms.items():
        payload[f"W_{name}"] = arm["W"]
        payload[f"b_{name}"] = arm["b"]
    temporary = state_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, state_path)
    return {
        "steps": step_now,
        "advanced": advancing,
        "target": int(schedule["max_steps"]),
        "schedule": schedule["name"],
        "stopped": bool(decision["stop"]),
        "stop_reason": decision["reason"],
        "best_validation_step": decision["best_step"],
        "control_mean_ce": control_mean,
        "trunk_frozen": True,
        "head_finetune": True,
        "noise_equation_frozen": True,
        "head_data": "subject0_nonshared_trunk_trained",
        "in_sample_for_trunk": True,
        "not_a_full_retrain": True,
        "substantive": True,
        "matched": True,
        "arms": list(PARTIAL_ARMS_15),
        "stop_on": schedule["stop_on"],
        "oov_excluded": loaded["oov"],
    }


def score_partial_directory(directory: str, dates: dict) -> dict:
    fit_path = os.path.join(directory, "noise", "fit.json")
    frozen_path = os.path.join(directory, "score_frozen.json")
    declaration_path = os.path.join(directory, "metric_declaration.json")
    state_path = os.path.join(directory, "partial", "state.npz")
    best_path = os.path.join(directory, "partial", "best_val.npz")
    for path in (fit_path, frozen_path, declaration_path, state_path, best_path):
        if not os.path.isfile(path):
            raise ProtocolError(f"partial score needs {os.path.basename(path)} written before the hit rates")
    with open(declaration_path, encoding="utf-8") as handle:
        declaration = json.load(handle)
    if declaration.get("primary_metric") != PRIMARY_METRIC or not declaration.get("declared_before_scoring"):
        raise ProtocolError("primary metric was not declared before scoring")
    with open(frozen_path, encoding="utf-8") as handle:
        frozen = json.load(handle)
    with np.load(state_path, allow_pickle=False) as handle:
        steps = int(handle["steps"])
        stopped = bool(int(handle["stopped"])) if "stopped" in handle.files else False
        schedule_name = str(handle["schedule"]) if "schedule" in handle.files else ""
    if schedule_name != PARTIAL_SCHEDULE_15["name"]:
        raise ProtocolError("this partial state is not the 1.5 substantive comparison")
    if not stopped and steps < int(PARTIAL_SCHEDULE_15["max_steps"]):
        raise ProtocolError(f"partial score refused at {steps} steps before the validation stop")
    with np.load(best_path, allow_pickle=False) as handle:
        arms = {name: {"W": np.asarray(handle[f"W_{name}"]), "b": np.asarray(handle[f"b_{name}"])} for name in PARTIAL_ARMS_15}
        scored_step = int(handle["steps"])
    paths = _phase_paths(directory, "eval")
    extension_paths = _phase_paths(directory, "eval_extension")
    original = _load_phase(paths, "eval", dates, keep_arrays=True)
    extension = _load_phase(extension_paths, "eval", dates, keep_quarters=tuple(EVAL_EXTENSION_QUARTERS), keep_arrays=True) if extension_paths else None
    _log_files(directory, paths, "score_partial", entered_coefficients=False, entered_selection=False, scored=True)
    if extension_paths:
        _log_files(directory, extension_paths, "score_partial_extension", entered_coefficients=False, entered_selection=False, scored=True)
    if extension is None or len(extension["labels"]) == 0:
        embeddings = original["embeddings"]
        labels = original["labels"]
        sessions = original["sessions"]
        oov = original["oov"]
    else:
        embeddings = np.concatenate([original["embeddings"], extension["embeddings"]], axis=0)
        labels = np.concatenate([original["labels"], extension["labels"]], axis=0)
        sessions = original["sessions"] + extension["sessions"]
        oov = {**original["oov"], **extension["oov"]}
    min_count = int(declaration["min_class_count"])
    scores = {}
    logits = {}
    for name in PARTIAL_ARMS_15:
        logits[name] = arm_logits(arms[name], embeddings)
        scores[name] = bacc10_present(logits[name], labels, min_count=min_count)
    if any(value is None for value in scores.values()):
        raise ProtocolError("primary metric is undefined; the declaration floor removed every class")
    fifty = {name: bacc10_from_logits(logits[name], labels) for name in PARTIAL_ARMS_15}
    contrasts = paired_bacc_contrasts(sessions, labels, logits, min_count=min_count, draws=BOOTSTRAP_DRAWS)
    decision = decision_from_contrasts(contrasts, frozen.get("nll_contrasts") or {"contrasts": {}})
    paired = scores["learned_state"] - scores["uniform"]
    return {
        "steps": scored_step,
        "trained_through_step": steps,
        "schedule": schedule_name,
        "substantive": True,
        "matched": True,
        "primary_metric": PRIMARY_METRIC,
        "min_class_count": min_count,
        "primary_bacc10_present": scores,
        "primary_defined": True,
        "primary_bacc10_50": fifty,
        "fifty_class_metric_defined": all(value is not None for value in fifty.values()),
        "learned_minus_uniform": float(paired),
        "reduced_minus_uniform": float(scores["reduced_state"] - scores["uniform"]),
        "bacc_contrasts": contrasts,
        "bacc_equivalence_margin": BACC_EQUIVALENCE_MARGIN,
        "decision": decision,
        "decisive": decision["decisive"],
        "exploratory_not_pristine": True,
        "trunk_frozen": True,
        "not_a_full_retrain": True,
        "effect_claimed": False,
        "eval_rows": int(len(labels)),
        "fit_rows": int(frozen.get("fit_rows") or 0),
        "oov_excluded": oov,
        "amendment_1_5": AMENDMENT_15,
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
    }


def score_heldout_directory(directory: str, dates: dict) -> dict:
    """Q4 of the coefficient recordings. Not used to fit, select, or choose the support floor."""
    fit_path = os.path.join(directory, "noise", "fit.json")
    declaration_path = os.path.join(directory, "metric_declaration.json")
    with open(fit_path, encoding="utf-8") as handle:
        fit = json.load(handle)
    with open(declaration_path, encoding="utf-8") as handle:
        declaration = json.load(handle)
    paths = _phase_paths(directory, "noise_pool")
    loaded = _load_phase(paths, "heldout", dates, quarter=HELD_OUT_QUARTER, keep_arrays=True)
    _log_files(directory, paths, "score_heldout_q4", entered_coefficients=False, entered_selection=False, scored=True)
    if len(loaded["labels"]) == 0:
        return {
            "heldout_rows": 0,
            "entered_coefficients": False,
            "entered_selection": False,
            "primary_bacc10_present": None,
            "decisive": False,
            "reason": "no in-vocabulary Q4 rows",
            "amendment_1_5": AMENDMENT_15,
        }
    matrix = transform_with_state(loaded["rows"], fit["state"])
    residual = decoder_residual(loaded["logits"], loaded["labels"])
    point = {}
    for name in CANDIDATES_15:
        params = params_from_json(fit["candidates"][name])
        log_var = _candidate_log_variance(matrix, fit["state"]["names"], name, params)
        point[name] = float(gaussian_nll(residual, params["mu"], log_var))
    primary = bacc10_present(loaded["logits"], loaded["labels"], min_count=int(declaration["min_class_count"]))
    return {
        "heldout_quarter": HELD_OUT_QUARTER,
        "heldout_rows": int(len(loaded["labels"])),
        "entered_coefficients": False,
        "entered_selection": False,
        "used_for_selection": False,
        "used_to_choose_support": False,
        "eval_nll": point,
        "primary_metric": PRIMARY_METRIC,
        "primary_bacc10_present": primary,
        "primary_bacc10_50": bacc10_from_logits(loaded["logits"], loaded["labels"]),
        "frozen_logits_only": True,
        "decisive": False,
        "effect_claimed": False,
        "dates_invented": False,
        "session_ordinal_used_as_day": False,
        "oov_excluded": loaded["oov"],
        "amendment_1_5": AMENDMENT_15,
    }


def _plan_summary(plan: dict) -> dict:
    summary = {}
    for phase, rows in plan["phases"].items():
        summary[phase] = [
            {"base": row["base"], "date": row["date"], "subject": row["subject"], "order_index": row["order_index"]}
            for row in rows
        ]
    summary["refused_holdout"] = plan["refused_holdout"]
    return summary


def run_dry(root: str) -> dict:
    directory = out_dir(root)
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, "pilot", "dry_run.json")
    if os.path.isfile(dest):
        with open(dest, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["already_written"] = True
        payload["rewritten"] = False
        return payload
    cache_cont, data_root = _cache_paths(root)
    if "holdout" in cache_cont.lower() or "holdout" in str(data_root).lower():
        raise ProtocolError("refusing a holdout path")
    cache_present = os.path.isdir(cache_cont)
    bases = list_cache_bases(cache_cont) if cache_present else []
    dates = read_dates(data_root) if os.path.isdir(data_root) else {}
    plan = plan_recordings(bases, dates)
    payload = honesty_15(
        stage="dry-run",
        next_action="run_next_for_the_live_measurement",
        meg_read=False,
        directory_listing_is_not_an_exposure=True,
        cache_present=cache_present,
        n_recordings_listed=len(bases),
        plan=_plan_summary(plan),
        power=power_report(),
        timers=timers_coherent(None),
        synthetic=False,
        measurement_complete=False,
        decisive=False,
    )
    _save_json(dest, payload)
    payload["wrote"] = dest
    return payload


def run_timing_pilot(root: str) -> dict:
    """Time the reduced arm and the paired bootstrap on synthetic rows. Do not read MEG."""
    directory = out_dir(root)
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, "pilot", "timing_pilot.json")
    if os.path.isfile(dest):
        with open(dest, encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["already_written"] = True
        payload["rewritten"] = False
        return payload
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    z = rng.normal(size=(64, 4))
    z[:, 3] = 0.0
    residual = z[:, 0].copy()
    names = ["run_fraction", "artifact_rms", "bandpower", "log_train_window_count"]
    reduced = fit_reduced_state(z, residual, names)
    sessions = np.array(["ses-a"] * 32 + ["ses-b"] * 32)
    values = {
        "uniform": np.zeros(64),
        "constant": np.full(64, 0.1),
        "random_covariate": np.full(64, 0.2),
        "learned_state": residual,
        "reduced_state": residual * 0.5,
    }
    contrasts = paired_mean_contrasts(sessions, values, draws=50)
    elapsed = time.perf_counter() - started
    payload = honesty_15(
        stage="timing-pilot",
        next_action="timing_pilot_is_not_the_measurement",
        synthetic=True,
        meg_read=False,
        competition_holdout_read=False,
        elapsed_sec=elapsed,
        inner_chunk_deadline_sec=None,
        runtime_ceiling_sec=None,
        timers=timers_coherent(None),
        reduced_kept_feature_names=reduced["kept_feature_names"],
        reduced_collapsed_to_constant=reduced["collapsed_to_constant"],
        bootstrap_defined=contrasts["defined"],
        measurement_complete=False,
        decisive=False,
        effect_claimed=False,
    )
    _save_json(dest, payload)
    payload["wrote"] = dest
    return payload


def _write_provenance(directory: str, base: str, phase: str, info: dict, dates: dict) -> None:
    path = os.path.join(directory, "provenance", base + ".json")
    if os.path.exists(path):
        return
    payload = {
        "amendment": AMENDMENT_15,
        "base": base,
        "phase": phase,
        "checkpoint_sha": info.get("sha"),
        "acquisition_date": dates.get(base) or None,
        "n_words": info.get("n_words"),
        "session_ordinal_used_as_day": False,
        "dates_invented": False,
        "when": utc_now(),
    }
    _save_json(path, payload)


def run_next(root: str) -> dict:
    directory = out_dir(root)
    os.makedirs(directory, exist_ok=True)
    cache_cont, data_root = _cache_paths(root)
    if "holdout" in cache_cont.lower() or "holdout" in str(data_root).lower():
        raise ProtocolError("refusing a holdout path")
    if not os.path.isdir(cache_cont):
        raise ProtocolError(f"missing continuous cache {cache_cont}")
    bases = list_cache_bases(cache_cont)
    dates = read_dates(data_root)
    plan = plan_recordings(bases, dates)
    state = _disk_state(directory, dates)
    action = next_action_15(plan, **state)
    report = honesty_15(
        stage=action["stage"],
        next_action=action["next_action"],
        reason=action.get("reason"),
        lock_held=True,
        schedule=action.get("schedule"),
    )
    if action["stage"] == "stop":
        partial_path = os.path.join(directory, "score_partial.json")
        if os.path.isfile(partial_path):
            with open(partial_path, encoding="utf-8") as handle:
                partial = json.load(handle)
            report["measurement_complete"] = bool(partial.get("measurement_complete", False))
            report["decisive"] = bool(partial.get("decisive", False))
            report["learned_minus_uniform"] = partial.get("learned_minus_uniform")
            report["primary_defined"] = partial.get("primary_defined")
            report["fifty_class_metric_defined"] = partial.get("fifty_class_metric_defined")
        else:
            report["measurement_complete"] = False
            report["decisive"] = False
        report["runs_completed"] = action.get("reason") == "stages_complete"
        return report
    if action["stage"] == "forward":
        begin_chunk(directory, action["base"], action["phase"])
        info = execute_forward(directory, root, action["base"], action["phase"])
        finish_chunk(directory, action["base"], action["phase"])
        _write_provenance(directory, action["base"], action["phase"], info, dates)
        append_exposure(directory, {
            "role": "forward",
            "base": action["base"],
            "phase": action["phase"],
            "when": utc_now(),
            "acquisition_date": dates.get(action["base"]) or None,
            "entered_coefficients": action["phase"] == "noise_pool",
            "entered_selection": action["phase"] == "selection",
            "scored": action["phase"] in ("eval", "eval_extension"),
            "meg_read": True,
            "checkpoint_sha": info.get("sha"),
            "n_words": info.get("n_words"),
        })
        report["base"] = info["base"]
        report["phase"] = info["phase"]
        report["n_words"] = info["n_words"]
        report["decoder"] = info["decoder"]
        return report
    if action["stage"] == "noise-fit":
        payload = fit_directory(directory, dates)
        _save_json(os.path.join(directory, "noise", "fit.json"), payload)
        report["chosen_on_validation"] = payload["chosen_on_validation"]
        report["fit_rows"] = payload["fit_rows"]
        report["day_terms_inestimable"] = payload["day_terms_inestimable"]
        report["day_terms_dropped"] = payload["day_terms_dropped"]
        return report
    if action["stage"] == "score-frozen":
        payload = score_frozen_directory(directory, dates)
        _save_json(os.path.join(directory, "score_frozen.json"), honesty_15(payload))
        report["primary_bacc10_present"] = payload["primary_bacc10_present"]
        report["primary_defined"] = payload["primary_defined"]
        report["fifty_class_metric_defined"] = payload["fifty_class_metric_defined"]
        report["measurement_complete"] = False
        return report
    if action["stage"] == "partial-train":
        info = partial_train_directory(directory, root, action["steps"], dates)
        report.update({key: info[key] for key in info if key != "oov_excluded"})
        report["oov_excluded"] = info.get("oov_excluded")
        return report
    if action["stage"] == "score-partial":
        payload = score_partial_directory(directory, dates)
        _ckpt, size, _sha = _require_checkpoint(root)
        complete = measurement_complete_15(
            checkpoint_bytes=size,
            fit_rows=payload["fit_rows"],
            primary_defined=payload["primary_defined"],
            lock_held=lock_held(),
        )
        payload["measurement_complete"] = complete
        payload["checkpoint_bytes"] = size
        saved = honesty_15(payload)
        saved["measurement_complete"] = complete
        saved["decisive"] = payload["decisive"]
        _save_json(os.path.join(directory, "score_partial.json"), saved)
        report["measurement_complete"] = complete
        report["decisive"] = payload["decisive"]
        report["learned_minus_uniform"] = payload["learned_minus_uniform"]
        report["primary_defined"] = True
        report["fifty_class_metric_defined"] = payload["fifty_class_metric_defined"]
        return report
    if action["stage"] == "score-heldout-q4":
        payload = score_heldout_directory(directory, dates)
        _save_json(os.path.join(directory, "score_heldout_q4.json"), honesty_15(payload))
        report["heldout_rows"] = payload["heldout_rows"]
        report["entered_coefficients"] = False
        return report
    raise ProtocolError(action["stage"])


def _status(payload: dict) -> dict:
    return {
        "protocol_id": payload.get("protocol_id", PROTOCOL_ID),
        "amendment": AMENDMENT_15,
        "stage": payload.get("stage"),
        "next_action": payload.get("next_action"),
        "reason": payload.get("reason"),
        "effect_claimed": False,
        "valid_scientific_claim": False,
        "measurement_complete": bool(payload.get("measurement_complete", False)),
        "decisive": payload.get("decisive"),
        "primary_defined": payload.get("primary_defined"),
        "fifty_class_metric_defined": payload.get("fifty_class_metric_defined"),
        "learned_minus_uniform": payload.get("learned_minus_uniform"),
        "fit_rows": payload.get("fit_rows"),
        "fit_row_floor": FIT_ROW_FLOOR,
        "runtime_ceiling_sec": None,
        "inner_chunk_deadline_sec": None,
        "real_meg_scientific_validation": False,
        "full_retrain_launched": False,
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    command = argv[0] if argv else "next"
    command = {"run": "next"}.get(command, command)
    try:
        if command == "manifest":
            print(json.dumps(_status(honesty_15(
                stage="manifest",
                next_action="dry-run",
                power=power_report(),
                measurement_complete=False,
            ))))
            return 0
        if command == "dry-run":
            print(json.dumps(_status(run_dry(root))))
            return 0
        if command == "timing-pilot":
            print(json.dumps(_status(run_timing_pilot(root))))
            return 0
        if command == "full-retrain":
            print(json.dumps(_status(honesty_15(stage="full-retrain", next_action="do_not_launch_full_retrain"))))
            print("blocker: full retrain is impractical under CPU 4G and is not launched", file=sys.stderr)
            return 2
        if os.environ.get("SDR_FORBID_LIVE_COMPUTE") == "1":
            print("blocker: refusing live compute because SDR_FORBID_LIVE_COMPUTE is set", file=sys.stderr)
            return 2
        if not lock_held():
            print("blocker: scheduler self-lock is not held; refusing to run beside MICrONS or ANN", file=sys.stderr)
            return 2
        hardware = verify_live_hardware()
        timers_coherent(hardware.get("runtime_max_sec"))
        if command != "next":
            print(
                "usage: python -m campaigns.sdr_uncertainty.strong_decoder_15_runner "
                "{manifest|dry-run|timing-pilot|next|full-retrain}",
                file=sys.stderr,
            )
            return 2
        payload = run_next(root)
        payload["timers"] = timers_coherent(hardware.get("runtime_max_sec"))
        print(json.dumps(_status(payload)))
        return 0
    except (CgroupError, ProtocolError, FileNotFoundError, OSError) as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        print(json.dumps(honesty_15(stage=command, blocker=str(exc), next_action="fix_blocker")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
