"""Checkpointed strong-decoder stages. One launch does one chunk and exits.

Does not start the 8-epoch deep_c fit. Does not overwrite deep_c_ep6.pt, the
linear-probe caches, or an earlier score file. effect_claimed stays false.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

from .cgroup import CgroupError, verify_live_hardware
from .strong_decoder import is_strong_decoder, load_strong_decoder
from .strong_forward import alignment_for, forward_recording, load_events_safe
from .strong_noise import (
    AMENDMENT,
    CANDIDATES,
    VAL_SUBJECTS,
    CHECKPOINT_BASENAME,
    CHECKPOINT_BYTES,
    ERROR_SCALING_FEATURES,
    ESTIMAND,
    FULL_RETRAIN,
    LPP_NOTE,
    PARTIAL_LR,
    PARTIAL_TARGET_STEPS,
    PROTOCOL_ID,
    SMOKE_SCHEDULE,
    SUBSTANTIVE_SCHEDULE,
    ProtocolError,
    arm_logits,
    assemble_features,
    bacc10_from_logits,
    coverage_report,
    decoder_residual,
    fit_noise,
    initial_partial_arms,
    log_variance,
    matched_partial_steps,
    measurement_status,
    quarter_labels,
    select_on_validation,
    temperature_changes_top10,
    transform_with_state,
    validation_stop,
    weighted_nll,
)
from .strong_plan import next_action, plan_order

LOCK_NAME = "libribrain-strong-decoder-noise"
OUT_DIRNAME = "strong_decoder_1_4"


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def assert_output_dir(path: str, root: str) -> str:
    real = os.path.realpath(path)
    repo = os.path.realpath(root)
    if real == repo or real.startswith(repo + os.sep):
        raise ProtocolError("refusing to write inside the git repository")
    lowered = real.lower()
    if "holdout" in lowered or "competition_holdout" in lowered:
        raise ProtocolError("refusing a holdout path")
    if os.path.basename(real) != OUT_DIRNAME:
        raise ProtocolError("outputs go only to strong_decoder_1_4")
    return real


def out_dir(root: str) -> str:
    base = os.environ.get("LIBRIBRAIN_SDR_OUT", "/mnt/ssd4tb/libribrain_sdr_out")
    return assert_output_dir(os.path.join(base, OUT_DIRNAME), root)


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_held() -> bool:
    if os.environ.get("SDR_STRONG_LOCK") != "1":
        return False
    path = os.environ.get("SDR_LOCK_TOKEN", "")
    if not path or not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip() == LOCK_NAME


def honesty(**extra) -> dict:
    payload = {
        "amendment": AMENDMENT,
        "protocol_id": PROTOCOL_ID,
        "estimand": ESTIMAND,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "valid_scientific_claim": False,
        "accuracy_claimed_from_temperature": False,
        "scalar_row_temperature_changes_top10": False,
        "frozen_trunk_subject0_days_are_holdout": False,
        "eval_subjects_are_pristine_confirmatory_holdout": False,
        "full_retrain": dict(FULL_RETRAIN),
        "old_outputs_not_rewritten": True,
        "lpp_pooled": False,
        "competition_holdout_read": False,
        "licences": {"libribrain": "CC-BY-NC-4.0", "lpp": "PDDL"},
    }
    payload.update(extra)
    payload["effect_claimed"] = False
    payload["valid_scientific_claim"] = False
    payload["exploratory"] = True
    payload["not_confirmatory"] = True
    payload["independent_confirmation_required"] = True
    payload["accuracy_claimed_from_temperature"] = False
    payload["frozen_trunk_subject0_days_are_holdout"] = False
    payload["eval_subjects_are_pristine_confirmatory_holdout"] = False
    payload["lpp_pooled"] = False
    payload["full_retrain"] = dict(FULL_RETRAIN)
    payload["numeric_precision"] = {
        "this_run": "fp32_cpu_when_cuda_hidden",
        "leaderboard_deep_c": "bf16_cuda_autocast",
        "direct_score_comparison_to_leaderboard": False,
    }
    return payload


def _stage_status(directory) -> dict:
    names = (
        "forward-selection", "forward-train", "noise-fit", "forward-eval",
        "score-frozen", "partial-train", "score-partial", "timing-control",
        "lpp-transfer", "full-retrain",
    )
    status = {name: "not_run" for name in names}
    status["full-retrain"] = "refused_impractical"
    if not directory or not os.path.isdir(directory):
        return status
    for phase in ("noise_pool", "selection", "train", "eval"):
        found = glob.glob(os.path.join(directory, "forward", phase, "*.npz"))
        if found:
            status[f"forward-{phase}"] = f"checkpointed_{len(found)}"
    if os.path.isfile(os.path.join(directory, "noise", "fit.json")):
        status["noise-fit"] = "complete_not_refit"
    if os.path.isfile(os.path.join(directory, "partial", "state.npz")):
        status["partial-train"] = "checkpointed"
    if os.path.isfile(os.path.join(directory, "score_frozen.json")):
        status["score-frozen"] = "written_not_overwritten"
    if os.path.isfile(os.path.join(directory, "score_partial.json")):
        status["score-partial"] = "written_not_overwritten"
    if glob.glob(os.path.join(directory, "timing", "*.json")):
        status["timing-control"] = "written"
    if glob.glob(os.path.join(directory, "lpp", "*.json")):
        status["lpp-transfer"] = "written_not_pooled"
    return status


def manifest(root: str) -> dict:
    code = {}
    for rel in ("src/v3lib.py", "src/models.py", "scripts/train_v3.py", "scripts/predict_holdout_v3.py", "src/split2.py"):
        code[rel] = _sha256_file(os.path.join(root, rel))
    ckpt = os.environ.get("LB_STRONG_CKPT", os.path.join(root, "runs", CHECKPOINT_BASENAME))
    cache = os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache")
    lpp = os.environ.get("LIBRIBRAIN_LPP_CACHE", os.path.join(cache, "ext_lpp_listen"))
    directory = None
    try:
        directory = out_dir(root)
    except ProtocolError:
        directory = None
    return honesty(
        runs_completed=False,
        code_sha256=code,
        checkpoint={
            "path": ckpt,
            "present": os.path.isfile(ckpt),
            "bytes": os.path.getsize(ckpt) if os.path.isfile(ckpt) else None,
            "expected_bytes_if_deep_c_ep6": CHECKPOINT_BYTES,
            "sha256": _sha256_file(ckpt),
        },
        cache={"path": os.path.join(cache, "cont"), "present": os.path.isdir(os.path.join(cache, "cont"))},
        lpp={"path": lpp, "present": os.path.isdir(lpp), "stage": "not_run", "pooling": "forbidden", "note": LPP_NOTE},
        leaderboard_not_comparable={"broad_top10": 0.72230, "deep_top10": 0.72929},
        stages=_stage_status(directory),
        commands={
            "next": 'cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh next',
            "timing-control": 'cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh timing-control',
            "lpp-transfer": 'cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh lpp-transfer',
            "full-retrain": "refused; partial-train inside next is the matched budget. A full 8x1500 retrain is not launched.",
        },
    )


def list_cache_bases(cache_cont: str) -> list[str]:
    bases = []
    for path in sorted(glob.glob(os.path.join(cache_cont, "*_ev.npz"))):
        base = os.path.basename(path)[:-7]
        if os.path.isfile(os.path.join(cache_cont, base + ".npy")):
            bases.append(base)
    return bases


def read_dates(data_root: str) -> dict:
    dates = {}
    if not data_root or not os.path.isdir(data_root):
        return dates
    scans = glob.glob(os.path.join(data_root, "*", "sub-*", "ses-*", "*_scans.tsv"))
    for path in scans:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) < 2 or not row[0].startswith("meg/"):
                    continue
                base = os.path.basename(row[0]).replace("_meg.fif", "")
                text = row[1][:10]
                if len(text) == 10 and text[4] == "-" and text[7] == "-":
                    dates[base] = text
    return dates


def _disk_state(directory: str) -> dict:
    done = []
    for phase in ("noise_pool", "selection", "train", "eval"):
        for path in glob.glob(os.path.join(directory, "forward", phase, "*.npz")):
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
        "partial_steps": partial,
        "partial_stopped": stopped,
        "score_frozen": os.path.isfile(os.path.join(directory, "score_frozen.json")),
        "score_partial": os.path.isfile(os.path.join(directory, "score_partial.json")),
    }


def _params_to_json(params: dict) -> dict:
    payload = dict(params)
    payload["slope"] = np.asarray(params["slope"], dtype=np.float64).tolist()
    return payload


def _params_from_json(payload: dict) -> dict:
    params = dict(payload)
    params["slope"] = np.asarray(payload["slope"], dtype=np.float64)
    return params


def _shared_session(base: str) -> str | None:
    if "ses-11_task-Sherlock1" in base:
        return "ses-11"
    if "ses-12_task-Sherlock1" in base:
        return "ses-12"
    return None


def _stimulus_tmax(paths: list[str]) -> dict:
    tmax = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            if "timechapter" not in handle.files:
                raise ProtocolError(f"{os.path.basename(path)} has no timechapter; refusing to invent a quarter")
            session = _shared_session(str(handle["base"]))
            if session is None:
                continue
            tmax[session] = max(tmax.get(session, 0.0), float(np.max(handle["timechapter"])))
    return tmax


def _rows_from_forward(path: str, split: str, dates: dict, *, quarter: str | None = None, t_max: float | None = None) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        base = str(handle["base"])
        subject = int(handle["subject"])
        labels = np.asarray(handle["label"], dtype=np.int64)
        logits = np.asarray(handle["logits"], dtype=np.float64)
        date = dates.get(base)
        if not date:
            raise ProtocolError(f"no acquisition date for {base}; refusing to invent day_rank")
        keep = np.ones(len(labels), dtype=bool)
        quarters = None
        if quarter is not None:
            if "timechapter" not in handle.files:
                raise ProtocolError(f"{base} has no timechapter; refusing to invent a quarter")
            quarters = quarter_labels(handle["timechapter"], t_max)
            keep = quarters == quarter
        residual = decoder_residual(logits, labels)
        rows = []
        for index in np.flatnonzero(keep):
            rows.append({
                "base": base,
                "subject": subject,
                "date": date,
                "split": split,
                "quarter": None if quarters is None else str(quarters[index]),
                "run_fraction": float(handle["run_fraction"][index]),
                "artifact_rms": float(handle["artifact_rms"][index]),
                "bandpower": float(handle["bandpower"][index]),
                "group_windows": int(np.sum(keep)),
                "residual": float(residual[index]),
            })
        return rows, logits[keep], labels[keep], np.full(int(np.sum(keep)), subject, dtype=np.int64)


def _candidate_matrix(z: np.ndarray, names: list[str], candidate: str):
    if candidate == "error_scaling":
        columns = [names.index(feature) for feature in ERROR_SCALING_FEATURES]
        return z[:, columns], list(ERROR_SCALING_FEATURES)
    return z, list(names)


def fit_noise_directory(directory: str, dates: dict) -> dict:
    """Fit on subject-0 shared Q3 only. Select on Broad validation rows. Do not open eval or head-train files.

    Q3 was outside deep_c train_mask. Q1/Q2 and non-shared subject-0 sessions were not.
    Validation rows choose the candidate and are not used as coefficient rows.
    """
    fit_paths = sorted(glob.glob(os.path.join(directory, "forward", "noise_pool", "*.npz")))
    val_paths = sorted(glob.glob(os.path.join(directory, "forward", "selection", "*.npz")))
    if not fit_paths or not val_paths:
        raise ProtocolError("noise fit needs the subject-0 shared forwards and every validation forward")
    tmax = _stimulus_tmax(fit_paths)
    rows = []
    for path in fit_paths:
        with np.load(path, allow_pickle=False) as handle:
            session = _shared_session(str(handle["base"]))
        if session is None or session not in tmax:
            raise ProtocolError(f"{os.path.basename(path)} is not a shared Sherlock run")
        part, _logits, _labels, _subjects = _rows_from_forward(path, "train", dates, quarter="Q3", t_max=tmax[session])
        rows.extend(part)
    n_fit = len(rows)
    if n_fit < 2:
        raise ProtocolError("Q3 trunk-unseen block has fewer than two rows; not fitting on trunk-trained sessions instead")
    for path in val_paths:
        part, _logits, _labels, _subjects = _rows_from_forward(path, "selection", dates)
        rows.extend(part)
    if any(row["split"] == "eval" for row in rows):
        raise ProtocolError("eval rows were passed to the noise fit")
    if any(row["quarter"] in ("other", "Q1", "Q2") for row in rows if row["split"] == "train"):
        raise ProtocolError("refusing in-sample trunk-trained residuals for the noise fit")
    fit_subjects = {int(row["subject"]) for row in rows if row["split"] == "train"}
    if fit_subjects != {0}:
        raise ProtocolError(f"noise fit subjects {sorted(fit_subjects)} are not subject 0 Q3")
    bundle = assemble_features(rows)
    residual = np.asarray([row["residual"] for row in rows], dtype=np.float64)
    train = bundle["train_index"]
    if int(np.sum(train)) != n_fit:
        raise ProtocolError("selection rows leaked into the coefficient fit")
    params = {}
    val_nll = {}
    for name in CANDIDATES:
        matrix, feat_names = _candidate_matrix(bundle["z"], bundle["names"], name)
        params[name] = fit_noise(matrix[train], residual[train], name, feature_names=feat_names, seed=0)
        val_nll[name] = weighted_nll(residual[~train], matrix[~train], params[name])
    chosen = select_on_validation(val_nll)
    return {
        "objective": "gaussian_nll",
        "objective_note": (
            "Gaussian NLL of decoder cross-entropy is a calibration comparison against uniform and constant. "
            "It is not evidence that a forecasting equation holds for MEG."
        ),
        "chosen_on_validation": chosen,
        "val_nll": {name: float(value) for name, value in val_nll.items()},
        "candidates": {name: _params_to_json(value) for name, value in params.items()},
        "state": bundle["state"],
        "fit_rows": int(n_fit),
        "train_rows": int(n_fit),
        "selection_rows": int(len(rows) - n_fit),
        "train_subjects": [0],
        "eval_subjects_in_train": [],
        "residual_fit_population": "subject0_shared_sherlock_Q3",
        "trunk_unseen_quarter": "Q3",
        "in_sample_trunk_trained_residual": False,
        "selection_population": "broad_val_subjects_1_5_10",
        "selection_rows_used_for_coefficients": False,
        "broad_eval_rows_used": False,
        "q4_not_used_for_fit_or_selection": True,
        "eval_files_read": False,
        "temperature_applied_to_reported_logits": False,
    }


def score_frozen_directory(directory: str, dates: dict) -> dict:
    with open(os.path.join(directory, "noise", "fit.json"), encoding="utf-8") as handle:
        fit = json.load(handle)
    paths = sorted(glob.glob(os.path.join(directory, "forward", "eval", "*.npz")))
    if not paths:
        raise ProtocolError("frozen score needs the exploratory eval forwards")
    rows = []
    logits = []
    labels = []
    subjects = []
    sessions = []
    for path in paths:
        part_rows, part_logits, part_labels, part_subjects = _rows_from_forward(path, "eval", dates)
        if not part_rows:
            continue
        rows.extend(part_rows)
        logits.append(part_logits)
        labels.append(part_labels)
        subjects.append(part_subjects)
        sessions.extend(part_rows[0]["base"] for _ in part_labels)
    if not rows:
        raise ProtocolError("exploratory eval forwards contained no rows")
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    subjects = np.concatenate(subjects, axis=0)
    matrix_z = transform_with_state(rows, fit["state"])
    residual = decoder_residual(logits, labels)
    nll = {}
    changed = False
    for name in CANDIDATES:
        matrix, _names = _candidate_matrix(matrix_z, fit["state"]["names"], name)
        params = _params_from_json(fit["candidates"][name])
        nll[name] = weighted_nll(residual, matrix, params)
        changed = changed or temperature_changes_top10(logits, log_variance(matrix, params))
    bacc = bacc10_from_logits(logits, labels)
    chosen = fit["chosen_on_validation"]
    coverage = coverage_report(logits, labels, subjects)
    return {
        "objective": "gaussian_nll",
        "objective_note": fit.get("objective_note"),
        "chosen_on_validation": chosen,
        "eval_nll": {name: float(value) for name, value in nll.items()},
        "paired_vs_uniform": float(nll[chosen] - nll["uniform"]),
        "primary_bacc10_50": bacc,
        "fifty_class_metric_defined": bacc is not None,
        "coverage": coverage,
        "eligible_metrics": coverage["eligible_metrics"],
        "temperature_changes_top10": bool(changed),
        "temperature_applied_to_reported_logits": False,
        "eval_rows": int(len(labels)),
        "eval_sessions": sorted(set(sessions)),
        "exploratory_not_pristine": True,
        "not_confirmatory": True,
        "independent_confirmation_required": True,
        "candidates_scored": list(CANDIDATES),
        "fit_rows": int(fit.get("fit_rows", fit.get("train_rows", 0))),
        "train_rows": int(fit["train_rows"]),
        "selection_rows": int(fit["selection_rows"]),
        "eval_subjects_in_train": list(fit["eval_subjects_in_train"]),
        "in_sample_trunk_trained_residual": bool(fit.get("in_sample_trunk_trained_residual", True)),
        "residual_fit_population": fit.get("residual_fit_population"),
    }


def _write_forward(directory, phase, base, subject, result, sha: str) -> str:
    folder = os.path.join(directory, "forward", phase)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, base + ".npz")
    if os.path.exists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    np.savez_compressed(
        path,
        base=np.array(base),
        subject=np.int64(subject),
        embedding=np.asarray(result["embedding"], dtype=np.float32),
        logits=np.asarray(result["logits"], dtype=np.float32),
        label=np.asarray(result["label"], dtype=np.int64),
        run_fraction=np.asarray(result["run_fraction"], dtype=np.float64),
        artifact_rms=np.asarray(result["artifact_rms"], dtype=np.float64),
        bandpower=np.asarray(result["bandpower"], dtype=np.float64),
        checkpoint_sha=np.array(sha),
        timing=np.array(result["timing"]),
        **({} if result.get("timechapter") is None else {"timechapter": np.asarray(result["timechapter"], dtype=np.float64)}),
    )
    return path


def _checkpoint_path(root: str) -> str:
    return os.environ.get("LB_STRONG_CKPT", os.path.join(root, "runs", CHECKPOINT_BASENAME))


def _require_checkpoint(root: str) -> tuple[str, int, str]:
    path = _checkpoint_path(root)
    if not os.path.isfile(path):
        raise ProtocolError(f"missing strong checkpoint {path}")
    size = os.path.getsize(path)
    if os.path.basename(path) == CHECKPOINT_BASENAME and size != CHECKPOINT_BYTES:
        raise ProtocolError(f"checkpoint size {size} != {CHECKPOINT_BYTES}")
    return path, size, _sha256_file(path)


def execute_forward(directory, root, base, phase, *, timing: str = "zero", write: bool = True) -> dict:
    cache = os.path.join(os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache"), "cont")
    events_path = os.path.join(cache, base + "_ev.npz")
    meg_path = os.path.join(cache, base + ".npy")
    if not os.path.isfile(events_path) or not os.path.isfile(meg_path):
        raise ProtocolError(f"missing cache files for {base}")
    ckpt, _size, sha = _require_checkpoint(root)
    model = load_strong_decoder(ckpt)
    if not is_strong_decoder(model):
        raise ProtocolError("loaded object is not src.v3lib.Model")
    subject = int(base.split("_")[0].split("-")[1])
    events = load_events_safe(events_path)
    meg = np.load(meg_path, mmap_mode="r")
    result = forward_recording(
        model, meg, events, alignment_for(model, subject),
        timing=timing, maxw=int(model.a.get("maxw", 12)),
    )
    written = None
    if write and timing == "zero":
        written = _write_forward(directory, phase, base, subject, result, sha)
    return {"base": base, "phase": phase, "timing": timing, "n_words": result["n_words"], "path": written, "decoder": "src.v3lib.Model", "result": result, "sha": sha}


def _save_json(path: str, payload: dict) -> None:
    if os.path.exists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_train_embeddings(directory: str):
    embeddings = []
    labels = []
    rows = []
    dates_path = os.path.join(directory, "noise", "fit.json")
    with open(dates_path, encoding="utf-8") as handle:
        fit = json.load(handle)
    # Dates are already inside the feature state. Rebuild rows from the npz plus the dates
    # stored indirectly: transform needs dates, which were required at fit time. Re-read scans.
    return fit


def partial_train_directory(directory: str, root: str, steps: int, dates: dict, schedule: dict | None = None, state_name: str = "state.npz") -> dict:
    """Matched head steps. The noise equation stays the Q3 fit. Validation chooses the stop. Eval is not read."""
    if any(os.path.basename(path) == CHECKPOINT_BASENAME for path in glob.glob(os.path.join(directory, "**"), recursive=True)):
        raise ProtocolError("refusing to keep a copy of the frozen checkpoint in the output tree")
    fit_path = os.path.join(directory, "noise", "fit.json")
    with open(fit_path, encoding="utf-8") as handle:
        fit = json.load(handle)
    paths = sorted(glob.glob(os.path.join(directory, "forward", "train", "*.npz")))
    if not paths:
        raise ProtocolError("partial train needs cached train embeddings from the real forward")
    embeddings = []
    labels = []
    rows = []
    sha = None
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            file_sha = str(handle["checkpoint_sha"])
            if sha is None:
                sha = file_sha
            elif file_sha != sha:
                raise ProtocolError("train forwards came from different checkpoints")
            part, _logits, part_labels, _subjects = _rows_from_forward(path, "train", dates)
            rows.extend(part)
            embeddings.append(np.asarray(handle["embedding"], dtype=np.float64))
            labels.append(part_labels)
    features = transform_with_state(rows, fit["state"])
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    ckpt_path = _checkpoint_path(root)
    if os.path.isfile(ckpt_path):
        live_sha = _sha256_file(ckpt_path)
        if sha and live_sha and sha != live_sha:
            raise ProtocolError("cached train forwards do not match the current checkpoint")
    schedule = dict(SUBSTANTIVE_SCHEDULE if schedule is None else schedule)
    state_path = os.path.join(directory, "partial", state_name)
    if os.path.isfile(state_path):
        with np.load(state_path, allow_pickle=False) as handle:
            arms = {}
            for name in ("uniform", "random_covariate", "learned_state"):
                arms[name] = {
                    "W": np.asarray(handle[f"W_{name}"], dtype=np.float64),
                    "b": np.asarray(handle[f"b_{name}"], dtype=np.float64),
                    "intercept": float(handle[f"intercept_{name}"]),
                    "slope": np.asarray(handle[f"slope_{name}"], dtype=np.float64),
                }
            done = int(handle["steps"])
            init_sha = str(handle["init_sha"])
            if "schedule" in handle.files and str(handle["schedule"]) != schedule["name"]:
                raise ProtocolError(
                    f"partial state is {str(handle['schedule'])}; refusing to switch to {schedule['name']} after it started"
                )
    else:
        ckpt, _size, _sha = _require_checkpoint(root)
        model = load_strong_decoder(ckpt)
        weight = model.head_ce.weight.detach().cpu().numpy().astype(np.float64).T
        bias = model.head_ce.bias.detach().cpu().numpy().astype(np.float64)
        del model
        arms = initial_partial_arms(weight, bias, features.shape[1])
        done = 0
        init_sha = hashlib.sha256(weight.tobytes() + bias.tobytes()).hexdigest()
    if embeddings.shape[1] != arms["uniform"]["W"].shape[0]:
        raise ProtocolError(
            f"embedding width {embeddings.shape[1]} does not match the classification head "
            f"{arms['uniform']['W'].shape[0]}"
        )
    if done >= int(schedule["max_steps"]):
        return {
            "steps": done, "advanced": 0, "target": int(schedule["max_steps"]),
            "schedule": schedule["name"], "substantive": bool(schedule["substantive"]),
            "trunk_frozen": True, "stopped": True,
        }
    advancing = min(int(steps), int(schedule["max_steps"]) - done)
    learned_params = _params_from_json(fit["candidates"]["learned_state"])
    random_params = _params_from_json(fit["candidates"]["random_covariate"])
    learned_z, _learned_names = _candidate_matrix(features, fit["state"]["names"], "learned_state")
    random_z, _random_names = _candidate_matrix(features, fit["state"]["names"], "random_covariate")
    frozen_log_var = {
        "learned_state": log_variance(learned_z, learned_params),
        "random_covariate": log_variance(random_z, random_params),
    }
    matched_partial_steps(
        arms, embeddings, labels, features, advancing,
        lr=PARTIAL_LR, seed=done, frozen_log_var=frozen_log_var,
    )
    val_paths = sorted(glob.glob(os.path.join(directory, "forward", "selection", "*.npz")))
    if not val_paths:
        raise ProtocolError("head finetune needs validation forwards for the stop curve")
    val_embeddings = []
    val_labels = []
    for path in val_paths:
        with np.load(path, allow_pickle=False) as handle:
            val_embeddings.append(np.asarray(handle["embedding"], dtype=np.float64))
            val_labels.append(np.asarray(handle["label"], dtype=np.int64))
    val_embeddings = np.concatenate(val_embeddings, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    step_now = done + advancing
    curve_name = "val_curve.json" if schedule["substantive"] else "smoke_curve.json"
    curve_path = os.path.join(directory, "partial", curve_name)
    curve = []
    if os.path.isfile(curve_path):
        with open(curve_path, encoding="utf-8") as handle:
            curve = json.load(handle)
    curve.append({
        "step": int(step_now),
        "learned_ce": float(np.mean(decoder_residual(arm_logits(arms["learned_state"], val_embeddings), val_labels))),
        "uniform_ce": float(np.mean(decoder_residual(arm_logits(arms["uniform"], val_embeddings), val_labels))),
        "random_ce": float(np.mean(decoder_residual(arm_logits(arms["random_covariate"], val_embeddings), val_labels))),
    })
    decision = validation_stop(curve, schedule)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(curve_path, "w", encoding="utf-8") as handle:
        json.dump(curve, handle)
    if schedule["substantive"] and decision["best_step"] == step_now:
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
        "init_sha": np.array(init_sha),
        "checkpoint_sha": np.array(sha or ""),
    }
    for name, arm in arms.items():
        payload[f"W_{name}"] = arm["W"]
        payload[f"b_{name}"] = arm["b"]
        payload[f"intercept_{name}"] = np.float64(arm["intercept"])
        payload[f"slope_{name}"] = arm["slope"]
    temporary = state_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, state_path)
    return {
        "steps": step_now,
        "advanced": advancing,
        "target": int(schedule["max_steps"]),
        "schedule": schedule["name"],
        "substantive": bool(schedule["substantive"]),
        "stopped": bool(decision["stop"]),
        "stop_reason": decision["reason"],
        "best_validation_step": decision["best_step"],
        "trunk_frozen": True,
        "head_finetune": True,
        "noise_equation_frozen_from_q3": True,
        "head_data": "subject0_nonshared_trunk_trained",
        "in_sample_for_trunk": True,
        "not_a_subject0_day_holdout": True,
        "init_saw_subject0_days": True,
        "subject0_day_holdout_from_this_init": False,
        "not_a_full_retrain": True,
        "init_sha": init_sha,
        "validation_learned_ce": curve[-1]["learned_ce"],
    }


def score_partial_directory(directory: str) -> dict:
    state_path = os.path.join(directory, "partial", "state.npz")
    best_path = os.path.join(directory, "partial", "best_val.npz")
    paths = sorted(glob.glob(os.path.join(directory, "forward", "eval", "*.npz")))
    if not os.path.isfile(state_path) or not os.path.isfile(best_path) or not paths:
        raise ProtocolError("partial score needs the validation-selected checkpoint and eval embeddings")
    with np.load(state_path, allow_pickle=False) as handle:
        steps = int(handle["steps"])
        stopped = bool(int(handle["stopped"])) if "stopped" in handle.files else False
        schedule_name = str(handle["schedule"]) if "schedule" in handle.files else ""
        best_step = int(handle["best_step"]) if "best_step" in handle.files else steps
    if schedule_name != SUBSTANTIVE_SCHEDULE["name"]:
        raise ProtocolError("smoke weights are not the substantive comparison")
    if not stopped and steps < int(SUBSTANTIVE_SCHEDULE["max_steps"]):
        raise ProtocolError(f"partial score refused at {steps} steps before the validation stop")
    with np.load(best_path, allow_pickle=False) as handle:
        arms = {}
        for name in ("uniform", "random_covariate", "learned_state"):
            arms[name] = {"W": np.asarray(handle[f"W_{name}"]), "b": np.asarray(handle[f"b_{name}"])}
        scored_step = int(handle["steps"])
    embeddings = []
    labels = []
    for path in paths:
        with np.load(path, allow_pickle=False) as handle:
            embeddings.append(np.asarray(handle["embedding"], dtype=np.float64))
            labels.append(np.asarray(handle["label"], dtype=np.int64))
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    scores = {}
    for name, arm in arms.items():
        scores[name] = bacc10_from_logits(arm_logits(arm, embeddings), labels)
    defined = all(value is not None for value in scores.values())
    paired = None if not defined else float(scores["learned_state"] - scores["uniform"])
    return {
        "steps": scored_step,
        "trained_through_step": steps,
        "best_validation_step": best_step,
        "schedule": schedule_name,
        "substantive": True,
        "matched": True,
        "primary_bacc10_50": scores,
        "fifty_class_metric_defined": defined,
        "learned_minus_uniform": paired,
        "exploratory_not_pristine": True,
        "trunk_frozen": True,
        "not_a_full_retrain": True,
        "effect_claimed": False,
    }


def _cache_paths(root: str):
    cache_root = os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache")
    data_root = os.environ.get("LIBRIBRAIN_DATA", "/mnt/ssd4tb/data/LibriBrain100")
    return os.path.join(cache_root, "cont"), data_root


def run_next(root: str) -> dict:
    directory = out_dir(root)
    os.makedirs(directory, exist_ok=True)
    cache_cont, data_root = _cache_paths(root)
    if not os.path.isdir(cache_cont):
        raise ProtocolError(f"missing continuous cache {cache_cont}")
    bases = list_cache_bases(cache_cont)
    if not bases:
        raise ProtocolError(f"no recordings under {cache_cont}")
    dates = read_dates(data_root)
    order = plan_order(bases, dates)
    state = _disk_state(directory)
    action = next_action(order, **state)
    report = honesty(stage=action["stage"], next_action=action["next_action"], stages=_stage_status(directory), lock_held=True)
    if action["stage"] == "stop":
        report["runs_completed"] = True
        return report
    if action["stage"] == "forward":
        info = execute_forward(directory, root, action["base"], action["phase"])
        report.update({"base": info["base"], "phase": info["phase"], "n_words": info["n_words"], "decoder": info["decoder"], "wrote": info["path"]})
        return report
    if action["stage"] == "noise-fit":
        payload = fit_noise_directory(directory, dates)
        _save_json(os.path.join(directory, "noise", "fit.json"), payload)
        report.update({"chosen_on_validation": payload["chosen_on_validation"], "val_nll": payload["val_nll"], "objective": "gaussian_nll"})
        return report
    if action["stage"] == "score-frozen":
        payload = score_frozen_directory(directory, dates)
        ckpt, size, _sha = _require_checkpoint(root)
        status = measurement_status({
            "decoder": "src.v3lib.Model",
            "checkpoint_bytes": size,
            "objective": payload["objective"],
            "candidates": payload["candidates_scored"],
            "eval_subjects_in_train": payload["eval_subjects_in_train"],
            "subject0_frozen_holdout": False,
            "temperature_applied_to_reported_logits": False,
            "in_sample_trunk_trained_residual": payload.get("in_sample_trunk_trained_residual"),
            "eval_rows": payload["eval_rows"],
            "fit_rows": payload.get("fit_rows", 0),
            "selection_rows": payload["selection_rows"],
            "lock_held": True,
            "fixture": False,
        })
        payload = honesty(**payload, **status, checkpoint=ckpt)
        _save_json(os.path.join(directory, "score_frozen.json"), payload)
        report.update({
            "measurement_complete": status["measurement_complete"],
            "valid_scientific_claim": False,
            "primary_bacc10_50": payload["primary_bacc10_50"],
            "eligible_metrics": payload.get("eligible_metrics"),
            "paired_vs_uniform": payload["paired_vs_uniform"],
        })
        return report
    if action["stage"] == "partial-train":
        info = partial_train_directory(directory, root, action["steps"], dates)
        report.update(info)
        return report
    if action["stage"] == "score-partial":
        payload = honesty(**score_partial_directory(directory))
        _save_json(os.path.join(directory, "score_partial.json"), payload)
        report.update({
            "learned_minus_uniform": payload["learned_minus_uniform"],
            "fifty_class_metric_defined": payload["fifty_class_metric_defined"],
            "matched_steps": payload["steps"],
        })
        return report
    raise ProtocolError(action["stage"])


def run_timing(root: str) -> dict:
    directory = out_dir(root)
    paths = sorted(glob.glob(os.path.join(directory, "forward", "eval", "*.npz")))
    if not paths:
        raise ProtocolError("timing control needs one exploratory eval forward already checkpointed")
    with np.load(paths[0], allow_pickle=False) as handle:
        base = str(handle["base"])
        cached = np.asarray(handle["logits"], dtype=np.float64)
    out = os.path.join(directory, "timing", base + ".json")
    if os.path.exists(out):
        raise ProtocolError(f"refusing to overwrite {out}")
    modes = {}
    for mode in ("zero", "shuffle", "timing_only"):
        info = execute_forward(directory, root, base, "eval", timing=mode, write=False)
        logits = np.asarray(info["result"]["logits"], dtype=np.float64)
        labels = np.asarray(info["result"]["label"])
        if mode == "zero" and np.max(np.abs(logits - cached)) > 1e-3:
            raise ProtocolError("zero-timing forward does not match the checkpointed primary logits")
        modes[mode] = {
            "mean_cross_entropy": float(np.mean(decoder_residual(logits, labels))),
            "primary_bacc10_50": bacc10_from_logits(logits, labels),
        }
    payload = honesty(
        base=base,
        modes=modes,
        primary_timing="zero",
        timing_only_is_a_control=True,
        shuffle_is_a_control=True,
        objective="mean_cross_entropy",
    )
    _save_json(out, payload)
    return payload


def run_lpp(root: str) -> dict:
    cache = os.environ.get("LIBRIBRAIN_LPP_CACHE", "/mnt/ssd4tb/data/libribrain_cache/ext_lpp_listen")
    directory = out_dir(root)
    if not os.path.isdir(cache):
        raise ProtocolError(f"missing LPP cache {cache}")
    events = sorted(glob.glob(os.path.join(cache, "*_ev.npz")))
    if not events:
        raise ProtocolError(f"LPP cache {cache} has no event files")
    chosen = None
    for path in events:
        base = os.path.basename(path)[:-7]
        meg = os.path.join(cache, base + ".npy")
        dest = os.path.join(directory, "lpp", base + ".json")
        if os.path.isfile(meg) and not os.path.exists(dest):
            chosen = (base, path, meg, dest)
            break
    if chosen is None:
        if not any(os.path.isfile(os.path.join(cache, os.path.basename(path)[:-7] + ".npy")) for path in events):
            raise ProtocolError("LPP event files have no sibling MEG array; no transfer metric was written")
        raise ProtocolError("every LPP recording with MEG already has a transfer file; not pooling and not overwriting")
    base, events_path, meg_path, dest = chosen
    ckpt, _size, sha = _require_checkpoint(root)
    model = load_strong_decoder(ckpt)
    # LPP listeners are not LibriBrain subjects. Use the mean Broad alignment, never a seen subject's A.
    result = forward_recording(
        model, np.load(meg_path, mmap_mode="r"), load_events_safe(events_path), alignment_for(model, VAL_SUBJECTS[0]),
        timing="zero", maxw=int(model.a.get("maxw", 12)),
    )
    labels = np.asarray(result["label"])
    same_vocab = bool(np.all((labels >= 0) & (labels < 50)))
    bacc = bacc10_from_logits(result["logits"], labels) if same_vocab else None
    payload = honesty(
        base=base,
        checkpoint_sha=sha,
        n_words=int(result["n_words"]),
        same_libribrain_vocab=same_vocab,
        primary_bacc10_50=bacc,
        fifty_class_metric_defined=bacc is not None,
        pooled_with_libribrain=False,
        entered_selection=False,
        decoder="src.v3lib.Model",
        wrote=dest,
        checkpoint=ckpt,
    )
    _save_json(dest, payload)
    return payload


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = repo_root()
    command = argv[0] if argv else "next"
    aliases = {"run": "next", "forward-broad": "next", "forward": "next"}
    command = aliases.get(command, command)
    try:
        if command == "manifest":
            payload = manifest(root)
            print(json.dumps({
                "protocol_id": payload["protocol_id"],
                "amendment": payload["amendment"],
                "checkpoint_present": payload["checkpoint"]["present"],
                "effect_claimed": False,
                "valid_scientific_claim": False,
                "stages": payload["stages"],
            }))
            return 0
        if command == "full-retrain":
            print(json.dumps(honesty(stage="full-retrain", next_action="do_not_launch_full_retrain")))
            print("blocker: full retrain is impractical under CPU 4G and is not launched", file=sys.stderr)
            return 2
        if os.environ.get("SDR_FORBID_LIVE_COMPUTE") == "1":
            print("blocker: refusing live compute because SDR_FORBID_LIVE_COMPUTE is set", file=sys.stderr)
            return 2
        if not lock_held():
            print("blocker: scheduler self-lock is not held; refusing to run beside MICrONS or ANN", file=sys.stderr)
            return 2
        hardware = verify_live_hardware()
        if command == "next":
            payload = run_next(root)
        elif command == "timing-control":
            payload = run_timing(root)
        elif command == "lpp-transfer":
            payload = run_lpp(root)
        elif command == "smoke":
            directory = out_dir(root)
            os.makedirs(directory, exist_ok=True)
            _cache, data_root = _cache_paths(root)
            info = partial_train_directory(
                directory, root, int(SMOKE_SCHEDULE["steps_per_launch"]), read_dates(data_root),
                schedule=SMOKE_SCHEDULE, state_name="smoke_state.npz",
            )
            payload = honesty(**info, stage="smoke", next_action="smoke_is_not_the_substantive_schedule")
        elif command in ("noise-fit", "partial-train", "score"):
            payload = run_next(root)
            if payload.get("stage") not in ("noise-fit", "partial-train", "score-frozen", "score-partial", "forward", "stop"):
                raise ProtocolError("the named stage is not the next pre-registered chunk")
        else:
            print(
                "usage: python -m campaigns.sdr_uncertainty.strong_decoder_runner "
                "{manifest|next|smoke|timing-control|lpp-transfer|full-retrain}",
                file=sys.stderr,
            )
            return 2
        payload = dict(payload)
        payload["cgroup"] = {"unit": hardware.get("unit"), "runtime_ceiling_required": False}
        payload["effect_claimed"] = False
        print(json.dumps({
            "protocol_id": payload.get("protocol_id", PROTOCOL_ID),
            "amendment": AMENDMENT,
            "stage": payload.get("stage"),
            "next_action": payload.get("next_action"),
            "effect_claimed": False,
            "valid_scientific_claim": False,
            "measurement_complete": bool(payload.get("measurement_complete", False)),
            "schedule": payload.get("schedule"),
            "real_meg_scientific_validation": False,
            "full_retrain_launched": False,
            "fifty_class_metric_defined": payload.get("fifty_class_metric_defined"),
            "blocker": payload.get("blocker"),
        }))
        return 0
    except (CgroupError, ProtocolError, FileNotFoundError, OSError) as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        print(json.dumps(honesty(stage=command, blocker=str(exc), next_action="fix_blocker")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
