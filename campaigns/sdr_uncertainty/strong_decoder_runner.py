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
    ProtocolError,
    arm_logits,
    assemble_features,
    bacc10_from_logits,
    decoder_residual,
    fit_noise,
    initial_partial_arms,
    log_variance,
    matched_partial_steps,
    measurement_claim,
    select_on_validation,
    temperature_changes_top10,
    transform_with_state,
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
    payload["accuracy_claimed_from_temperature"] = False
    payload["frozen_trunk_subject0_days_are_holdout"] = False
    payload["eval_subjects_are_pristine_confirmatory_holdout"] = False
    payload["lpp_pooled"] = False
    payload["full_retrain"] = dict(FULL_RETRAIN)
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
    for phase in ("selection", "train", "eval"):
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
    for phase in ("selection", "train", "eval"):
        for path in glob.glob(os.path.join(directory, "forward", phase, "*.npz")):
            done.append(os.path.basename(path)[:-4])
    partial = 0
    state_path = os.path.join(directory, "partial", "state.npz")
    if os.path.isfile(state_path):
        with np.load(state_path, allow_pickle=False) as handle:
            partial = int(handle["steps"])
    return {
        "done_bases": done,
        "noise_fit": os.path.isfile(os.path.join(directory, "noise", "fit.json")),
        "partial_steps": partial,
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


def _rows_from_forward(path: str, split: str, dates: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        base = str(handle["base"])
        subject = int(handle["subject"])
        labels = np.asarray(handle["label"], dtype=np.int64)
        logits = np.asarray(handle["logits"], dtype=np.float64)
        date = dates.get(base)
        if not date:
            raise ProtocolError(f"no acquisition date for {base}; refusing to invent day_rank")
        residual = decoder_residual(logits, labels)
        rows = []
        for index in range(len(labels)):
            rows.append({
                "base": base,
                "subject": subject,
                "date": date,
                "split": split,
                "run_fraction": float(handle["run_fraction"][index]),
                "artifact_rms": float(handle["artifact_rms"][index]),
                "bandpower": float(handle["bandpower"][index]),
                "group_windows": int(len(labels)),
                "residual": float(residual[index]),
            })
        return rows, logits, labels


def _candidate_matrix(z: np.ndarray, names: list[str], candidate: str):
    if candidate == "error_scaling":
        columns = [names.index(feature) for feature in ERROR_SCALING_FEATURES]
        return z[:, columns], list(ERROR_SCALING_FEATURES)
    return z, list(names)


def fit_noise_directory(directory: str, dates: dict) -> dict:
    """Fit on train forwards. Score every candidate on selection forwards. Eval files are not opened."""
    train_paths = sorted(glob.glob(os.path.join(directory, "forward", "train", "*.npz")))
    val_paths = sorted(glob.glob(os.path.join(directory, "forward", "selection", "*.npz")))
    if not train_paths or not val_paths:
        raise ProtocolError("noise fit needs the budgeted train forwards and every selection forward")
    rows = []
    for path in train_paths:
        part, _logits, _labels = _rows_from_forward(path, "train", dates)
        rows.extend(part)
    n_train = len(rows)
    for path in val_paths:
        part, _logits, _labels = _rows_from_forward(path, "selection", dates)
        rows.extend(part)
    if any(row["split"] == "eval" for row in rows):
        raise ProtocolError("eval rows were passed to the noise fit")
    bundle = assemble_features(rows)
    residual = np.asarray([row["residual"] for row in rows], dtype=np.float64)
    train = bundle["train_index"]
    params = {}
    val_nll = {}
    for name in CANDIDATES:
        matrix, feat_names = _candidate_matrix(bundle["z"], bundle["names"], name)
        params[name] = fit_noise(matrix[train], residual[train], name, feature_names=feat_names, seed=0)
        val_nll[name] = weighted_nll(residual[~train], matrix[~train], params[name])
    chosen = select_on_validation(val_nll)
    subjects = sorted({int(row["subject"]) for row in rows if row["split"] == "train"})
    return {
        "objective": "gaussian_nll",
        "chosen_on_validation": chosen,
        "val_nll": {name: float(value) for name, value in val_nll.items()},
        "candidates": {name: _params_to_json(value) for name, value in params.items()},
        "state": bundle["state"],
        "train_rows": int(n_train),
        "selection_rows": int(len(rows) - n_train),
        "train_subjects": subjects,
        "eval_subjects_in_train": [],
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
    sessions = []
    for path in paths:
        part_rows, part_logits, part_labels = _rows_from_forward(path, "eval", dates)
        rows.extend(part_rows)
        logits.append(part_logits)
        labels.append(part_labels)
        sessions.extend(part_rows[0]["base"] for _ in part_labels)
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
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
    return {
        "objective": "gaussian_nll",
        "chosen_on_validation": chosen,
        "eval_nll": {name: float(value) for name, value in nll.items()},
        "paired_vs_uniform": float(nll[chosen] - nll["uniform"]),
        "primary_bacc10_50": bacc,
        "fifty_class_metric_defined": bacc is not None,
        "metric_undefined_reason": None if bacc is not None else "at least one of the 50 classes is absent from the exploratory eval forwards",
        "temperature_changes_top10": bool(changed),
        "temperature_applied_to_reported_logits": False,
        "eval_rows": int(len(labels)),
        "eval_sessions": sorted(set(sessions)),
        "exploratory_not_pristine": True,
        "candidates_scored": list(CANDIDATES),
        "train_rows": int(fit["train_rows"]),
        "selection_rows": int(fit["selection_rows"]),
        "eval_subjects_in_train": list(fit["eval_subjects_in_train"]),
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


def partial_train_directory(directory: str, root: str, steps: int, dates: dict) -> dict:
    """Matched steps on cached train embeddings. The trunk is not updated. deep_c is not written."""
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
            part, _logits, part_labels = _rows_from_forward(path, "train", dates)
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
    state_path = os.path.join(directory, "partial", "state.npz")
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
    if done >= PARTIAL_TARGET_STEPS:
        return {"steps": done, "advanced": 0, "target": PARTIAL_TARGET_STEPS, "trunk_frozen": True}
    advancing = min(int(steps), PARTIAL_TARGET_STEPS - done)
    matched_partial_steps(arms, embeddings, labels, features, advancing, lr=PARTIAL_LR, seed=done)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    payload = {"steps": np.int64(done + advancing), "init_sha": np.array(init_sha), "checkpoint_sha": np.array(sha or "")}
    for name, arm in arms.items():
        payload[f"W_{name}"] = arm["W"]
        payload[f"b_{name}"] = arm["b"]
        payload[f"intercept_{name}"] = np.float64(arm["intercept"])
        payload[f"slope_{name}"] = arm["slope"]
    temporary = state_path + ".tmp.npz"
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, state_path)
    return {
        "steps": done + advancing,
        "advanced": advancing,
        "target": PARTIAL_TARGET_STEPS,
        "trunk_frozen": True,
        "head_finetune": True,
        "init_saw_subject0_days": True,
        "subject0_day_holdout_from_this_init": False,
        "not_a_full_retrain": True,
        "init_sha": init_sha,
    }


def score_partial_directory(directory: str) -> dict:
    state_path = os.path.join(directory, "partial", "state.npz")
    paths = sorted(glob.glob(os.path.join(directory, "forward", "eval", "*.npz")))
    if not os.path.isfile(state_path) or not paths:
        raise ProtocolError("partial score needs the matched checkpoint and eval embeddings")
    with np.load(state_path, allow_pickle=False) as handle:
        steps = int(handle["steps"])
        arms = {}
        for name in ("uniform", "random_covariate", "learned_state"):
            arms[name] = {"W": np.asarray(handle[f"W_{name}"]), "b": np.asarray(handle[f"b_{name}"])}
    if steps != PARTIAL_TARGET_STEPS:
        raise ProtocolError(f"partial score refused at {steps} steps; the matched budget is {PARTIAL_TARGET_STEPS}")
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
        "steps": steps,
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
        claimed = measurement_claim({
            "decoder": "src.v3lib.Model",
            "checkpoint_bytes": size,
            "objective": payload["objective"],
            "candidates": payload["candidates_scored"],
            "eval_subjects_in_train": payload["eval_subjects_in_train"],
            "subject0_frozen_holdout": False,
            "temperature_applied_to_reported_logits": False,
            "eval_rows": payload["eval_rows"],
            "train_rows": payload["train_rows"],
            "selection_rows": payload["selection_rows"],
            "lock_held": True,
        })
        payload = honesty(**payload, valid_scientific_claim=claimed, checkpoint=ckpt)
        _save_json(os.path.join(directory, "score_frozen.json"), payload)
        report.update({
            "valid_scientific_claim": claimed,
            "primary_bacc10_50": payload["primary_bacc10_50"],
            "fifty_class_metric_defined": payload["fifty_class_metric_defined"],
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
        elif command in ("noise-fit", "partial-train", "score"):
            payload = run_next(root)
            if payload.get("stage") not in ("noise-fit", "partial-train", "score-frozen", "score-partial", "forward", "stop"):
                raise ProtocolError("the named stage is not the next pre-registered chunk")
        else:
            print(
                "usage: python -m campaigns.sdr_uncertainty.strong_decoder_runner "
                "{manifest|next|timing-control|lpp-transfer|full-retrain}",
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
            "valid_scientific_claim": bool(payload.get("valid_scientific_claim", False)),
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
