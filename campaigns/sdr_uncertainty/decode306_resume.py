"""One bounded batch of the 306-channel probe, resumed on a frozen date list.

A batch is not the scientific gate. The gate remains 3 observations of all 50
words on every held-out day and in the days that train it. When every run on
the frozen dates has been scanned and that gate is still unmet, the missing
classes are a property of those recordings, and another batch must not be read.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .decode306 import evaluate_decode306, normalize_raw_window, planned_budget
from .protocol import (
    DECODE306_BATCH_READ_BYTES,
    DECODE306_CACHE_VERSION,
    DECODE306_FEATURE_CACHE_MAX_BYTES,
    DECODE306_MAX_DAYS,
    DECODE306_MIN_PER_CLASS,
    DECODE306_N_CLASSES,
    DECODE306_NONOVERLAP_SEC,
    EXCLUDED_SMOKE_DATE,
    VOCAB,
)

PREPROCESS_ID = "scale_torchstd_lowermedian_clip10_pool5"
SELECTION_RULE = "frozen_earliest_dates_then_every_run_in_name_order"


def events_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h5_fingerprint(path: str, dataset) -> dict:
    return {
        "nbytes": int(os.path.getsize(path)),
        "shape": [int(v) for v in dataset.shape],
        "dtype": str(dataset.dtype),
    }


def freeze_runs(discovered: list[dict]) -> dict:
    """Earliest dates, then every run on those dates. Later runs on disk do not join a frozen manifest."""
    dates = []
    for spec in discovered:
        if spec["date"] not in dates:
            dates.append(spec["date"])
        if len(dates) == DECODE306_MAX_DAYS:
            break
    chosen = [spec for spec in discovered if spec["date"] in dates]
    chosen.sort(key=lambda spec: (spec["date"], spec["base"]))
    return {
        "cache_version": DECODE306_CACHE_VERSION,
        "preprocess_id": PREPROCESS_ID,
        "selection_rule": SELECTION_RULE,
        "excluded_date": EXCLUDED_SMOKE_DATE,
        "nonoverlap_sec": DECODE306_NONOVERLAP_SEC,
        "min_per_class": DECODE306_MIN_PER_CLASS,
        "n_classes": DECODE306_N_CLASSES,
        "dates": dates,
        "runs": chosen,
    }


def _manifest_key(manifest: dict) -> dict:
    return {
        "dates": manifest["dates"],
        "runs": [(row["date"], row["base"], row["events_sha256"], row["h5_nbytes"], row["h5_shape"], row["h5_dtype"])
                 for row in manifest["runs"]],
        "preprocess_id": manifest["preprocess_id"],
        "selection_rule": manifest["selection_rule"],
    }


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _empty_progress() -> dict:
    return {
        "run_index": 0,
        "word_index": 0,
        "last_kept_onset": None,
        "last_kept_run": None,
        "kept": 0,
        "skipped_overlap": 0,
        "in_vocab_considered": 0,
    }


def _class_matrix(batches: list[dict], n_dates: int) -> np.ndarray:
    counts = np.zeros((n_dates, DECODE306_N_CLASSES), dtype=np.int64)
    for batch in batches:
        for day, label in zip(batch["day_index"], batch["label"]):
            if 0 <= int(label) < DECODE306_N_CLASSES:
                counts[int(day), int(label)] += 1
    return counts


def _load_batches(cache_dir: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(cache_dir)):
        if not (name.startswith("batch_") and name.endswith(".npz")):
            continue
        with np.load(os.path.join(cache_dir, name)) as archive:
            rows.append({
                "features": archive["features"],
                "label": archive["label"],
                "day_index": archive["day_index"],
                "onset_s": archive["onset_s"],
                "run_index": archive["run_index"],
            })
    return rows


def inventory(cache_dir: str) -> dict:
    manifest = _load_json(os.path.join(cache_dir, "manifest.json"))
    progress = _load_json(os.path.join(cache_dir, "progress.json")) or _empty_progress()
    if manifest is None:
        return {"claim_status": "no_manifest", "effect_claimed": False}
    batches = _load_batches(cache_dir)
    date_index = {date: i for i, date in enumerate(manifest["dates"])}
    counts = _class_matrix(batches, len(manifest["dates"]))
    per_date = []
    runs_by_date = {date: [] for date in manifest["dates"]}
    for index, run in enumerate(manifest["runs"]):
        runs_by_date[run["date"]].append(index)
    for date in manifest["dates"]:
        indexes = runs_by_date[date]
        consumed = all(index < progress["run_index"] for index in indexes) if indexes else True
        row_counts = counts[date_index[date]]
        per_date.append({
            "date": date,
            "runs": [manifest["runs"][index]["base"] for index in indexes],
            "runs_exhausted": consumed,
            "kept_nonoverlapping": int(row_counts.sum()),
            "classes_present": int(np.sum(row_counts > 0)),
            "classes_below_min": [VOCAB[i] for i in range(DECODE306_N_CLASSES) if row_counts[i] < DECODE306_MIN_PER_CLASS],
            "per_class": [int(v) for v in row_counts],
        })
    heldout_block = []
    for held, date in enumerate(manifest["dates"]):
        test = counts[held]
        train = counts.sum(axis=0) - test
        heldout_block.append({
            "date": date,
            "test_min": int(test.min()),
            "train_min": int(train.min()),
            "classes_absent_from_train": [VOCAB[i] for i in range(DECODE306_N_CLASSES) if train[i] == 0],
            "test_classes_present": int(np.sum(test > 0)),
            "fifty_class_metric_defined": bool(np.all(test >= DECODE306_MIN_PER_CLASS) and np.all(train >= DECODE306_MIN_PER_CLASS)),
        })
    exhausted = progress["run_index"] >= len(manifest["runs"])
    defined = all(row["fifty_class_metric_defined"] for row in heldout_block) if heldout_block else False
    if defined:
        claim = "ready_to_score"
    elif exhausted:
        claim = "dataset_limitation"
    else:
        claim = "resume_required"
    return {
        "claim_status": claim,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "dates": manifest["dates"],
        "runs_consumed": int(progress["run_index"]),
        "runs_total": len(manifest["runs"]),
        "kept": int(progress["kept"]),
        "skipped_overlap": int(progress["skipped_overlap"]),
        "in_vocab_considered": int(progress["in_vocab_considered"]),
        "per_date": per_date,
        "heldout_coverage": heldout_block,
        "science_gate": "fifty_class_min3_on_every_day_and_its_complement",
        "batch_read_cap_is_not_the_science_gate": True,
        "next_action": {
            "ready_to_score": "score_from_checkpoint",
            "dataset_limitation": "stop",
            "resume_required": "rerun_this_command",
        }[claim],
    }


def _feature_bytes(cache_dir: str) -> int:
    total = 0
    for name in os.listdir(cache_dir):
        if name.startswith("batch_") and name.endswith(".npz"):
            total += os.path.getsize(os.path.join(cache_dir, name))
    return total


def resume_batch(cache_dir: str, runs: list[dict], read_raw, *, batch_bytes: int = DECODE306_BATCH_READ_BYTES,
                 itemsize: int = 4) -> dict:
    """Read the next chronological non-overlapping slice. `runs` must match a manifest once one exists."""
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    existing = _load_json(manifest_path)
    proposed = {
        "cache_version": DECODE306_CACHE_VERSION,
        "preprocess_id": PREPROCESS_ID,
        "selection_rule": SELECTION_RULE,
        "excluded_date": EXCLUDED_SMOKE_DATE,
        "nonoverlap_sec": DECODE306_NONOVERLAP_SEC,
        "min_per_class": DECODE306_MIN_PER_CLASS,
        "n_classes": DECODE306_N_CLASSES,
        "dates": [],
        "runs": runs,
    }
    # The caller passes the already frozen run list, including only the selected dates.
    proposed["dates"] = []
    for row in runs:
        if row["date"] not in proposed["dates"]:
            proposed["dates"].append(row["date"])
    if existing is None:
        _write_json(manifest_path, proposed)
        manifest = proposed
    else:
        if _manifest_key(existing) != _manifest_key(proposed):
            raise RuntimeError("refusing to append: the frozen date, run, or source fingerprint changed")
        manifest = existing
    if _feature_bytes(cache_dir) > DECODE306_FEATURE_CACHE_MAX_BYTES:
        report = inventory(cache_dir)
        report["claim_status"] = "dataset_limitation"
        report["reason"] = "The normalized-feature cache reached its 2 GiB guard before the 50-class gate."
        report["next_action"] = "stop"
        _write_json(os.path.join(cache_dir, "status.json"), report)
        return report
    progress = _load_json(os.path.join(cache_dir, "progress.json")) or _empty_progress()
    if progress["run_index"] >= len(manifest["runs"]):
        return _finish(cache_dir, manifest, progress, batch_bytes, itemsize, read_more=False)
    features, labels, days, onsets, run_ids = [], [], [], [], []
    used = 0
    per_window = 306 * 250 * int(itemsize)
    run_index = int(progress["run_index"])
    word_index = int(progress["word_index"])
    while run_index < len(manifest["runs"]):
        words = manifest["runs"][run_index]["words"]
        while word_index < len(words):
            onset, label = words[word_index]
            same_run = progress["last_kept_run"] == run_index and progress["last_kept_onset"] is not None
            if same_run and float(onset) < float(progress["last_kept_onset"]) + DECODE306_NONOVERLAP_SEC:
                progress["skipped_overlap"] += 1
                progress["in_vocab_considered"] += 1
                word_index += 1
                continue
            if used + per_window > batch_bytes:
                progress["run_index"] = run_index
                progress["word_index"] = word_index
                _save_partial(cache_dir, features, labels, days, onsets, run_ids, progress)
                return _finish(cache_dir, manifest, progress, batch_bytes, itemsize, read_more=True)
            raw = read_raw(run_index, float(onset))
            progress["in_vocab_considered"] += 1
            word_index += 1
            if raw is None:
                continue
            features.append(normalize_raw_window(raw).astype(np.float32))
            labels.append(int(label))
            days.append(manifest["dates"].index(manifest["runs"][run_index]["date"]))
            onsets.append(float(onset))
            run_ids.append(run_index)
            used += per_window
            progress["kept"] += 1
            progress["last_kept_onset"] = float(onset)
            progress["last_kept_run"] = run_index
        run_index += 1
        word_index = 0
        progress["last_kept_onset"] = None
        progress["last_kept_run"] = None
    progress["run_index"] = run_index
    progress["word_index"] = 0
    _save_partial(cache_dir, features, labels, days, onsets, run_ids, progress)
    return _finish(cache_dir, manifest, progress, batch_bytes, itemsize, read_more=False)


def _save_partial(cache_dir, features, labels, days, onsets, run_ids, progress) -> None:
    if features:
        names = [name for name in os.listdir(cache_dir) if name.startswith("batch_")]
        path = os.path.join(cache_dir, f"batch_{len(names):04d}.npz")
        np.savez_compressed(
            path,
            features=np.stack(features),
            label=np.asarray(labels, dtype=np.int16),
            day_index=np.asarray(days, dtype=np.int16),
            onset_s=np.asarray(onsets, dtype=np.float64),
            run_index=np.asarray(run_ids, dtype=np.int16),
        )
    _write_json(os.path.join(cache_dir, "progress.json"), progress)


def _finish(cache_dir, manifest, progress, batch_bytes, itemsize, *, read_more: bool) -> dict:
    report = inventory(cache_dir)
    report["batch_read_cap_bytes"] = int(batch_bytes)
    report["planned_budget"] = planned_budget(itemsize)
    report["batch_read_cap_is_not_the_science_gate"] = True
    if report["claim_status"] == "ready_to_score":
        batches = _load_batches(cache_dir)
        features = np.concatenate([batch["features"] for batch in batches])
        labels = np.concatenate([batch["label"] for batch in batches])
        day = np.concatenate([batch["day_index"] for batch in batches])
        scored = evaluate_decode306(features, labels, day, manifest["dates"])
        scored["effect_claimed"] = False
        scored["real_meg_scientific_validation"] = False
        scored["inventory"] = report
        scored["checkpoint"] = cache_dir
        _write_json(os.path.join(cache_dir, "status.json"), {k: v for k, v in scored.items() if k != "inventory"})
        # status.json must stay small. The full score is returned to the runner, which writes one JSON.
        return scored
    if read_more and report["claim_status"] != "dataset_limitation":
        report["claim_status"] = "resume_required"
        report["next_action"] = "rerun_this_command"
    report["reason"] = {
        "resume_required": "Coverage is still short and later runs on the frozen dates remain. Rerun the same command.",
        "dataset_limitation": (
            "Every run on the frozen dates has been scanned, or the feature cache is full, "
            "and at least one held-out day still lacks 3 observations of some vocabulary word. "
            "Further empty reads will not create those words."
        ),
        "ready_to_score": "Coverage met.",
    }.get(report["claim_status"], report["claim_status"])
    _write_json(os.path.join(cache_dir, "status.json"), report)
    return report
