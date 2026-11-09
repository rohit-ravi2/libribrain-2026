"""Chronological subject-0 dates after the frozen earliest-4 cache.

The directory decode306_cache is provenance. This module reads it and never
writes it. New windows go to decode306_expand_cache, one 512 MiB raw batch
per launch, with batch sha256 recorded before the next launch may append.

A date is a held-out day only when its own labeled counts are at least 3 for
every vocabulary word. Dates below that stay in the train pool and are not
scored. The 50-class metric is defined only when at least two held-out days
exist and each one's train fold also has 3 of every class. Partial-class
accuracy is not reported. effect_claimed stays false.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .decode306 import normalize_raw_window, planned_budget, score_consumed_days
from .decode306_resume import (
    PREPROCESS_ID,
    SELECTION_RULE,
    _batch_names,
    _empty_progress,
    _load_count_rows,
    _load_json,
    _write_json,
    events_sha256,
    h5_fingerprint,
    inventory,
)
from .protocol import (
    DECODE306_BATCH_READ_BYTES,
    DECODE306_CACHE_VERSION,
    DECODE306_EXPAND_CACHE_DIRNAME,
    DECODE306_EXPAND_CACHE_VERSION,
    DECODE306_EXPAND_SELECTION_RULE,
    DECODE306_FEATURE_CACHE_MAX_BYTES,
    DECODE306_MIN_HELDOUT_DAYS,
    DECODE306_FROZEN_CACHE_DIRNAME,
    DECODE306_MIN_PER_CLASS,
    DECODE306_N_CLASSES,
    DECODE306_NONOVERLAP_SEC,
    DECODE306_VERIFIED_FROZEN_BATCH_COUNT,
    DECODE306_VERIFIED_FROZEN_DATES,
    DECODE306_VERIFIED_FROZEN_RUNS,
    EXCLUDED_SMOKE_DATE,
    N_SENSORS,
    VOCAB,
    WINDOW_SAMPLES,
)

VERIFIED_BATCHES = tuple(
    f"batch_{index:04d}.npz" for index in range(DECODE306_VERIFIED_FROZEN_BATCH_COUNT)
)

AMENDED_PROTOCOL_ID = "decode306-expand-exploratory-coverage-amendment-1"


def amended_protocol_label() -> dict:
    """Names the estimand. It does not decide which dates pass the count gate."""
    return {
        "amended_protocol_id": AMENDED_PROTOCOL_ID,
        "estimand": (
            "Paired difference, across held-out dates, of 50-word BAcc@10 for "
            "train-only shrunk channel whitening minus the uniform ridge head. "
            "A date is held out only when that date has at least 3 observations of "
            "all 50 words. Dates below that coverage, including the exhausted earliest "
            "four, are training rows only and are not replication days."
        ),
        "selection_rule": DECODE306_EXPAND_SELECTION_RULE,
        "score_selection": False,
        "original_frozen_four_date_replication": False,
        "protocol_role": "exploratory_amendment_conditional_on_coverage",
        "old_dates_status": "dataset_limitation",
        "old_dates": list(DECODE306_VERIFIED_FROZEN_DATES),
        "earlier_data_exposures": [
            "out-of-day residual proxy on four channel means; M2-M0 not distinguishable from zero",
            "four-sensor decode-probe; labeled per day 27, 20, 20, 28; decoding_inadequate",
            "decode-306 earliest four dates 2024-07-31, 2024-08-07, 2024-08-12, 2024-08-14; "
            "class presence 40/42/36/44; train_min 0; dataset_limitation; 21 runs; "
            "batches batch_0000.npz through batch_0006.npz",
        ],
        "future_independent_confirmation_required": True,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_provenance(cache_dir: str) -> list[dict]:
    rows = []
    for name in _batch_names(cache_dir):
        path = os.path.join(cache_dir, name)
        rows.append({
            "name": name,
            "sha256": sha256_file(path),
            "nbytes": int(os.path.getsize(path)),
        })
    return rows


def select_later_runs(discovered: list[dict], frozen_dates: list[str]) -> list[dict]:
    """Dates strictly after the frozen prefix, in date then run-name order.

    A `score` field is not a sort key. A date inside or before the frozen prefix
    that is not one of those frozen dates is an inventory change, not a candidate.
    """
    if list(frozen_dates) != sorted(frozen_dates):
        raise RuntimeError("frozen dates are not chronological")
    if len(frozen_dates) != len(set(frozen_dates)):
        raise RuntimeError("frozen dates are not unique")
    last = frozen_dates[-1]
    frozen = set(frozen_dates)
    ordered = sorted(discovered, key=lambda spec: (spec["date"], spec["base"]))
    selected = []
    for spec in ordered:
        date = spec["date"]
        if date == EXCLUDED_SMOKE_DATE:
            raise RuntimeError(f"refusing excluded date {EXCLUDED_SMOKE_DATE}")
        if date in frozen:
            continue
        if date <= last:
            raise RuntimeError(
                f"refusing date {date}; it is not after the frozen prefix ending {last}"
            )
        selected.append(spec)
    return selected


def assert_frozen_rediscovery(discovered: list[dict], manifest: dict) -> None:
    frozen = set(manifest["dates"])
    expected = [run["base"] for run in manifest["runs"]]
    found = [spec["base"] for spec in sorted(discovered, key=lambda spec: (spec["date"], spec["base"]))
             if spec["date"] in frozen]
    if found != expected:
        raise RuntimeError("frozen date run list changed; refusing to expand")


def _verified_batches(cache_dir: str) -> None:
    names = _batch_names(cache_dir)
    if tuple(names) != VERIFIED_BATCHES:
        raise RuntimeError(
            "verified frozen cache batch files are not batch_0000 through "
            f"batch_{DECODE306_VERIFIED_FROZEN_BATCH_COUNT - 1:04d}"
        )


def frozen_action(cache_dir: str) -> str:
    """`absent`, `resume_frozen`, or `expand`. Raises if the verified cache was mutated."""
    manifest = _load_json(os.path.join(cache_dir, "manifest.json"))
    if manifest is None:
        return "absent"
    if manifest.get("dates") != list(DECODE306_VERIFIED_FROZEN_DATES):
        return "resume_frozen"
    progress = _load_json(os.path.join(cache_dir, "progress.json")) or _empty_progress()
    exhausted = int(progress.get("run_index", 0)) >= len(manifest.get("runs", []))
    if not exhausted:
        return "resume_frozen"
    if manifest.get("cache_version") != DECODE306_CACHE_VERSION:
        raise RuntimeError("verified frozen cache version changed")
    if manifest.get("selection_rule") != SELECTION_RULE:
        raise RuntimeError("verified frozen selection rule changed")
    if manifest.get("excluded_date") != EXCLUDED_SMOKE_DATE:
        raise RuntimeError("verified frozen cache no longer excludes 2024-10-04")
    if manifest.get("preprocess_id") != PREPROCESS_ID:
        raise RuntimeError("verified frozen preprocess changed")
    if len(manifest.get("runs", [])) != DECODE306_VERIFIED_FROZEN_RUNS:
        raise RuntimeError(
            f"verified frozen run count is {len(manifest.get('runs', []))}, expected {DECODE306_VERIFIED_FROZEN_RUNS}"
        )
    _verified_batches(cache_dir)
    report = inventory(cache_dir)
    if report.get("claim_status") != "dataset_limitation":
        raise RuntimeError("verified frozen cache is exhausted but is not a dataset_limitation")
    if report.get("effect_claimed"):
        raise RuntimeError("verified frozen cache claims an effect; refusing to expand")
    if any(row.get("fifty_class_metric_defined") for row in report.get("heldout_coverage", [])):
        raise RuntimeError("verified frozen cache defines the 50-class metric; refusing to expand")
    return "expand"


def _frozen_summary(cache_dir: str) -> dict:
    manifest = _load_json(os.path.join(cache_dir, "manifest.json"))
    progress = _load_json(os.path.join(cache_dir, "progress.json")) or _empty_progress()
    report = inventory(cache_dir)
    coverage = []
    for row in report.get("heldout_coverage", []):
        coverage.append({
            "date": row["date"],
            "test_min": row["test_min"],
            "train_min": row["train_min"],
            "classes_absent_from_train": row["classes_absent_from_train"],
            "test_classes_present": row["test_classes_present"],
            "fifty_class_metric_defined": False,
        })
    return {
        "cache_dirname": DECODE306_FROZEN_CACHE_DIRNAME,
        "claim_status": "dataset_limitation",
        "next_action": "stop",
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "fifty_class_metric_defined": False,
        "dates": list(manifest["dates"]),
        "runs_consumed": int(progress["run_index"]),
        "runs_total": len(manifest["runs"]),
        "kept": int(progress.get("kept", 0)),
        "heldout_coverage": coverage,
        "batches": batch_provenance(cache_dir),
        "cache_preserved": True,
    }


def gate_from_counts(counts: np.ndarray, dates: list[str], consumed: list[bool]) -> dict:
    counts = np.asarray(counts, dtype=np.int64)
    if len(dates) != len(consumed) or counts.shape != (len(dates), DECODE306_N_CLASSES):
        raise RuntimeError("coverage counts do not match the date list")
    consumed_idx = [index for index, flag in enumerate(consumed) if flag]
    rows = []
    for index, date in enumerate(dates):
        test = counts[index]
        if not consumed[index]:
            rows.append({
                "date": date,
                "consumed": False,
                "heldout_eligible": False,
                "train_gate": False,
                "test_min": None,
                "train_min": None,
                "classes_below_min": [],
                "classes_absent_from_train": [],
                "fifty_class_metric_defined": False,
            })
            continue
        train = np.zeros(DECODE306_N_CLASSES, dtype=np.int64)
        for other in consumed_idx:
            if other != index:
                train += counts[other]
        test_ok = bool(np.all(test >= DECODE306_MIN_PER_CLASS))
        train_ok = bool(np.all(train >= DECODE306_MIN_PER_CLASS))
        rows.append({
            "date": date,
            "consumed": True,
            "heldout_eligible": test_ok,
            "train_gate": bool(train_ok),
            "test_min": int(test.min()),
            "train_min": int(train.min()),
            "classes_below_min": [VOCAB[cls] for cls in range(DECODE306_N_CLASSES) if test[cls] < DECODE306_MIN_PER_CLASS],
            "classes_absent_from_train": [VOCAB[cls] for cls in range(DECODE306_N_CLASSES) if train[cls] == 0],
            "fifty_class_metric_defined": bool(test_ok and train_ok),
        })
    eligible = [index for index, row in enumerate(rows) if row["heldout_eligible"]]
    label_gate = (
        len(eligible) >= DECODE306_MIN_HELDOUT_DAYS
        and all(rows[index]["train_gate"] for index in eligible)
    )
    return {
        "rows": rows,
        "eligible_days": eligible,
        "label_gate_met": bool(label_gate),
        "not_held_out_below_min": [
            rows[index]["date"] for index in consumed_idx if not rows[index]["heldout_eligible"]
        ],
        "partial_class_accuracy_reported": False,
    }


def _load_combined_counts(frozen_dir: str, expand_dir: str, n_frozen: int, n_expand: int) -> np.ndarray:
    counts = np.zeros((n_frozen + n_expand, DECODE306_N_CLASSES), dtype=np.int64)
    for batch in _load_count_rows(frozen_dir):
        for day, label in zip(batch["day_index"], batch["label"]):
            day = int(day)
            label = int(label)
            if day < 0 or day >= n_frozen:
                raise RuntimeError("frozen batch day index is outside the frozen date list")
            if 0 <= label < DECODE306_N_CLASSES:
                counts[day, label] += 1
    if n_expand == 0:
        return counts
    for batch in _load_count_rows(expand_dir):
        for day, label in zip(batch["day_index"], batch["label"]):
            day = int(day)
            label = int(label)
            if day < 0 or day >= n_expand:
                raise RuntimeError("expansion batch day index is outside the expansion date list")
            if 0 <= label < DECODE306_N_CLASSES:
                counts[n_frozen + day, label] += 1
    return counts


def _consumed_flags(n_frozen: int, runs: list[dict], run_index: int, n_expand_dates: int) -> list[bool]:
    flags = [True] * n_frozen
    dates = []
    for run in runs:
        if run["date"] not in dates:
            dates.append(run["date"])
    if len(dates) != n_expand_dates:
        raise RuntimeError("expansion date list does not match the run list")
    for date in dates:
        indexes = [index for index, run in enumerate(runs) if run["date"] == date]
        flags.append(bool(indexes) and all(index < run_index for index in indexes))
    return flags


def _audit_expand(cache_dir: str, progress: dict) -> None:
    names = _batch_names(cache_dir)
    recorded = progress.get("batch_sha256", [])
    kept = sum(len(row["label"]) for row in _load_count_rows(cache_dir))
    if kept != int(progress.get("kept", 0)) or len(names) != len(recorded):
        raise RuntimeError("expansion checkpoint does not match batch rows; refusing to append or rewrite")
    for rec, name in zip(recorded, names):
        path = os.path.join(cache_dir, name)
        if rec.get("name") != name or int(rec.get("nbytes", -1)) != os.path.getsize(path):
            raise RuntimeError("expansion batch fingerprint changed")
        if rec.get("sha256") != sha256_file(path):
            raise RuntimeError("expansion batch fingerprint changed")


def _feature_bytes(cache_dir: str) -> int:
    total = 0
    for name in _batch_names(cache_dir):
        total += os.path.getsize(os.path.join(cache_dir, name))
    return total


def _run_key(manifest: dict) -> dict:
    return {
        "dates": manifest["dates"],
        "runs": [
            (row["date"], row["base"], row["events_sha256"], row["h5_nbytes"], row["h5_shape"], row["h5_dtype"])
            for row in manifest["runs"]
        ],
        "preprocess_id": manifest["preprocess_id"],
        "selection_rule": manifest["selection_rule"],
        "cache_version": manifest["cache_version"],
    }


def _dates_of(runs: list[dict]) -> list[str]:
    dates = []
    for run in runs:
        if run["date"] not in dates:
            dates.append(run["date"])
    return dates


def _save_batch(cache_dir: str, features, labels, days, onsets, run_ids, progress) -> None:
    recorded = progress.setdefault("batch_sha256", [])
    if features:
        name = f"batch_{len(_batch_names(cache_dir)):04d}.npz"
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            raise RuntimeError(f"refusing to overwrite {name}")
        np.savez_compressed(
            path,
            features=np.stack(features),
            label=np.asarray(labels, dtype=np.int16),
            day_index=np.asarray(days, dtype=np.int16),
            onset_s=np.asarray(onsets, dtype=np.float64),
            run_index=np.asarray(run_ids, dtype=np.int16),
        )
        recorded.append({
            "name": name,
            "sha256": sha256_file(path),
            "nbytes": int(os.path.getsize(path)),
            "rows": len(features),
        })
    _write_json(os.path.join(cache_dir, "progress.json"), progress)


def _chunk_factory(frozen_dir: str, expand_dir: str, n_frozen: int, consumed: list[bool]):
    expand_keep = {index - n_frozen for index, flag in enumerate(consumed) if flag and index >= n_frozen}

    def factory():
        for name in _batch_names(frozen_dir):
            with np.load(os.path.join(frozen_dir, name)) as archive:
                features = np.array(archive["features"], dtype=np.float64, copy=True)
                labels = np.array(archive["label"], dtype=np.int64, copy=True)
                day = np.array(archive["day_index"], dtype=np.int64, copy=True)
            yield {"features": features, "label": labels, "day": day}
        for name in _batch_names(expand_dir):
            with np.load(os.path.join(expand_dir, name)) as archive:
                features = np.array(archive["features"], dtype=np.float64, copy=True)
                labels = np.array(archive["label"], dtype=np.int64, copy=True)
                day = np.array(archive["day_index"], dtype=np.int64, copy=True)
            keep = np.array([int(value) in expand_keep for value in day], dtype=bool)
            if not np.any(keep):
                continue
            yield {
                "features": features[keep],
                "label": labels[keep],
                "day": day[keep] + n_frozen,
            }

    return factory


def _seal(record: dict) -> dict:
    label = amended_protocol_label()
    record["amended_protocol"] = label
    record["amended_protocol_id"] = label["amended_protocol_id"]
    record["estimand"] = label["estimand"]
    record["selection_rule"] = label["selection_rule"]
    record["score_selection"] = False
    record["original_frozen_four_date_replication"] = False
    record["protocol_role"] = label["protocol_role"]
    record["old_dates_status"] = label["old_dates_status"]
    record["old_dates"] = label["old_dates"]
    record["earlier_data_exposures"] = label["earlier_data_exposures"]
    record["future_independent_confirmation_required"] = True
    record["effect_claimed"] = False
    record["real_meg_scientific_validation"] = False
    record["partial_class_accuracy_reported"] = False
    record["competition_encoder"] = False
    record["residual_proxy_is_not_evidence"] = True
    record["batch_read_cap_is_not_the_science_gate"] = True
    record["runtime_ceiling_required"] = False
    if not record.get("fifty_class_metric_defined"):
        record["frozen_eval"] = None
        record["fifty_class_metric_defined"] = False
    return record


def _finish(frozen_dir, expand_dir, manifest, progress, counts, *, batch_bytes, itemsize, runs_remaining: bool) -> dict:
    frozen_dates = manifest["frozen_provenance"]["dates"]
    expand_dates = manifest["dates"]
    dates = list(frozen_dates) + list(expand_dates)
    consumed = _consumed_flags(len(frozen_dates), manifest["runs"], int(progress["run_index"]), len(expand_dates))
    gate = gate_from_counts(counts, dates, consumed)
    scored = None
    if gate["label_gate_met"]:
        scored = score_consumed_days(
            _chunk_factory(frozen_dir, expand_dir, len(frozen_dates), consumed),
            dates,
            gate["eligible_days"],
        )
        if scored.get("effect_claimed"):
            raise RuntimeError("refusing effect_claimed from the 50-class score")
        claim = scored["claim_status"]
        nxt = "stop"
    elif runs_remaining:
        claim = "resume_required"
        nxt = "rerun_this_command"
    else:
        claim = "dataset_limitation"
        nxt = "stop"
    reason = {
        "resume_required": (
            "The earliest-4 cache stays as provenance. Later subject-0 dates remain, "
            "in acquisition order, and the 50-class gate is not met. Rerun the same command."
        ),
        "dataset_limitation": (
            "Every later subject-0 run has been scanned, or the expansion feature cache is full, "
            "and the held-out days still do not each have 3 observations of every class in the "
            "test day and in its train fold. No partial-class accuracy is reported."
        ),
        "decode306_probe_estimated": (
            "Every held-out day and its train fold has 3 observations of all 50 words. "
            "effect_claimed remains false."
        ),
        "label_gate_met_score_blocked_by_memory_cap": scored["reason"] if scored else "",
        "engineering_cap_metric_undefined": scored["reason"] if scored else "",
    }.get(claim, claim)
    record = {
        "claim_status": claim,
        "next_action": nxt,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "fifty_class_metric_defined": bool(scored and scored.get("fifty_class_metric_defined")),
        "frozen_eval": None if not scored else scored.get("frozen_eval"),
        "reason": reason,
        "science_gate": "fifty_class_min3_on_every_heldout_day_and_its_train_fold",
        "heldout_rule": (
            "A fully scanned date is held out only when that date has at least 3 of every class. "
            "Dates below that are train-only and are not scored. At least 2 held-out dates are required. "
            "Every held-out date's train fold must also have at least 3 of every class."
        ),
        "acquisition": DECODE306_EXPAND_SELECTION_RULE,
        "frozen_provenance": manifest["frozen_provenance"],
        "expansion_dates": expand_dates,
        "dates": dates,
        "runs_consumed": int(progress["run_index"]),
        "runs_total": len(manifest["runs"]),
        "kept": int(progress["kept"]),
        "skipped_overlap": int(progress["skipped_overlap"]),
        "in_vocab_considered": int(progress["in_vocab_considered"]),
        "heldout_coverage": gate["rows"],
        "not_held_out_below_min": gate["not_held_out_below_min"],
        "eligible_heldout_dates": [dates[index] for index in gate["eligible_days"]],
        "batch_read_cap_bytes": int(batch_bytes),
        "planned_budget": planned_budget(itemsize),
        "checkpoint": expand_dir,
        "frozen_cache": frozen_dir,
        "partial_class_accuracy_reported": False,
    }
    if scored and scored.get("fifty_class_metric_defined"):
        record["frozen_eval"] = scored["frozen_eval"]
    record = _seal(record)
    _write_json(os.path.join(expand_dir, "status.json"), {
        key: value for key, value in record.items() if key not in ("planned_budget",)
    })
    return record


def launch_expansion(frozen_dir: str, expand_dir: str, runs: list[dict], read_raw, *,
                     batch_bytes: int = DECODE306_BATCH_READ_BYTES, itemsize: int = 4) -> dict:
    frozen_real = os.path.realpath(frozen_dir)
    os.makedirs(expand_dir, exist_ok=True)
    expand_real = os.path.realpath(expand_dir)
    if expand_real == frozen_real or expand_real.startswith(frozen_real + os.sep):
        raise RuntimeError("refusing to write inside the frozen cache")
    frozen_manifest = _load_json(os.path.join(frozen_dir, "manifest.json"))
    if frozen_manifest is None:
        raise RuntimeError("frozen cache has no manifest")
    if frozen_manifest.get("preprocess_id") != PREPROCESS_ID:
        raise RuntimeError("frozen cache preprocess does not match; refusing to pool it")
    provenance = _frozen_summary(frozen_dir)
    proposed = {
        "cache_version": DECODE306_EXPAND_CACHE_VERSION,
        "preprocess_id": PREPROCESS_ID,
        "selection_rule": DECODE306_EXPAND_SELECTION_RULE,
        "excluded_date": EXCLUDED_SMOKE_DATE,
        "nonoverlap_sec": DECODE306_NONOVERLAP_SEC,
        "min_per_class": DECODE306_MIN_PER_CLASS,
        "n_classes": DECODE306_N_CLASSES,
        "dates": _dates_of(runs),
        "runs": runs,
        "frozen_provenance": provenance,
    }
    manifest_path = os.path.join(expand_dir, "manifest.json")
    existing = _load_json(manifest_path)
    if existing is None:
        _write_json(manifest_path, proposed)
        manifest = proposed
    else:
        if _run_key(existing) != _run_key(proposed):
            raise RuntimeError("refusing to append: the expansion date, run, or source fingerprint changed")
        if existing["frozen_provenance"]["batches"] != provenance["batches"]:
            raise RuntimeError("frozen cache fingerprint changed; refusing to append")
        if existing["frozen_provenance"]["dates"] != provenance["dates"]:
            raise RuntimeError("frozen date list changed; refusing to append")
        manifest = existing
    progress = _load_json(os.path.join(expand_dir, "progress.json")) or _empty_progress()
    progress.setdefault("batch_sha256", [])
    _audit_expand(expand_dir, progress)
    frozen_dates = provenance["dates"]
    expand_dates = manifest["dates"]
    counts = _load_combined_counts(frozen_dir, expand_dir, len(frozen_dates), len(expand_dates))
    dates = list(frozen_dates) + list(expand_dates)
    consumed = _consumed_flags(len(frozen_dates), manifest["runs"], int(progress["run_index"]), len(expand_dates))
    if gate_from_counts(counts, dates, consumed)["label_gate_met"]:
        return _finish(
            frozen_dir, expand_dir, manifest, progress, counts,
            batch_bytes=batch_bytes, itemsize=itemsize, runs_remaining=False,
        )
    if _feature_bytes(expand_dir) > DECODE306_FEATURE_CACHE_MAX_BYTES:
        report = _finish(
            frozen_dir, expand_dir, manifest, progress, counts,
            batch_bytes=batch_bytes, itemsize=itemsize, runs_remaining=False,
        )
        report["claim_status"] = "dataset_limitation"
        report["next_action"] = "stop"
        report["fifty_class_metric_defined"] = False
        report["frozen_eval"] = None
        report["reason"] = "The expansion feature cache reached its 2 GiB guard before the 50-class gate."
        return _seal(report)
    if int(progress["run_index"]) >= len(manifest["runs"]):
        return _finish(
            frozen_dir, expand_dir, manifest, progress, counts,
            batch_bytes=batch_bytes, itemsize=itemsize, runs_remaining=False,
        )
    features, labels, days, onsets, run_ids = [], [], [], [], []
    used = 0
    per_window = N_SENSORS * WINDOW_SAMPLES * int(itemsize)
    run_index = int(progress["run_index"])
    word_index = int(progress["word_index"])
    stop_for_gate = False
    while run_index < len(manifest["runs"]) and not stop_for_gate:
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
                _save_batch(expand_dir, features, labels, days, onsets, run_ids, progress)
                return _finish(
                    frozen_dir, expand_dir, manifest, progress, counts,
                    batch_bytes=batch_bytes, itemsize=itemsize, runs_remaining=True,
                )
            raw = read_raw(run_index, float(onset))
            progress["in_vocab_considered"] += 1
            word_index += 1
            if raw is None:
                continue
            local_day = expand_dates.index(manifest["runs"][run_index]["date"])
            features.append(normalize_raw_window(raw).astype(np.float32))
            labels.append(int(label))
            days.append(local_day)
            onsets.append(float(onset))
            run_ids.append(run_index)
            counts[len(frozen_dates) + local_day, int(label)] += 1
            used += per_window
            progress["kept"] += 1
            progress["last_kept_onset"] = float(onset)
            progress["last_kept_run"] = run_index
            del raw
        finished_date = (
            run_index + 1 == len(manifest["runs"])
            or manifest["runs"][run_index + 1]["date"] != manifest["runs"][run_index]["date"]
        )
        run_index += 1
        word_index = 0
        progress["last_kept_onset"] = None
        progress["last_kept_run"] = None
        if finished_date:
            consumed_now = _consumed_flags(len(frozen_dates), manifest["runs"], run_index, len(expand_dates))
            if gate_from_counts(counts, dates, consumed_now)["label_gate_met"]:
                stop_for_gate = True
    progress["run_index"] = run_index
    progress["word_index"] = word_index
    _save_batch(expand_dir, features, labels, days, onsets, run_ids, progress)
    remaining = run_index < len(manifest["runs"])
    return _finish(
        frozen_dir, expand_dir, manifest, progress, counts,
        batch_bytes=batch_bytes, itemsize=itemsize, runs_remaining=remaining,
    )


def fingerprint_runs(specs: list[dict]) -> tuple[list[dict], int]:
    import numpy as np
    from .stream import open_data_dataset, parse_events, vocab_index
    runs = []
    itemsize = None
    for spec in specs:
        with open(spec["events"], encoding="utf-8") as handle:
            text = handle.read()
        rows, refused = parse_events(text)
        words = []
        for row in rows:
            label = vocab_index(row["segment"], VOCAB)
            if label < 0:
                continue
            words.append([float(row["timemeg"]), int(label)])
        words.sort()
        handle, dataset = open_data_dataset(spec["h5"])
        try:
            this_item = int(np.dtype(dataset.dtype).itemsize)
            if itemsize is None:
                itemsize = this_item
            elif this_item != itemsize:
                raise RuntimeError("refusing mixed h5 dtypes inside one probe")
            fingerprint = h5_fingerprint(spec["h5"], dataset)
        finally:
            handle.close()
        runs.append({
            "date": spec["date"],
            "base": spec["base"],
            "h5": spec["h5"],
            "events_sha256": events_sha256(spec["events"]),
            "h5_nbytes": fingerprint["nbytes"],
            "h5_shape": fingerprint["shape"],
            "h5_dtype": fingerprint["dtype"],
            "words": words,
            "fields_seen_not_used": refused,
        })
    return runs, itemsize or 4


def run_expansion(destination: str, data_root: str) -> dict:
    from .stream import list_subject0_runs
    frozen_dir = os.path.join(destination, DECODE306_FROZEN_CACHE_DIRNAME)
    if frozen_action(frozen_dir) != "expand":
        raise RuntimeError("expansion was requested before the verified earliest-4 cache was exhausted")
    discovered = list_subject0_runs(data_root)
    manifest = _load_json(os.path.join(frozen_dir, "manifest.json"))
    assert_frozen_rediscovery(discovered, manifest)
    later = select_later_runs(discovered, list(manifest["dates"]))
    runs, itemsize = fingerprint_runs(later)
    expand_dir = os.path.join(destination, DECODE306_EXPAND_CACHE_DIRNAME)
    from .decode306 import read_window
    from .protocol import FS_HZ
    from .stream import open_data_dataset
    open_handles = {}

    def read_raw(run_index, onset):
        if run_index not in open_handles:
            handle, dataset = open_data_dataset(runs[run_index]["h5"])
            open_handles[run_index] = (handle, dataset)
        dataset = open_handles[run_index][1]
        start = int(round(float(onset) * FS_HZ))
        stop = start + WINDOW_SAMPLES
        if start < 0 or stop > int(dataset.shape[1]):
            return None
        block, _used = read_window(dataset, start, stop, 0, N_SENSORS * WINDOW_SAMPLES * itemsize)
        return block

    try:
        return launch_expansion(frozen_dir, expand_dir, runs, read_raw, itemsize=itemsize)
    finally:
        for handle, _dataset in open_handles.values():
            handle.close()
