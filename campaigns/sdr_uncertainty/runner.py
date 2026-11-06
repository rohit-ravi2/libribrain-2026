"""Local and synthetic entry points.

numpy is imported only after a live smoke has passed the cgroup and thread checks.
Synthetic evaluation does not read MEG and is not a scientific result.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time

from .cgroup import (
    CgroupError,
    nvidia_majors_from_dev,
    runtime_environment_errors,
    verify_fixture,
    verify_live,
    verify_live_hardware,
)
from .protocol import (
    ALREADY_EXPLORED,
    MAX_OUTPUT_BYTES,
    OOD_SELECTION_RULE,
    OOD_SOFT_DEADLINE_SEC,
    PROTOCOL_VERSION,
    protocol_sha256,
)


def _repo_root() -> str:
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def assert_output_dir(path: str, *, synthetic: bool) -> str:
    real = os.path.realpath(path)
    if "holdout" in real.lower():
        raise RuntimeError("refusing an output path that mentions holdout")
    repo = _repo_root()
    if real == repo or real.startswith(repo + os.sep):
        raise RuntimeError("refusing to write campaign output inside the git repository")
    if synthetic and os.environ.get("SDR_SYNTHETIC") == "1" and os.environ.get("SDR_TEST_HOOK") == "1":
        return real
    prefix = os.path.realpath(os.environ.get("LIBRIBRAIN_SDR_SSD_PREFIX", "/mnt/ssd4tb"))
    if real != prefix and not real.startswith(prefix + os.sep):
        raise RuntimeError(f"output {real} is not under the SSD prefix {prefix}")
    return real


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _repo_root(), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes.
    return int(rss) * 1024


def _stamp(payload: dict, started: float) -> dict:
    payload["protocol_version"] = PROTOCOL_VERSION
    payload["protocol_sha256"] = protocol_sha256()
    payload["git_commit"] = _git_commit()
    payload["python"] = sys.version.split()[0]
    payload["utc_unix"] = time.time()
    payload["wall_sec"] = time.perf_counter() - started
    payload["peak_rss_bytes"] = _peak_rss_bytes()
    payload["real_meg_scientific_validation"] = False
    payload["already_explored_not_reused"] = ALREADY_EXPLORED
    payload["argv"] = sys.argv
    return payload


def _write(path: str, payload: dict) -> str:
    text = json.dumps(payload, indent=1, sort_keys=True)
    raw = text.encode()
    if len(raw) > MAX_OUTPUT_BYTES:
        raise RuntimeError(f"refusing to write {len(raw)} bytes; cap is {MAX_OUTPUT_BYTES}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    return path


def _feature_checksum(array) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def smoke_synthetic(out_dir: str) -> dict:
    started = time.perf_counter()
    destination = assert_output_dir(out_dir, synthetic=True)
    from .crossfit import run_crossfit
    from .synthetic import make_dataset
    batch = make_dataset("hetero", n_days=4, n_per=160)
    result = run_crossfit(batch)
    result = _stamp(result, started)
    result["data_regime"] = "synthetic_dgp"
    result["raw_meg_read"] = False
    path = os.path.join(destination, "sdr_uncertainty_synthetic.json")
    _write(path, result)
    result["output_path"] = path
    return result


def _live_preflight(*, require_runtime: bool = True):
    if os.environ.get("SDR_FORBID_LIVE_SMOKE") == "1" or os.environ.get("SDR_FORBID_LIVE_COMPUTE") == "1":
        raise CgroupError(["refusing live compute because SDR_FORBID_LIVE_SMOKE or SDR_FORBID_LIVE_COMPUTE is set"])
    env_errors = runtime_environment_errors(dict(os.environ), nvidia_majors_from_dev())
    if env_errors:
        raise CgroupError(env_errors)
    fixture = os.environ.get("SDR_CGROUP_FIXTURE")
    if fixture:
        if os.environ.get("SDR_TEST_HOOK") != "1":
            raise CgroupError(["SDR_CGROUP_FIXTURE is a test hook and is refused for live compute"])
        verify_fixture(fixture)
        return {"source": "test_fixture", "local_cgroup_verified": False}
    info = verify_live() if require_runtime else verify_live_hardware()
    info["local_cgroup_verified"] = True
    return info


def _open_run(spec):
    from .stream import open_data_dataset, parse_events
    with open(spec["events"], encoding="utf-8") as handle:
        text = handle.read()
    rows, refused = parse_events(text)
    handle, dataset = open_data_dataset(spec["h5"])
    return text, rows, refused, dataset, handle.close


def smoke_local() -> dict:
    started = time.perf_counter()
    cgroup_info = _live_preflight()
    out = os.environ.get("LIBRIBRAIN_SDR_OUT")
    if not out:
        raise RuntimeError("blocker: LIBRIBRAIN_SDR_OUT is not set")
    destination = assert_output_dir(out, synthetic=False)
    data_root = os.environ.get("LIBRIBRAIN_DATA")
    if not data_root:
        raise RuntimeError("blocker: LIBRIBRAIN_DATA is not set")
    from .protocol import VOCAB
    from .stream import discover_smoke_run, load_smoke_run
    spec = discover_smoke_run(data_root)
    batch, exposure = load_smoke_run(spec, VOCAB)
    from .crossfit import smoke_one_series
    result = smoke_one_series(batch)
    result = _stamp(result, started)
    result["cgroup"] = cgroup_info
    result["selection"] = {key: spec[key] for key in ("date", "base", "selection_rule", "excluded_date")}
    result["exposure"] = exposure
    result["feature_checksum_sha256"] = _feature_checksum(batch["y"])
    result["n_windows"] = int(batch["y"].shape[0])
    result["data_regime"] = "local_smoke_one_training_day_group"
    result["raw_meg_transferred"] = False
    path = os.path.join(destination, "sdr_uncertainty_local_smoke.json")
    _write(path, result)
    result["output_path"] = path
    return result


def out_of_day() -> dict:
    started_perf = time.perf_counter()
    cgroup_info = _live_preflight()
    out = os.environ.get("LIBRIBRAIN_SDR_OUT")
    if not out:
        raise RuntimeError("blocker: LIBRIBRAIN_SDR_OUT is not set")
    destination = assert_output_dir(out, synthetic=False)
    data_root = os.environ.get("LIBRIBRAIN_DATA")
    if not data_root:
        raise RuntimeError("blocker: LIBRIBRAIN_DATA is not set")
    from .outday import evaluate_out_of_day, stream_out_of_day
    from .protocol import VOCAB
    from .stream import discover_out_of_day_runs
    specs = discover_out_of_day_runs(data_root)
    if len(specs) < 2:
        record = evaluate_out_of_day(
            [],
            regime="local_out_of_day_stream",
            timed_out=False,
            cgroup_live=bool(cgroup_info["local_cgroup_verified"]),
            selection_rule=OOD_SELECTION_RULE,
        )
        record["dates_discovered"] = [spec["date"] for spec in specs]
        record["meg_opened"] = False
    else:
        streamed = stream_out_of_day(
            specs,
            VOCAB,
            deadline=time.monotonic() + OOD_SOFT_DEADLINE_SEC,
            opener=_open_run,
        )
        record = evaluate_out_of_day(
            streamed["parts"],
            regime="local_out_of_day_stream",
            timed_out=bool(streamed["timed_out"]),
            cgroup_live=bool(cgroup_info["local_cgroup_verified"]),
            selection_rule=streamed["selection_rule"],
        )
        record["read_budget_hit"] = streamed["read_budget_hit"]
        record["dates_discovered"] = [spec["date"] for spec in specs]
        record["meg_opened"] = True
        record["exposure"] = {
            "files": [item for part in streamed["parts"] for item in part.get("exposure_files", [])],
            "fields_seen_not_used": streamed["fields_seen_not_used"],
            "fields_stored_from_events": ["timemeg_slice_only_not_in_features", "segment_task_label_only"],
            "onset_in_feature_matrix": False,
            "full_meg_loaded": False,
            "holdout_opened": False,
            "quarters_scored": False,
        }
    science = bool(record["real_meg_scientific_validation"])
    record = _stamp(record, started_perf)
    record["real_meg_scientific_validation"] = science
    record["cgroup"] = cgroup_info
    record["data_regime"] = "local_out_of_day_stream"
    record["raw_meg_transferred"] = False
    record["caps_relaxed"] = False
    path = os.path.join(destination, "sdr_uncertainty_out_of_day.json")
    _write(path, record)
    record["output_path"] = path
    return record


def decode_probe() -> dict:
    started_perf = time.perf_counter()
    cgroup_info = _live_preflight()
    out = os.environ.get("LIBRIBRAIN_SDR_OUT")
    if not out:
        raise RuntimeError("blocker: LIBRIBRAIN_SDR_OUT is not set")
    destination = assert_output_dir(out, synthetic=False)
    data_root = os.environ.get("LIBRIBRAIN_DATA")
    if not data_root:
        raise RuntimeError("blocker: LIBRIBRAIN_DATA is not set")
    import numpy as np
    from .decode import collect_decode_windows, evaluate_decoding
    from .protocol import OOD_MAX_DAYS, OOD_MAX_READ_BYTES, OOD_MAX_WINDOWS_PER_DAY, VOCAB
    from .stream import discover_out_of_day_runs
    specs = discover_out_of_day_runs(data_root, max_days=OOD_MAX_DAYS)
    parts = []
    bytes_used = 0
    if len(specs) < 2:
        record = {
            "claim_status": "decoding_inadequate",
            "real_meg_scientific_validation": False,
            "effect_claimed": False,
            "endpoint_is_decoding": False,
            "adequate_for_competition_claim": False,
            "reason": "Fewer than two eligible days. No decoding score is reported.",
            "dates": [spec["date"] for spec in specs],
            "frozen_eval": None,
            "meg_opened": False,
        }
        record = _stamp(record, started_perf)
        record["real_meg_scientific_validation"] = False
        record["effect_claimed"] = False
        record["cgroup"] = cgroup_info
        record["bytes_read"] = 0
        record["caps_relaxed"] = False
        path = os.path.join(destination, "sdr_uncertainty_decode_probe.json")
        _write(path, record)
        record["output_path"] = path
        return record
    for spec in specs:
        remaining = OOD_MAX_READ_BYTES - bytes_used
        if remaining <= 0:
            break
        text, rows, refused, dataset, closer = _open_decode(spec)
        try:
            part = collect_decode_windows(dataset, rows, VOCAB, max_windows=OOD_MAX_WINDOWS_PER_DAY, max_bytes=remaining)
        finally:
            closer()
        part["day"] = np.full(part["n_windows"], len(parts), dtype=np.int64)
        part["date"] = spec["date"]
        part["fields_seen_not_used"] = refused
        part["events_bytes"] = len(text.encode())
        parts.append(part)
        bytes_used += int(part["bytes_read"])
    record = evaluate_decoding({
            "y": np.concatenate([part["y"] for part in parts]),
            "state": np.concatenate([part["state"] for part in parts]),
            "label": np.concatenate([part["label"] for part in parts]),
            "day": np.concatenate([part["day"] for part in parts]),
            "time_index": np.concatenate([part["time_index"] for part in parts]),
            "dates": [part["date"] for part in parts],
        })
    record = _stamp(record, started_perf)
    record["real_meg_scientific_validation"] = False
    record["effect_claimed"] = False
    record["cgroup"] = {**cgroup_info, "local_cgroup_verified": bool(cgroup_info.get("local_cgroup_verified"))}
    record["bytes_read"] = int(bytes_used) if parts else 0
    record["caps_relaxed"] = False
    path = os.path.join(destination, "sdr_uncertainty_decode_probe.json")
    _write(path, record)
    record["output_path"] = path
    return record


def _open_decode(spec):
    from .stream import open_data_dataset, parse_events
    with open(spec["events"], encoding="utf-8") as handle:
        text = handle.read()
    rows, refused = parse_events(text)
    handle, dataset = open_data_dataset(spec["h5"])
    return text, rows, refused, dataset, handle.close


def _decode306_runs(data_root: str):
    import numpy as np
    from .decode306_resume import events_sha256, h5_fingerprint
    from .protocol import DECODE306_MAX_DAYS, VOCAB
    from .stream import list_subject0_runs, open_data_dataset, parse_events, vocab_index
    dates = []
    selected = []
    for spec in list_subject0_runs(data_root):
        if spec["date"] not in dates:
            if len(dates) >= DECODE306_MAX_DAYS:
                continue
            dates.append(spec["date"])
        selected.append(spec)
    runs = []
    itemsize = None
    for spec in selected:
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


def decode306() -> dict:
    started_perf = time.perf_counter()
    cgroup_info = _live_preflight(require_runtime=False)
    out = os.environ.get("LIBRIBRAIN_SDR_OUT")
    if not out:
        raise RuntimeError("blocker: LIBRIBRAIN_SDR_OUT is not set")
    destination = assert_output_dir(out, synthetic=False)
    data_root = os.environ.get("LIBRIBRAIN_DATA")
    if not data_root:
        raise RuntimeError("blocker: LIBRIBRAIN_DATA is not set")
    from .decode306 import read_window
    from .decode306_resume import resume_batch
    from .protocol import FS_HZ, N_SENSORS, WINDOW_SAMPLES
    from .stream import open_data_dataset
    runs, itemsize = _decode306_runs(data_root)
    if len({row["date"] for row in runs}) < 2:
        raise RuntimeError("blocker: fewer than two subject-0 dates remain after excluding 2024-10-04")
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
        block, _ = read_window(dataset, start, stop, 0, N_SENSORS * WINDOW_SAMPLES * itemsize)
        return block

    cache_dir = os.path.join(destination, "decode306_cache")
    try:
        record = resume_batch(cache_dir, runs, read_raw, itemsize=itemsize)
    finally:
        for handle, _dataset in open_handles.values():
            handle.close()
    record = _stamp(record, started_perf)
    record["real_meg_scientific_validation"] = False
    record["effect_claimed"] = False
    record["cgroup"] = cgroup_info
    record["caps_relaxed"] = False
    record["runtime_ceiling_required"] = False
    record["microns_lock"] = "not_acquired_here; coordinator queues after the MICrONS CPU job"
    record["status_correction"] = "campaigns/sdr_uncertainty/STATUS_CORRECTION.md"
    record["full_meg_loaded"] = False
    path = os.path.join(destination, "sdr_uncertainty_decode306.json")
    _write(path, record)
    record["output_path"] = path
    return record


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else ""
    try:
        if command == "verify-cgroup":
            info = verify_live()
            print(json.dumps({"ok": True, **info, "compute": False}))
            return 0
        if command == "smoke-synthetic":
            out = os.environ.get("LIBRIBRAIN_SDR_OUT", "")
            if not out:
                print("blocker: LIBRIBRAIN_SDR_OUT is not set", file=sys.stderr)
                return 2
            result = smoke_synthetic(out)
            print(json.dumps({
                "output_path": result["output_path"],
                "claim_status": result["claim_status"],
                "real_meg_scientific_validation": False,
                "data_regime": "synthetic_dgp",
            }))
            return 0
        if command == "smoke-local":
            result = smoke_local()
            print(json.dumps({
                "output_path": result["output_path"],
                "claim_status": result["claim_status"],
                "real_meg_scientific_validation": False,
                "data_regime": result["data_regime"],
                "n_windows": result["n_windows"],
                "peak_rss_bytes": result["peak_rss_bytes"],
                "wall_sec": result["wall_sec"],
                "acquisition_date": result["selection"]["date"],
                "run_base": result["selection"]["base"],
                "banner": "Local smoke only. Not a cross-day result and not scientific validation.",
            }))
            return 0
        if command == "decode-306":
            result = decode306()
            print(json.dumps({
                "output_path": result["output_path"],
                "claim_status": result["claim_status"],
                "real_meg_scientific_validation": False,
                "effect_claimed": False,
                "fifty_class_metric_defined": result.get("fifty_class_metric_defined"),
                "planned_max_read_bytes": (result.get("planned_budget") or {}).get("max_read_bytes"),
                "next_action": result.get("next_action"),
                "kept": result.get("kept"),
                "banner": "One 512 MiB batch. Rerun only while next_action is rerun_this_command. Not an effect claim.",
            }))
            return 0
        if command == "decode-probe":
            result = decode_probe()
            print(json.dumps({
                "output_path": result["output_path"],
                "claim_status": result["claim_status"],
                "real_meg_scientific_validation": False,
                "effect_claimed": False,
                "endpoint_is_decoding": result.get("endpoint_is_decoding"),
                "adequate_for_competition_claim": False,
                "banner": "Decode probe only. Not a competition-decoder result and not evidence that weighting helps.",
            }))
            return 0
        if command == "out-of-day":
            result = out_of_day()
            science = bool(result["real_meg_scientific_validation"])
            print(json.dumps({
                "output_path": result["output_path"],
                "claim_status": result["claim_status"],
                "real_meg_scientific_validation": science,
                "data_regime": "local_out_of_day_stream",
                "n_days": result.get("n_days"),
                "timed_out": result.get("timed_out"),
                "peak_rss_bytes": result["peak_rss_bytes"],
                "wall_sec": result["wall_sec"],
                "banner": (
                    "Out-of-day cross-fit finished inside the original caps."
                    if science else
                    "Not a scientific result. See claim_status. Caps were not relaxed."
                ),
            }))
            return 0
        print(
            "usage: python -m campaigns.sdr_uncertainty.runner "
            "{verify-cgroup|smoke-synthetic|smoke-local|out-of-day|decode-probe|decode-306}",
            file=sys.stderr,
        )
        return 2
    except (CgroupError, RuntimeError) as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
