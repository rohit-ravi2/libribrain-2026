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

from .cgroup import CgroupError, nvidia_majors_from_dev, runtime_environment_errors, verify_fixture, verify_live
from .protocol import ALREADY_EXPLORED, MAX_OUTPUT_BYTES, PROTOCOL_VERSION, protocol_sha256


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


def smoke_local() -> dict:
    started = time.perf_counter()
    if os.environ.get("SDR_FORBID_LIVE_SMOKE") == "1":
        raise CgroupError(["refusing live smoke because SDR_FORBID_LIVE_SMOKE=1"])
    env_errors = runtime_environment_errors(dict(os.environ), nvidia_majors_from_dev())
    if env_errors:
        raise CgroupError(env_errors)
    fixture = os.environ.get("SDR_CGROUP_FIXTURE")
    if fixture:
        if os.environ.get("SDR_TEST_HOOK") != "1":
            raise CgroupError(["SDR_CGROUP_FIXTURE is a test hook and is refused for a local smoke"])
        verify_fixture(fixture)
        cgroup_info = {"source": "test_fixture", "local_cgroup_verified": False}
    else:
        cgroup_info = verify_live()
        cgroup_info["local_cgroup_verified"] = True
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
        print(
            "usage: python -m campaigns.sdr_uncertainty.runner "
            "{verify-cgroup|smoke-synthetic|smoke-local}",
            file=sys.stderr,
        )
        return 2
    except (CgroupError, RuntimeError) as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
