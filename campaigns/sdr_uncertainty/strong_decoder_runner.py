"""Local entry for the frozen-trunk noise prior. Does not launch a fit by itself.

The shell must hold the scheduler lock. This process does not acquire a second lock
and does not wait. Missing checkpoint, cache, or torch fails closed with no metric.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from .cgroup import CgroupError, verify_live_hardware
from .strong_decoder import CHECKPOINT_BASENAME, CHECKPOINT_BYTES
from .strong_noise import (
    ESTIMAND,
    LPP_NOTE,
    PROTOCOL_ID,
    ProtocolError,
)


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(root: str) -> dict:
    code = {}
    for rel in ("src/v3lib.py", "src/models.py", "scripts/train_v3.py", "scripts/predict_holdout_v3.py", "src/split2.py"):
        code[rel] = _sha256_file(os.path.join(root, rel))
    ckpt = os.environ.get("LB_STRONG_CKPT", os.path.join(root, "runs", CHECKPOINT_BASENAME))
    cache = os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache")
    lpp = os.environ.get("LIBRIBRAIN_LPP", "/mnt/ssd4tb/data/extmeg/lpp_listen")
    return {
        "protocol_id": PROTOCOL_ID,
        "estimand": ESTIMAND,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "valid_scientific_claim": False,
        "runs_completed": False,
        "code_sha256": code,
        "checkpoint": {
            "path": ckpt,
            "present": os.path.isfile(ckpt),
            "bytes": os.path.getsize(ckpt) if os.path.isfile(ckpt) else None,
            "expected_bytes_if_deep_c_ep6": CHECKPOINT_BYTES,
            "sha256": _sha256_file(ckpt),
        },
        "cache": {"path": os.path.join(cache, "cont"), "present": os.path.isdir(os.path.join(cache, "cont"))},
        "lpp": {"path": lpp, "present": os.path.isdir(lpp), "stage": "not_run", "pooling": "forbidden", "note": LPP_NOTE},
        "competition_holdout_read": False,
        "licences": {"libribrain": "CC-BY-NC-4.0", "lpp": "PDDL"},
        "old_probe_not_reinterpreted": True,
        "leaderboard_not_comparable": {"broad_top10": 0.72230, "deep_top10": 0.72929},
    }


def run(root: str) -> dict:
    if os.environ.get("SDR_FORBID_LIVE_COMPUTE") == "1":
        raise CgroupError(["refusing live compute because SDR_FORBID_LIVE_COMPUTE is set"])
    if os.environ.get("SDR_STRONG_LOCK") != "1":
        raise ProtocolError("scheduler self-lock is not held; refusing to run beside MICrONS or ANN")
    hardware = verify_live_hardware()
    info = manifest(root)
    info["cgroup"] = hardware
    info["runtime_ceiling_required"] = False
    if not info["checkpoint"]["present"]:
        info["blocker"] = f"missing strong checkpoint {info['checkpoint']['path']}"
        return info
    if info["checkpoint"]["bytes"] != CHECKPOINT_BYTES and os.path.basename(info["checkpoint"]["path"]) == CHECKPOINT_BASENAME:
        info["blocker"] = "checkpoint size does not match the local deep_c_ep6.pt manifest"
        return info
    if not info["cache"]["present"]:
        info["blocker"] = f"missing continuous cache {info['cache']['path']}"
        return info
    try:
        from .strong_decoder import load_strong_decoder
        model = load_strong_decoder(info["checkpoint"]["path"])
    except Exception as exc:
        info["blocker"] = f"strong decoder did not load: {exc}"
        return info
    info["strong_decoder_loaded"] = True
    info["decoder_identity"] = "src.v3lib.Model"
    info["checkpoint_sha256"] = model._sdr_sha256
    info["blocker"] = (
        "Checkpoint loaded. A full session fit is a separate resumed job and was not started by this import. "
        "No test metric was computed."
    )
    info["valid_scientific_claim"] = False
    return info


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    command = argv[0] if argv else ""
    try:
        if command == "manifest":
            payload = manifest(root)
            print(json.dumps({"protocol_id": payload["protocol_id"], "checkpoint_present": payload["checkpoint"]["present"], "runs_completed": False}))
            return 0
        if command == "run":
            payload = run(root)
            print(json.dumps({
                "protocol_id": payload["protocol_id"],
                "valid_scientific_claim": False,
                "effect_claimed": False,
                "runs_completed": False,
                "blocker": payload.get("blocker"),
                "strong_decoder_loaded": payload.get("strong_decoder_loaded", False),
            }))
            return 0 if payload.get("strong_decoder_loaded") else 2
        print("usage: python -m campaigns.sdr_uncertainty.strong_decoder_runner {manifest|run}", file=sys.stderr)
        return 2
    except (CgroupError, ProtocolError, FileNotFoundError) as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
