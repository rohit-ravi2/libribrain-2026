"""Bind the noise prior to src.v3lib.Model. A linear probe cannot authorize a claim.

Cloud runners do not have the 24.7 MB checkpoint or torch. Missing either fails closed
before a metric is written. Loading the real module also imports src.split, which reads
local session tables; that import happens only inside load_strong_decoder.
"""
from __future__ import annotations

import hashlib
import os

from .strong_noise import CHECKPOINT_BASENAME, CHECKPOINT_BYTES, ProtocolError

_LOADED = set()
REQUIRED = ("enc", "head_ce", "head_t5", "logit_s", "A", "subs", "log_prior", "args")


def _count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if hasattr(value, "shape"):
        total = 1
        for dim in value.shape:
            total *= int(dim)
        return int(total)
    if isinstance(value, dict):
        return sum(_count(item) for item in value.values())
    return 0


def attest_checkpoint_dict(payload: dict) -> dict:
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ProtocolError(f"checkpoint is not a v3 train_v3 dict, missing {missing}")
    args = payload["args"]
    for key in ("d", "D", "maxw", "t5", "ctx", "rel"):
        if key not in args:
            raise ProtocolError(f"checkpoint args missing {key}")
    if int(np_shape0(payload["head_ce"])) != 50 or int(np_shape0(payload["head_t5"])) != 1024:
        raise ProtocolError("heads are not the 50-way CE and 1024-d T5 projections")
    count = _count(payload["enc"]) + _count(payload["head_ce"]) + _count(payload["head_t5"])
    if payload.get("ctx") is not None:
        count += _count(payload["ctx"])
    if count < 1_000_000:
        raise ProtocolError(f"parameter count {count} is not the strong v3 decoder")
    if int(args.get("rel", 0)) != 0:
        raise ProtocolError("primary checkpoint must be the timing-zeroed deep_c recipe")
    return {
        "family": "src.v3lib.Model",
        "parameter_count": count,
        "d": args["d"],
        "D": args["D"],
        "t5": args["t5"],
        "ctx_layers": args.get("ctx_layers"),
        "rel": 0,
        "timing_in_primary_forward": "zero",
    }


def np_shape0(module_state: dict) -> int:
    weight = module_state["weight"]
    return int(weight.shape[0])


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strong_decoder(path: str):
    """Load runs/deep_c_ep6.pt through src.v3lib.Model. Does not run a training job."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    if os.path.basename(path) == CHECKPOINT_BASENAME and size != CHECKPOINT_BYTES:
        raise ProtocolError(f"{CHECKPOINT_BASENAME} is {size} bytes, expected {CHECKPOINT_BYTES}")
    os.environ["LB_DEV"] = "cpu"
    import torch
    from src.v3lib import Model
    model = Model(path)
    token = ("src.v3lib.Model", id(model))
    _LOADED.add(token)
    model._sdr_token = token
    model._sdr_sha256 = checkpoint_sha256(path)
    model._sdr_bytes = size
    return model


def is_strong_decoder(model) -> bool:
    token = getattr(model, "_sdr_token", None)
    return token in _LOADED and token[0] == "src.v3lib.Model"


def forward_smoke(model, meg, valid, alignment, group_index, pad, rel) -> dict:
    """One real forward. rel must already be zeros for the primary arm."""
    if not is_strong_decoder(model):
        raise ProtocolError("refusing forward: object was not loaded by load_strong_decoder")
    if float(np_max_abs(rel)) != 0.0:
        raise ProtocolError("primary forward was given nonzero relative timing")
    embedding, ce, t5 = model.forward(meg, valid, alignment, group_index, pad, rel)
    return {"embedding": embedding, "ce": ce, "t5": t5, "decoder": "src.v3lib.Model"}


def np_max_abs(rel) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(rel))))


def claim_from_decoder(model, *, metric_defined: bool) -> dict:
    strong = is_strong_decoder(model)
    valid = bool(strong and metric_defined)
    return {
        "decoder_identity": "src.v3lib.Model" if strong else "rejected_not_v3lib.Model",
        "strong_decoder_loaded": strong,
        "metric_defined": bool(metric_defined),
        "valid_scientific_claim": valid,
        "effect_claimed": False,
        "real_meg_scientific_validation": False,
        "linear_probe_is_not_this_result": True,
        "old_fifty_class_probe_is_not_a_decode_gain": True,
    }
