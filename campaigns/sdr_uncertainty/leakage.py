"""Leakage checks. A violation stops the fit; it is not recorded as a model result."""
from __future__ import annotations

from .protocol import ALLOWED_STATE_NAMES, FEATURE_SET, NEVER_STORE_FIELDS


class LeakageError(RuntimeError):
    pass


def assert_batch_allowed(batch: dict) -> None:
    if batch.get("feature_set") != FEATURE_SET:
        raise LeakageError(
            f"feature set {batch.get('feature_set')!r} is not the frozen set {FEATURE_SET}"
        )
    names = tuple(batch.get("state_names") or ())
    if not names:
        raise LeakageError("state_names missing")
    for name in names:
        if name not in ALLOWED_STATE_NAMES:
            raise LeakageError(f"state feature {name!r} is outside the frozen conditioning set")
    for key in NEVER_STORE_FIELDS + ("timemeg", "segment", "word", "phoneme"):
        if key in batch:
            raise LeakageError(f"forbidden field {key} is stored on the batch")
    y = batch["y"]
    state = batch["state"]
    if y.shape[0] != state.shape[0]:
        raise LeakageError("y and state row counts differ")
    if state.shape[1] != len(names):
        raise LeakageError("state width does not match state_names")


def assert_channels_disjoint(target, state) -> None:
    overlap = set(target) & set(state)
    if overlap:
        raise LeakageError(f"target and state channels overlap: {sorted(overlap)}")
    if len(set(target)) != len(tuple(target)) or len(set(state)) != len(tuple(state)):
        raise LeakageError("duplicate channel index")
    for index in list(target) + list(state):
        if not isinstance(index, int) or index < 0 or index >= 306:
            raise LeakageError(f"channel index out of range: {index}")


def null_adaptive_search(reason: str, candidates) -> dict:
    """Mining features invalidates the whole search. Scores are omitted on purpose."""
    return {
        "claim_status": "null_adaptive_search",
        "real_meg_scientific_validation": False,
        "evidence_class": "null_control",
        "metrics_omitted": True,
        "reason": reason,
        "candidates": list(candidates) if candidates is not None else None,
        "neural_inference_allowed": False,
        "decoder_intervention_executed": False,
    }
