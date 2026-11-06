"""Out-of-day residual comparison. Synthetic calls stay non-scientific."""
from __future__ import annotations

import time

from .crossfit import run_crossfit
from .metrics import day_group_uncertainty
from .protocol import (
    EXCLUDED_SMOKE_DATE,
    LADDER,
    OOD_MAX_READ_BYTES,
    OOD_MAX_WINDOWS_PER_DAY,
    OOD_MIN_DAYS,
    OOD_MIN_WINDOWS_PER_DAY,
    OOD_SELECTION_RULE,
    PRIMARY_CONTRAST,
    PRIMARY_ENDPOINT,
    next_rung,
)
from .stream import assemble_days, stream_windows


def _power(windows_per_day: list[int]) -> bool:
    return (
        len(windows_per_day) >= OOD_MIN_DAYS
        and all(count >= OOD_MIN_WINDOWS_PER_DAY for count in windows_per_day)
    )


def classify_out_of_day(*, regime: str, n_days: int, windows_per_day: list[int], timed_out: bool,
                        cgroup_live: bool, dates: list[str], q3q4_used: bool,
                        selection_rule: str) -> dict:
    """Science requires a finished live stream. Fixtures, smoke, and synthetic runs never qualify."""
    base = {
        "real_meg_scientific_validation": False,
        "neural_inference_allowed": False,
        "rung": LADDER[1],
        "rung_completed_as_science": False,
    }
    if regime in ("local_smoke_one_training_day_group", "plumbing"):
        base["claim_status"] = "plumbing"
        base["reason"] = "A smoke or a cgroup check is not the out-of-day comparison."
        return base
    if regime != "local_out_of_day_stream":
        base["claim_status"] = "synthetic_evaluation_only" if regime == "synthetic_dgp" else "not_a_scientific_result"
        base["reason"] = "This regime is not a live multi-day MEG stream."
        return base
    if q3q4_used or EXCLUDED_SMOKE_DATE in dates or selection_rule != OOD_SELECTION_RULE:
        base["claim_status"] = "protocol_violation"
        base["reason"] = "Excluded day, quarter scores, or a non-frozen day rule entered the run."
        return base
    if not cgroup_live:
        base["claim_status"] = "cgroup_not_live"
        base["reason"] = "The live cgroup was not verified. A fixture cap is not this rung."
        return base
    if timed_out:
        base["claim_status"] = "timeout"
        base["reason"] = "The soft deadline inside the 600s cap fired before the frozen day list finished."
        return base
    if not _power(windows_per_day) or n_days < OOD_MIN_DAYS:
        base["claim_status"] = "underpowered"
        base["reason"] = (
            f"Need at least {OOD_MIN_DAYS} days with {OOD_MIN_WINDOWS_PER_DAY} windows each. "
            "The caps were not raised."
        )
        return base
    base["real_meg_scientific_validation"] = True
    base["rung_completed_as_science"] = True
    base["claim_status"] = "out_of_day_replication"
    base["reason"] = "Pre-registered multi-day cross-fit finished inside the original caps."
    return base


def uncertainty_report(fitted: dict) -> dict:
    """Every held-out day stays in the summary. No day and no model is picked after seeing scores."""
    per_model = {}
    for name, summary in fitted["models"].items():
        per_model[name] = {
            "mean": summary["mean"],
            "std": summary["std"],
            "se": summary["se"],
            "n_days": summary["n_days"],
            "per_day": list(summary["per_day"]),
        }
    deltas = []
    for fold in fitted["folds"]:
        deltas.append(
            fold["models"]["M2_state_variance"]["logscore"] - fold["models"]["M0_fixed_gaussian"]["logscore"]
        )
    return {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_contrast": PRIMARY_CONTRAST,
        "primary_contrast_by_day": day_group_uncertainty(deltas),
        "models": per_model,
        "best_day_index": None,
        "selected_model": None,
        "day_selected_by_score": False,
        "model_selected_by_score": False,
        "confirmation_quarters_used": False,
    }


def _empty_record(status: dict, *, dates, windows, timed_out: bool, bytes_read: int) -> dict:
    return {
        **status,
        "cross_day": False,
        "fit_completed": False,
        "dates": list(dates),
        "windows_per_day": list(windows),
        "n_days": len(windows),
        "timed_out": timed_out,
        "bytes_read": bytes_read,
        "models": None,
        "uncertainty": None,
        "next_rung": next_rung(LADDER[1]),
        "decoder_intervention_executed": False,
    }


def evaluate_out_of_day(parts: list[dict], *, regime: str, timed_out: bool, cgroup_live: bool,
                        selection_rule: str, q3q4_used: bool = False, fit=None) -> dict:
    dates = [part["date"] for part in parts]
    windows = [int(part["n_windows"]) for part in parts]
    bytes_read = int(sum(int(part.get("bytes_read", 0)) for part in parts))
    status = classify_out_of_day(
        regime=regime,
        n_days=len(parts),
        windows_per_day=windows,
        timed_out=timed_out,
        cgroup_live=cgroup_live,
        dates=dates,
        q3q4_used=q3q4_used,
        selection_rule=selection_rule,
    )
    if timed_out or len(parts) < 2 or status["claim_status"] == "protocol_violation":
        return _empty_record(status, dates=dates, windows=windows, timed_out=timed_out, bytes_read=bytes_read)
    batch = assemble_days(parts)
    if regime == "synthetic_dgp":
        batch["evidence_class"] = "synthetic_dgp"
    elif regime != "local_out_of_day_stream":
        batch["evidence_class"] = regime
    fitter = fit if fit is not None else run_crossfit
    fitted = fitter(batch)
    # run_crossfit always clears the science flag. Restore only what classify allows.
    fitted.update(status)
    fitted["cross_day"] = True
    fitted["fit_completed"] = True
    fitted["dates"] = dates
    fitted["windows_per_day"] = windows
    fitted["n_days"] = len(parts)
    fitted["timed_out"] = False
    fitted["bytes_read"] = bytes_read
    fitted["uncertainty"] = uncertainty_report(fitted)
    fitted["next_rung"] = next_rung(LADDER[1])
    fitted["evidence_class"] = batch["evidence_class"]
    return fitted


def stream_out_of_day(specs: list[dict], vocab: tuple[str, ...], *, deadline: float,
                      opener, max_bytes: int = OOD_MAX_READ_BYTES,
                      max_windows: int = OOD_MAX_WINDOWS_PER_DAY) -> dict:
    """Stream each discovered day until the frozen byte cap or the soft deadline."""
    parts = []
    bytes_used = 0
    timed_out = False
    read_budget_hit = False
    refused_fields = []
    for spec in specs:
        if time.monotonic() > deadline:
            timed_out = True
            break
        remaining = max_bytes - bytes_used
        if remaining <= 0:
            read_budget_hit = True
            break
        if spec["date"] == EXCLUDED_SMOKE_DATE:
            raise RuntimeError(f"discovery returned excluded date {EXCLUDED_SMOKE_DATE}")
        text, rows, refused, dataset, closer = opener(spec)
        refused_fields.extend(refused)
        try:
            streamed = stream_windows(dataset, rows, vocab, max_windows=max_windows, max_bytes=remaining)
        finally:
            closer()
        streamed["date"] = spec["date"]
        streamed["n_windows"] = int(streamed["y"].shape[0])
        streamed["exposure_files"] = [
            {"path": spec["h5"], "role": "meg_h5_hyperslab", "bytes_read": streamed["bytes_read"]},
            {"path": spec["events"], "role": "events_tsv_whitelist", "bytes_read": len(text.encode())},
        ]
        if streamed["stop_reason"] == "max_bytes" and streamed["n_windows"] < OOD_MIN_WINDOWS_PER_DAY:
            read_budget_hit = True
            break
        parts.append(streamed)
        bytes_used += int(streamed["bytes_read"])
        if streamed["stop_reason"] == "max_bytes":
            read_budget_hit = True
            break
    return {
        "parts": parts,
        "timed_out": timed_out,
        "read_budget_hit": read_budget_hit,
        "bytes_read": bytes_used,
        "fields_seen_not_used": sorted(set(refused_fields)),
        "selection_rule": OOD_SELECTION_RULE,
    }
