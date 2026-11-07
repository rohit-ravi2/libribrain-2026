"""Pre-registered run order and the next checkpointed stage. Scores are not an input."""
from __future__ import annotations

from .strong_noise import (
    EVAL_SUBJECTS,
    PARTIAL_STEPS_PER_LAUNCH,
    PARTIAL_TARGET_STEPS,
    TRAIN_FORWARD_RUNS,
    VAL_SUBJECTS,
    frozen_trunk_role,
)


def classify_base(base: str) -> dict:
    subject = int(base.split("_")[0].split("-")[1])
    shared = ("ses-11_task-Sherlock1" in base) or ("ses-12_task-Sherlock1" in base)
    role = frozen_trunk_role(subject, shared)
    role["base"] = base
    role["shared"] = shared
    return role


def plan_order(bases, dates=None) -> list[dict]:
    """Selection, then a frozen budget of subject-0 train runs, then exploratory eval.

    Dates break ties inside a phase. A missing date sorts last and does not drop the run.
    Subject-0 runs past the budget stay listed and are not forwarded by `next`.
    """
    dates = dates or {}
    rows = [classify_base(base) for base in bases]
    phase_order = {"selection": 0, "train": 1, "eval": 2, "train_shared": 3, "seen_or_late": 4, "unused": 5}
    rows.sort(key=lambda row: (phase_order[row["phase"]], dates.get(row["base"]) or "9999-99-99", row["base"]))
    train_seen = 0
    planned = []
    for row in rows:
        item = dict(row)
        if row["phase"] == "train":
            item["in_budget"] = train_seen < TRAIN_FORWARD_RUNS
            train_seen += 1
        else:
            item["in_budget"] = row["phase"] in ("selection", "eval")
        item["date"] = dates.get(row["base"])
        planned.append(item)
    return planned


def budgeted(order: list[dict], phase: str) -> list[dict]:
    return [row for row in order if row["in_budget"] and row["phase"] == phase]


def next_action(order: list[dict], *, done_bases, noise_fit: bool, partial_steps: int, score_frozen: bool, score_partial: bool) -> dict:
    """One chunk. Eval recordings are forwarded only after the noise fit is on disk."""
    done = set(done_bases)
    for phase in ("selection", "train"):
        for row in budgeted(order, phase):
            if row["base"] not in done:
                return {"stage": "forward", "base": row["base"], "phase": phase, "next_action": "rerun_this_command"}
    if not noise_fit:
        return {"stage": "noise-fit", "base": None, "phase": "selection", "next_action": "rerun_this_command"}
    for row in budgeted(order, "eval"):
        if row["base"] not in done:
            return {"stage": "forward", "base": row["base"], "phase": "eval", "next_action": "rerun_this_command"}
    if not score_frozen:
        return {"stage": "score-frozen", "base": None, "phase": "eval", "next_action": "rerun_this_command"}
    steps = int(partial_steps)
    if steps < PARTIAL_TARGET_STEPS:
        return {
            "stage": "partial-train",
            "base": None,
            "phase": "train",
            "steps": min(PARTIAL_STEPS_PER_LAUNCH, PARTIAL_TARGET_STEPS - steps),
            "next_action": "rerun_this_command",
        }
    if not score_partial:
        return {"stage": "score-partial", "base": None, "phase": "eval", "next_action": "rerun_this_command"}
    return {"stage": "stop", "base": None, "phase": None, "next_action": "stop"}


def assert_known_subjects():
    if EVAL_SUBJECTS != (3, 6, 9, 12) or VAL_SUBJECTS != (1, 5, 10):
        raise RuntimeError("split literals drifted from src/split2.py")
