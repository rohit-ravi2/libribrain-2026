"""Pre-registered run order and the next checkpointed stage. Scores are not an input."""
from __future__ import annotations

from .strong_noise import (
    EVAL_SUBJECTS,
    SUBSTANTIVE_SCHEDULE,
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
    """Noise-pool forwards, then selection, then exploratory eval, then a head-finetune budget.

    Subject-0 shared Sherlock runs are the noise-equation pool (Q3 words only, applied later).
    They are trunk-unseen quarters, not Broad eval and not Broad validation.
    Subject-0 non-shared runs are the head-finetune cache, capped, and are not the noise fit.
    Dates break ties inside a phase. A missing date sorts last and does not drop the run.
    """
    dates = dates or {}
    rows = []
    for base in bases:
        row = classify_base(base)
        if row["phase"] == "train_shared" and row["subject"] == 0:
            row["phase"] = "noise_pool"
        rows.append(row)
    phase_order = {
        "noise_pool": 0, "selection": 1, "eval": 2, "train": 3,
        "train_shared": 4, "seen_or_late": 5, "unused": 6,
    }
    rows.sort(key=lambda row: (phase_order[row["phase"]], dates.get(row["base"]) or "9999-99-99", row["base"]))
    train_seen = 0
    planned = []
    for row in rows:
        item = dict(row)
        if row["phase"] == "train":
            item["in_budget"] = train_seen < TRAIN_FORWARD_RUNS
            train_seen += 1
        else:
            item["in_budget"] = row["phase"] in ("noise_pool", "selection", "eval")
        item["date"] = dates.get(row["base"])
        planned.append(item)
    return planned


def budgeted(order: list[dict], phase: str) -> list[dict]:
    return [row for row in order if row["in_budget"] and row["phase"] == phase]


def next_action(order: list[dict], *, done_bases, noise_fit: bool, partial_steps: int, score_frozen: bool, score_partial: bool, partial_stopped: bool = False, schedule: dict | None = None) -> dict:
    """One chunk. Eval recordings are forwarded only after the noise fit is on disk.

    The head finetune starts only after the frozen exploratory score is written, so
    its validation stop cannot be influenced by rewriting that score. Smoke is not
    a stage of this gate.
    """
    schedule = dict(SUBSTANTIVE_SCHEDULE if schedule is None else schedule)
    done = set(done_bases)
    for phase in ("noise_pool", "selection"):
        for row in budgeted(order, phase):
            if row["base"] not in done:
                return {"stage": "forward", "base": row["base"], "phase": phase, "next_action": "rerun_this_command", "schedule": schedule["name"]}
    if not noise_fit:
        return {"stage": "noise-fit", "base": None, "phase": "noise_pool", "next_action": "rerun_this_command", "schedule": schedule["name"]}
    for row in budgeted(order, "eval"):
        if row["base"] not in done:
            return {"stage": "forward", "base": row["base"], "phase": "eval", "next_action": "rerun_this_command", "schedule": schedule["name"]}
    if not score_frozen:
        return {"stage": "score-frozen", "base": None, "phase": "eval", "next_action": "rerun_this_command", "schedule": schedule["name"]}
    for row in budgeted(order, "train"):
        if row["base"] not in done:
            return {"stage": "forward", "base": row["base"], "phase": "train", "next_action": "rerun_this_command", "schedule": schedule["name"]}
    steps = int(partial_steps)
    if not partial_stopped and steps < int(schedule["max_steps"]):
        return {
            "stage": "partial-train",
            "base": None,
            "phase": "train",
            "steps": min(int(schedule["steps_per_launch"]), int(schedule["max_steps"]) - steps),
            "next_action": "rerun_this_command",
            "schedule": schedule["name"],
        }
    if not score_partial:
        return {"stage": "score-partial", "base": None, "phase": "eval", "next_action": "rerun_this_command", "schedule": schedule["name"]}
    return {"stage": "stop", "base": None, "phase": None, "next_action": "stop", "schedule": schedule["name"]}


def assert_known_subjects():
    if EVAL_SUBJECTS != (3, 6, 9, 12) or VAL_SUBJECTS != (1, 5, 10):
        raise RuntimeError("split literals drifted from src/split2.py")
