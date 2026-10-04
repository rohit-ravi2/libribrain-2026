"""Local validation split, disjoint in BOTH subject and stimulus (RULES.md §6).

Stimulus axis: Sherlock1 ses-11/12 is the only stimulus shared by subj0 and subj1-32. Inside each of those two
sessions, block positions are defined on the shared stimulus clock `timechapter`, with T_q = q * max(timechapter).
  - Block B (local test stimulus) = words with timechapter >= T75 + GUARD. Nobody trains on it (subj0 included).
  - Words in [T75 - GUARD, T75 + GUARD) are dropped for everyone (1 s windows must not overlap B's MEG span).
Subject axis: EVAL_SUBJECTS (from the 100% group, the only one that has data in block B) are never in training.
  - zero-shot proxy : score on block B, no data from that subject used at all.
  - few-shot proxies: subject's own words with timechapter < T_q - GUARD for q in .25/.50/.75
                      (.25 = 25% group ~10 min, .50 = 50% group ~20 min, .75 approximates the 100% group, ~30 of 40 min).
Composite = holdout-row-weighted mean: (12*s100 + 10*s50 + 10*s25 + 7*s0) / 39.
Fixed 2026-10-04 before any model run.
"""
import glob, os
import numpy as np
from .common import CACHE

EVAL_SUBJECTS = (3, 6, 9, 12)
GUARD = 1.0
SHARED = ("ses-11_task-Sherlock1", "ses-12_task-Sherlock1")
T_MAX = {}  # filled from subj0's full ses-11/12 events at import time
GROUP_W = {"s100": 12, "s50": 10, "s25": 10, "s0": 7}
FEWSHOT_Q = {"s25": 0.25, "s50": 0.50, "s100": 0.75}


def _tmax():
    import pandas as pd
    for ses in SHARED:
        ev = glob.glob(f"/mnt/ssd4tb/data/LibriBrain100/Sherlock1/derivatives/events/sub-0_{ses}_run-*_events.tsv")[0]
        T_MAX[ses] = float(pd.read_csv(ev, sep="\t").timechapter.max())


_tmax()


def parse(base):
    sub = int(base.split("_")[0].split("-")[1])
    ses = "_".join(base.split("_")[1:3])
    return sub, ses


def block_masks(base, timechapter):
    """Return dict of boolean masks over a run's windows: 'B', 'pre75', and few-shot 'fs_q' masks."""
    sub, ses = parse(base)
    n = len(timechapter)
    if ses not in SHARED:
        return {"B": np.zeros(n, bool), "pre75": np.ones(n, bool)}
    t = np.asarray(timechapter, float)
    T = T_MAX[ses]
    m = {"B": t >= 0.75 * T + GUARD, "pre75": t < 0.75 * T - GUARD}
    for g, q in FEWSHOT_Q.items():
        m["fs_" + g] = t < q * T - GUARD
    return m


def train_mask(base, timechapter):
    """Windows any model may train on (excludes eval subjects entirely and block B + guard for everyone)."""
    sub, _ = parse(base)
    if sub in EVAL_SUBJECTS:
        return np.zeros(len(timechapter), bool)
    return block_masks(base, timechapter)["pre75"]


def composite(scores):
    return sum(GROUP_W[g] * scores[g] for g in GROUP_W) / sum(GROUP_W.values())
