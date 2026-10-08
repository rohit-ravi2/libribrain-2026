"""Split v2 (2026-10-08, after the execution audit): stimulus-rotated, selection and test on disjoint stimulus.

Shared stimulus = Sherlock1 ses-11/12, divided into quarters Q1..Q4 of the stimulus clock (timechapter / T_MAX).
- Training uses only Q1-Q2 of the shared stimulus (plus all other subj0 sessions) for every subject, subj0 included.
  Windows within GUARD of the Q2/Q3 or Q3/Q4 boundary, or whose 1 s span crosses one, are dropped for everyone.
- Fold A: select on Q3, test on Q4.  Fold B: select on Q4, test on Q3.  Same training set for both folds.
- Subjects: EVAL (never trained; zero-shot/few-shot test), VAL (never trained; zero-shot selection),
  SEEN (100%-group subjects trained on Q1-Q2; scored on Q3/Q4 with their own jointly trained input layer =
  the deployment case for subjects 1-32). Subjects 13-32 train on whatever of their data lies in Q1-Q2.
- Deployment mixture: (32 * seen + 7 * zero_shot) / 39.
The shared stimulus contains 48/50 words ("really", "new" never occur); Q3 covers 47, Q4 38.
"""
import numpy as np
from .split import parse, T_MAX, SHARED, CACHE

EVAL = (3, 6, 9, 12); VAL = (1, 5, 10); SEEN = (2, 4, 7, 8, 11)
GUARD = 1.0
FOLDS = {"A": ("Q3", "Q4"), "B": ("Q4", "Q3")}   # fold: (select block, test block)


def quarters(base, timechapter):
    """Per-window quarter label 'Q1'..'Q4' for shared sessions (guarded), '' if dropped, 'other' for non-shared."""
    sub, ses = parse(base)
    t = np.asarray(timechapter, float)
    if ses not in SHARED:
        return np.array(["other"] * len(t), dtype=object)
    T = T_MAX[ses]; q = np.clip((t / T * 4).astype(int), 0, 3)
    out = np.array([f"Q{k + 1}" for k in q], dtype=object)
    for k in (2, 3):                                  # boundaries Q2|Q3 and Q3|Q4
        bnd = k * T / 4
        bad = (t + 1.0 > bnd - GUARD) & (t < bnd + GUARD)
        out[bad] = ""
    return out


def train_mask(base, timechapter):
    sub, _ = parse(base)
    if sub in EVAL or sub in VAL:
        return np.zeros(len(timechapter), bool)
    q = quarters(base, timechapter)
    return np.isin(q, ["other", "Q1", "Q2"])


def block_mask(base, timechapter, block):
    return quarters(base, timechapter) == block


def fewshot_mask(base, timechapter, g):
    """Eval-subject adaptation data: s25 = Q1, s50 = Q1-Q2 (the 100% proxy is unavailable under this split)."""
    q = quarters(base, timechapter)
    return q == "Q1" if g == "s25" else np.isin(q, ["Q1", "Q2"])


def deploy(seen, zs):
    return (32 * seen + 7 * zs) / 39
