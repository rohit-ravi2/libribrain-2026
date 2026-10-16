"""Rerun of ideas 7/14 with thresholds matched to the actual post-SLP probability scale (0.3/0.4 never fire)."""
import numpy as np
from adjacency import *
D = load(); M = np.concatenate([m for b in D for m in b["slp"]])
print("row-max quantiles 50/90/99:", np.quantile(M.max(1), [.5, .9, .99]).round(3))
print("func max q90/q99:", np.quantile(M[:, FUNC].max(1), [.9, .99]).round(3), "conj max q90/q99:", np.quantile(M[:, CONJ].max(1), [.9, .99]).round(3))
for thr in (0.08, 0.12):
    for m in (0.85, 0.7): print("7 func-repeat | thr", thr, "x", m, "|", score(r7(thr, m), D), flush=True)
for thr in (0.08, 0.12):
    for f in (0.1, 0.3): print("14 coord copy | thr", thr, "frac", f, "|", score(r14(thr, f), D), flush=True)
