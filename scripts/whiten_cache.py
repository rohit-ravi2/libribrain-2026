"""Write a spatially whitened copy of the 50 Hz cache (spatial noise equation eps ~ N(0, Sigma), Sigma^-1/2 applied).
Sigma is estimated once from subject-0 TRAINING runs only (the clean screen's held-out day ses-11/12 Sherlock1 is
excluded), on run-z-scored, clipped data, with shrinkage 0.1 toward its mean diagonal; the same matrix is applied to
every run. Event files are symlinked. Usage: whiten_cache.py DST  (e.g. cont_wh)"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
PER_RUN = "--per_run" in sys.argv; SRC = f"{CACHE}/cont"; dst = f"{CACHE}/{sys.argv[1]}"; os.makedirs(dst, exist_ok=True)
clean = lambda Y: np.nan_to_num(np.asarray(Y, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
mp = f"{dst}/_whitener.npz"
if not os.path.exists(mp):
    runs = sorted(f for f in glob.glob(f"{SRC}/sub-0_*.npy") if "ses-11_task-Sherlock1" not in f and "ses-12_task-Sherlock1" not in f)
    C = np.zeros((306, 306)); n = 0
    for f in np.random.default_rng(0).choice(runs, 20, replace=False):
        Y = clean(np.load(f, mmap_mode="r")); Y = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6); Y = np.clip(Y, -10, 10)
        C += Y @ Y.T; n += Y.shape[1]
    C /= n; C = 0.9 * C + 0.1 * np.trace(C) / 306 * np.eye(306); e, U = np.linalg.eigh(C)
    np.savez(mp, W=(U * e ** -0.5) @ U.T, eig=e); print("eig range", e.min(), e.max(), flush=True)
W = np.load(mp)["W"].astype(np.float32)
for ev in sorted(glob.glob(f"{SRC}/*_ev.npz")):
    b = os.path.basename(ev)[:-7]; out = f"{dst}/{b}.npy"
    if not os.path.exists(f"{dst}/{b}_ev.npz"): os.symlink(ev, f"{dst}/{b}_ev.npz")
    if os.path.exists(out): continue
    Y = clean(np.load(f"{SRC}/{b}.npy", mmap_mode="r")); sd = Y.std(1, keepdims=True) + 1e-6; mu = Y.mean(1, keepdims=True)
    Yz = np.clip((Y - mu) / sd, -10, 10)
    if PER_RUN:                                        # day-level noise equation: this run's own covariance (label-free)
        C = Yz @ Yz.T / Yz.shape[1]; C = 0.9 * C + 0.1 * np.trace(C) / 306 * np.eye(306); e, U = np.linalg.eigh(C); Wr = ((U * e ** -0.5) @ U.T).astype(np.float32)
    Z = (Wr if PER_RUN else W) @ Yz                    # same per-run z-scoring as the fit, then Sigma^-1/2
    np.save(out + ".tmp.npy", Z.astype(np.float16)); os.replace(out + ".tmp.npy", out); print(b, flush=True)
