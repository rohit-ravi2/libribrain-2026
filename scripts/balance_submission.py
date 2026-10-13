"""BAL-2: per-subject class-balance re-ranking of an existing submission (within-subject only).
For each subject, Sinkhorn-normalise its rows' probability matrix so each of the 50 words receives ~equal total mass
(uniform target), then renormalise rows. Monotone within a row is NOT preserved across columns, so rankings change."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--iters", type=int, default=10); a = ap.parse_args()
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False); md = H.metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P /= P.sum(1, keepdims=True); out = P.copy()
for s, g in md.groupby("subject"):
    rows = g["index"].values; Q = P[rows].copy(); n = len(rows)
    for _ in range(a.iters):
        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0)
        Q = Q / Q.sum(1, keepdims=True)
    out[rows] = Q
write_submission(a.out, df.iloc[:, 0].values, out)
t0, t1 = P.argmax(1), out.argmax(1)
print(f"{a.track}: top-1 changed on {np.mean(t0 != t1):.3f} of rows; max top-1 share {np.bincount(t0, minlength=50).max() / len(P):.3f} -> {np.bincount(t1, minlength=50).max() / len(out):.3f}")
