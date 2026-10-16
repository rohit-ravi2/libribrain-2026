"""ISO-1: per-subject Sinkhorn on the isolated-word block only (the 92 `source == "word"` rows), applied on top of a finished
file (SLP + all-row Sinkhorn). Diagnosis: on isolated 1 s windows the model collapses onto a few sentence-initial classes
(so / and / good / but), so every isolated row of a subject gets nearly the same top-10 list and isolated rows score ~chance.
The block-level balance removes that condition-specific bias with 50 column scalings estimated from the subject's own
isolated rows, anchored to uniform (pattern 3). Sentence rows are left untouched.
Usage: iso_block.py --track broad --inp final.csv --out out.csv --T 1.0 [--alpha 1.0]
alpha < 1 blends the block correction in log space (0 = input unchanged)."""
import argparse, os, sys
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, pandas as pd
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB

ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--alpha", type=float, default=1.0)
a = ap.parse_args()
md = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False).metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P = np.clip(P, 1e-12, None); P /= P.sum(1, keepdims=True); OUT = P.copy()


def jacc(X):
    t = np.argsort(-X, 1)[:, :10]; c = np.bincount(t.ravel(), minlength=50) / len(X); return (c > 0).sum(), c.max()


stats = []
for s, g in md.groupby("subject"):
    iso = g[g.source == "word"]["index"].values
    Q = P[iso] ** (1.0 / a.T); Q /= Q.sum(1, keepdims=True); n = len(iso)
    for _ in range(50):
        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0); Q = Q / Q.sum(1, keepdims=True)
    L = (1 - a.alpha) * np.log(P[iso]) + a.alpha * np.log(Q); L = np.exp(L - L.max(1, keepdims=True)); OUT[iso] = L / L.sum(1, keepdims=True)
    stats.append(jacc(P[iso]) + jacc(OUT[iso]))
write_submission(a.out, df.iloc[:, 0].values, OUT)
st = np.array(stats, float)
print(f"{a.track} T {a.T} alpha {a.alpha}: isolated rows, words ever in a top-10 {st[:, 0].mean():.1f} -> {st[:, 2].mean():.1f}; "
      f"max top-10 inclusion rate {st[:, 1].mean():.2f} -> {st[:, 3].mean():.2f}")
