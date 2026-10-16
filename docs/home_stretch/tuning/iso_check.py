"""Isolated-word rows vs sentence rows in existing raw/final CSVs: sharpness and per-source class-mass skew (no labels)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from pnpl.competition import LibriBrainCompetitionHoldout
from src.common import HOLDOUT
for track, f in (("broad", "fita2_broad_ens5.csv"), ("broad", "fita2_broad_ens5_k20.csv"), ("deep", "final7_deep_bn.csv"), ("deep", "final7_deep_bn_k20.csv")):
    md = LibriBrainCompetitionHoldout(track=track, data_path=os.path.dirname(HOLDOUT), download=False).metadata_frame()
    P = pd.read_csv("submissions/" + f).iloc[:, 1:].to_numpy(float); P /= P.sum(1, keepdims=True)
    for src in ("sentence", "word"):
        r = md[md.source == src]["index"].values; sk = []
        for s, g in md[md.source == src].groupby("subject"):
            cs = P[g["index"].values].mean(0) * 50; sk.append(cs.max())
        print(f"{f:28s} {src:8s} mean maxp {P[r].max(1).mean():.3f} entropy {(-(P[r]*np.log(np.clip(P[r],1e-12,None))).sum(1)).mean():.3f} | max class mass x uniform (median over subjects) {np.median(sk):.2f}")
