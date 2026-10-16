"""Which classes absorb the isolated-row mass (diagnostic only; nothing is mapped between subjects)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from pnpl.competition import LibriBrainCompetitionHoldout
from src.common import HOLDOUT, VOCAB
for track, f in (("broad", "fita2_broad_ens5.csv"), ("deep", "final7_deep_bn.csv"), ("broad", "fita2_broad_ens5_k20.csv")):
    md = LibriBrainCompetitionHoldout(track=track, data_path=os.path.dirname(HOLDOUT), download=False).metadata_frame()
    P = pd.read_csv("submissions/" + f).iloc[:, 1:].to_numpy(float); P /= P.sum(1, keepdims=True)
    for src in ("word", "sentence"):
        r = md[md.source == src]["index"].values; m = P[r].mean(0) * 50; o = np.argsort(-m)[:5]
        top1 = np.bincount(P[r].argmax(1), minlength=50) / len(r) * 50; t = np.argsort(-top1)[:3]
        print(f"{f:26s} {src:8s} mass x uniform: " + ", ".join(f"{VOCAB[i]} {m[i]:.1f}" for i in o) + " | top-1 share x uniform: " + ", ".join(f"{VOCAB[i]} {top1[i]:.1f}" for i in t))
