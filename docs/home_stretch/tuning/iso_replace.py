"""ISO-REPLACE: in a raw (pre-SLP/Sinkhorn) Deep file, replace the isolated-word rows (source == word) by
w * B0 + (1 - w) * original (B0 = dedicated 1 s subj0 decoder, docs/home_stretch/tuning/b0_deep_raw.csv). Sentence rows untouched.
Then run hmm_prior.py --track deep --T 3 --kappa 20 on the output as usual. Usage: iso_replace.py raw.csv b0.csv out.csv w"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB
raw, b0, out, w = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
md = LibriBrainCompetitionHoldout(track="deep", data_path=os.path.dirname(HOLDOUT), download=False).metadata_frame()
A = pd.read_csv(raw); B = pd.read_csv(b0); assert (A.iloc[:, 0].values == B.iloc[:, 0].values).all() and list(A.columns[1:]) == VOCAB == list(B.columns[1:])
P = A.iloc[:, 1:].to_numpy(float); P /= P.sum(1, keepdims=True); Q = B.iloc[:, 1:].to_numpy(float); Q /= Q.sum(1, keepdims=True)
r = md[md.source == "word"]["index"].values; O = P.copy(); O[r] = w * Q[r] + (1 - w) * P[r]
write_submission(out, A.iloc[:, 0].values, O)
t = lambda X: np.bincount(X[r].argmax(1), minlength=50) / len(r) * 50
print(f"iso rows {len(r)}; top-1 max share x uniform: original {t(P).max():.1f}, B0 {t(Q).max():.1f}, out {t(O).max():.1f}; top-1 agreement orig vs B0 {np.mean(P[r].argmax(1) == Q[r].argmax(1)):.3f}")
