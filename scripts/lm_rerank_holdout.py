"""LM-rerank of an existing Broad submission: within each holdout SENTENCE (one subject's own rows, in word order),
replace per-row probabilities by exact forward-backward marginals under the external trigram LM over the 50 words
(src/lm50.py), emission p^beta. Isolated-word rows are untouched. Within-subject only; no cross-subject correspondence."""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB
from src.lm50 import marginals, uni
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--delta", type=float, default=0.0); a = ap.parse_args()
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False); md = H.metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P = P / P.sum(1, keepdims=True); out = P.copy()
sent = md[md.source == "sentence"]; n = 0
for (s, e), g in sent.groupby(["subject", "epoch"]):
    rows = g.sort_values("word")["index"].values
    if len(rows) < 2: continue
    ell = np.log(np.clip(P[rows], 1e-9, 1)); m = marginals(ell, a.beta) - a.delta * np.log(uni)[None]
    q = np.exp(m - m.max(1, keepdims=True)); out[rows] = q / q.sum(1, keepdims=True); n += 1
write_submission(a.out, df.iloc[:, 0].values, out)
top0, top1 = P.argmax(1), out.argmax(1)
print(f"reranked {n} sentences; top-1 changed on {np.mean(top0 != top1):.3f} of rows; max top-1 share {np.bincount(top1, minlength=50).max() / len(out):.3f}")
