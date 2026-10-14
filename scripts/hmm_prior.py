"""SLP-1: self-learned within-subject sentence prior (approved Oct-13 plan, item b).

Per subject, its 156 holdout sentences are treated as an HMM over the 50 words: emissions = the model's row posteriors
divided by the subject's mean posterior (scaled likelihoods, so the class prior is not counted twice), transitions A
(50x50) and start distribution learned by EM from THAT subject's rows only, with Dirichlet smoothing toward uniform of
strength kappa per row. Output = forward-backward marginals for sentence rows (isolated-word rows unchanged), then the
usual per-subject Sinkhorn balance at temperature T. No cross-subject information of any kind.
Usage: hmm_prior.py --inp raw.csv --out out.csv --track broad --kappa 50 --beta 1.0 --iters 10 --T 2"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB

ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--kappa", type=float, default=50.0); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--iters", type=int, default=10); ap.add_argument("--T", type=float, default=2.0); a = ap.parse_args()
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False); md = H.metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P /= P.sum(1, keepdims=True); OUT = P.copy(); V = 50


def fb(E, A, pi):
    """Scaled forward-backward for one sentence. E (n,V) emissions; returns marginals (n,V), pairwise sum (V,V)."""
    n = len(E); al = np.zeros((n, V)); be = np.ones((n, V)); c = np.zeros(n)
    al[0] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n):
        al[t] = (al[t - 1] @ A) * E[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1):
        be[t] = (A @ (E[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True)
    xi = np.zeros((V, V))
    for t in range(n - 1):
        x = al[t][:, None] * A * (E[t + 1] * be[t + 1])[None]; xi += x / x.sum()
    return g, xi


for s, g in md.groupby("subject"):
    sent = g[g.source == "sentence"].sort_values(["epoch", "word"])
    groups = [grp["index"].values for _, grp in sent.groupby("epoch")]
    rows = g["index"].values; marg = P[rows].mean(0)
    E_all = {tuple(r): (P[r] / marg[None]) ** a.beta for r in groups}
    A = np.full((V, V), 1.0 / V); pi = np.full(V, 1.0 / V)
    for it in range(a.iters):
        XI = np.zeros((V, V)); G0 = np.zeros(V)
        for r in groups:
            gm, xi = fb(E_all[tuple(r)], A, pi); XI += xi; G0 += gm[0]
        A = (XI + a.kappa / V) / (XI + a.kappa / V).sum(1, keepdims=True); pi = (G0 + a.kappa / V) / (G0 + a.kappa / V).sum()
    for r in groups:
        gm, _ = fb(E_all[tuple(r)], A, pi)
        OUT[r] = gm * marg[None] / (gm * marg[None]).sum(1, keepdims=True)   # back to posterior scale (re-attach the marginal)
    # Sinkhorn balance over all the subject's rows (as balance_submission.py)
    Q = np.clip(OUT[rows], 1e-12, None) ** (1.0 / a.T); Q /= Q.sum(1, keepdims=True); n = len(rows)
    for _ in range(10):
        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0); Q = Q / Q.sum(1, keepdims=True)
    OUT[rows] = Q
write_submission(a.out, df.iloc[:, 0].values, OUT)
t0, t1 = P.argmax(1), OUT.argmax(1)
print(f"{a.track} kappa {a.kappa} beta {a.beta}: top-1 changed on {np.mean(t0 != t1):.3f} of rows; max top-1 share {np.bincount(t1, minlength=50).max() / len(OUT):.3f}")
