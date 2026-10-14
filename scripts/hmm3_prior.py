"""SLP-T: trigram-backoff version of the self-learned within-subject sentence prior (scripts/hmm_prior.py).

Hidden state at position t = (previous word u, current word v); u = START (index 50) at t = 0. Transition
(u, v) -> (v, w) with P3(w | u, v) = (C3[u,v,w] + k3 * P2(w | v)) / (C3[u,v,:].sum() + k3), where the bigram
P2(w | v) = (C2[v,w] + k2 / V) / (C2[v,:].sum() + k2) and the start pi(v) = (C1[v] + k2 / V) / (C1.sum() + k2).
All counts are EM expectations from THIS subject's own sentence rows. Emissions = posteriors / subject mean posterior.
k3 -> inf reduces exactly to the bigram SLP. Output: marginals for sentence rows (isolated rows unchanged) -> Sinkhorn T.
Usage: hmm3_prior.py --inp raw.csv --out out.csv --track broad --k2 20 --k3 20 --iters 10 --T 2"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB

ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--k2", type=float, default=20.0); ap.add_argument("--k3", type=float, default=20.0)
ap.add_argument("--iters", type=int, default=10); ap.add_argument("--T", type=float, default=2.0); a = ap.parse_args()
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False); md = H.metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P /= P.sum(1, keepdims=True); OUT = P.copy(); V = 50; S = V + 1   # S: prev index incl. START


def fb3(E, P3, pi):
    """Pair-state forward-backward. E (n,V). P3 (S,V,V). Returns word marginals (n,V), trigram counts (S,V,V), start marginal (V,)."""
    n = len(E); al = np.zeros((n, S, V)); be = np.ones((n, S, V)); c = np.zeros(n)
    al[0, V] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n):
        nxt = np.einsum("uv,uvw->vw", al[t - 1], P3) * E[t][None]          # (V prev, V cur)
        al[t, :V] = nxt; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1):
        be[t] = np.einsum("uvw,w,vw->uv", P3, E[t + 1], be[t + 1, :V]) / c[t + 1]
    gam = al * be; gam /= gam.sum((1, 2), keepdims=True)
    xi = np.zeros((S, V, V))
    for t in range(n - 1):
        x = al[t][:, :, None] * P3 * (E[t + 1][None, :] * be[t + 1, :V])[None]; xi += x / x.sum()
    return gam.sum(1), xi, gam[0, V]


for s, g in md.groupby("subject"):
    sent = g[g.source == "sentence"].sort_values(["epoch", "word"])
    groups = [grp["index"].values for _, grp in sent.groupby("epoch")]
    rows = g["index"].values; marg = P[rows].mean(0)
    E_all = {tuple(r): P[r] / marg[None] for r in groups}
    P3 = np.full((S, V, V), 1.0 / V); pi = np.full(V, 1.0 / V)
    for it in range(a.iters):
        C3 = np.zeros((S, V, V)); C1 = np.zeros(V)
        for r in groups:
            _, xi, g0 = fb3(E_all[tuple(r)], P3, pi); C3 += xi; C1 += g0
        C2 = C3.sum(0); P2 = (C2 + a.k2 / V) / (C2.sum(1, keepdims=True) + a.k2)
        P3 = (C3 + a.k3 * P2[None]) / (C3.sum(2, keepdims=True) + a.k3)
        pi = (C1 + a.k2 / V) / (C1.sum() + a.k2)
    for r in groups:
        gm, _, _ = fb3(E_all[tuple(r)], P3, pi)
        OUT[r] = gm * marg[None] / (gm * marg[None]).sum(1, keepdims=True)
    Q = np.clip(OUT[rows], 1e-12, None) ** (1.0 / a.T); Q /= Q.sum(1, keepdims=True); n = len(rows)
    for _ in range(10):
        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0); Q = Q / Q.sum(1, keepdims=True)
    OUT[rows] = Q
write_submission(a.out, df.iloc[:, 0].values, OUT)
t0, t1 = P.argmax(1), OUT.argmax(1)
print(f"{a.track} k2 {a.k2} k3 {a.k3}: top-1 changed on {np.mean(t0 != t1):.3f} of rows; max top-1 share {np.bincount(t1, minlength=50).max() / len(OUT):.3f}")
