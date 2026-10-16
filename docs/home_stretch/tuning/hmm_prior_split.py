"""ISO-SPLIT copy of scripts/hmm_prior.py (home-stretch TUNING.md): adds --sk_split / --T_iso / --marg_sent; defaults reproduce hmm_prior.py.
SLP-1: self-learned within-subject sentence prior (approved Oct-13 plan, item b).

Per subject, its 156 holdout sentences are treated as an HMM over the 50 words: emissions = the model's row posteriors
divided by the subject's mean posterior (scaled likelihoods, so the class prior is not counted twice), transitions A
(50x50) and start distribution learned by EM from THAT subject's rows only, with Dirichlet smoothing toward uniform of
strength kappa per row. Output = forward-backward marginals for sentence rows (isolated-word rows unchanged), then the
usual per-subject Sinkhorn balance at temperature T. No cross-subject information of any kind.
Usage: hmm_prior.py --inp raw.csv --out out.csv --track broad --kappa 50 --beta 1.0 --iters 10 --T 2"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import HOLDOUT, VOCAB

ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--kappa", type=float, default=50.0); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--iters", type=int, default=10); ap.add_argument("--T", type=float, default=2.0)
ap.add_argument("--end", type=int, default=0, help="1 = also learn an end-of-sentence word distribution (smoothed like the start)")
ap.add_argument("--bal_em", type=int, default=0, help="1 = enforce class balance inside EM (damped per-class emission weights)")
ap.add_argument("--sk_split", type=int, default=0, help="1 = Sinkhorn the isolated-word rows (source==word) separately from the sentence rows, each to uniform over 50")
ap.add_argument("--T_iso", type=float, default=None, help="temperature for the isolated block when --sk_split 1 (default = --T)")
ap.add_argument("--marg_sent", type=int, default=0, help="1 = SLP emission marginal from sentence rows only (isolated rows are a collapsed OOD block)")
ap.add_argument("--base", default="uniform", help="uniform | vocab: Dirichlet base measure (vocab = bigram of results/vocab_sentences.txt)")
a = ap.parse_args()
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False); md = H.metadata_frame()
df = pd.read_csv(a.inp); assert list(df.columns[1:]) == VOCAB and len(df) == len(md)
P = df.iloc[:, 1:].to_numpy().astype(np.float64); P /= P.sum(1, keepdims=True); OUT = P.copy(); V = 50


def fb(E, A, pi, eta=None):
    """Scaled forward-backward for one sentence. E (n,V) emissions; eta optional end distribution; returns marginals (n,V), pairwise sum (V,V)."""
    n = len(E); al = np.zeros((n, V)); be = np.ones((n, V)); c = np.zeros(n)
    if eta is not None: be[n - 1] = eta * V
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


VI = {w: i for i, w in enumerate(VOCAB)}
if a.base == "vocab":
    Cb = np.full((V, V), 0.5); Cs = np.full(V, 0.5); Ce = np.full(V, 0.5)
    for line in open(os.path.expanduser("~/projects/libribrain-2026/results/vocab_sentences.txt")):
        ids = [VI[w] for w in line.split() if w in VI]
        if not ids: continue
        Cs[ids[0]] += 1; Ce[ids[-1]] += 1
        for x, y in zip(ids[:-1], ids[1:]): Cb[x, y] += 1
    BASE_A = Cb / Cb.sum(1, keepdims=True); BASE_PI = Cs / Cs.sum(); BASE_ETA = Ce / Ce.sum()
else:
    BASE_A = np.full((V, V), 1.0 / V); BASE_PI = np.full(V, 1.0 / V); BASE_ETA = np.full(V, 1.0 / V)

for s, g in md.groupby("subject"):
    sent = g[g.source == "sentence"].sort_values(["epoch", "word"])
    groups = [grp["index"].values for _, grp in sent.groupby("epoch")]
    rows = g["index"].values; marg = P[sent["index"].values].mean(0) if a.marg_sent else P[rows].mean(0)
    E_all = {tuple(r): (P[r] / marg[None]) ** a.beta for r in groups}
    A = BASE_A.copy(); pi = BASE_PI.copy(); eta = BASE_ETA.copy() if a.end else None; wcls = np.ones(V); nrows = sum(map(len, groups))
    for it in range(a.iters):
        XI = np.zeros((V, V)); G0 = np.zeros(V); GE = np.zeros(V); GT = np.zeros(V)
        for r in groups:
            gm, xi = fb(E_all[tuple(r)] * wcls[None], A, pi, eta); XI += xi; G0 += gm[0]; GE += gm[-1]; GT += gm.sum(0)
        A = (XI + a.kappa * BASE_A) / (XI + a.kappa * BASE_A).sum(1, keepdims=True); pi = (G0 + a.kappa * BASE_PI) / (G0 + a.kappa * BASE_PI).sum()
        if a.end: eta = (GE + a.kappa * BASE_ETA) / (GE + a.kappa * BASE_ETA).sum()
        if a.bal_em: wcls = wcls * np.clip(nrows / V / np.clip(GT, 1e-9, None), 0.2, 5.0) ** 0.5
    for r in groups:
        gm, _ = fb(E_all[tuple(r)] * wcls[None], A, pi, eta)
        OUT[r] = gm * marg[None] / (gm * marg[None]).sum(1, keepdims=True)   # back to posterior scale (re-attach the marginal)
    # Sinkhorn balance over all the subject's rows (as balance_submission.py)
    blocks = [(g[g.source == "sentence"]["index"].values, a.T), (g[g.source == "word"]["index"].values, a.T_iso or a.T)] if a.sk_split else [(rows, a.T)]
    for rb, Tb in blocks:
        Q = np.clip(OUT[rb], 1e-12, None) ** (1.0 / Tb); Q /= Q.sum(1, keepdims=True); n = len(rb)
        for _ in range(10):
            Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0); Q = Q / Q.sum(1, keepdims=True)
        OUT[rb] = Q
write_submission(a.out, df.iloc[:, 0].values, OUT)
t0, t1 = P.argmax(1), OUT.argmax(1)
print(f"{a.track} sk_split {a.sk_split} T_iso {a.T_iso} marg_sent {a.marg_sent} kappa {a.kappa} beta {a.beta} end {a.end} bal_em {a.bal_em} base {a.base}: top-1 changed on {np.mean(t0 != t1):.3f} of rows; max top-1 share {np.bincount(t1, minlength=50).max() / len(OUT):.3f}")
