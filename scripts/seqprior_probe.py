"""MOON-2: sentence-prior moonshots vs the learned bigram (SLP), on runs of consecutive vocab words inside local prose
sentences (split-v2 Q3/Q4; Broad 12 subj identity A + subj0), frozen deep_c ep6. All within one subject-block.
 (2) CA prior: words -> 6-bit codes (rank of subject mean posterior); elementary CA rule r maps code -> next code;
     transition = smoothed deterministic map; rule chosen per subject on Q3 by forward likelihood, applied to Q4.
 (10) rule-array / discrete dynamical generator = class HMM with K hidden states (K 4, 8) emitting words, EM on posteriors.
 (8) multiway branching: beam of K partial sentences (K 5, 20) scored by emissions x learned bigram, marginals from beam.
Metric: top-10 bacc + Sinkhorn (uniform over 50) over all vocab rows (rows outside runs unchanged); relative to SLP."""
import sys, os, json, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p
M = Model("runs/deep_c_ep6.pt"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev); u = torch.full((50,), 1 / 50); V = 50
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
W = Words(runs=runs, load_meg=True); q = W.quarters()
def runs_of(idx):
    rp = {int(w): r for r, w in enumerate(idx)}; G = []
    for s_ in np.unique(W.sent[idx]):
        ws = np.flatnonzero(W.sent == s_); ws = ws[np.argsort(W.onset[ws])]; cur = []
        for w in ws:
            if int(w) in rp: cur.append(rp[int(w)])
            else:
                if len(cur) >= 2: G.append(np.array(cur))
                cur = []
        if len(cur) >= 2: G.append(np.array(cur))
    return G
def fb(E, A, pi):
    n = len(E); al = np.zeros((n, A.shape[0])); be = np.ones_like(al); c = np.zeros(n)
    al[0] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n): al[t] = (al[t - 1] @ A) * E[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1): be[t] = (A @ (E[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True); xi = np.zeros_like(A)
    for t in range(n - 1): x = al[t][:, None] * A * (E[t + 1] * be[t + 1])[None]; xi += x / x.sum()
    return g, xi, np.log(c).sum()
def slp_fit(E, G, kappa=20.0, iters=10):
    A = np.full((V, V), 1 / V); pi = np.full(V, 1 / V)
    for _ in range(iters):
        XI = np.zeros((V, V)); G0 = np.zeros(V)
        for g in G: gm, xi, _ = fb(E[g], A, pi); XI += xi; G0 += gm[0]
        A = (XI + kappa / V) / (XI + kappa / V).sum(1, keepdims=True); pi = (G0 + kappa / V) / (G0 + kappa / V).sum()
    return A, pi
def apply_word_hmm(P, E, G, A, pi, marg):
    out = P.copy()
    for g in G: gm, _, _ = fb(E[g], A, pi); o = gm * marg[None]; out[g] = o / o.sum(1, keepdims=True)
    return out
def ca_map(rule, codes):
    bits = (codes[:, None] >> np.arange(6)) & 1; l, r = np.roll(bits, 1, 1), np.roll(bits, -1, 1); idx = 4 * l + 2 * bits + r
    nb = (rule >> idx) & 1; return (nb * (1 << np.arange(6))).sum(1)
def class_hmm(P, E, G, K, iters=15, kappa=5.0):
    g = np.random.default_rng(K); B = g.dirichlet(np.ones(V), K); A = np.full((K, K), 1 / K); pi = np.full(K, 1 / K)
    for _ in range(iters):
        AB = np.zeros((K, K)); BB = np.full((K, V), kappa / V); P0 = np.ones(K)
        for gg in G:
            Ek = E[gg] @ B.T; gm, xi, _ = fb(Ek, A, pi); AB += xi; P0 += gm[0]
            for t, row in enumerate(gg):
                r = gm[t][:, None] * B * E[row][None]; BB += r / r.sum()
        A = (AB + 1) / (AB + 1).sum(1, keepdims=True); B = BB / BB.sum(1, keepdims=True); pi = P0 / P0.sum()
    out = P.copy(); marg = P.mean(0)
    for gg in G:
        gm, _, _ = fb(E[gg] @ B.T, A, pi)
        for t, row in enumerate(gg):
            w = (gm[t][:, None] * B).sum(0) * E[row] * marg; out[row] = w / w.sum()
    return out
def beam(P, E, G, A, pi, K):
    out = P.copy(); lA = np.log(A); lp = np.log(pi)
    for gg in G:
        paths = [((), 0.0)]
        for t, row in enumerate(gg):
            le = np.log(E[row] + 1e-12); cand = []
            for pth, sc in paths:
                base = (lp if t == 0 else lA[pth[-1]]) + le
                for w in np.argsort(-base)[:K]: cand.append((pth + (w,), sc + base[w]))
            cand.sort(key=lambda c: -c[1]); paths = cand[:K]
        sc = np.array([c[1] for c in paths]); wts = np.exp(sc - sc.max()); wts /= wts.sum()
        for t, row in enumerate(gg):
            m = np.full(V, 1e-4)
            for (pth, _), wt in zip(paths, wts): m[pth[t]] += wt
            o = 0.5 * m / m.sum() + 0.5 * P[row]; out[row] = o / o.sum()
    return out
R = {}; CA = {}
for blk in ("Q3", "Q4"):
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        Pp = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = Pp["y"] >= 0
        P = F.softmax(M.readout(Pp["ce"].to(dev), Pp["zt"].to(dev), "ce", 0.5), 1).cpu().numpy()[k].astype(np.float64); y = Pp["y"][k]; idx = Pp["idx"][k.numpy()]
        G = runs_of(idx); marg = P.mean(0); E = P / marg[None]; T = 3.0 if s == 0 else 2.0
        A, pi = slp_fit(E, G); variants = {"none": P, "slp": apply_word_hmm(P, E, G, A, pi, marg)}
        codes = np.argsort(np.argsort(-marg))
        if blk == "Q3":
            best = max(range(256), key=lambda r: sum(fb(E[g], (lambda m: (np.eye(V)[np.minimum(m, V - 1)] * 0.7 + 0.3 / V))(ca_map(r, codes)), np.full(V, 1 / V))[2] for g in G))
            CA[s] = best
        Aca = np.eye(V)[np.minimum(ca_map(CA.get(s, 110), codes), V - 1)] * 0.7 + 0.3 / V; variants["ca"] = apply_word_hmm(P, E, G, Aca, np.full(V, 1 / V), marg)
        for K in (4, 8): variants[f"classhmm{K}"] = class_hmm(P, E, G, K)
        for K in (5, 20): variants[f"beam{K}"] = beam(P, E, G, A, pi, K)
        for nm, Z in variants.items():
            R.setdefault("deep" if s == 0 else "broad", {}).setdefault(nm, []).append(bacc_null(sinkhorn_p(torch.tensor(Z, dtype=torch.float32), T, u).log(), y)[0])
        print(blk, s, len(G), "runs", flush=True)
out = {g: {nm: round(float(np.mean(v)), 4) for nm, v in d.items()} for g, d in R.items()}; out["ca_rules"] = CA
print("SEQ", json.dumps(out)); json.dump(out, open("runs/seqprior_probe.json", "w"), indent=1, default=str)
