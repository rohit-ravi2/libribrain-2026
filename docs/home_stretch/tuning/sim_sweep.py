"""TUNE-SIM: SIM-1 (the calibrated instrument for sentence priors: SIM +0.016 vs LB +0.018 Broad) re-run on the CPU bank
(FINAL-6c 5-seed prob-mean, ce tau 0.5, Broad subjects 1-12 identity A, out-of-sample), sweeping SLP knobs never mapped:
EM iterations, kappa x iterations, emission beta, Sinkhorn T after SLP. 156 LLM vocab-only sentences per simulated subject,
each word = a random held-out posterior row of that label from that subject; R draws per subject.
Matches hmm_prior.py: emissions (P/marg)^beta, Dirichlet base uniform with strength kappa per row, uniform init, Sinkhorn after."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from src.common import VOCAB_ID, norm_word
B = np.load(sys.argv[1]); CE, Y, S, LP = B["ce"].astype(np.float64), B["y"], B["sub"], B["log_prior"].astype(np.float64)
R = int(os.environ.get("R", "2")); V = 50; rng = np.random.default_rng(0)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
P_all = softmax(CE - 0.5 * LP[None, None]).mean(0)
def bacc(P, y):
    top = np.argpartition(-P, 10, 1)[:, :10]; hit = (top == y[:, None]).any(1)
    return np.mean([hit[y == c].mean() for c in np.unique(y)])
def sk(P, T, it=10):
    Q = np.clip(P, 1e-12, None) ** (1 / T); Q /= Q.sum(1, keepdims=True); n = len(Q)
    for _ in range(it): Q = Q / Q.sum(0, keepdims=True) * (n / 50); Q /= Q.sum(1, keepdims=True)
    return Q
def fb(E, A, pi):
    n = len(E); al = np.zeros((n, V)); be = np.ones((n, V)); c = np.zeros(n)
    al[0] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n): al[t] = (al[t - 1] @ A) * E[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1): be[t] = (A @ (E[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True); xi = np.zeros((V, V))
    for t in range(n - 1):
        x = al[t][:, None] * A * (E[t + 1] * be[t + 1])[None]; xi += x / x.sum()
    return g, xi
def slp_path(Es, kappa, iters_list):
    """returns {iters: marginals list} along one EM path"""
    A = np.full((V, V), 1 / V); pi = np.full(V, 1 / V); out = {}; mx = max(iters_list)
    for it in range(1, mx + 1):
        XI = np.zeros((V, V)); G0 = np.zeros(V)
        for E in Es: g, xi = fb(E, A, pi); XI += xi; G0 += g[0]
        A = (XI + kappa / V) / (XI + kappa / V).sum(1, keepdims=True); pi = (G0 + kappa / V) / (G0 + kappa / V).sum()
        if it in iters_list: out[it] = [fb(E, A, pi)[0] for E in Es]
    return out
SENTS = []
for l in open(os.path.expanduser("~/projects/libribrain-2026/results/vocab_sentences.txt")):
    ids = [VOCAB_ID.get(norm_word(w)) for w in l.split()]
    if 5 <= len(ids) <= 9 and None not in ids: SENTS.append(ids)
res = {}
KAP = (10.0, 20.0, 40.0); ITS = (3, 5, 10, 20, 40); BETAS = (0.7, 1.0, 1.4); TS = (2.0, 3.0, 4.0)
for s in [x for x in np.unique(S) if x != 0]:
    r_s = np.flatnonzero(S == s); p, y = P_all[r_s], Y[r_s]; byl = {c: np.flatnonzero(y == c) for c in range(50)}
    ok = [Z for Z in SENTS if all(len(byl[c]) for c in Z)]
    for d in range(R):
        chosen = [ok[i] for i in rng.choice(len(ok), 156, replace=False)]
        rows = [[rng.choice(byl[c]) for c in Z] for Z in chosen]; P = np.concatenate([p[rr] for rr in rows]); Yt = np.concatenate(chosen)
        marg = P.mean(0)
        for T in TS: res.setdefault(f"base_T{T}", []).append(bacc(sk(P, T), Yt))
        for beta in BETAS:
            Es = [(p[rr] / marg) ** beta for rr in rows]
            for kap in KAP:
                if beta != 1.0 and kap != 20.0: continue
                path = slp_path(Es, kap, ITS if beta == 1.0 else (10,))
                for it, G in path.items():
                    Gs = np.concatenate([gg * marg for gg in G]); Gs /= Gs.sum(1, keepdims=True)
                    for T in TS:
                        if (T != 2.0) and not (kap == 20.0 and beta == 1.0 and it in (10, 40)): continue
                        res.setdefault(f"k{kap:g}_it{it}_b{beta}_T{T}", []).append(bacc(sk(Gs, T), Yt))
    print("subj", s, "done", flush=True)
out = {k: [round(float(np.mean(v)), 4), round(float(np.std(v) / np.sqrt(len(v))), 4)] for k, v in res.items()}
ref = np.array(res["k20_it10_b1.0_T2.0"])
out = {k: v + ([round(float(np.mean(np.array(res[k]) - ref)), 4), round(float(np.std(np.array(res[k]) - ref) / np.sqrt(len(ref))), 4)] if len(res[k]) == len(ref) else []) for k, v in out.items()}
for k in sorted(out): print(k, out[k])
json.dump(out, open(sys.argv[2], "w"), indent=1)
