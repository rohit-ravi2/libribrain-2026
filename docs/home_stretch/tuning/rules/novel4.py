"""Novel post-processing ideas on SIM-RULE (no y access in rules)."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from simrule import load, score, slp, sk, V
D = load(); t0 = time.time()
def rep(name, r): print(name, r, f"{time.time()-t0:.0f}s", flush=True)
def nrm(m): m = np.clip(m, 1e-12, None); return m / m.sum(1, keepdims=True)

# 1) confidence-dependent raw/SLP mixing: w_raw = a * (rowmax_raw)^p  (confident raw rows keep more raw evidence)
for a in [0.1, 0.25, 0.5]:
    def r1(b, a=a):
        out = []
        for R, G in zip(b["raw"], b["slp"]):
            c = R.max(1, keepdims=True); c = c / 1.0; w = a * c
            out.append(nrm((1 - w) * G + w * R))
        return out
    rep(f"IDEA1 confmix a={a}", score(r1, D))
# 1b) geometric version (log-linear) with constant weight
for a in [0.1, 0.25]:
    rep(f"IDEA1b geomix a={a}", score(lambda b, a=a: [nrm(G ** (1 - a) * R ** a) for R, G in zip(b["raw"], b["slp"])], D))

# 2) posterior averaging over kappa {10,20,40}
cache = {}
for i, b in enumerate(D):
    for k in [10, 40]: cache[(i, k)] = slp(b["raw"], kappa=k)[0]
rep("slp cache built", {})
idx = {id(b): i for i, b in enumerate(D)}
for wt in [(0.25, 0.5, 0.25), (1/3, 1/3, 1/3), (0.5, 0.5, 0)]:
    def r2(b, wt=wt):
        i = idx[id(b)]
        return [nrm(wt[0] * g10 + wt[1] * g20 + wt[2] * g40) for g10, g20, g40 in zip(cache[(i, 10)], b["slp"], cache[(i, 40)])]
    rep(f"IDEA2 kappaavg w(10,20,40)={wt}", score(r2, D))

# 3) balance raw before SLP (Sinkhorn on raw emissions at temperature T0, then SLP), balance after as usual
for T0 in [1.0, 1.75, 3.0]:
    pre = {}
    for i, b in enumerate(D):
        cat = np.concatenate(b["raw"]); Q = sk(cat, T=1.0, gamma=1.0) if T0 == 1.0 else sk(cat, T=T0)
        if T0 != 1.0: Q = nrm(Q ** (T0))  # undo temperature flattening, keep column balancing factors
        lens = np.cumsum([len(r) for r in b["raw"]])[:-1]; pre[i] = slp(np.split(Q, lens))[0]
    rep(f"IDEA3 balance-before T0={T0}", score(lambda b: pre[idx[id(b)]], D))

# 4) max-marginal (Viterbi) blending using the block's A, pi
def maxmarg(E, A, pi):
    n = len(E); lA = np.log(A); la = np.zeros((n, V)); lb = np.zeros((n, V)); lE = np.log(np.clip(E, 1e-300, None))
    la[0] = np.log(pi) + lE[0]
    for t in range(1, n): la[t] = (la[t-1][:, None] + lA).max(0) + lE[t]
    for t in range(n-2, -1, -1): lb[t] = (lA + (lE[t+1] + lb[t+1])[None]).max(1)
    m = la + lb; return m - m.max(1, keepdims=True)
MM = {}
for i, b in enumerate(D):
    marg = np.concatenate(b["raw"]).mean(0)
    MM[i] = [maxmarg(R / marg, b["A"], b["pi"]) for R in b["raw"]]
rep("maxmarg built", {})
for lam in [0.1, 0.3, 0.6]:
    rep(f"IDEA4 maxmarg-boost lam={lam}", score(lambda b, lam=lam: [nrm(G * np.exp(lam * m)) for G, m in zip(b["slp"], MM[idx[id(b)]])], D))
