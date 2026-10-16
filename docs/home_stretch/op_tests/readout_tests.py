"""Readout-layer priors on a cached posterior bank (no GPU). Two folds per subject: fit on one quarter, score on the other.
VS  = per-class temperature+bias (vector scaling) fitted on the subject's own labelled quarter, shrunk to identity.
VSP = empirical-Bayes: Broad per-class params shrunk toward the pooled Broad mean (subject's own fit + population prior).
IB  = label-free top-10 inclusion balancing after Sinkhorn (each class in top-10 of 10/50 rows).
UOT = unbalanced Sinkhorn (KL-relaxed column marginals, exponent rho)."""
import os, numpy as np, torch
torch.set_num_threads(4); V = 50
B = np.load(os.path.join(os.path.dirname(__file__), "bank.npz"))
subs = sorted({int(k.split("_")[0][1:]) for k in B.files})
def sk(P, T, it=10, rho=1.0):
    P = torch.as_tensor(P).double().clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / (P.sum(0, keepdim=True) / (n / V)) ** rho; P = P / P.sum(1, keepdim=True)
    return P
def score(P, y):
    P = torch.as_tensor(P); y = torch.as_tensor(y); top = P.topk(10, 1).indices; hit = (top == y[:, None]).any(1).float()
    return float(torch.stack([hit[y == c].mean() for c in torch.unique(y)]).mean())
def fit_vs(lo, y, lam=1.0, steps=300):
    lo = torch.as_tensor(lo).float(); y = torch.as_tensor(y).long()
    la = torch.zeros(V, requires_grad=True); b = torch.zeros(V, requires_grad=True); opt = torch.optim.LBFGS([la, b], max_iter=steps)
    w = 1.0 / torch.bincount(y, minlength=V).clamp_min(1).float()[y]  # balanced CE: the metric is balanced
    def cl():
        opt.zero_grad(); l = (F_ce(lo * la.exp() + b, y) * w).sum() / w.sum() + lam * 1e-2 * (la.pow(2).sum() + b.pow(2).sum()); l.backward(); return l
    opt.step(cl); return la.detach(), b.detach()
F_ce = lambda z, y: torch.nn.functional.cross_entropy(z, y, reduction="none")
def apply(lo, la, b): return torch.softmax(torch.as_tensor(lo).float() * la.exp() + b, 1)
def ib(P, it=30, eta=0.5):
    L = torch.as_tensor(P).double().log(); off = torch.zeros(V, dtype=torch.double)
    for _ in range(it):
        top = (L + off).topk(10, 1).indices; r = torch.bincount(top.flatten(), minlength=V).double() / len(L)
        off -= eta * (r.clamp_min(1e-3).log() - np.log(10 / V))
    return torch.softmax(L + off, 1)
res = {}
fits = {}
for s in subs:
    for tr, te in (("Q3", "Q4"), ("Q4", "Q3")):
        if f"s{s}_{tr}_lo" in B.files and f"s{s}_{te}_lo" in B.files:
            fits[(s, te)] = fit_vs(B[f"s{s}_{tr}_lo"], B[f"s{s}_{tr}_y"])
for s in subs:
    T = 3 if s == 0 else 2
    for te in ("Q3", "Q4"):
        if (s, te) not in fits: continue
        lo, y = B[f"s{s}_{te}_lo"], B[f"s{s}_{te}_y"]; P0 = torch.softmax(torch.as_tensor(lo).float(), 1)
        r = {"base": score(sk(P0, T), y)}
        la, b = fits[(s, te)]; r["VS"] = score(sk(apply(lo, la, b), T), y)
        r["VS_la_only"] = score(sk(apply(lo, la, torch.zeros(V)), T), y)
        if s != 0:
            others = [fits[k] for k in fits if k[0] not in (0, s) and k[1] == te]
            if others:
                mla = torch.stack([o[0] for o in others]).mean(0); mb = torch.stack([o[1] for o in others]).mean(0)
                r["VSP_pop_only"] = score(sk(apply(lo, mla, mb), T), y)
                r["VSP_shrunk"] = score(sk(apply(lo, 0.5 * la + 0.5 * mla, 0.5 * b + 0.5 * mb), T), y)
        Ps = sk(P0, T); r["IB"] = score(ib(Ps), y)
        for rho in (0.5, 0.8): r[f"UOT{rho}"] = score(sk(P0, T, rho=rho), y)
        for T2 in (T - 1, T + 1): r[f"T{T2}"] = score(sk(P0, T2), y)
        res[(s, te)] = r
keys = sorted({k for r in res.values() for k in r})
print("sub blk n " + " ".join(f"{k:>12}" for k in keys))
for (s, te), r in res.items(): print(s, te, len(B[f's{s}_{te}_y']), " ".join(f"{r.get(k, np.nan):12.4f}" for k in keys))
for grp, sel in (("subj0", lambda s: s == 0), ("broad", lambda s: s != 0)):
    rows = [r for (s, _), r in res.items() if sel(s)]
    if not rows: continue
    print(f"== {grp} mean delta vs base (n blocks {len(rows)}; wins/blocks)")
    for k in keys:
        d = [r[k] - r["base"] for r in rows if k in r]
        if d: print(f"  {k:14s} {np.mean(d):+.4f}  {sum(x > 0 for x in d)}/{len(d)}")
