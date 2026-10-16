import sys, numpy as np; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from simrule import load, score, slp, sk, bacc
D = load()
def rk(M):  # rank 0 = best, per row
    return np.argsort(np.argsort(-M, 1), 1).astype(float)
for b in D: b["k10"] = slp(b["raw"], 10.0)[0]
def lp(M): return np.log(np.clip(M, 1e-12, None))
def exps(S): S = S - S.max(1, keepdims=True); E = np.exp(S); return E / E.sum(1, keepdims=True)
def post(fn):  # rule operating on balanced matrices, custom loop
    base, new = [], []
    for b in D:
        y = np.concatenate(b["y"]); B = sk(np.concatenate(b["slp"]))
        base.append(bacc(B, y)); new.append(bacc(fn(b, B), y))
    d = np.array(new) - np.array(base)
    return dict(delta=round(d.mean(), 4), se=round(d.std()/np.sqrt(len(d)), 4), wins=int((d > 0).sum()))
R = {}
R["rankavg raw+slp exp(-r/5)"] = score(lambda b: [exps(-(rk(s)+rk(r))/2/5) for s, r in zip(b["slp"], b["raw"])], D)
R["borda raw/slp/k10 exp(-r/5)"] = score(lambda b: [exps(-(rk(s)+rk(r)+rk(k))/3/5) for s, r, k in zip(b["slp"], b["raw"], b["k10"])], D)
R["RRF k60 slp+raw (log)"] = score(lambda b: [exps(200*(1/(60+rk(s))+1/(60+rk(r)))) for s, r in zip(b["slp"], b["raw"])], D)
R["slp + rank tiebreak 0.02*raw rank"] = score(lambda b: [exps(lp(s) - 0.02*rk(r)) for s, r in zip(b["slp"], b["raw"])], D)
R["slp*0.8 rankavg blend"] = score(lambda b: [0.8*s + 0.2*exps(-(rk(s)+rk(r))/2/5) for s, r in zip(b["slp"], b["raw"])], D)
def top10union(b, B):
    Br = sk(np.concatenate(b["raw"])); out = lp(B).copy()
    both = (rk(B) < 10) & (rk(Br) < 15); return exps(out + 2.0*both)
R["post: boost slp top10 also in raw top15"] = post(top10union)
R["post: rankavg sk(slp)+sk(raw)"] = post(lambda b, B: exps(-(rk(B)+0.5*rk(sk(np.concatenate(b["raw"]))))))
R["post: borda sk slp/raw/k10"] = post(lambda b, B: exps(-(rk(B)+rk(sk(np.concatenate(b["raw"])))+rk(sk(np.concatenate(b["k10"]))))))
R["post: slp rank + 0.3 k10 rank"] = post(lambda b, B: exps(-(rk(B)+0.3*rk(sk(np.concatenate(b["k10"]))))))
R["post: tiebreak slp ranks 8-12 by raw"] = post(lambda b, B: exps(-(rk(B) + 0.4*((rk(B)>=7)&(rk(B)<=12))*rk(sk(np.concatenate(b["raw"])))/10)))
for k, v in R.items(): print(k, v, flush=True)
