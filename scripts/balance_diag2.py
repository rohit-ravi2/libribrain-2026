"""BAL-3 map on local rows (deep_c ep6): Sinkhorn temperature/iterations; prototype-pooling EM in T5-projection space;
and the vocab-only-context check (accuracy on words inside runs of >=4 consecutive vocab words vs all)."""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words
from src.v3lib import Model, segments, predict, bacc_null, dev
M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5
W = Words(load_meg=True); q = W.quarters(); scored = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)


def sinkhorn(lo, iters=10, T=1.0, tgt=None):
    P = F.softmax(lo / T, 1); n, C = P.shape; t = torch.full((C,), 1 / C) if tgt is None else tgt
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (t * n)[None]; P = P / P.sum(1, keepdim=True)
    return P.log()


def pool_em(lo, z, tgt, iters=5, alpha=1.0, kappa=20.0):
    """Prototype pooling: posteriors R (n,50) from balanced model probs; class prototypes mu_c = R-weighted mean of unit
    embeddings z; new log-score = alpha*log p_model + kappa*cos(z, mu_c); re-balance; iterate."""
    z = F.normalize(z, dim=1); logp = F.log_softmax(lo, 1); R = sinkhorn(lo, tgt=tgt).exp()
    for _ in range(iters):
        mu = F.normalize((R.T @ z), dim=1)                                 # (50, D)
        sc = alpha * logp + kappa * (z @ mu.T)
        R = sinkhorn(sc, tgt=tgt).exp()
    return R.log()


res = {}
for blk in ("Q3", "Q4"):
    segs = segments(W, (q == blk) & np.isin(W.sub, scored), maxw)
    P = predict(M, W, segs, {s: I for s in scored}); lo_all = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau).cpu()
    z_all = P["zt"]; y_all, s_all, idx_all = P["y"], P["sub"], P["idx"]
    # vocab-only-context flag: word is inside a run of >=4 consecutive vocab words (by onset order within sentence)
    vrun = np.zeros(len(idx_all), bool)
    order = np.lexsort((W.onset[idx_all], W.sent[idx_all])); lab = W.label[idx_all][order]; sent = W.sent[idx_all][order]
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and lab[j] >= 0 and sent[j] == sent[i]: j += 1
        if j - i >= 4: vrun[order[i:j]] = True
        i = j if j > i else i + 1
    k = y_all >= 0
    for s in scored:
        m = (s_all == s) & k; lo, y, z, vr = lo_all[m], y_all[m], z_all[m], torch.from_numpy(vrun)[m]
        if len(y) == 0: continue
        pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
        r = {"asis": bacc_null(lo, y)[0]}
        r["sink_T2"] = bacc_null(sinkhorn(lo, 10, 2.0, tgt), y)[0]
        e = P["e"][m]
        for kappa in (5.0, 20.0):
            r[f"poolE_k{kappa}"] = bacc_null(pool_em(lo, e, tgt, kappa=kappa), y)[0]
            r[f"poolL_k{kappa}"] = bacc_null(pool_em(lo, F.log_softmax(lo, 1), tgt, kappa=kappa), y)[0]
        for T in (3.0, 4.0):
            r[f"sink_T{T}_i10"] = bacc_null(sinkhorn(lo, 10, T, tgt), y)[0]
        if vr.sum() >= 20 and (~vr).sum() >= 20:
            r["asis_vocabrun"] = bacc_null(lo[vr], y[vr])[0]; r["asis_other"] = bacc_null(lo[~vr], y[~vr])[0]; r["n_vocabrun"] = int(vr.sum())
        res[f"{blk}/{s}"] = r
keys = sorted({k for v in res.values() for k in v}); agg = {k: float(np.mean([v[k] for v in res.values() if k in v])) for k in keys}
print("MEAN", json.dumps({k: round(v, 4) for k, v in agg.items()}))
json.dump({"per": res, "mean": agg}, open("runs/balance_diag3.json", "w"), indent=1)
