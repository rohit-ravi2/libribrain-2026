"""BAL-1: does per-subject class-balance re-ranking help? Uses a split-v2 checkpoint (Q3/Q4 never trained)."""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words
from src.v3lib import Model, segments, predict, bacc_null, dev
ck = sys.argv[1]; M = Model(ck); maxw = M.a["maxw"]; tau = 0.5
W = Words(load_meg=True); q = W.quarters(); scored = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
I = torch.eye(306, device=dev); rng = np.random.default_rng(0)


def sinkhorn(lo, iters=10, target=None):
    """lo (n,50) logits -> log-probs with column mass pushed to target (uniform over present classes by default)."""
    P = F.softmax(lo, 1); n, C = P.shape
    t = torch.full((C,), 1.0 / C) if target is None else target
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (t * n)[None]
        P = P / P.sum(1, keepdim=True)
    return P.log()


def marginal_fix(lo):
    P = F.softmax(lo, 1); return lo - P.mean(0).clamp_min(1e-9).log()[None]


res = {}
for blk in ("Q3", "Q4"):
    segs = segments(W, (q == blk) & np.isin(W.sub, scored), maxw)
    P = predict(M, W, segs, {s: I for s in scored}); lo_all = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau).cpu()
    y_all, s_all = P["y"], P["sub"]; k = y_all >= 0; lo_all, y_all, s_all = lo_all[k], y_all[k], s_all[k]
    for s in scored:
        m = s_all == s; lo, y = lo_all[m], y_all[m]
        if len(y) == 0: continue
        pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1.0 / len(pres)
        r = {"n": int(len(y)), "classes": int(len(pres)),
             "asis": bacc_null(lo, y)[0], "asis_marg": bacc_null(marginal_fix(lo), y)[0], "asis_sink": bacc_null(sinkhorn(lo, target=tgt), y)[0]}
        # balanced resamples
        b0, b1, b2 = [], [], []
        for d in range(5):
            idx = np.concatenate([rng.choice(np.flatnonzero((y == c).numpy()), min(20, int((y == c).sum())), replace=False) for c in pres.tolist()])
            lo_b, y_b = lo[idx], y[idx]; tb = torch.zeros(50); tb[pres] = 1.0 / len(pres)
            b0.append(bacc_null(lo_b, y_b)[0]); b1.append(bacc_null(marginal_fix(lo_b), y_b)[0]); b2.append(bacc_null(sinkhorn(lo_b, target=tb), y_b)[0])
        r.update({"bal": float(np.mean(b0)), "bal_marg": float(np.mean(b1)), "bal_sink": float(np.mean(b2))})
        res[f"{blk}/{s}"] = r
        # top-1 skew
        top = lo.argmax(1); r["top1_max_share"] = float(torch.bincount(top, minlength=50).max() / len(top))
agg = {k: float(np.mean([v[k] for v in res.values()])) for k in ("asis", "asis_marg", "asis_sink", "bal", "bal_marg", "bal_sink", "top1_max_share")}
for k, v in res.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("MEAN", json.dumps({k: round(v, 4) for k, v in agg.items()}))
json.dump({"ckpt": ck, "per": res, "mean": agg}, open("runs/balance_diag.json", "w"), indent=1)
