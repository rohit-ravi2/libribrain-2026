"""TKB-1: metric-aligned balancing. Sinkhorn (per-class column scaling so tempered probability MASS is uniform) is one choice
of 50 per-class log-biases v_c. The metric (balanced Recall@10) counts top-10 INCLUSIONS, so pick v_c so that every class
lands in a top-10 list of ~10n/50 rows instead (soft inclusion via a per-row threshold, sigmoid width eps). Rows are then
ranked by log p + v_c. Same parameter count, same anchor (uniform), same data (the subject's own rows) as Sinkhorn.
Instrument: split-v2 Q3/Q4 blocks, deep_c ep7, identity A (Broad) / A0 (subj0), ce tau 0.5; natural blocks and
class-balanced (<= 18/class, holdout-like) subsamples.
Usage: LB_DEV=cpu topk_balance.py ckpt subjs out.json"""
import sys, os, json, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev

M = Model(sys.argv[1]); SUBS = [int(s) for s in sys.argv[2].split(",")]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS], load_meg=True); q = W.quarters()


def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P


def topk_bal(L, eps=0.5, k=10, iters=200, lr=0.5):
    """L (n,50) log-scores. Returns L + v with v chosen so soft top-k inclusion counts per class ~ k n / 50."""
    n = len(L); tgt = k * n / 50; v = torch.zeros(50, dtype=L.dtype)
    for _ in range(iters):
        S = L + v; lo = S.min(1, keepdim=True).values - 10 * eps; hi = S.max(1, keepdim=True).values + 10 * eps
        for _ in range(40):   # per-row threshold so that the soft count per row = k
            mid = (lo + hi) / 2; c = torch.sigmoid((S - mid) / eps).sum(1, keepdim=True); big = c > k; lo = torch.where(big, mid, lo); hi = torch.where(big, hi, mid)
        cnt = torch.sigmoid((S - (lo + hi) / 2) / eps).sum(0)
        v = v - lr * eps * torch.log(cnt.clamp_min(1e-3) / tgt); v = v - v.mean()
    return L + v


res = {}; rng = np.random.default_rng(0)
for s in SUBS:
    A = {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}; T0 = 3.0 if s == 0 else 2.0
    for b in ("Q3", "Q4"):
        segs = segments(W, (q == b) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        with torch.no_grad(): P = predict(M, W, segs, A)
        kk = P["y"] >= 0; p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[kk].double(); y = P["y"][kk]
        views = {"nat": [np.arange(len(y))]}
        byc = {c: np.flatnonzero(y.numpy() == c) for c in range(50)}
        views["bal"] = [np.concatenate([rng.choice(v_, min(18, len(v_)), replace=False) for v_ in byc.values() if len(v_)]) for _ in range(5)]
        for vn, idxs in views.items():
            r = {}
            for ix in idxs:
                pp, yy = p[ix], y[ix]; lp = pp.log()
                cand = {"raw": lp, f"sk_T{T0:g}": sk(pp, T0).log(), "sk_T4": sk(pp, 4.0).log()}
                for e in (0.25, 0.5, 1.0): cand[f"tkb_e{e}"] = topk_bal(lp, e)
                cand["sk+tkb_e0.5"] = topk_bal(sk(pp, T0).log(), 0.5)
                for kname, L in cand.items(): r.setdefault(kname, []).append(bacc_null(L.float(), yy)[0])
            res.setdefault(vn, {}).setdefault(f"{s}_{b}", {k_: round(float(np.mean(v_)), 4) for k_, v_ in r.items()})
            print(s, b, vn, res[vn][f"{s}_{b}"], flush=True)
summ = {}
for vn, blocks in res.items():
    for grp, sel in (("deep", lambda k: k.startswith("0_")), ("broad", lambda k: not k.startswith("0_"))):
        ks = [k for k in blocks if sel(k)]
        if ks: summ[f"{vn}/{grp}"] = {m: round(float(np.mean([blocks[k][m] for k in ks])), 4) for m in blocks[ks[0]]}
print("SUMMARY", json.dumps(summ, indent=1)); json.dump({"blocks": res, "summary": summ}, open(sys.argv[3], "w"), indent=1)
