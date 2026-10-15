"""INORM-2: inference-time per-channel instance norm on (a) day-novel subj0 (lso1, lso_dw on the held-out recording day,
all vocab rows) and (b) the width model dw_s0/dw_s1 on split-v2 Q3/Q4 (Broad + subj0). Also 'half' = per-channel sd
shrunk halfway to the type-median sd. + Sinkhorn all-50 (T3 subj0, T2 Broad)."""
import sys, os, glob, json, re
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(10); R = os.path.expanduser("~/projects/libribrain-2026/runs/")
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
MODE = {}
def make_win(W0):
    def f(idx, *a, **k):
        x, v = W0(idx, *a, **k)
        if MODE.get("m") is None: return x, v
        vf = v[:, None].float(); n = vf.sum(-1, keepdim=True).clamp_min(1)
        mu = (x.float() * vf).sum(-1, keepdim=True) / n; sd = ((((x.float() - mu) * vf) ** 2).sum(-1, keepdim=True) / n).sqrt().clamp_min(1e-12)
        if MODE["m"] == "half": sd = (sd * sd.median(1, keepdim=True).values).sqrt()
        return (((x.float() - mu) / sd) * vf).to(x.dtype), v
    return f
def run(M, W, segs, s, T):
    o = {}
    for m in (None, "inst", "half"):
        MODE["m"] = m; P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; o[str(m)] = bacc_null(sk(p, T).log(), P["y"][k])[0]
    MODE["m"] = None; return o
out = {}
PXR = "sub-0_ses-1[12]_task-Sherlock1"
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if re.search(PXR, b)], load_meg=True); W.windows = make_win(W.windows)
for ck in ("lso1_ep6", "lso_dw_ep6"):
    M = Model(R + ck + ".pt"); segs = segments(W, (W.sub == 0), M.a["maxw"]); out["daynovel_" + ck] = {k: round(v, 4) for k, v in run(M, W, segs, 0, 3.0).items()}
    print(ck, out["daynovel_" + ck], flush=True)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]], load_meg=True); q = W.quarters(); W.windows = make_win(W.windows)
for ck in ("dw_s0_ep6", "dw_s1_ep6"):
    M = Model(R + ck + ".pt"); acc = {}
    for blk in ("Q3", "Q4"):
        for s in BROAD + [0]:
            segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
            if not segs: continue
            for k_, v_ in run(M, W, segs, s, 3.0 if s == 0 else 2.0).items(): acc.setdefault(("deep" if s == 0 else "broad", k_), []).append(v_)
    out[ck] = {f"{g}/{k}": round(float(np.mean(v)), 4) for (g, k), v in acc.items()}; print(ck, out[ck], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inorm2.json"), "w"), indent=1)
