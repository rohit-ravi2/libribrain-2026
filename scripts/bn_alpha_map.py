"""BN-α map (local, split v2 Q3/Q4): partial AdaBN. Encoder BatchNorm running stats = (1-α)·training + α·subject-block
(subject's own rows; running_var blended likewise), α ∈ {0, .25, .5, .75, 1}; + Sinkhorn. Broad subjects identity A, subj0 A0.
Usage: bn_alpha_map.py runs/<ckpt>.pt"""
import sys, os, json, copy, time, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev

M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5; t0 = time.time()
import re as _re
PX = os.environ.get("PX")                       # regex of held-out subj0 runs (leave-day-out) -> blocks PX_all / PX_short (5-9-word sentences)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = [0] if PX else BROAD + [0]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if _re.search(PX, b)] if PX else [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev)
if PX:
    q = np.array(["PX_all"] * W.n, dtype=object); _, cnt = np.unique(W.sent, return_counts=True); short = (cnt[W.sent] >= 5) & (cnt[W.sent] <= 9)
BNS = [m for m in M.enc.modules() if isinstance(m, nn.BatchNorm1d)]; TRAIN = [(m.running_mean.clone(), m.running_var.clone()) for m in BNS]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def sinkhorn(lo, T, tgt, iters=10):
    P = F.softmax(lo / T, 1); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P.log()


@torch.no_grad()
def subject_stats(segs, A):
    for m in BNS: m.reset_running_stats(); m.momentum = None
    M.enc.train()
    for mod in M.enc.modules():
        if isinstance(mod, nn.Dropout): mod.eval()
    for i in range(0, len(segs), 40):
        idx, gi, pad, rel = pack(W, segs[i:i + 40]); x, v = W.windows(idx); M.forward(x, v, A, gi, pad, rel)
    M.enc.eval(); return [(m.running_mean.clone(), m.running_var.clone()) for m in BNS]


res = {}
for blk in (("PX_all", "PX_short") if PX else ("Q3", "Q4")):
    for s in SUBS:
        segs = segments(W, ((q == "PX_all") & short if blk == "PX_short" else (q == blk)) & (W.sub == s), maxw)
        if not segs: continue
        A = M.A[M.sid[0]] if s == 0 else I; T = 3.0 if s == 0 else 2.0; r = {}
        SUBJ = subject_stats(segs, A)
        for al in ALPHAS:
            for m, (mt, vt), (ms, vs) in zip(BNS, TRAIN, SUBJ):
                m.running_mean.copy_((1 - al) * mt + al * ms); m.running_var.copy_((1 - al) * vt + al * vs)
            P = predict(M, W, segs, {s: A}); lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau).cpu(); y = P["y"]; k = y >= 0
            lo, y = lo[k], y[k]; pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
            r[f"a{al}"] = bacc_null(lo, y)[0]; r[f"a{al}+sk"] = bacc_null(sinkhorn(lo, T, tgt), y)[0]
        for m, (mt, vt) in zip(BNS, TRAIN): m.running_mean.copy_(mt); m.running_var.copy_(vt)
        res[f"{blk}/{s}"] = r; print(blk, s, json.dumps({k: round(v, 3) for k, v in r.items()}), round(time.time() - t0), "s", flush=True)
out = {}
for grp, members in (("broad", BROAD), ("deep", [0])):
    for blkset in ((("PX_all",), ("PX_short",)) if PX else (("Q3", "Q4"),)):
        rows = [v for k, v in res.items() if int(k.split("/")[1]) in members and k.split("/")[0] in blkset]
        if not rows: continue
        out[grp + "/" + "+".join(blkset)] = {k: round(float(np.mean([v[k] for v in rows])), 4) for k in rows[0]}
        print("MEAN", grp, "+".join(blkset), json.dumps(out[grp + "/" + "+".join(blkset)]), flush=True)
json.dump({"per": res, "mean": out}, open(os.environ.get("OUT", "runs/bn_alpha_map.json"), "w"), indent=1)
