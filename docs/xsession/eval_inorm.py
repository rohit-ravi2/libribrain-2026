"""Score a checkpoint trained with instance norm (patched normalize) vs deep_c (original) on split-v2 Q3/Q4, + Sinkhorn all-50;
and, with PX, on subj0's held-out recording day."""
import sys, os, glob, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import inorm_patch  # noqa
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
M = Model(sys.argv[1]); PX = os.environ.get("PX"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
if PX: W = Words(runs=[b for b in runs if re.search(PX, b)], load_meg=True); blocks = [("PX", 0, W.sub == 0)]
else:
    W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]], load_meg=True); q = W.quarters()
    blocks = [(b, s, (q == b) & (W.sub == s)) for b in ("Q3", "Q4") for s in BROAD + [0]]
acc = {}
for b, s, m in blocks:
    segs = segments(W, m, M.a["maxw"])
    if not segs: continue
    P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
    p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]
    acc.setdefault("deep" if s == 0 else "broad", []).append(bacc_null(sk(p, 3.0 if s == 0 else 2.0).log(), P["y"][k])[0])
print("EVAL", sys.argv[1], PX or "Q3Q4", {g: round(float(np.mean(v)), 4) for g, v in acc.items()}, flush=True)
