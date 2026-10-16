"""Heavy-tail noise prior: robust per-channel winsorising of the continuous input (median +- k*1.4826*MAD, per run),
inference only, deep_c ep6, same 7-subject Q3/Q4 bank, Sinkhorn all-50 (T3 subj0, T2 Broad)."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, dev
torch.set_num_threads(4); V = 50
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt"))
SUBS = [0] + sorted(set(S2.EVAL + S2.VAL + S2.SEEN))[:6]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS], load_meg=True); q = W.quarters()
print("meg shapes", [m.shape for m in W.meg[:2]], W.meg[0].dtype, flush=True)
RAW = [np.array(m) for m in W.meg]
def sk(P, T, it=10):
    P = P.double().clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / V); P = P / P.sum(1, keepdim=True)
    return P
def score(P, y):
    top = P.topk(10, 1).indices; hit = (top == y[:, None]).any(1).float()
    return float(torch.stack([hit[y == c].mean() for c in torch.unique(y)]).mean())
def run_all():
    out = {}
    for s in SUBS:
        for blk in ("Q3", "Q4"):
            segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
            with torch.no_grad():
                P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
                lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5).cpu()[k]
            out[(s, blk)] = score(sk(torch.softmax(lo, 1), 3 if s == 0 else 2), P["y"][k])
    return out
base = run_all(); print("base", {k: round(v, 4) for k, v in base.items()}, flush=True)
ax = 1 if RAW[0].shape[0] in (306,) else 0   # time axis
for kk in (3.0, 5.0, 8.0):
    for i, r in enumerate(RAW):
        med = np.median(r, axis=ax, keepdims=True); mad = np.median(np.abs(r - med), axis=ax, keepdims=True) * 1.4826 + 1e-12
        W.meg[i] = np.clip(r, med - kk * mad, med + kk * mad).astype(r.dtype)
    res = run_all(); d = np.array([res[k] - base[k] for k in base])
    print(f"k={kk}: subj0 {d[:2].mean():+.4f}  broad {d[2:].mean():+.4f} ({(d[2:] > 0).sum()}/{len(d) - 2} up)", flush=True)
