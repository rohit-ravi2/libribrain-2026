"""SPEQ-1: per-subject spectral equalisation to subj0 (label-free, subject's own block rows). Frozen deep_c ep6,
split-v2 Q3/Q4, 12 Broad subjects (identity A). Gain G(f) = clip(sqrt(PSD_subj0 / PSD_subj), 0.5, 2), smoothed over
5 bins, per sensor type ('type') or per channel ('chan'); applied in the FFT domain to each window, then re-masked.
Also a 'chan_half' variant (sqrt of chan gain = half-way). Scored raw and + Sinkhorn (all-50 uniform)."""
import sys, os, glob, json
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE, IS_MAG
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(8)
def sk(P, T=2.0, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")); I = torch.eye(306, device=dev)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
W = Words(runs=runs, load_meg=True); q = W.quarters(); WIN0 = W.windows
def psd(idx):
    acc = None
    for i in range(0, len(idx), 200):
        x, v = WIN0(idx[i:i + 200]); x = x.float() * v[:, None].float()
        x = x - x.sum(-1, keepdim=True) / v.float().sum(-1)[:, None, None].clamp_min(1)
        pw = (torch.fft.rfft(x, dim=-1).abs() ** 2).sum(0); acc = pw if acc is None else acc + pw
    return acc / len(idx)
ref = psd(np.flatnonzero(np.isin(q, ["Q1", "Q2"]) & (W.sub == 0) & (W.label >= 0)))
mag = torch.from_numpy(np.asarray(IS_MAG))
def gains(Ps):
    def sm(G): return F.avg_pool1d(G[:, None], 5, 1, 2, count_include_pad=False)[:, 0]
    r = (ref / Ps.clamp_min(1e-20))
    out = {"chan": sm(r.sqrt()).clamp(0.5, 2.0)}
    t = torch.zeros_like(r)
    for m_ in (mag, ~mag): t[m_] = r[m_].mean(0, keepdim=True)
    out["type"] = sm(t.sqrt()).clamp(0.5, 2.0); out["chan_half"] = out["chan"].sqrt()
    return out
G = None
def eq_windows(idx, *a, **k):
    x, v = WIN0(idx, *a, **k)
    if G is None: return x, v
    X = torch.fft.rfft(x.float(), dim=-1) * G[None]; y = torch.fft.irfft(X, n=x.shape[-1], dim=-1) * v[:, None].float()
    return y.to(x.dtype), v
W.windows = eq_windows; R = {}
for blk in ("Q3", "Q4"):
    for s in BROAD:
        sel = (q == blk) & (W.sub == s); segs = segments(W, sel, M.a["maxw"])
        if not segs: continue
        Gs = gains(psd(np.flatnonzero(sel & (W.label > -2))))
        for nm in ("base", "type", "chan", "chan_half"):
            G = None if nm == "base" else Gs[nm]
            P = predict(M, W, segs, {s: I}); k = P["y"] >= 0
            p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]
            d = R.setdefault(nm, [[], []]); d[0].append(bacc_null(p.log(), y)[0]); d[1].append(bacc_null(sk(p).log(), y)[0])
        G = None
        print(blk, s, {n: round(v[1][-1] - R["base"][1][-1], 3) for n, v in R.items()}, flush=True)
out = {n: dict(raw=round(float(np.mean(v[0])), 4), sk=round(float(np.mean(v[1])), 4),
               d_sk=round(float(np.mean(np.array(v[1]) - np.array(R["base"][1]))), 4),
               n_pos=int((np.array(v[1]) > np.array(R["base"][1])).sum()), n=len(v[1])) for n, v in R.items()}
print("SPEQ", json.dumps(out)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "speq.json"), "w"), indent=1)
