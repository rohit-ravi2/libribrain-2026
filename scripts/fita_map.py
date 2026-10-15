"""FITA-2: robustness map of the labelled per-subject input layer (ORACLE-1 O6) on split-v2 Q3/Q4, Broad subjects.
A_s (from identity, trunk frozen, CE on the shared head) is fitted on the subject's labelled Q1-Q2 words, subsampled to a
fraction f of its sentences (f = 0.25 / 0.5 / 1.0 ~ the 25% / 50% / 100% supervision groups), for 100 / 300 / 600 steps.
Scored on both blocks, raw and + Sinkhorn (uniform over all 50). Usage: fita_map.py runs/<ckpt>.pt [out_tag]"""
import sys, os, json, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p
M = Model(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "deepc"; BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD]
W = Words(runs=runs, load_meg=True); q = W.quarters(); rng = np.random.default_rng(0); maxw = M.a["maxw"]; u = torch.full((50,), 1 / 50)
XYZ = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")), float)
_d = ((XYZ[:, None] - XYZ[None]) ** 2).sum(-1); _nn = np.argsort(_d, 1)[:, 1:10]; Adj = np.zeros((306, 306))
for i in range(306): Adj[i, _nn[i]] = 1
Adj = np.maximum(Adj, Adj.T); LAP = torch.tensor(np.diag(Adj.sum(1)) - Adj, dtype=torch.float32, device=dev)
_c = XYZ - XYZ.mean(0); _lat = np.abs(_c[:, 0]) / np.abs(_c[:, 0]).max(); _low = (_c[:, 2].max() - _c[:, 2]) / np.ptp(_c[:, 2])
TEMPW = torch.tensor(1 + 0.5 * (_lat * _low) / (_lat * _low).max(), dtype=torch.float32, device=dev)   # 1 .. 1.5, lateral-inferior up-weighted
REG = os.environ.get("REG", "plain")
def fit(segs, steps, lr=3e-4):
    base = torch.diag(TEMPW) if REG == "temporal" else I
    if REG == "lowrank":
        U = nn.Parameter(1e-3 * torch.randn(306, 16, device=dev)); V = nn.Parameter(1e-3 * torch.randn(16, 306, device=dev)); ps = [U, V]; mk = lambda: base + U @ V
    else:
        D = nn.Parameter(torch.zeros(306, 306, device=dev)); ps = [D]; mk = lambda: base + D
    opt = torch.optim.Adam(ps, lr=lr)
    for _ in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]; idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        A = mk(); _, ce, _ = M.forward(x, v, A, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        Dl = A - base; reg = 1e-3 * (Dl ** 2).sum() + (1e-2 * torch.trace(Dl.T @ LAP @ Dl) if REG == "lap" else 0)
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + reg; opt.zero_grad(); loss.backward(); opt.step()
    return mk().detach()
def score(segs, A):
    P = predict(M, W, segs, {int(W.sub[segs[0][0]]): A}); k = P["y"] >= 0
    p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]
    return bacc_null(p.log(), y)[0], bacc_null(sinkhorn_p(p, 2.0, u).log(), y)[0]
res = {}
for s in BROAD:
    tr = segments(W, np.isin(q, ["Q1", "Q2"]) & (W.sub == s), maxw); tests = {b: segments(W, (q == b) & (W.sub == s), maxw) for b in ("Q3", "Q4")}
    if not tr: continue
    r = {b: {"base": score(t, I), "base_temporal": score(t, torch.diag(TEMPW))} for b, t in tests.items() if t}
    for f in ((0.25, 1.0) if REG != "plain" or os.environ.get("QUICK") else (0.25, 0.5, 1.0)):
        sub = [tr[i] for i in rng.permutation(len(tr))[:max(4, int(f * len(tr)))]]
        for st in ((300,) if os.environ.get("QUICK") else (100, 300, 600)):
            A = fit(sub, st)
            for b, t in tests.items():
                if t: r[b][f"f{f}_s{st}"] = score(t, A)
    res[s] = r; print(s, json.dumps({b: {k: [round(x, 3) for x in v] for k, v in d.items()} for b, d in r.items()}), flush=True)
keys = [k for k in next(iter(next(iter(res.values())).values()))]
out = {k: [round(float(np.mean([res[s][b][k][j] for s in res for b in res[s]])), 4) for j in (0, 1)] for k in keys}
out = {k: v + [round(v[1] - out["base"][1], 4)] for k, v in out.items()}
print("MEAN [raw, +sk, d_sk]", json.dumps(out)); json.dump({"per": {str(k): v for k, v in res.items()}, "mean": out}, open(f"runs/fita_map_{tag}_{REG}.json", "w"), indent=1)
