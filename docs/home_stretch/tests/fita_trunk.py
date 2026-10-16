"""PX-FT1: does FITA (labelled per-subject 306x306 A, frozen trunk) remove the Broad penalty of the Deep-only trunk
changes (width, stretch)? Same protocol as fita_map.py (split-v2, A fitted on the subject's Q1-Q2 words, f=1.0, 300 steps,
CE + label smoothing 0.1, ridge 1e-3 to identity), scored on Q3/Q4 raw and + Sinkhorn T2 all-50.
Usage: LB_DEV=cpu fita_trunk.py ck1,ck2,... subj1,subj2,... out.json"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p

CKS = sys.argv[1].split(","); SUBS = [int(s) for s in sys.argv[2].split(",")]; OUT = sys.argv[3]; STEPS = int(os.environ.get("STEPS", "300"))
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); u = torch.full((50,), 1 / 50); I = torch.eye(306, device=dev)


def fit(M, segs, steps, seed, lr=3e-4):
    rng = np.random.default_rng(seed); D = nn.Parameter(torch.zeros(306, 306, device=dev)); opt = torch.optim.Adam([D], lr=lr)
    for _ in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]; idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, I + D, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * (D ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return (I + D).detach()


def score(M, segs, A):
    with torch.no_grad():
        P = predict(M, W, segs, {int(W.sub[segs[0][0]]): A})
    k = P["y"] >= 0; p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]
    return round(bacc_null(p.log(), y)[0], 4), round(bacc_null(sinkhorn_p(p, 2.0, u).log(), y)[0], 4)


res = {}
for ck in CKS:
    M = Model(ck); name = os.path.basename(ck); res[name] = {}
    for s in SUBS:
        t0 = time.time(); maxw = M.a["maxw"]
        tr = segments(W, np.isin(q, ["Q1", "Q2"]) & (W.sub == s), maxw); tests = {b: segments(W, (q == b) & (W.sub == s), maxw) for b in ("Q3", "Q4")}
        A = fit(M, tr, STEPS, seed=s)
        r = {b: {"base": score(M, t, I), "fita": score(M, t, A)} for b, t in tests.items() if t}
        res[name][s] = r; print(name, s, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
    m = lambda k, j: round(float(np.mean([res[name][s][b][k][j] for s in res[name] for b in res[name][s]])), 4)
    res[name]["mean"] = {k: [m(k, 0), m(k, 1)] for k in ("base", "fita")}; print(name, "MEAN", res[name]["mean"], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
