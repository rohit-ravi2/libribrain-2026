"""VIEW-1: does preceding-word context hurt? Score every word with (i) the full [-1,+2] s sentence-clipped window and
(ii) the window clipped at its own onset (pre-onset masked, as for a sentence-initial word); ensemble (mean p). Local
split-v2 Q3/Q4, deep_c ep6; report all rows and mid-sentence rows, raw and + Sinkhorn (uniform over all 50)."""
import sys, os, json, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p
M = Model("runs/deep_c_ep6.pt"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
W = Words(runs=runs, load_meg=True); q = W.quarters(); WIN0 = W.windows
def clipped(idx, iso=None, clip=None): return WIN0(idx, iso, (np.zeros(len(idx)), np.full(len(idx), 9.0)))
R = {}
for blk in ("Q3", "Q4"):
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        ps = []
        for view in ("full", "clip"):
            W.windows = WIN0 if view == "full" else clipped
            P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); W.windows = WIN0; k = P["y"] >= 0
            ps.append(F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]); y = P["y"][k]; idx = P["idx"][k.numpy()]
        mid = torch.from_numpy(np.array([np.any((W.sent == W.sent[w]) & (W.onset < W.onset[w])) for w in idx]))
        T = 3.0 if s == 0 else 2.0; u = torch.full((50,), 1 / 50)
        for name, p in (("full", ps[0]), ("clip", ps[1]), ("ens", (ps[0] + ps[1]) / 2), ("ens_w", 0.7 * ps[0] + 0.3 * ps[1])):
            r = R.setdefault("deep" if s == 0 else "broad", {}).setdefault(name, {"all": [], "all+sk": [], "mid_hit": []})
            r["all"].append(bacc_null(p.log(), y)[0]); r["all+sk"].append(bacc_null(sinkhorn_p(p, T, u).log(), y)[0])
            r["mid_hit"].append(float((p[mid].topk(10, 1).indices == y[mid][:, None]).any(1).float().mean()))
out = {g: {n: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for n, d in G.items()} for g, G in R.items()}
print(json.dumps(out)); json.dump(out, open("runs/clip_view.json", "w"), indent=1)
