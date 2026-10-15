"""ENS-T: late fusion of two frozen models with different target spaces (log-prob average with weight w on model B),
split-v2 Q3/Q4, Broad (identity A) + subj0 (own A), raw and + Sinkhorn (all-50 uniform). Usage: fuse_eval.py ckptA ckptB"""
import sys, os, json, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p
MA, MB = Model(sys.argv[1]), Model(sys.argv[2]); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev); u = torch.full((50,), 1 / 50)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
W = Words(runs=runs, load_meg=True); q = W.quarters(); R = {}
for blk in ("Q3", "Q4"):
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), MA.a["maxw"])
        if not segs: continue
        L = []
        for M in (MA, MB):
            P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
            L.append(F.log_softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]); y = P["y"][k]
        T = 3.0 if s == 0 else 2.0
        for w in (0.0, 0.2, 0.35, 0.5, 1.0):
            p = F.softmax((1 - w) * L[0] + w * L[1], 1); d = R.setdefault("deep" if s == 0 else "broad", {}).setdefault(w, [[], []])
            d[0].append(bacc_null(p.log(), y)[0]); d[1].append(bacc_null(sinkhorn_p(p, T, u).log(), y)[0])
out = {g: {str(w): [round(float(np.mean(v[0])), 4), round(float(np.mean(v[1])), 4)] for w, v in G.items()} for g, G in R.items()}
print("FUSE", json.dumps(out)); json.dump(out, open("runs/fuse_eval.json", "w"), indent=1)
