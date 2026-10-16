"""Build a posterior bank (deep_c ep6, split-v2 Q3/Q4) for subj0 + Broad subjects on CPU; same recipe as SIM-1."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, dev
torch.set_num_threads(4)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 6
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt"))
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))[:NB]; SUBS = [0] + BROAD
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS], load_meg=True); q = W.quarters()
out = {}
for s in SUBS:
    for blk in ("Q3", "Q4"):
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        with torch.no_grad():
            P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
            lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5).cpu()[k]
        out[f"s{s}_{blk}_lo"] = lo.numpy(); out[f"s{s}_{blk}_y"] = P["y"][k].numpy(); print(s, blk, len(lo), flush=True)
np.savez(os.path.join(os.path.dirname(__file__), "bank.npz"), **out)
