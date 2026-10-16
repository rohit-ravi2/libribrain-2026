"""ISO-0 (+ randgroup7_1s: single 1 s words grouped at random into 7-word pseudo-segments of the same block): how well does the current recipe decode an isolated 1 s word (its own 1-word segment, [0,1) s window) for subj0?
deep_c ep6 (Q3/Q4 held out, out-of-sample), subj0 own A0. Arms: full context 3 s (deploy), 1 s windows inside context segments,
single-word 1 s segments (= how holdout2_word rows are fed). Top-10 balanced acc over present classes, raw and + Sinkhorn T3."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(4)
ck = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and b.startswith("sub-0_")], load_meg=True); q = W.quarters()
M = Model(ck); A = {0: M.A[M.sid[0]]}
def sk(P, T):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(10): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
for blk in ("Q3", "Q4"):
    segs = segments(W, (q == blk) & (W.sub == 0), M.a["maxw"]); flat = np.concatenate(segs)
    g = np.random.default_rng(0); perm = g.permutation(flat); rnd = [np.sort(perm[i:i + 7]) for i in range(0, len(perm), 7)]
    ARMS = (("ctx3s", segs, False), ("ctx1s", segs, True), ("single1s", [np.array([i]) for i in flat], True), ("randgroup7_1s", rnd, True))
    if os.environ.get("ONLY_NEW"): ARMS = ARMS[2:]
    for name, sg, iso in ARMS:
        P = predict(M, W, sg, A, iso=iso); k = P["y"] >= 0
        p = F.softmax(M.readout(P["ce"], P["zt"], "ce", 0.5), 1)[k]; y = P["y"][k]
        print(blk, name, "raw", round(float(bacc_null(p.log(), y)[0]), 4), "sk3", round(float(bacc_null(sk(p.double(), 3.0).log(), y)[0]), 4), flush=True)
