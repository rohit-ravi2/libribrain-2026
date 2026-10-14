"""PURITY-1: can same-word rows be found without labels? kNN (k=10) label purity of vocab rows within a subject-block in
several label-free spaces built from the subject's own raw MEG (no model): post-onset windows [0, L] s, per-window
normalised, projected on the block's top-K spatial PCs, optionally z-scored per feature over the block; cosine. Compared
with chance (sum of squared class frequencies) and with the model's logit/embedding purity (ORACLE-1 O4).
Usage: purity_scan.py [SUBJ=..] [BLOCKS=..]"""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize, ON, FS
from src.common import CACHE
torch.set_num_threads(8)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = [int(x) for x in os.environ.get("SUBJ", ",".join(map(str, BROAD + [0]))).split(",")]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters()
res = {}
for blk in os.environ.get("BLOCKS", "Q3,Q4").split(","):
    for s in SUBS:
        idx = np.flatnonzero((q == blk) & (W.sub == s) & (W.label >= 0))
        if len(idx) < 100: continue
        x, v = W.windows(idx); x = normalize(x, v).numpy(); y = W.label[idx]
        freq = np.bincount(y, minlength=50) / len(y); r = {"chance": float((freq ** 2).sum())}
        for L in (0.4, 0.8):
            seg = x[:, :, ON:ON + int(L * FS)]                               # (n, 306, t) post-onset
            C = np.einsum("nct,ndt->cd", seg, seg) / (seg.shape[0] * seg.shape[2]); ev, U = np.linalg.eigh(C); U = U[:, ::-1]
            for K in (10, 40):
                F_ = np.einsum("ck,nct->nkt", U[:, :K], seg).reshape(len(idx), -1)
                for z in (0, 1):
                    G = (F_ - F_.mean(0)) / (F_.std(0) + 1e-6) if z else F_
                    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9); S = G @ G.T; np.fill_diagonal(S, -9)
                    nn_ = np.argsort(-S, 1)[:, :10]; r[f"L{L}_K{K}_z{z}"] = float((y[nn_] == y[:, None]).mean())
        res[f"{blk}/{s}"] = r; print(blk, s, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
keys = list(next(iter(res.values())).keys())
for grp, mem in (("broad", BROAD), ("deep", [0])):
    rows = [v for k, v in res.items() if int(k.split("/")[1]) in mem]
    if rows: print("MEAN", grp, json.dumps({k: round(float(np.mean([r_[k] for r_ in rows])), 4) for k in keys}), flush=True)
json.dump(res, open("runs/purity_scan.json", "w"), indent=1)
