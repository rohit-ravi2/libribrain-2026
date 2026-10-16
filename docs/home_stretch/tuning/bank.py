"""TUNE-BANK: CPU posterior bank for post-processing knobs. FINAL-6c checkpoints (subj0-only recipe) on split-v2 Q3+Q4,
subj0 (own A0) + Broad subjects (identity A). Saves raw ce logits per model + labels + subject. CPU only (LB_DEV=cpu).
Usage: LB_DEV=cpu OMP_NUM_THREADS=4 bank.py out.npz subj_list ckpt1,ckpt2,..."""
import sys, os, glob, time
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, dev
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
out, subs, cks = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3].split(",")
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in subs], load_meg=True); q = W.quarters()
CE, Y, S = [], None, None
for ck in cks:
    M = Model(ck); t0 = time.time(); ce, ys, ss = [], [], []
    for s in subs:
        segs = segments(W, np.isin(q, ["Q3", "Q4"]) & (W.sub == s), M.a["maxw"])
        P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
        ce.append((P["ce"] - 0 * M.log_prior.cpu())[k].numpy()); ys.append(P["y"][k].numpy()); ss.append(np.full(int(k.sum()), s))
    CE.append(np.concatenate(ce)); Y = np.concatenate(ys); S = np.concatenate(ss)
    print(ck, "rows", len(Y), f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out, ce=np.stack(CE), y=Y, sub=S, log_prior=M.log_prior.cpu().numpy())   # incremental save after each model
