"""ISO-BANK: isolated-word analog on CPU. FINAL-6c seed 0, Broad subjects 1-12 (identity A), split-v2 Q3/Q4 vocab words, each word
as its OWN 1-word segment with the [0,1) s window only (as holdout2_word rows are fed). Saves ce logits in the same row order
as bank.py (segments order flattened) so rows align with bank_final6.npz model 0. Also marks sentence-initial words."""
import sys, os, glob, time
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, dev
torch.set_num_threads(4)
out, subs, ck = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in subs], load_meg=True); q = W.quarters()
M = Model(ck); CE, Y, S, FIRST = [], [], [], []
for s in subs:
    segs = segments(W, np.isin(q, ["Q3", "Q4"]) & (W.sub == s), M.a["maxw"]); flat = np.concatenate(segs)
    first = np.zeros(len(flat), bool); o = 0
    for sg in segs: first[o] = True; o += len(sg)
    P = predict(M, W, [np.array([i]) for i in flat], {s: torch.eye(306, device=dev)}, iso=True); k = P["y"] >= 0
    CE.append(P["ce"][k].numpy()); Y.append(P["y"][k].numpy()); S.append(np.full(int(k.sum()), s)); FIRST.append(first[k.numpy()])
    print(s, int(k.sum()), flush=True)
np.savez(out, ce=np.concatenate(CE), y=np.concatenate(Y), sub=np.concatenate(S), first=np.concatenate(FIRST))
