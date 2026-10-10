"""Post-hoc ablations of a train_v3 checkpoint on split v2 blocks: (i) relative word timing zeroed at inference,
(ii) context segments broken into single words, (iii) each readout with its label-permutation null.
Usage: ablate_v3.py runs/<tag>_ep<k>.pt"""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words
from src.v3lib import Model, segments, pack, group, dev

M = Model(sys.argv[1]); maxw = M.a["maxw"]
W = Words(load_meg=True); q = W.quarters(); scored = set(S2.EVAL + S2.VAL + S2.SEEN)


@torch.no_grad()
def run(blk, zero_rel=False, single=False):
    segs = segments(W, (q == blk) & np.isin(W.sub, list(scored)), 1 if single else maxw)
    CE, ZT, IDX = [], [], []
    for i in range(0, len(segs), 40):
        idx, gi, pad, rel = pack(W, segs[i:i + 40], rel_on=bool(M.a.get('rel', 1)))
        if zero_rel: rel = rel * 0
        Am = torch.stack([M.A[M.sid[int(s)]] if int(s) in S2.SEEN else M.Amean for s in W.sub[idx]])
        x, v = W.windows(idx); _, ce, zt = M.forward(x, v, Am, gi, pad, rel); CE.append(ce); ZT.append(zt); IDX.append(idx)
    idx = np.concatenate(IDX); y = torch.from_numpy(W.label[idx]); s = torch.from_numpy(W.sub[idx]); k = y >= 0
    out = {}
    for ro in (("ce", "cos", "ce+cos") if M.proto is not None else ("ce",)):
        lo = M.readout(torch.cat(CE), torch.cat(ZT), ro, 0.5 if ro == "ce" else 0.0).cpu()[k]
        out[ro] = {"seen": group(lo, y[k], s[k], S2.SEEN), "val_zs": group(lo, y[k], s[k], S2.VAL), "eval_zs": group(lo, y[k], s[k], S2.EVAL)}
    return out


for blk in ("Q3", "Q4"):
    for name, kw in [("full", {}), ("rel_zero", {"zero_rel": True}), ("single_word", {"single": True})]:
        r = run(blk, **kw)
        print(blk, name, {ro: {g: [round(v[0], 3), round(v[1], 3)] for g, v in rr.items()} for ro, rr in r.items()}, flush=True)
