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


for blk in (("Q3", "Q4") if os.environ.get("NEIGHBOUR") != "1" else ()):
    for name, kw in [("full", {}), ("rel_zero", {"zero_rel": True}), ("single_word", {"single": True})]:
        r = run(blk, **kw)
        print(blk, name, {ro: {g: [round(v[0], 3), round(v[1], 3)] for g, v in rr.items()} for ro, rr in r.items()}, flush=True)


@torch.no_grad()
def neighbour_random(blk):
    """Structure kept (segment length, positions), but for each scored word all OTHER words' windows are replaced by
    random windows of other words from the same subject and block. Each word is scored from the copy where it is real."""
    rng = np.random.default_rng(0)
    segs = segments(W, (q == blk) & np.isin(W.sub, list(scored)), maxw)
    pool = {s: np.flatnonzero((q == blk) & (W.sub == s)) for s in scored}
    CE, ZT, IDX = [], [], []
    for i in range(0, len(segs), 6):
        copies, keep_pos = [], []
        for s_ in segs[i:i + 6]:
            L = len(s_); sub = int(W.sub[s_[0]])
            for j in range(L):
                c = rng.choice(pool[sub], L); c[j] = s_[j]; copies.append(c); keep_pos.append(j)
        idx, gi, pad, rel = pack(W, copies, rel_on=False)
        Am = torch.stack([M.A[M.sid[int(s)]] if int(s) in S2.SEEN else M.Amean for s in W.sub[idx]])
        x, v = W.windows(idx); _, ce, zt = M.forward(x, v, Am, gi, pad, rel)
        # pick the real word of each copy
        gi_t = torch.as_tensor(gi); sel = torch.stack([gi_t[k, j] for k, j in enumerate(keep_pos)])
        CE.append(ce[sel]); ZT.append(zt[sel]); IDX.append(idx[sel.numpy()])
    idx = np.concatenate(IDX); y = torch.from_numpy(W.label[idx]); s = torch.from_numpy(W.sub[idx]); k = y >= 0
    out = {}
    for ro in (("ce", "cos", "ce+cos") if M.proto is not None else ("ce",)):
        lo = M.readout(torch.cat(CE), torch.cat(ZT), ro, 0.5 if ro == "ce" else 0.0).cpu()[k]
        out[ro] = {"seen": group(lo, y[k], s[k], S2.SEEN), "val_zs": group(lo, y[k], s[k], S2.VAL), "eval_zs": group(lo, y[k], s[k], S2.EVAL)}
    return out


if os.environ.get("NEIGHBOUR") == "1":
    for blk in ("Q3", "Q4"):
        r = neighbour_random(blk)
        print(blk, "neighbour_random", {ro: {g: [round(v[0], 3), round(v[1], 3)] for g, v in rr.items()} for ro, rr in r.items()}, flush=True)
