"""Timing + MEG combined (leakage paper, Section 4.1). Does stimulus timing add to a timing-free MEG decoder?

Rows: the Deep subject's holdout-format runs in the local evaluation blocks (Q3/Q4 of the shared stimulus): maximal runs
of >= 5 consecutive vocabulary words in one sentence, chunked to <= 9 words, as in timing_holdout_format.py.
MEG posteriors: a timing-free checkpoint (rel=0) run over the ordinary sentence segments of Q3/Q4 (prose context
windows, not clipped to the run; a known optimistic difference from the holdout), CE readout with tau 0.5.
Timing posteriors: gradient boosting on holdout-format timing features, trained on all other subj0 runs.
Fusion: log P = log P_meg + w log P_timing, w chosen on one block and scored on the other (and the full curve).
Usage: LB_DEV=cpu LB_MMAP=1 timing_meg_fusion.py runs/<ckpt>.pt [tag]  ->  timing_meg_fusion_<tag>.json"""
import os, sys, glob, json
import numpy as np, torch, torch.nn.functional as F
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, R); os.chdir(R)
from sklearn.ensemble import HistGradientBoostingClassifier
from src.common import CACHE
from src.cont import Words
from src.v3lib import Model, segments, predict, dev

CK = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(CK)[:-3]; V = 50
SH = ("ses-11_task-Sherlock1", "ses-12_task-Sherlock1")


def bacc10(P, y):
    top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))


def chunks(W, mask):
    """holdout-format epochs: (word indices, features) for runs of >= 5 vocab words within one sentence, <= 9 words."""
    idx = np.flatnonzero(mask); idx = idx[np.lexsort((W.onset[idx], W.sent[idx]))]
    out_i, out_x = [], []
    for s in np.split(idx, np.flatnonzero(np.diff(W.sent[idx])) + 1):
        k = 0
        while k < len(s):
            if W.label[s[k]] < 0: k += 1; continue
            j = k
            while j < len(s) and W.label[s[j]] >= 0: j += 1
            for c in range(k, j, 9):
                ch = s[c:min(c + 9, j)]
                if len(ch) < 5: continue
                on = W.onset[ch]; end = on[-1] + 1.0; n = len(ch); nxt = np.append(on[1:], end) - on
                out_i.extend(ch); out_x.extend([[on[t] - on[0], nxt[t], t, n, n - 1 - t, end - on[0]] for t in range(n)])
            k = j
    return np.array(out_i, int), np.array(out_x, float)


# ---- timing model: all subj0 runs except the two shared-stimulus sessions' Q3/Q4 (here: the whole shared sessions)
all0 = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz")]
Wt = Words(runs=[b for b in all0 if not any(s in b for s in SH)], load_meg=False)
ti, tx = chunks(Wt, np.ones(Wt.n, bool)); ty = Wt.label[ti]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0).fit(tx, ty)

# ---- MEG posteriors on Q3/Q4 of the shared sessions
W = Words(runs=[b for b in all0 if any(s in b for s in SH)], load_meg=True); q = W.quarters()
M = Model(CK); assert not M.a.get("rel", 0), "checkpoint must be timing-free"
res = {"ckpt": CK, "n_train_timing_rows": int(len(ty))}
blocks = {}
for blk in ("Q3", "Q4"):
    m = q == blk
    P = predict(M, W, segments(W, m, M.a["maxw"]), {0: M.A[M.sid[0]]} if 0 in M.sid else {})
    lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5).cpu(); pm = F.softmax(lo, 1).numpy()
    pos = {int(i): r for r, i in enumerate(P["idx"])}
    ci, cx = chunks(W, m); keep = np.array([i in pos for i in ci]); ci, cx = ci[keep], cx[keep]
    Pm = pm[[pos[int(i)] for i in ci]]; Pt = np.full((len(ci), V), 1e-9); Pt[:, clf.classes_] = clf.predict_proba(cx); Pt /= Pt.sum(1, keepdims=True)
    blocks[blk] = (np.log(np.clip(Pm, 1e-9, None)), np.log(Pt), W.label[ci])
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
for blk, (lm, lt, y) in blocks.items():
    res[blk] = {"n_rows": int(len(y)), "n_words": int(len(np.unique(y))), "meg": round(bacc10(lm, y), 4), "timing": round(bacc10(lt, y), 4),
                "curve": {str(w): round(bacc10(lm + w * lt, y), 4) for w in W_GRID}}
cv = []
for sel, tst in (("Q3", "Q4"), ("Q4", "Q3")):
    wb = max(W_GRID, key=lambda w: res[sel]["curve"][str(w)]); cv.append({"select": sel, "test": tst, "w": wb, "fused": res[tst]["curve"][str(wb)], "meg": res[tst]["meg"]})
res["crossfit"] = cv; res["mean_meg"] = round(float(np.mean([c["meg"] for c in cv])), 4); res["mean_fused"] = round(float(np.mean([c["fused"] for c in cv])), 4)
res["mean_timing"] = round(float(np.mean([res[b]["timing"] for b in ("Q3", "Q4")])), 4)
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(R, "docs/writeups/analyses", f"timing_meg_fusion_{TAG}.json"), "w"), indent=1)
