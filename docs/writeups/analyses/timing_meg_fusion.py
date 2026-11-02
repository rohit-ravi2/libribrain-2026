"""Timing-only and timing + MEG in the holdout's epoch format (leakage paper, Section 4.1). No held-out labels are used.

Two local analogues of the holdout's sentence epochs, from the Deep subject's own recordings:
  sent59  every sentence of 5 to 9 words (vocabulary and other words), framed as a holdout epoch: relative onsets,
          gap to the next onset, last word's gap = 1 s to the epoch end, position, length, epoch duration. Only
          vocabulary words are scored. The holdout's sentences are all-vocabulary; these are prose.
  runs5   maximal runs of >= 5 consecutive vocabulary words inside a sentence, cut to <= 9 words (closest in content,
          but rare in prose: tens of rows, reported for completeness).
Test rows: the local evaluation blocks (Q3/Q4 of the shared stimulus). Timing model: gradient boosting on the same
features, trained on all other subj0 sessions, 3 seeds. MEG posteriors: a timing-free checkpoint (rel=0) run over the
ordinary sentence segments of Q3/Q4 (prose context windows, not clipped to the epoch: optimistic for MEG), CE readout,
tau 0.5. Fusion: log P = log P_meg + w log P_timing; w chosen on one block and scored on the other.
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


def epochs(W, mask, fmt):
    """(word indices, features) of holdout-format epochs; only vocabulary words are returned (features use all words)."""
    idx = np.flatnonzero(mask); idx = idx[np.lexsort((W.onset[idx], W.sent[idx]))]
    out_i, out_x = [], []
    def add(ch, score):
        on = W.onset[ch]; end = on[-1] + 1.0; n = len(ch); nxt = np.append(on[1:], end) - on
        for t in range(n):
            if score[t]: out_i.append(ch[t]); out_x.append([on[t] - on[0], nxt[t], t, n, n - 1 - t, end - on[0]])
    for s in np.split(idx, np.flatnonzero(np.diff(W.sent[idx])) + 1):
        if fmt == "sent59":
            if 5 <= len(s) <= 9: add(s, W.label[s] >= 0)
            continue
        k = 0
        while k < len(s):
            if W.label[s[k]] < 0: k += 1; continue
            j = k
            while j < len(s) and W.label[s[j]] >= 0: j += 1
            for c in range(k, j, 9):
                ch = s[c:min(c + 9, j)]
                if len(ch) >= 5: add(ch, np.ones(len(ch), bool))
            k = j
    return np.array(out_i, int), np.array(out_x, float).reshape(-1, 6)


# ---- MEG posteriors on Q3/Q4 of the shared sessions (computed once)
all0 = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz")]
Wt = Words(runs=[b for b in all0 if not any(s in b for s in SH)], load_meg=False)
W = Words(runs=[b for b in all0 if any(s in b for s in SH)], load_meg=True); q = W.quarters()
M = Model(CK); assert not M.a.get("rel", 0), "checkpoint must be timing-free"
# blocks: Q3/Q4 for a same-day checkpoint; for a leave-day-out checkpoint (args.exclude covers both shared sessions)
# every word of each held-out session is MEG-novel, so each session is a block
DAY = bool(M.a.get("exclude")) and all(__import__("re").search(M.a["exclude"], b) for b in W.bases)
BM = ({"ses11": W.ses[W.run] == "ses-11_task-Sherlock1", "ses12": W.ses[W.run] == "ses-12_task-Sherlock1"} if DAY else {"Q3": q == "Q3", "Q4": q == "Q4"})
BK = list(BM)
MEG = {}
for blk in BK:
    P = predict(M, W, segments(W, BM[blk], M.a["maxw"]), {0: M.A[M.sid[0]]} if 0 in M.sid else {})
    lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5).cpu()
    MEG[blk] = ({int(i): r for r, i in enumerate(P["idx"])}, np.log(np.clip(F.softmax(lo, 1).numpy(), 1e-9, None)))
res = {"ckpt": CK, "blocks": BK}; W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]; SEEDS = (0, 1, 2)
for fmt in ("sent59", "runs5"):
    ti, tx = epochs(Wt, np.ones(Wt.n, bool), fmt); ty = Wt.label[ti]; r = {"n_train_rows": int(len(ty))}
    clfs = [HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=sd).fit(tx, ty) for sd in SEEDS]
    blocks = {}
    for blk in BK:
        pos, lm_all = MEG[blk]; ci, cx = epochs(W, BM[blk], fmt); keep = np.array([int(i) in pos for i in ci], bool); ci, cx = ci[keep], cx[keep]
        y = W.label[ci]; lm = lm_all[[pos[int(i)] for i in ci]]; lts = []
        for c in clfs:
            Pt = np.full((len(ci), V), 1e-9); Pt[:, c.classes_] = c.predict_proba(cx); lts.append(np.log(Pt / Pt.sum(1, keepdims=True)))
        blocks[blk] = (lm, lts, y)
        r[blk] = {"n_rows": int(len(y)), "n_words": int(len(np.unique(y))), "meg": round(bacc10(lm, y), 4),
                  "timing_seeds": [round(bacc10(lt, y), 4) for lt in lts],
                  "curve_seed0": {str(w): round(bacc10(lm + w * lts[0], y), 4) for w in W_GRID}}
    cv = []
    for sel, tst in ((BK[0], BK[1]), (BK[1], BK[0])):
        for k in range(len(SEEDS)):
            lm, lts, y = blocks[sel]; wb = max(W_GRID, key=lambda w: bacc10(lm + w * lts[k], y))
            lm2, lts2, y2 = blocks[tst]; cv.append({"select": sel, "test": tst, "seed": SEEDS[k], "w": wb, "fused": round(bacc10(lm2 + wb * lts2[k], y2), 4)})
    r["crossfit"] = cv
    r["mean_meg"] = round(float(np.mean([r[b]["meg"] for b in BK])), 4)
    r["mean_timing"] = round(float(np.mean([r[b]["timing_seeds"] for b in BK])), 4)
    r["mean_fused"] = round(float(np.mean([c["fused"] for c in cv])), 4)
    r["sd_fused_over_seeds"] = round(float(np.std([np.mean([c["fused"] for c in cv if c["seed"] == sd]) for sd in SEEDS], ddof=1)), 4)
    res[fmt] = r
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(R, "docs/writeups/analyses", f"timing_meg_fusion_{TAG}.json"), "w"), indent=1)
