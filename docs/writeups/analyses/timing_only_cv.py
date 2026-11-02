"""Timing-only baseline with large test sets (leakage paper, Section 4.1). No MEG, no held-out labels.

Every labelled vocabulary word of the Deep subject's recordings, featurised from stimulus timing alone, in two
formats: 'prose' (the original timing_only_baseline.py features: true sentence end) and 'sent59' (holdout epoch
framing, sentences of 5 to 9 words only: last word's gap = 1 s to the epoch end). Five-fold cross-validation grouped by
recording session (no session contributes to both training and test), three model seeds. Top-10 balanced accuracy,
chance 0.20. Output: timing_only_cv.json"""
import os, sys, glob, json
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, R)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from src.common import CACHE
from src.cont import Words

V = 50; SEEDS = (0, 1, 2)
W = Words(runs=[os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz")], load_meg=False)


def bacc10(P, y):
    top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))


def feats(fmt):
    idx = np.lexsort((W.onset, W.sent)); X, y, g = [], [], []
    for s in np.split(idx, np.flatnonzero(np.diff(W.sent[idx])) + 1):
        n = len(s)
        if n < 2 or (fmt == "sent59" and not 5 <= n <= 9): continue
        on = W.onset[s]; end = on[-1] + 1.0 if fmt == "sent59" else W.s1[s[0]]
        dur = end - on[0] if fmt == "sent59" else W.s1[s[0]] - W.s0[s[0]]
        nxt = np.append(on[1:], end) - on
        for t in range(n):
            if W.label[s[t]] >= 0:
                X.append([on[t] - on[0], nxt[t], t, n, n - 1 - t, dur]); y.append(W.label[s[t]]); g.append(W.ses[W.run[s[t]]])
    return np.array(X, float), np.array(y), np.array(g)


res = {}
for fmt in ("sent59", "prose"):
    X, y, g = feats(fmt); r = {"n_rows": int(len(y)), "n_sessions": int(len(np.unique(g)))}
    for name, cols in (("all", slice(None)), ("gap_only", [1]), ("position_only", [2, 3, 4])):
        sc = []
        for sd in SEEDS:
            P = np.zeros((len(y), V))
            for tr, te in GroupKFold(5).split(X, y, g):
                c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=sd).fit(X[tr][:, cols], y[tr])
                P[te][:, :] = 0; Pt = np.full((len(te), V), 1e-9); Pt[:, c.classes_] = c.predict_proba(X[te][:, cols]); P[te] = Pt
            sc.append(bacc10(P, y))
        r[name] = {"mean": round(float(np.mean(sc)), 4), "sd": round(float(np.std(sc, ddof=1)), 4), "seeds": [round(v, 4) for v in sc]}
        print(fmt, name, r[name], flush=True)
    res[fmt] = r
json.dump(res, open(os.path.join(R, "docs/writeups/analyses", "timing_only_cv.json"), "w"), indent=1); print(json.dumps(res, indent=1))
