"""TIMING-ONLY baseline: how much of word identity is recoverable from stimulus metadata alone (no MEG)?
Per word, only what a holdout row exposes: onset relative to the sentence start, gap to the next word's onset (or to
sentence end for the last word), position in sentence, number of words, sentence duration. Train on subject 0's
training sentences (split-v2 train mask: everything except the shared stimulus' Q3/Q4), score vocabulary words of the
shared stimulus' Q3/Q4 (the same blocks as the local MEG evaluation). Metric: top-10 balanced accuracy, chance 0.20.
Two training regimes: (a) prose sentences as recorded; (b) vocabulary-only runs (>= 4 consecutive vocab words),
closer to the holdout's constructed sentences."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from src.common import CACHE
from src import split2 as S2
from sklearn.ensemble import HistGradientBoostingClassifier

def feats(f):
    d = np.load(f, allow_pickle=True); base = os.path.basename(f)[:-7]
    on, lab, sid, s0, s1 = d["onset_s"], d["label"], d["sentenceidx"], d["sent_start_s"], d["sent_end_s"]
    q = S2.quarters(base, d["timechapter"]) if any(s in base for s in ("ses-11_task-Sherlock1", "ses-12_task-Sherlock1")) else np.full(len(on), "other", object)
    X, y, Q = [], [], []
    for s in np.unique(sid):
        w = np.flatnonzero(sid == s); w = w[np.argsort(on[w])]; n = len(w)
        if n < 2: continue
        rel = on[w] - on[w[0]]; nxt = np.append(on[w[1:]], s1[w[0]]) - on[w]
        for k, i in enumerate(w):
            if lab[i] < 0: continue
            X.append([rel[k], nxt[k], k, n, n - 1 - k, s1[w[0]] - s0[w[0]]]); y.append(lab[i]); Q.append(q[i])
    return np.array(X, float), np.array(y), np.array(Q, object)

def bacc10(P, y):
    top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))

parts = [feats(f) for f in sorted(glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz"))]
X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); Q = np.concatenate([p[2] for p in parts])
tr = ~np.isin(Q, ["Q3", "Q4"]); te = np.isin(Q, ["Q3", "Q4"])
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=0).fit(X[tr], y[tr])
P = np.full((te.sum(), 50), 1e-9); P[:, clf.classes_] = clf.predict_proba(X[te])
res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "timing_only_bacc10": round(bacc10(P, y[te]), 4)}
# duration feature alone and position-only, to attribute
for name, cols in (("gap_only", [1]), ("position_only", [2, 3, 4])):
    c2 = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0).fit(X[tr][:, cols], y[tr])
    P2 = np.full((te.sum(), 50), 1e-9); P2[:, c2.classes_] = c2.predict_proba(X[te][:, cols]); res[name] = round(bacc10(P2, y[te]), 4)
print(json.dumps(res)); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_only_baseline.json"), "w"), indent=1)
