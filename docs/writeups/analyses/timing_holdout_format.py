"""Timing-only baseline in the holdout's own format, without touching held-out labels (leakage paper, Section 4.1).

The holdout's sentences are constructed from the 50 vocabulary words, 5 to 9 words long, and each epoch runs from the
first word's onset to one second after the last word's onset. We build the closest labelled analogue from the Deep
subject's own recordings: maximal runs of >= 5 consecutive vocabulary words inside one sentence (all words labelled),
cut to at most 9 words, framed as holdout epochs (relative onsets, gap to the next onset, last word's gap = 1 s to the
epoch end, position, length, epoch duration). Test = runs in the shared stimulus' Q3/Q4 (the local MEG evaluation
blocks), train = everything else. Two training formats are compared: (a) holdout-format runs, (b) prose sentences with
the original features (the 0.632 model of timing_only_baseline.py), scored on the same holdout-format test rows.
Seeds vary the boosting model's random state (early-stopping split). Output: timing_holdout_format.json."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from src.common import CACHE
from src import split2 as S2
from sklearn.ensemble import HistGradientBoostingClassifier

V = 50; SEEDS = (0, 1, 2)


def bacc10(P, y):
    top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))


def epoch_feats(on, end):
    n = len(on); nxt = np.append(on[1:], end) - on
    return [[on[k] - on[0], nxt[k], k, n, n - 1 - k, end - on[0]] for k in range(n)]


def load(f):
    d = np.load(f, allow_pickle=True); base = os.path.basename(f)[:-7]
    on, lab, sid, s1 = d["onset_s"], d["label"], d["sentenceidx"], d["sent_end_s"]
    q = S2.quarters(base, d["timechapter"]) if any(s in base for s in ("ses-11_task-Sherlock1", "ses-12_task-Sherlock1")) else np.full(len(on), "other", object)
    H, P = ([], [], []), ([], [], [])                       # (X, y, Q) holdout-format runs / prose sentences
    for s in np.unique(sid):
        w = np.flatnonzero(sid == s); w = w[np.argsort(on[w])]
        if len(w) < 2: continue
        rel = on[w] - on[w[0]]; nxt = np.append(on[w[1:]], s1[w[0]]) - on[w]
        for k, i in enumerate(w):
            if lab[i] >= 0:
                P[0].append([rel[k], nxt[k], k, len(w), len(w) - 1 - k, s1[w[0]] - s0(d, w)]); P[1].append(lab[i]); P[2].append(q[i])
        k = 0                                                # maximal runs of vocabulary words, split into <= 9-word chunks
        while k < len(w):
            if lab[w[k]] < 0: k += 1; continue
            j = k
            while j < len(w) and lab[w[j]] >= 0: j += 1
            run = w[k:j]
            for c in range(0, len(run), 9):
                ch = run[c:c + 9]
                if len(ch) < 5: continue
                X = epoch_feats(on[ch], on[ch[-1]] + 1.0)
                H[0].extend(X); H[1].extend(lab[ch]); H[2].extend([q[i] for i in ch])
            k = j
    return H, P


def s0(d, w):
    return d["sent_start_s"][w[0]]


Hs, Ps = zip(*[load(f) for f in sorted(glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz"))])
cat = lambda parts, i: np.concatenate([np.asarray(p[i], dtype=float if i == 0 else (int if i == 1 else object)) for p in parts if len(p[1])])
Xh, yh, Qh = (cat(Hs, i) for i in range(3)); Xp, yp, Qp = (cat(Ps, i) for i in range(3))
te = np.isin(Qh, ["Q3", "Q4"]); trh = ~te; trp = ~np.isin(Qp, ["Q3", "Q4"])
res = {"n_test_rows": int(te.sum()), "n_test_words": int(len(np.unique(yh[te]))), "n_train_holdout_format": int(trh.sum()),
       "n_train_prose": int(trp.sum())}


def fit_score(X, y, Xt, yt, seed):
    c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced", random_state=seed).fit(X, y)
    P = np.full((len(Xt), V), 1e-9); P[:, c.classes_] = c.predict_proba(Xt); return bacc10(P, yt)


for name, (X, y) in {"train_holdout_format": (Xh[trh], yh[trh]), "train_prose": (Xp[trp], yp[trp])}.items():
    sc = [fit_score(X, y, Xh[te], yh[te], s) for s in SEEDS]
    res[name] = {"mean": round(float(np.mean(sc)), 4), "sd": round(float(np.std(sc, ddof=1)), 4), "seeds": [round(x, 4) for x in sc]}
# class prior alone on the same rows (frequency ranking of the training labels): the floor timing must clear
freq = np.bincount(yh[trh], minlength=V).astype(float); res["prior_only"] = round(bacc10(np.tile(freq, (te.sum(), 1)), yh[te]), 4)
# gap (duration proxy) and position alone, holdout format
for nm, cols in (("gap_only", [1]), ("position_only", [2, 3, 4])):
    sc = [fit_score(Xh[trh][:, cols], yh[trh], Xh[te][:, cols], yh[te], s) for s in SEEDS]; res[nm] = round(float(np.mean(sc)), 4)
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_holdout_format.json"), "w"), indent=1)
