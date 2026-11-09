"""Timing floor without a network (plan stage 0b, 2026-11-09). How much of a word's identity is readable from the word
onsets that fall inside a fixed window [onset, onset + W) - the only timing a fixed-window isolated decoder can see
through onset-locked responses. Uses event files only (onsets, labels); no MEG.

Split: the clean screen's - train on every other subject-0 run, test on Sherlock1 ses-11 and ses-12 run-2 (one held-out
recording day and a held-out text: no 5-gram of it occurs in a training run). Rows: vocabulary words; onsets of ALL
words (vocabulary or not) form the timing pattern, since real MEG responds to every word.

Two estimators:
  1. Gradient-boosted classifier on [count of onsets in window, offsets of the next K onsets (W if absent)]:
     balanced top-1 / top-10 over the 50 words, as in clean_screen.py (mean over the two sessions).
  2. Packard (1990) conditional-set information, cross-fitted: cells C = next-onset offset in 20 ms bins (and count);
     P_C(y) estimated on train with add-1/2 smoothing; held-out information = mean log2 P_C(y)/P_0(y) in bits.
Usage: timing_floor.py  ->  timing_floor.json"""
import os, sys, glob, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; K = 4; BIN = 0.02
TEST = ("sub-0_ses-11_task-Sherlock1_run-2", "sub-0_ses-12_task-Sherlock1_run-2")


def rows(base, W):
    e = np.load(f"{D}/{base}_ev.npz", allow_pickle=True); on = e["onset_s"].astype(float); y = e["label"].astype(int)
    o = np.sort(on); X = []
    for t in on[y >= 0]:
        nxt = o[(o > t + 1e-6) & (o < t + W)] - t
        X.append([len(nxt)] + [nxt[k] if k < len(nxt) else W for k in range(K)])
    return np.array(X, float), y[y >= 0]


def bacc(P, y, k):
    top = np.argsort(-P, 1)[:, :k]; hit = (top == y[:, None]).any(1)
    return float(np.mean([hit[y == c].mean() for c in np.unique(y)]))


def packard_bits(Xtr, ytr, Xte, yte, W):
    cell = lambda X: (np.minimum(X[:, 1], W) / BIN).astype(int) * 16 + np.minimum(X[:, 0], 15).astype(int)
    ctr, cte = cell(Xtr), cell(Xte); P0 = (np.bincount(ytr, minlength=50) + 0.5) / (len(ytr) + 25)
    tab = {}
    for c in np.unique(ctr): tab[c] = np.bincount(ytr[ctr == c], minlength=50) + 0.5
    lp = [np.log2(tab[c][y] / tab[c].sum()) if c in tab else np.log2(P0[y]) for c, y in zip(cte, yte)]
    return float(np.mean(lp) - np.mean(np.log2(P0[yte]))), float(np.log2(50) - (-np.sum(P0 * np.log2(P0))))


out = {}
train = [os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz") if os.path.basename(f)[:-7] not in TEST]
for W in (0.5, 1.0, 2.0, 3.0):
    tr = [rows(b, W) for b in train]; Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    r = {"n_train": int(len(ytr))}
    for b in TEST:
        Xte, yte = rows(b, W); P = np.zeros((len(yte), 50)); P[:, clf.classes_] = clf.predict_proba(Xte)
        bits, _ = packard_bits(Xtr, ytr, Xte, yte, W)
        r[b.split("_")[1]] = {"n": int(len(yte)), "top1": round(bacc(P, yte, 1), 4), "top10": round(bacc(P, yte, 10), 4), "packard_bits_heldout": round(bits, 4)}
    r["mean"] = {m: round(float(np.mean([r[s][m] for s in ("ses-11", "ses-12")])), 4) for m in ("top1", "top10", "packard_bits_heldout")}
    out[f"W{W}"] = r; print(W, json.dumps(r["mean"]), flush=True)
json.dump(out, open(os.path.join(R, "docs/writeups/analyses/timing_floor.json"), "w"), indent=1)
