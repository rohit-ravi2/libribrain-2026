"""C1: within-subject linear positive control on the eval subjects (own fs_q data -> own block B)."""
import glob, json, os, sys
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
sele = lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None
X, y, se, be = load_sets({"e": sele})["e"]
F = normalize(X).reshape(len(X), 306, 10, 5).mean(-1).reshape(len(X), -1).numpy()
tc = np.zeros(len(X)); masks = {k: np.zeros(len(X), bool) for k in ["B", "fs_s25", "fs_s50", "fs_s100"]}
for b in set(be):
    idx = np.array([i for i, x in enumerate(be) if x == b]); m = S.block_masks(b, np.load(f"{S.CACHE}/runs/{b}.npz")["timechapter"])
    for k in masks: masks[k][idx] = m[k]
res = {g: {} for g in ["s25", "s50", "s100"]}
for s in S.EVAL_SUBJECTS:
    ms = (se == s).numpy(); B = ms & masks["B"]
    for g in res:
        tr = ms & masks["fs_" + g]
        clf = LogisticRegression(C=1e-3, class_weight="balanced", max_iter=300).fit(F[tr], y[tr].numpy())
        P = np.full((B.sum(), 50), -1e9); P[:, clf.classes_] = clf.decision_function(F[B])
        res[g][s] = bacc_at_k(torch.from_numpy(P), y[torch.from_numpy(B)])
    print(s, {g: round(res[g][s], 3) for g in res}, flush=True)
summ = {g: float(np.mean(list(v.values()))) for g, v in res.items()}
json.dump({"summary": summ, "per_subject": res}, open("runs/c1_within_linear.json", "w"), indent=1); print(json.dumps(summ))
