"""Ensemble of train_v3 runs on split v2: per fold, average softmax of each run's fold-selected checkpoint (own readout/tau)
over the test block. Usage: ens_v3.py tagA,tagB[,...] [out_tag]"""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words
from src.v3lib import Model, segments, predict, group

tags = sys.argv[1].split(","); out_tag = sys.argv[2] if len(sys.argv) > 2 else "ens_" + "_".join(tags)
R = {t: json.load(open(f"runs/{t}.json")) for t in tags}; maxw = R[tags[0]]["args"]["maxw"]
W = Words(load_meg=True); q = W.quarters(); scored = set(S2.EVAL + S2.VAL + S2.SEEN)
res = {}
for f, (sel, test) in S2.FOLDS.items():
    segs = segments(W, (q == test) & np.isin(W.sub, list(scored)), maxw)
    P = None
    for t in tags:
        fd = R[t]["folds"][f]; M = Model(f"runs/{t}_ep{fd['epoch']}.pt")
        pr = predict(M, W, segs, {s: M.A[M.sid[s]] for s in S2.SEEN})
        lo = M.readout(pr["ce"].cuda(), pr["zt"].cuda(), fd["readout"], fd["tau"]).cpu()
        p = F.softmax(lo, 1); P = p if P is None else P + p; y, s = pr["y"], pr["sub"]
        print(f, t, "single deploy", round(S2.deploy(group(lo[y >= 0], y[y >= 0], s[y >= 0], S2.SEEN)[0], group(lo[y >= 0], y[y >= 0], s[y >= 0], S2.EVAL)[0]), 4), flush=True)
    k = y >= 0; lo = P.log()[k]
    seen, zs = group(lo, y[k], s[k], S2.SEEN), group(lo, y[k], s[k], S2.EVAL)
    res[f] = {"seen": seen, "eval_zs": zs, "deploy": S2.deploy(seen[0], zs[0]), "deploy_null": S2.deploy(seen[1], zs[1])}
    print(f, "ENSEMBLE", json.dumps(res[f]), flush=True)
summ = {k: float(np.mean([res[f][k] for f in res])) for k in ("deploy", "deploy_null")}; summ["deploy_lift"] = summ["deploy"] - summ["deploy_null"]
json.dump({"tags": tags, "folds": res, "summary": summ}, open(f"runs/{out_tag}.json", "w"), indent=1); print("SUMMARY", json.dumps(summ))
