"""XR-1: cross-recipe ensembling for Deep. Width and stretch do not stack inside one model (dwstr_s0 0.739 vs width 0.742,
stretch 0.737), so test whether they stack as ensemble members. subj0 split-v2 Q3/Q4 (same-day), A0, ce tau 0.5, +
Sinkhorn T3 all-50. Compares equal-size pairs: same-recipe 2-seed vs cross-recipe, and 4-member mixes.
Usage: LB_DEV=cpu xrecipe_ens.py out.json"""
import sys, os, json, glob, itertools
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev

R = os.path.expanduser("~/projects/libribrain-2026/runs/"); NAMES = ["deep_c", "deepc_s1", "dw_s0", "dw_s1", "stretch_s0", "stretch_s1", "dwstr_s0"]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and b.startswith("sub-0_")], load_meg=True); q = W.quarters()


def sk(P, T=3.0, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P


CACHEF = os.environ.get("POSTF"); POST = torch.load(CACHEF) if CACHEF else {}
for nm in ([] if CACHEF else NAMES):
    f = R + nm + "_ep7.pt"
    if not os.path.exists(f): continue
    M = Model(f); out = []
    for b in ("Q3", "Q4"):
        segs = segments(W, (q == b) & (W.sub == 0), M.a["maxw"])
        with torch.no_grad(): P = predict(M, W, segs, {0: M.A[M.sid[0]]})
        k = P["y"] >= 0; order = np.argsort(P["idx"][k.numpy()])
        out.append((F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k][order], P["y"][k][order]))
    POST[nm] = out; print("pred", nm, flush=True)


torch.save(POST, os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "xrecipe_post.pt"))
TT = float(os.environ.get("TT", "3")); GEO = bool(os.environ.get("GEO"))


def mix(ps):
    if GEO: L = torch.stack([p.clamp_min(1e-12).log() for p in ps]).mean(0); return F.softmax(L, 1)
    return torch.stack(ps).mean(0)


def score(members):
    return round(float(np.mean([bacc_null(sk(mix([POST[m][j][0] for m in members]), TT).log(), POST[members[0]][j][1])[0] for j in (0, 1)])), 4)


res = {"single": {m: score([m]) for m in POST}}
fam = {"c": ["deep_c", "deepc_s1"], "w": ["dw_s0", "dw_s1"], "s": ["stretch_s0", "stretch_s1"]}
res["pairs"] = {"+".join(p): score(list(p)) for p in itertools.combinations([m for m in POST if m != "dwstr_s0"], 2)}
res["same_recipe_pairs"] = {k: res["pairs"]["+".join(v)] for k, v in fam.items()}
cross = [score([a, b]) for a in fam["w"] for b in fam["s"]]
res["w+s_cross_pair_mean"] = round(float(np.mean(cross)), 4)
res["four"] = {"w2+s2": score(fam["w"] + fam["s"]), "c2+w2": score(fam["c"] + fam["w"]), "c2+s2": score(fam["c"] + fam["s"]), "c2+w2+s2": score(fam["c"] + fam["w"] + fam["s"])}
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1], "w"), indent=1)
