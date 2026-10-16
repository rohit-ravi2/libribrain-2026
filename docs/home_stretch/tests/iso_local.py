"""ISO-1 local check: score isolated 1 s windows (single-word segments, window masked to [0,1) s, as the holdout's 92
isolated words) on split-v2 Q3/Q4, in holdout-shaped mixes: 868 contextual rows + 92 isolated rows (<= 2 per class,
near-uniform). Compares (a) Sinkhorn over all 960 rows (the current pipeline) vs (b) + block Sinkhorn on the isolated rows.
Scored on the isolated rows (bacc@10 over present classes) and on all 960 rows. Local continuous speech has no
pre-word silence, so this checks the balancing mechanics, not the holdout's post-silence onset bias.
Usage: LB_DEV=cpu iso_local.py ckpt subj[,subj] out.json"""
import sys, os, json, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev

M = Model(sys.argv[1]); SUBS = [int(s) for s in sys.argv[2].split(",")]; OUT = sys.argv[3]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS], load_meg=True); q = W.quarters()


def sk(P, T, it=50):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P


res = {}
for s in SUBS:
    A = {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}; T = 3.0 if s == 0 else 2.0
    with torch.no_grad():
        segs = segments(W, np.isin(q, ["Q3", "Q4"]) & (W.sub == s), M.a["maxw"]); Pc = predict(M, W, segs, A)
        kc = Pc["y"] >= 0; pc = F.softmax(M.readout(Pc["ce"].to(dev), Pc["zt"].to(dev), "ce", 0.5), 1).cpu()[kc]; yc = Pc["y"][kc]
        single = [[int(i)] for i in Pc["idx"][kc.numpy()]]; Pi = predict(M, W, single, A, iso=True)
        pi = F.softmax(M.readout(Pi["ce"].to(dev), Pi["zt"].to(dev), "ce", 0.5), 1).cpu(); yi = Pi["y"]
    rng = np.random.default_rng(s); out = {"iso_raw_all": round(bacc_null(pi.log(), yi)[0], 4), "ctx_raw_all": round(bacc_null(pc.log(), yc)[0], 4)}
    acc = {k: [] for k in ("iso_joint", "iso_block", "all_joint", "all_block", "iso_distinct_top10_joint", "iso_distinct_top10_block")}
    for b in range(30):
        byc = {c: np.flatnonzero(yi.numpy() == c) for c in range(50)}
        iso = np.concatenate([rng.choice(v, min(2, len(v)), replace=False) for v in byc.values() if len(v)]); iso = rng.choice(iso, min(92, len(iso)), replace=False)
        if os.environ.get("BAL"):   # holdout-like: contextual rows near class-balanced too (<= 18 per class)
            byk = {c: np.flatnonzero(yc.numpy() == c) for c in range(50)}
            ctx = np.concatenate([rng.choice(v, min(18, len(v)), replace=False) for v in byk.values() if len(v)]); ctx = rng.choice(ctx, min(868, len(ctx)), replace=False)
        else: ctx = rng.choice(len(yc), 868, replace=False)
        nc = len(ctx); P = torch.cat([pc[ctx], pi[iso]]); Y = torch.cat([yc[ctx], yi[iso]]); J = sk(P, T); B = J.clone(); B[nc:] = sk(J[nc:], 1.0)
        for nm, X in (("joint", J), ("block", B)):
            acc[f"iso_{nm}"].append(bacc_null(X[nc:].log(), Y[nc:])[0]); acc[f"all_{nm}"].append(bacc_null(X.log(), Y)[0])
            acc[f"iso_distinct_top10_{nm}"].append(len(np.unique(X[nc:].topk(10, 1).indices.numpy())))
    out.update({k: round(float(np.mean(v)), 4) for k, v in acc.items()}); res[s] = out; print(s, json.dumps(out), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
