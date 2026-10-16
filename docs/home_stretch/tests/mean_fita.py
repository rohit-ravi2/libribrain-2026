"""MFA-1: population-mean FITA map as the input layer for the 7 zero-shot subjects (33-39).
Neighbour of two measured wins: FITA (labelled per-subject 306x306 A on a frozen trunk, +0.03..+0.07 local, 18/18) and the
fixed temporal-lobe prior for no-data subjects (+0.033, 23/24). The average of the labelled maps of many Broad subjects
estimates the systematic part of "new Broad brain -> subj0 trunk" (cohort head position, session, sensor gain), learned
from released labelled training data only (no holdout rows, no cross-subject row correspondence).
Protocol (leave-subjects-out, split-v2 Q3/Q4 of the TEST subjects only):
  fit A_j on all labelled Q1-Q4 words of each FIT subject j (frozen trunk, CE, ridge to identity, as predict_holdout_v3),
  Abar = mean_j A_j; score each TEST subject (never used in fitting) on Q3/Q4 with I, diag(temporal k3), Abar,
  (I+Abar)/2, Abar @ diag(temporal k3); raw and + Sinkhorn T2 all-50.
Trunk: a subj0-only checkpoint (broad_frac 0, so no Broad subject was ever trained on).
Usage: LB_DEV=cuda|cpu mean_fita.py ckpt fit_subjs test_subjs out.json   (STEPS env, default 300)"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p

M = Model(sys.argv[1]); FIT = [int(s) for s in sys.argv[2].split(",")]; TEST = [int(s) for s in sys.argv[3].split(",")]; OUT = sys.argv[4]
STEPS = int(os.environ.get("STEPS", "300")); I = torch.eye(306, device=dev); u = torch.full((50,), 1 / 50)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in FIT + TEST], load_meg=True); q = W.quarters()
XYZ = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")), float)
_c = XYZ - XYZ.mean(0); _lat = np.abs(_c[:, 0]) / np.abs(_c[:, 0]).max(); _low = (_c[:, 2].max() - _c[:, 2]) / np.ptp(_c[:, 2])
TEMP = torch.diag(torch.tensor(1 + 3.0 * (_lat * _low) / (_lat * _low).max(), dtype=torch.float32, device=dev))


def fit(segs, steps, seed, lr=3e-4):
    rng = np.random.default_rng(seed); D = nn.Parameter(torch.zeros(306, 306, device=dev)); opt = torch.optim.Adam([D], lr=lr)
    for _ in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]; idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, I + D, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * (D ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return (I + D).detach()


def score(segs, A):
    with torch.no_grad(): P = predict(M, W, segs, {int(W.sub[segs[0][0]]): A})
    k = P["y"] >= 0; p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]
    return round(bacc_null(p.log(), y)[0], 4), round(bacc_null(sinkhorn_p(p, 2.0, u).log(), y)[0], 4)


As = []
for j in FIT:
    t0 = time.time(); As.append(fit(segments(W, W.sub == j, M.a["maxw"]), STEPS, j)); print("fit", j, f"{time.time() - t0:.0f}s", flush=True)
Abar = torch.stack(As).mean(0); off = (Abar - I).norm().item(); print("||Abar - I||_F", round(off, 3), "mean ||A_j - I||", round(float(np.mean([(A - I).norm().item() for A in As])), 3), flush=True)
VAR = {"identity": I, "temporal_k3": TEMP, "meanA": Abar, "half_meanA": 0.5 * (I + Abar), "meanA@temporal": Abar @ TEMP}
res = {}
for s in TEST:
    for b in ("Q3", "Q4"):
        segs = segments(W, (q == b) & (W.sub == s), M.a["maxw"])
        if segs: res[f"{s}_{b}"] = {k: score(segs, A) for k, A in VAR.items()}; print(s, b, res[f"{s}_{b}"], flush=True)
mean = {k: [round(float(np.mean([r[k][i] for r in res.values()])), 4) for i in (0, 1)] for k in VAR}
wins = {k: int(sum(r[k][1] > r["identity"][1] for r in res.values())) for k in VAR}
print("MEAN [raw, +sk]", json.dumps(mean), "blocks beating identity (+sk)", wins, "of", len(res), flush=True)
json.dump({"fit": FIT, "test": TEST, "blocks": res, "mean": mean, "wins": wins}, open(OUT, "w"), indent=1)
