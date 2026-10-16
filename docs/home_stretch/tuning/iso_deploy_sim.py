"""ISO-DEPLOY: subj0 holdout-shaped simulation of replacing the isolated-word rows' posteriors with a dedicated 1 s decoder (B0),
under the DEPLOYED joint Sinkhorn (T3 over all 960 rows). Context rows: deep_c ep6 3 s context predictions (Q3+Q4, out-of-sample).
Isolated rows: 92 Q4 words (B0 held-out block) as lone 1 s windows, disjoint from the context rows. 30 draws, natural and
class-balanced isolated draws. Score: pooled top-10 balanced accuracy over all 960 rows (as the LB) and on the isolated rows."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src.split import SHARED
from src.cont import Words, ON, FS
from src.common import CACHE
from src.data import normalize as norm_b0
from src.models import WordCNN
from src.v3lib import Model, segments, predict, dev
torch.set_num_threads(4)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and b.startswith("sub-0_")], load_meg=True); q = W.quarters()
ck = torch.load(os.path.expanduser("~/projects/libribrain-2026/runs/b0.pt"), map_location="cpu", weights_only=False)
B0 = WordCNN(d=ck.get("args", {}).get("d", 256), p=ck.get("args", {}).get("p", 0.2)); B0.load_state_dict(ck["model"]); B0.eval()
adj = ck.get("tau", 0.75) * ck["log_prior"].float(); M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt"))
A0 = {0: M.A[M.sid[0]]}
segs = segments(W, np.isin(q, ["Q3", "Q4"]) & (W.sub == 0), M.a["maxw"])
P = predict(M, W, segs, A0); k = (P["y"] >= 0).numpy()
PC = F.softmax(M.readout(P["ce"], P["zt"], "ce", 0.5), 1)[k].double().numpy(); YC = P["y"][k].numpy(); IC = P["idx"][k]
s4 = segments(W, (q == "Q4") & (W.sub == 0), M.a["maxw"]); flat = np.concatenate(s4)
P = predict(M, W, [np.array([i]) for i in flat], A0, iso=True); k = (P["y"] >= 0).numpy()
PV = F.softmax(M.readout(P["ce"], P["zt"], "ce", 0.5), 1)[k].double().numpy(); YI = P["y"][k].numpy(); II = P["idx"][k]
x, v = W.windows(II, iso=np.ones(len(II), bool))
with torch.no_grad(): PB = torch.softmax(B0(norm_b0(x[:, :, ON:ON + FS].float())).float() - adj, 1).double().numpy()
def bacc(P, y):
    top = np.argpartition(-P, 10, 1)[:, :10]; hit = (top == y[:, None]).any(1); return np.mean([hit[y == c].mean() for c in np.unique(y)])
def sk(P, T, it=10):
    Q = np.clip(P, 1e-12, None) ** (1 / T); Q /= Q.sum(1, keepdims=True); n = len(Q)
    for _ in range(it): Q = Q / Q.sum(0, keepdims=True) * (n / 50); Q /= Q.sum(1, keepdims=True)
    return Q
rng = np.random.default_rng(0); R = {}
for mode in ("natural", "balanced"):
    for d in range(30):
        p = rng.permutation(len(YI))
        iso = p[:92] if mode == "natural" else np.concatenate([p[YI[p] == c][:2] for c in np.unique(YI)])[:92]
        used = set(II[iso].tolist()); okc = np.array([j for j in range(len(YC)) if IC[j] not in used]); ctx = rng.choice(okc, 868, replace=False)
        for name, PI in (("v3", PV), ("B0", PB), ("mix0.5", (PV + PB) / 2), ("mix0.75", 0.25 * PV + 0.75 * PB)):
            X = np.concatenate([PC[ctx], PI[iso]]); y = np.concatenate([YC[ctx], YI[iso]])
            Q = sk(X, 3.0); R.setdefault((mode, name), []).append((bacc(Q, y), bacc(Q[868:], y[868:])))
for mode in ("natural", "balanced"):
    ref = np.array(R[(mode, "v3")])
    for name in ("v3", "B0", "mix0.5", "mix0.75"):
        a = np.array(R[(mode, name)]); d = a[:, 0] - ref[:, 0]
        print(f"{mode:8s} iso={name:8s} all-960 {a[:,0].mean():.4f} (d {d.mean():+.4f} SE {d.std(ddof=1)/np.sqrt(len(d)):.4f}) | iso rows {a[:,1].mean():.4f}", flush=True)
