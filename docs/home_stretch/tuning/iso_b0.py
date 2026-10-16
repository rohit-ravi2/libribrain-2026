"""ISO-B0: a dedicated 1 s single-window decoder (B0 WordCNN, subj0-only, v1 split: block B ~ split-v2 Q4 held out) vs the
3 s context recipe fed a lone 1 s window (deep_c ep6), on subj0 Q4 single 1 s windows. Same rows, same metric (bacc_null top-10).
CPU only. Also: the fused (mean of probabilities) and Sinkhorn T3 versions."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src.split import SHARED
from src.cont import Words, ON, FS
from src.common import CACHE
from src.data import normalize as norm_b0
from src.models import WordCNN
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(4)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and b.startswith("sub-0_")], load_meg=True); q = W.quarters()
ck = torch.load(os.path.expanduser("~/projects/libribrain-2026/runs/b0.pt"), map_location="cpu", weights_only=False)
B0 = WordCNN(d=ck.get("args", {}).get("d", 256), p=ck.get("args", {}).get("p", 0.2)); B0.load_state_dict(ck["model"]); B0.eval()
adj = ck.get("tau", 0.75) * ck["log_prior"].float()
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt"))
def sk(P, T):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(10): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
for blk in ("Q4", "Q3"):
    segs = segments(W, (q == blk) & (W.sub == 0), M.a["maxw"]); flat = np.concatenate(segs)
    P = predict(M, W, [np.array([i]) for i in flat], {0: M.A[M.sid[0]]}, iso=True); k = (P["y"] >= 0).numpy()
    pv3 = F.softmax(M.readout(P["ce"], P["zt"], "ce", 0.5), 1)[k]; y = P["y"][k]; idx = P["idx"][k]
    x, v = W.windows(idx, iso=np.ones(len(idx), bool)); x1 = x[:, :, ON:ON + FS].float()
    with torch.no_grad(): pb0 = torch.softmax(B0(norm_b0(x1)).float() - adj, 1)
    out = {"v3_single1s": pv3, "B0_1s": pb0, "mean": (pv3 + pb0) / 2}
    print(blk, "(Q3 is in B0's training; Q4 ~ B0 block B, held out)", {kk: (round(float(bacc_null(vv.log(), y)[0]), 4), round(float(bacc_null(sk(vv.double(), 3.0).log(), y)[0]), 4)) for kk, vv in out.items()}, "null", round(float(bacc_null(pv3.log(), y)[1]), 4), flush=True)
