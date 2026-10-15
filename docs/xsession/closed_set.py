"""CLOSED-1: ceiling of sentence-level retrieval from a candidate pool that contains the truth.
Real held-out posteriors (deep_c ep6, split-v2 Q3/Q4; 12 Broad identity A + subj0 A0). Rows grouped into pseudo-sentences
of n=7 consecutive rows (segment order). Candidate pool = truth + (K-1) distractors:
  'rand'  : label sequences sampled from the block's empirical label distribution
  'near1/2/3': truth with 1/2/3 positions substituted (sampled from the empirical distribution) -- grammatical near-misses.
Sentence score = sum_i log p_i(c_i); posterior over pool = softmax(score); row marginal q_i; final = 0.5 q + 0.5 p, then
Sinkhorn all-50 (T2 Broad / T3 subj0). Reports retrieval top-1 and bacc vs baseline (p + Sinkhorn)."""
import sys, os, glob, json
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(10); rng = np.random.default_rng(0)
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]], load_meg=True); q = W.quarters()
N = 7; R = {}
for blk in ("Q3", "Q4"):
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k].double(); y = P["y"][k].numpy(); n = (len(y) // N) * N
        p, y = p[:n], y[:n]; lp = p.clamp_min(1e-12).log().numpy(); T = 3.0 if s == 0 else 2.0; g = "deep" if s == 0 else "broad"
        freq = np.bincount(y, minlength=50) / len(y)
        base = bacc_null(sk(p, T).log(), torch.from_numpy(y))[0]; R.setdefault((g, "base"), []).append(base)
        for kind in ("rand", "near1", "near2", "near3"):
            for K in (10, 100, 1000, 10000):
                if kind != "rand" and K > 1000: continue
                Q = np.zeros((n, 50)); hit = 0
                for j in range(0, n, N):
                    t = y[j:j + N]; L = lp[j:j + N]
                    if kind == "rand": C = rng.choice(50, size=(K - 1, N), p=freq)
                    else:
                        m = int(kind[-1]); C = np.repeat(t[None], K - 1, 0)
                        for r_ in range(K - 1): pos = rng.choice(N, m, replace=False); C[r_, pos] = rng.choice(50, m, p=freq)
                    C = np.vstack([t[None], C]); sc = L[np.arange(N)[None], C].sum(1); w = np.exp(sc - sc.max()); w /= w.sum()
                    hit += int(np.argmax(sc) == 0)
                    for i in range(N): np.add.at(Q[j + i], C[:, i], w)
                fin = torch.from_numpy(0.5 * Q + 0.5 * p.numpy())
                R.setdefault((g, f"{kind}_K{K}"), []).append(bacc_null(sk(fin, T).log(), torch.from_numpy(y))[0])
                R.setdefault((g, f"{kind}_K{K}_retr"), []).append(hit / (n // N))
        print(blk, s, flush=True)
out = {f"{g}/{k}": round(float(np.mean(v)), 4) for (g, k), v in R.items()}
print("CLOSED", json.dumps(out)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "closed_set.json"), "w"), indent=1)
