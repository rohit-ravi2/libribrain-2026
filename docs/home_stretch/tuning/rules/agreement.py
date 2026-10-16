"""Agreement-group hard-coded rules (ideas 2, 12, 9, 8) scored on simrule. Never reads block['y']."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from simrule import load, score
from src.common import VOCAB
I = {w: i for i, w in enumerate(VOCAB)}
ids = lambda ws: [I[w] for w in ws]
PAIRS = {  # trigger -> agreeing targets (idea 2)
    "they": ["are", "were", "have"], "he": ["is", "was", "has"], "she": ["is", "was", "has"], "it": ["is", "was", "has"],
    "i": ["am", "was", "have", "think"], "we": ["are", "were", "have"],
    "will": ["be", "have", "think"], "can": ["be", "have", "think"], "do": ["be", "have", "think"],
    "not": ["be", "have", "think", "do"]}
PAIRS = {I[k]: ids(v) for k, v in PAIRS.items()}

def norm(M): M = np.clip(M, 1e-12, None); return M / M.sum(1, keepdims=True)

def r_pairs(thr, s, soft=False, maxd=3, maxp=5):
    def f(b):
        out = []
        for P in b["slp"]:
            Q = P.copy(); n = len(P); cand = []
            for t in range(n):
                for w, tg in PAIRS.items():
                    p = P[t, w]
                    if (soft and p > 0.05) or p > thr: cand.append((p, t, w))
            cand.sort(reverse=True)
            for p, t, w in cand[:maxp]:
                wt = p if soft else 1.0
                for d in range(1, maxd + 1):
                    if t + d < n: Q[t + d, PAIRS[w]] += s * wt
            out.append(norm(Q))
        return out
    return f

SG, PL = ids(["he", "she", "it"]), ids(["we", "they"])
SGV, PLV, IV = ids(["is", "was", "has"]), ids(["are", "were", "have"]), ids(["am", "was", "have"])
ALLV = sorted(set(SGV + PLV + IV))
def r_lock(s_up, s_dn, thr=0.5):
    def f(b):
        out = []
        for P in b["slp"]:
            Q = P.copy(); n = len(P)
            for t in range(n):
                for grp, match in ((SG, SGV), (PL, PLV), ([I["i"]], IV)):
                    if P[t, grp].max() > thr:
                        mis = [v for v in ALLV if v not in match]
                        Q[t + 1:, match] += s_up; Q[t + 1:, mis] -= s_dn
            out.append(norm(Q))
        return out
    return f

NEGT = ids(["be", "have", "think", "do"])
def r_neg(thr, s):
    def f(b):
        out = []
        for P in b["slp"]:
            Q = P.copy(); n = len(P)
            for t in np.flatnonzero(P[:, I["not"]] > thr):
                for d in (1, 2):
                    if t + d < n: Q[t + d, NEGT] += s
            out.append(norm(Q))
        return out
    return f

AUX = ids(["is", "was", "are", "were", "do", "can", "will", "have", "has", "had", "am"])
PRON = ids(["i", "we", "he", "she", "it", "they", "there"])
def r_q(thr, s):
    def f(b):
        out = []
        for P in b["slp"]:
            Q = P.copy()
            if P[0, AUX].sum() > thr: Q[1:4, PRON] += s
            out.append(norm(Q))
        return out
    return f

if __name__ == "__main__":
    D = load(); R = {}
    tests = {
        "pairs thr0.4 s0.03": r_pairs(0.4, 0.03), "pairs thr0.4 s0.08": r_pairs(0.4, 0.08), "pairs thr0.4 s0.15": r_pairs(0.4, 0.15),
        "pairs thr0.2 s0.08": r_pairs(0.2, 0.08), "pairs soft s0.08": r_pairs(0, 0.08, soft=True), "pairs soft s0.15": r_pairs(0, 0.15, soft=True),
        "lock +0.02/-0.01": r_lock(0.02, 0.01), "lock +0.06/-0.04": r_lock(0.06, 0.04), "lock +0.12/-0.08": r_lock(0.12, 0.08),
        "neg thr0.4 s0.03": r_neg(0.4, 0.03), "neg thr0.4 s0.1": r_neg(0.4, 0.1), "neg thr0.4 s0.2": r_neg(0.4, 0.2),
        "q thr0.25 s0.03": r_q(0.25, 0.03), "q thr0.25 s0.1": r_q(0.25, 0.1), "q thr0.25 s0.2": r_q(0.25, 0.2)}
    for k, f in tests.items():
        R[k] = score(f, D); print(k, R[k], flush=True)
    json.dump(R, open(os.path.join(HERE, "agreement.json"), "w"), indent=1)
