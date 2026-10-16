"""Position rules (ideas 1,5,6): sentence-position PMI priors from TRAINING transcripts (cache *_ev.npz), never vocab_sentences."""
import sys, os, glob, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from simrule import load, score
from src.common import VOCAB_ID, norm_word, CACHE
V = 50; CLS = ["first", "second", "mid", "penult", "last"]
def cls(t, n): return 0 if t == 0 else 4 if t == n - 1 else 1 if t == 1 else 3 if t == n - 2 else 2

# training sentences (unique texts), full word sequences incl. non-vocab words
seen, C = set(), np.ones((5, V))  # add-1 smoothing
for f in sorted(glob.glob(f"{CACHE}/cont/*_ev.npz")):
    d = np.load(f, allow_pickle=True); w, s = d["word"], d["sentenceidx"]
    for k in np.unique(s[s >= 0]):
        ws = tuple(norm_word(x) for x in w[s == k]); 
        if ws in seen or len(ws) < 3: continue
        seen.add(ws); n = len(ws)
        for t, x in enumerate(ws):
            if x in VOCAB_ID: C[cls(t, n), VOCAB_ID[x]] += 1
print("train sentences", len(seen), "vocab tokens", int(C.sum() - C.size))
Pw = C.sum(0) / C.sum(); PMI = np.log(C / C.sum(1, keepdims=True) / Pw)   # log p(w|pos)/p(w)
from src.common import VOCAB
for c in (0, 4): print(CLS[c], "top:", [VOCAB[i] for i in np.argsort(-PMI[c])[:8]])
D = load()

def apply(mats, f):
    out = []
    for m in mats:
        n = len(m); L = np.log(np.clip(m, 1e-12, None)) + np.array([f(t, n) for t in range(n)]); L -= L.max(1, keepdims=True)
        e = np.exp(L); out.append(e / e.sum(1, keepdims=True))
    return out
iI = VOCAB_ID["i"]; endset = np.argsort(-PMI[4])[:10]
print("end words:", [VOCAB[i] for i in endset])
res = []
def run(name, f, side="slp"):
    if side == "slp": r = score(lambda b: apply(b["slp"], f), D)
    else:
        from simrule import slp
        cache = {}
        r = score(lambda b: slp(apply(b["raw"], f))[0], D)
    print(name, side, r, flush=True); res.append((name, side, r))
z = np.zeros(V)
for eps in (0.05, 0.1, 0.2, 0.4): run(f"pos_pmi eps={eps}", lambda t, n, e=eps: e * PMI[cls(t, n)])
for eps in (0.1, 0.2): run(f"pos_pmi eps={eps}", lambda t, n, e=eps: e * PMI[cls(t, n)], side="raw")
for b_ in (0.05, 0.1, 0.15, 0.3):
    def f(t, n, b_=b_): v = z.copy(); v[iI] = b_ if t == 0 else 0; return v
    run(f"i_initial +{b_}", f)
for b_ in (0.04, 0.08, 0.12, 0.25):
    def f(t, n, b_=b_): v = z.copy(); v[endset] = b_ if t == n - 1 else 0; return v
    run(f"end_boost +{b_}", f)
for eps in (0.05, 0.1, 0.2):   # graded end version: PMI on last row only
    run(f"end_pmi eps={eps}", lambda t, n, e=eps: e * PMI[4] if t == n - 1 else z)
