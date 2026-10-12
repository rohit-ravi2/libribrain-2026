"""External trigram LM over the 50-word vocab (Gutenberg vocab-only runs) + exact forward-backward marginals. Extracted from scripts/sim_lm2.py (verified against brute force, max err 6e-15)."""
import glob, os, re
import numpy as np
from .common import VOCAB_ID
V = 50
# ---- trigram LM from external text (vocab-only runs inside sentences; START=50 marks run start) ----
c1 = np.zeros(V); c2 = np.zeros((V + 1, V)); c3 = np.zeros((V + 1, V + 1, V))
LM_CORPUS = os.environ.get("LM_CORPUS", "gutenberg")   # gutenberg (prose) | vocab (LLM-generated vocab-only 5-9-word sentences)
_files = glob.glob("/mnt/ssd4tb/data/gutenberg/*.txt") if LM_CORPUS == "gutenberg" else [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/vocab_sentences.txt")]
for f in _files:
    txt = open(f, encoding="utf-8", errors="ignore").read().lower().replace("’", "'")
    for sent in re.split(r"[.!?;:\n]{1,}", txt):
        run = []
        for w in re.findall(r"[a-z']+", sent) + [None]:
            if w is not None and w in VOCAB_ID: run.append(VOCAB_ID[w]); continue
            if len(run) >= 2:
                p2, p1 = V, V
                for x in run: c1[x] += 1; c2[p1, x] += 1; c3[p2, p1, x] += 1; p2, p1 = p1, x
            run = []
uni = (c1 + 0.1) / (c1 + 0.1).sum()
bi = (c2 + 0.1) / (c2 + 0.1).sum(-1, keepdims=True)
tri = (c3 + 0.1) / (c3 + 0.1).sum(-1, keepdims=True)
P = 0.6 * tri + 0.3 * bi[None] + 0.1 * uni[None, None]          # P[w_{i-2}, w_{i-1}, w_i], index V = START
if os.environ.get("LM_ORDER") == "1":
    P = np.broadcast_to(uni[None, None], (V + 1, V + 1, V)).copy()   # control: frequency prior only, no context
logP = np.log(P)
print("LM tokens", int(c1.sum()), flush=True)


def marginals(ell, beta):
    """Exact forward-backward over pair states (a=w_{i-1}, b=w_i). ell: (n,V) brain log-scores. Returns (n,V) log marginals."""
    n = len(ell); E = beta * ell
    fw = [None] * n
    a0 = logP[V, V] + E[0]                                     # first word: context (START, START)
    fw[0] = np.full((V + 1, V), -np.inf); fw[0][V] = a0
    for i in range(1, n):
        # fw[i][a, b] = logsum_{z} fw[i-1][z, a] + logP[z, a, b] + E[i, b]
        prev = fw[i - 1]                                        # (V+1, V): (z, a)
        t = prev[:, :, None] + logP[:, :V, :]                   # (z, a, b)
        f = np.logaddexp.reduce(t, axis=0) + E[i][None]         # (a, b)
        fw[i] = np.full((V + 1, V), -np.inf); fw[i][:V] = f
    bw = [None] * n; bw[n - 1] = np.zeros((V + 1, V))
    for i in range(n - 2, -1, -1):
        # bw[i][z, a] = logsum_b logP[z, a, b] + E[i+1, b] + bw[i+1][a, b]
        nxt = np.full((V + 1, V), -np.inf)
        nb = bw[i + 1][:V]                                      # (a, b)
        nxt = np.logaddexp.reduce(logP[:, :V, :] + (E[i + 1][None, None] + nb[None]), axis=2)   # (z, a)
        bw[i] = nxt
    out = np.zeros((n, V))
    for i in range(n):
        j = fw[i] + bw[i]; out[i] = np.logaddexp.reduce(j, axis=0)
    return out - np.logaddexp.reduce(out, axis=1, keepdims=True)


