"""LM2: exact forward-backward sentence decoding with an external-text trigram LM over the 50-word vocab (CPU)."""
import glob, hashlib, json, os, re, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.common import VOCAB_ID, norm_word
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN
torch.set_num_threads(8); rng = np.random.default_rng(0); V = 50; NMAX = 300
ESVAL = "sub-0_ses-10_task-Sherlock1_run-1"

# ---- trigram LM from external text (vocab-only runs inside sentences; START=50 marks run start) ----
c1 = np.zeros(V); c2 = np.zeros((V + 1, V)); c3 = np.zeros((V + 1, V + 1, V))
for f in glob.glob("/mnt/ssd4tb/data/gutenberg/*.txt"):
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


# ---- sentences + folds (identical to LM1) ----
runs = set()
for f in sorted(glob.glob("/mnt/ssd4tb/data/LibriBrain100/*/derivatives/events/sub-0_*.tsv")):
    w = pd.read_csv(f, sep="\t"); w = w[w.kind == "word"]; ws = [norm_word(x) for x in w.segment]; sid = list(w.sentenceidx.values)
    cur, cs = [], None
    for x, s in zip(ws + [None], sid + [None]):
        if x is not None and x in VOCAB_ID and (not cur or s == cs): cur.append(x); cs = s
        else:
            if 5 <= len(cur) <= 9: runs.add(" ".join(cur))
            cur = [x] if (x is not None and x in VOCAB_ID) else []; cs = s
runs = sorted(runs); fold = {r: int(hashlib.md5(r.encode()).hexdigest(), 16) % 2 for r in runs}


def pool(ckpt, X, use_meanA):
    ck = torch.load(ckpt, map_location="cpu"); a = ck.get("args", {})
    m = WordCNN(d=a.get("d", 256), p=a.get("p", 0.2)); m.load_state_dict(ck["model"]); m.eval()
    A = None
    if use_meanA:
        subs = ck["subs"]; A = ck["A"][[subs.index(s) for s in subs if s != 0]].mean(0)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            x = normalize(X[i:i + 1024])
            if A is not None: x = torch.einsum("ij,bjt->bit", A, x)
            out.append(torch.log_softmax(m(x).float(), 1) - ck["log_prior"])
    return torch.cat(out), ck["log_prior"]


D = load_sets({"X": lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None,
               "S": lambda b, s, ses, t: (np.ones(len(t), bool) if b == ESVAL else S.block_masks(b, t)["B"]) if s == 0 else None})
results = {}
for name, ck, meanA in [("X", "runs/b5a.pt", True), ("S", "runs/b0.pt", False)]:
    X, y, _, _ = D[name]; L, lp = pool(ck, X, meanA); Pp = {c: L[y == c] for c in range(50) if (y == c).any()}
    sents = [r.split() for r in runs if all(VOCAB_ID[w] in Pp for w in r.split())]
    by = {f_: [s for s in sents if fold[" ".join(s)] == f_][:NMAX] for f_ in (0, 1)}
    samp = {f_: [(torch.stack([Pp[VOCAB_ID[w]][rng.integers(len(Pp[VOCAB_ID[w]]))] for w in s]).numpy(), [VOCAB_ID[w] for w in s]) for s in ss] for f_, ss in by.items()}
    lpn = lp.numpy()
    def score(f_, cfg):
        S_, Y_ = [], []
        for ell, true in samp[f_]:
            S_.append(torch.from_numpy(ell if cfg is None else marginals(ell, cfg[0]) - cfg[1] * lpn)); Y_ += true
        return bacc_at_k(torch.cat(S_).float(), torch.tensor(Y_))
    t0 = time.time(); r = {"noLM": {f_: score(f_, None) for f_ in by}}
    grid = [(b, d) for b in (0.5, 1, 2, 4, 8) for d in (0.0, 0.5, 1.0)]
    devg = {f"{b},{d}": score(0, (b, d)) for b, d in grid}; best = max(devg, key=devg.get); b, d = map(float, best.split(","))
    r.update(dev_grid=devg, best=best, LM_test=score(1, (b, d)), t=round(time.time() - t0)); r["gain_test"] = r["LM_test"] - r["noLM"][1]
    results[name] = r; print(name, json.dumps({k: v for k, v in r.items() if k != "dev_grid"}), flush=True)
json.dump(results, open("runs/lm2_sim%s.json" % ("_unigram" if os.environ.get("LM_ORDER") == "1" else ""), "w"), indent=1)
