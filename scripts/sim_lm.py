"""LM1: simulate noisy-channel sentence decoding on holdout-style sentences (PREDICTIONS.md LM1).
Brain evidence is sampled from real out-of-sample decoder outputs; the LM prior is GPT-2 (never sees holdout data)."""
import glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from src import split as S
from src.common import VOCAB, VOCAB_ID, norm_word
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN
dev = "cuda"; rng = np.random.default_rng(0)
NMAX = int(os.environ.get("NMAX", 300))   # sentences per fold
ESVAL = "sub-0_ses-10_task-Sherlock1_run-1"

# ---- sentences: natural vocab-only runs, length 5-9 ----
runs = set()
for f in sorted(glob.glob(f"{S.CACHE}/../LibriBrain100/*/derivatives/events/sub-0_*.tsv")) or sorted(glob.glob("/mnt/ssd4tb/data/LibriBrain100/*/derivatives/events/sub-0_*.tsv")):
    w = pd.read_csv(f, sep="\t"); w = w[w.kind == "word"]; ws = [norm_word(x) for x in w.segment]; sid = list(w.sentenceidx.values)
    cur, cs = [], None
    for x, s in zip(ws + [None], sid + [None]):
        if x is not None and x in VOCAB_ID and (not cur or s == cs): cur.append(x); cs = s
        else:
            if 5 <= len(cur) <= 9: runs.add(" ".join(cur))
            cur = [x] if (x is not None and x in VOCAB_ID) else []; cs = s
runs = sorted(runs)
fold = {r: int(hashlib.md5(r.encode()).hexdigest(), 16) % 2 for r in runs}
print("sentences", len(runs), flush=True)


# ---- brain evidence pools ----
def pool(ckpt, X, use_meanA):
    ck = torch.load(ckpt, map_location=dev); a = ck.get("args", {})
    m = WordCNN(d=a.get("d", 256), p=a.get("p", 0.2)).to(dev); m.load_state_dict(ck["model"]); m.eval()
    A = None
    if use_meanA:
        subs = ck["subs"]; A = ck["A"][[subs.index(s) for s in subs if s != 0]].to(dev).mean(0)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            x = normalize(X[i:i + 2048].to(dev))
            if A is not None: x = torch.einsum("ij,bjt->bit", A, x)
            out.append(torch.log_softmax(m(x).float(), 1) - ck["log_prior"].to(dev))   # tau = 1
    return torch.cat(out).cpu(), ck["log_prior"].cpu()


D = load_sets({"X": lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None,
               "S": lambda b, s, ses, t: (np.ones(len(t), bool) if b == ESVAL else S.block_masks(b, t)["B"]) if s == 0 else None})
pools = {}
for name, ck, meanA in [("X", "runs/b5a.pt", True), ("S", "runs/b0.pt", False)]:
    X, y, _, _ = D[name]; L, lp = pool(ck, X, meanA)
    pools[name] = ({c: L[y == c] for c in range(50) if (y == c).any()}, lp)
    print(name, "windows", len(X), "classes covered", len(pools[name][0]), flush=True)
del D

# ---- GPT-2 sentence scorer ----
tok = AutoTokenizer.from_pretrained("gpt2"); lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
LMW = [("I" if norm_word(v) == "i" else norm_word(v)) for v in VOCAB]


@torch.no_grad()
def lm_logprob(sents):
    enc = tok([tok.eos_token + " " + " ".join(s) for s in sents], return_tensors="pt", padding=True)
    ids = enc.input_ids.to(dev); att = enc.attention_mask.to(dev)
    lo = torch.log_softmax(lm(ids, attention_mask=att).logits[:, :-1].float(), -1)
    tl = lo.gather(-1, ids[:, 1:, None]).squeeze(-1) * att[:, 1:]
    return tl.sum(1).cpu()


tok.pad_token = tok.eos_token


def decode(ell, true, beta, delta, lp, oracle=False, sweeps=2):
    """ell: (n,50) brain scores. Returns (n,50) final candidate scores."""
    n = len(ell); cur = [int(ell[i].argmax()) for i in range(n)] if not oracle else list(true)
    sc = ell.clone()
    for _ in range(sweeps):
        for i in range(n):
            ctx = list(true) if oracle else cur
            cands = [[LMW[c] if j == i else LMW[ctx[j]] for j in range(n)] for c in range(50)]
            lmlp = lm_logprob(cands)
            sc[i] = beta * ell[i] + lmlp - delta * lp
            if not oracle: cur[i] = int(sc[i].argmax())
        if oracle: break
    return sc


results = {}
for name in ["X", "S"]:
    P, lp = pools[name]
    sents = [r.split() for r in runs if all(VOCAB_ID[w] in P for w in r.split())]
    by = {f_: [s for s in sents if fold[" ".join(s)] == f_] for f_ in (0, 1)}
    by = {f_: [ss[j] for j in np.random.default_rng(10 + f_).permutation(len(ss))[:NMAX]] for f_, ss in by.items()}   # random, not alphabetical (2026-10-08 fix)
    samp = {}
    for f_, ss in by.items():   # fixed brain samples per sentence
        samp[f_] = [(torch.stack([P[VOCAB_ID[w]][rng.integers(len(P[VOCAB_ID[w]]))] for w in s]), [VOCAB_ID[w] for w in s]) for s in ss]
    def score(f_, cfg):
        S_, Y_ = [], []
        for ell, true in samp[f_]:
            S_.append(ell if cfg is None else decode(ell, true, *cfg[:2], lp, oracle=cfg[2])); Y_ += true
        return bacc_at_k(torch.cat(S_), torch.tensor(Y_))
    t0 = time.time(); r = {"n_sent": {f_: len(v) for f_, v in by.items()}, "noLM": {f_: score(f_, None) for f_ in by}}
    grid = [(b, d) for b in (0.5, 1.0, 2.0, 4.0) for d in (0.0, 0.5, 1.0)]
    dev_scores = {f"{b},{d}": score(0, (b, d, False)) for b, d in grid}
    best = max(dev_scores, key=dev_scores.get); b, d = map(float, best.split(","))
    r.update(dev_grid=dev_scores, best=best, LM_test=score(1, (b, d, False)),
             oracle_test=score(1, (b, d, True)), t=round(time.time() - t0))
    r["gain_test"] = r["LM_test"] - r["noLM"][1]; r["oracle_gain_test"] = r["oracle_test"] - r["noLM"][1]
    results[name] = r; print(name, json.dumps({k: v for k, v in r.items() if k != "dev_grid"}), flush=True)
    json.dump(results, open("runs/lm1_sim.json", "w"), indent=1)
