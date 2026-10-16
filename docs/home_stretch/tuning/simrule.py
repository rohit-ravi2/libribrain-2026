"""SIM-RULE harness: the calibrated sentence-prior simulator (SIM-1; sim_sweep.py) with the SLP kappa 20 / 10 EM iterations
already run and cached, so a post-processing rule can be scored in seconds.
Simulated holdout: Broad subjects 1-12 (FINAL-6c 5-seed bank, identity A, out-of-sample rows), 156 LLM vocab-only sentences
(results/vocab_sentences.txt) per subject x R=2 draws; each word = a random real held-out posterior row of that label.
NOTE: the sentence texts are the simulator's ground truth -> never estimate statistics from them (use training text / priors only).

Usage in a test script:
    from simrule import load, score
    D = load()          # list of blocks; each block: dict(sub, raw=[n_i x 50 per sentence], slp=[...], y=[...]) (probabilities)
    def rule(block): return [modified per-sentence matrices]   # gets the block, returns list like block["slp"]
    print(score(rule))  # paired delta vs baseline (SLP k20 -> Sinkhorn T1.75 with gamma 1), mean + SE over 24 blocks
"""
import sys, os, numpy as np, pickle
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from src.common import VOCAB_ID, VOCAB, norm_word
HERE = os.path.dirname(os.path.abspath(__file__)); CACHEF = os.path.join(HERE, "simrule_cache.pkl"); V = 50


def bacc(P, y):
    top = np.argpartition(-P, 10, 1)[:, :10]; hit = (top == y[:, None]).any(1)
    return np.mean([hit[y == c].mean() for c in np.unique(y)])


def sk(P, T=1.75, it=10, gamma=1.0):
    Q0 = np.clip(P, 1e-12, None) ** (1 / T); Q0 /= Q0.sum(1, keepdims=True); c = np.ones(V); n = len(Q0)
    for _ in range(it):
        Q = Q0 * c; Q /= Q.sum(1, keepdims=True); c *= (n / V) / Q.sum(0).clip(1e-12)
    Q = Q0 * c ** gamma; return Q / Q.sum(1, keepdims=True)


def _fb(E, A, pi):
    n = len(E); al = np.zeros((n, V)); be = np.ones((n, V)); c = np.zeros(n)
    al[0] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n): al[t] = (al[t - 1] @ A) * E[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1): be[t] = (A @ (E[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True); xi = np.zeros((V, V))
    for t in range(n - 1):
        x = al[t][:, None] * A * (E[t + 1] * be[t + 1])[None]; xi += x / x.sum()
    return g, xi


def slp(raws, kappa=20.0, iters=10):
    marg = np.concatenate(raws).mean(0); Es = [r / marg for r in raws]
    A = np.full((V, V), 1 / V); pi = np.full(V, 1 / V)
    for _ in range(iters):
        XI = np.zeros((V, V)); G0 = np.zeros(V)
        for E in Es: g, xi = _fb(E, A, pi); XI += xi; G0 += g[0]
        A = (XI + kappa / V) / (XI + kappa / V).sum(1, keepdims=True); pi = (G0 + kappa / V) / (G0 + kappa / V).sum()
    out = []
    for E in Es:
        g = _fb(E, A, pi)[0] * marg; out.append(g / g.sum(1, keepdims=True))
    return out, A, pi


def build():
    B = np.load(os.path.join(HERE, "bank_final6.npz")); CE, Y, S, LP = B["ce"].astype(np.float64), B["y"], B["sub"], B["log_prior"].astype(np.float64)
    z = CE - 0.5 * LP[None, None]; z = z - z.max(-1, keepdims=True); e = np.exp(z); P_all = (e / e.sum(-1, keepdims=True)).mean(0)
    SENTS = []
    for l in open(os.path.expanduser("~/projects/libribrain-2026/results/vocab_sentences.txt")):
        ids = [VOCAB_ID.get(norm_word(w)) for w in l.split()]
        if 5 <= len(ids) <= 9 and None not in ids: SENTS.append(ids)
    rng = np.random.default_rng(0); D = []
    for s in [x for x in np.unique(S) if x != 0]:
        r_s = np.flatnonzero(S == s); p, y = P_all[r_s], Y[r_s]; byl = {c: np.flatnonzero(y == c) for c in range(50)}
        ok = [Z for Z in SENTS if all(len(byl[c]) for c in Z)]
        for d in range(2):
            chosen = [ok[i] for i in rng.choice(len(ok), 156, replace=False)]
            raws = [p[[rng.choice(byl[c]) for c in Z]] for Z in chosen]
            G, A, pi = slp(raws)
            D.append(dict(sub=int(s), raw=raws, slp=G, y=[np.array(Z) for Z in chosen], A=A, pi=pi))
        print("subj", s, flush=True)
    pickle.dump(D, open(CACHEF, "wb"))


def load():
    if not os.path.exists(CACHEF): build()
    return pickle.load(open(CACHEF, "rb"))


def score(rule, D=None, T=1.75, gamma=1.0):
    """rule(block) -> list of per-sentence matrices (applied before the final Sinkhorn). Returns dict with paired delta."""
    D = D or load(); base, new = [], []
    for b in D:
        y = np.concatenate(b["y"]); base.append(bacc(sk(np.concatenate(b["slp"]), T, gamma=gamma), y))
        M = [np.clip(m, 1e-12, None) for m in rule(b)]; M = [m / m.sum(1, keepdims=True) for m in M]
        new.append(bacc(sk(np.concatenate(M), T, gamma=gamma), y))
    base, new = np.array(base), np.array(new); d = new - base
    return dict(base=round(base.mean(), 4), new=round(new.mean(), 4), delta=round(d.mean(), 4), se=round(d.std() / np.sqrt(len(d)), 4), wins=int((d > 0).sum()), n=len(d))


if __name__ == "__main__":
    D = load(); print("identity rule:", score(lambda b: b["slp"], D)); print("no-SLP (raw):", score(lambda b: b["raw"], D))
