"""Adjacency-group hand rules (ideas 7, 10, 11, 13, 14, adjacent-repeat) on post-SLP posteriors. Never reads y."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, score, VOCAB
ID = {w: i for i, w in enumerate(VOCAB)}
FUNC = [ID[w] for w in "the a of to and an in for at on".split()]
ART = [ID[w] for w in "the a an".split()]
NOUN = [ID[w] for w in "time people good new".split()]
BE = [ID[w] for w in "is are was were".split()]
CONJ = [ID[w] for w in "and but".split()]
THERE = ID["there"]


def per_sent(f):
    return lambda b: [f(m.copy()) for m in b["slp"]]


def r7(thr, mult):
    def f(m):
        for t in range(1, len(m)):
            for w in FUNC:
                if m[t - 1, w] > thr and m[t, w] > thr: m[t, FUNC] *= mult; break
        return m
    return per_sent(f)


def r10(sub, gap=4):
    def f(m):
        top3 = np.argsort(-m, 1)[:, :3]
        for t in range(len(m)):
            for s in range(t - gap):  # s < t - gap  => >4 apart
                for w in set(top3[t]) & set(top3[s]): m[t, w] = max(m[t, w] - sub, 1e-6)
        return m
    return per_sent(f)


def r11(thr, add):
    def f(m):
        for t in range(len(m) - 1):
            if m[t, ART].max() > thr: m[t + 1, NOUN] += add
        return m
    return per_sent(f)


def r13(add):
    def f(m):
        if len(m) > 1 and THERE in np.argsort(-m[0])[:5]: m[1, BE] += add
        return m
    return per_sent(f)


def r14(thr, frac):
    def f(m):
        o = m.copy()
        for t in range(1, len(m) - 1):
            if m[t, CONJ].max() > thr: o[t + 1] = (1 - frac) * o[t + 1] + frac * m[t - 1]
        return o
    return per_sent(f)


def rrep(mult):
    def f(m):
        top = m.argmax(1)
        for t in range(1, len(m)):
            if top[t] == top[t - 1]: m[t, top[t]] *= mult
        return m
    return per_sent(f)


if __name__ == "__main__":
    D = load()
    tests = [("7 func-repeat", f"thr0.3 x{m}", r7(0.3, m)) for m in (0.95, 0.85, 0.7)]
    tests += [("10 far-repeat top3", f"-{s}", r10(s)) for s in (0.02, 0.05, 0.08)]
    tests += [("11 article->noun", f"thr0.35 +{a}", r11(0.35, a)) for a in (0.02, 0.04, 0.07)]
    tests += [("13 there->be", f"+{a}", r13(a)) for a in (0.03, 0.06, 0.1)]
    tests += [("14 coord copy", f"thr0.4 {f}", r14(0.4, f)) for f in (0.1, 0.2, 0.3)]
    tests += [("adj top1 repeat penalty", f"x{m}", rrep(m)) for m in (0.9, 0.7, 0.5)]
    print("identity:", score(lambda b: b["slp"], D), flush=True)
    for idea, var, r in tests:
        print(idea, "|", var, "|", score(r, D), flush=True)
