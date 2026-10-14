"""INF-1: inference-only map on a v3 checkpoint (split v2 Q3/Q4). Variants: base, onset-jitter TTA, channel-dropout TTA,
per-subject BatchNorm recalibration on the subject's own unlabelled block rows, and a frozen-trunk input-layer fit (A_s)
for SEEN subjects on their own labelled training words. Every variant is followed by per-subject Sinkhorn balance.
Only each subject's own rows are used. Usage: infer_map.py runs/<ckpt>.pt"""
import sys, os, json, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words, FS
from src.v3lib import Model, segments, pack, predict, bacc_null, dev

SLIM = os.environ.get("SLIM") == "1"; OUT = os.environ.get("OUT", "runs/infer_map.json")
M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5; t0 = time.time()
W = Words(load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev); rng = np.random.default_rng(0); torch.manual_seed(0)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); A0 = M.A[M.sid[0]]
ONSET0 = W.onset.copy(); WIN0 = W.windows
trainable = np.isin(q, ["other", "Q1", "Q2"])
print("load", round(time.time() - t0), "s", flush=True)


def sinkhorn(lo, T, tgt, iters=10):
    P = F.softmax(lo / T, 1); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P.log()


def logits(segs, A, passes=None):
    """Mean softmax over TTA passes -> log-probs for vocab rows. passes: list of callables applied before predict."""
    acc = None
    for setup in (passes or [lambda: None]):
        setup(); P = predict(M, W, segs, {s: A for s in set(int(W.sub[g[0]]) for g in segs)})
        W.onset[:] = ONSET0; W.windows = WIN0
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau), 1).cpu()
        acc = p if acc is None else acc + p
        y = P["y"]
    k = y >= 0
    return (acc / len(passes or [0])).clamp_min(1e-9).log()[k], y[k]


def jitter(d):
    def f(): W.onset[:] = ONSET0 + d / FS
    return f


def chdrop(seed):
    def f():
        g = np.random.default_rng(seed); m = torch.from_numpy(g.random(306) < 0.1)
        def win(idx, iso=None, clip=None):
            x, v = WIN0(idx, iso, clip); x[:, m] = 0; return x, v
        W.windows = win
    return f


@torch.no_grad()
def bn_recal(segs, A):
    """Reset and re-estimate every BatchNorm's running stats from this subject's own rows (cumulative average)."""
    saved = copy.deepcopy(M.enc.state_dict())
    for mod in M.enc.modules():
        if isinstance(mod, nn.BatchNorm1d): mod.reset_running_stats(); mod.momentum = None
    M.enc.train()
    for i in range(0, len(segs), 40):
        idx, gi, pad, rel = pack(W, segs[i:i + 40]); x, v = W.windows(idx); M.forward(x, v, A, gi, pad, rel)
    M.enc.eval(); return saved


def fit_A(s, steps, lr=3e-4):
    segs = segments(W, trainable & (W.sub == s), maxw); A_s = nn.Parameter(I.clone())
    opt = torch.optim.Adam([A_s], lr=lr)
    for _ in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]
        idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, A_s, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * ((A_s - I) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach()


res = {}
for blk in ("Q3", "Q4"):
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), maxw)
        if not segs: continue
        A = A0 if s == 0 else I; T = 3.0 if s == 0 else 2.0; r = {}
        def add(name, lo, y):
            pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
            r[name] = bacc_null(lo, y)[0]; r[name + "+sk"] = bacc_null(sinkhorn(lo, T, tgt), y)[0]
        lo, y = logits(segs, A); add("base", lo, y)
        if not SLIM:
            add("jit", *logits(segs, A, [jitter(d) for d in (-2, -1, 0, 1, 2)]))
            add("chd", *logits(segs, A, [chdrop(k) for k in range(8)]))
        saved = bn_recal(segs, A); add("bn", *logits(segs, A)); M.enc.load_state_dict(saved)
        if s == 0:
            add("idA", *logits(segs, I)); saved = bn_recal(segs, I); add("idA_bn", *logits(segs, I)); M.enc.load_state_dict(saved)
        if s in S2.SEEN:
            for st in (100, 300):
                As = fit_A(s, st); add(f"fitA{st}", *logits(segs, As))
            saved = bn_recal(segs, As); add("bn+fitA300", *logits(segs, As)); M.enc.load_state_dict(saved)
        res[f"{blk}/{s}"] = r; print(blk, s, json.dumps({k: round(v, 3) for k, v in r.items()}), round(time.time() - t0), "s", flush=True)

out = {}
for grp, members in (("broad", BROAD), ("deep", [0]), ("seen", list(S2.SEEN))):
    rows = [v for k, v in res.items() if int(k.split("/")[1]) in members]
    keys = sorted({k for v in rows for k in v}); out[grp] = {k: round(float(np.mean([v[k] for v in rows if k in v])), 4) for k in keys}
    print("MEAN", grp, json.dumps(out[grp]), flush=True)
json.dump({"per": res, "mean": out, "ckpt": sys.argv[1]}, open(OUT, "w"), indent=1)
