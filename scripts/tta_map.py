"""INF-2: within-subject self-training of the input layer on the subject's OWN unlabelled block rows (split v2 Q3/Q4).
Per subject: A_s (init identity; subj0: A0) is fitted to soft pseudo-labels = Sinkhorn-balanced posteriors (T 2/3) of the
current model, re-estimated every `every` steps (EM); KL(target || model) + ridge to init. Variants: steps 50/150, with and
without BN recalibration first. Scored transductively on the same rows (the holdout condition), then Sinkhorn.
Usage: tta_map.py runs/<ckpt>.pt"""
import sys, os, json, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
from src.common import CACHE
import glob

M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5; t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = BROAD + [0]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev)
print("load", round(time.time() - t0), "s", flush=True)


def sinkhorn_p(P, T, tgt, iters=10):
    P = P.clamp_min(1e-9) ** (1 / T); P = P / P.sum(1, keepdim=True); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P


def fwd_all(segs, A, grad=False):
    out = []; ys = []
    for i in range(0, len(segs), 40):
        idx, gi, pad, rel = pack(W, segs[i:i + 40]); x, v = W.windows(idx)
        with torch.set_grad_enabled(grad):
            _, ce, _ = M.forward(x, v, A, gi, pad, rel)
        out.append(ce - tau * M.log_prior); ys.append(torch.from_numpy(W.label[idx]))
    return torch.cat(out), torch.cat(ys)


@torch.no_grad()
def bn_recal(segs, A, dropout_off=False):
    saved = copy.deepcopy(M.enc.state_dict())
    for mod in M.enc.modules():
        if isinstance(mod, nn.BatchNorm1d): mod.reset_running_stats(); mod.momentum = None
    M.enc.train()
    if dropout_off:
        for mod in M.enc.modules():
            if isinstance(mod, nn.Dropout): mod.eval()
    fwd_all(segs, A); M.enc.eval(); return saved


def bn_affine_train(segs, A, T, steps=100, every=25, lr=1e-3):
    """TENT-style: train only BN affine params (running stats frozen at the recalibrated values) on balanced pseudo-labels."""
    params = [p for mod in M.enc.modules() if isinstance(mod, nn.BatchNorm1d) for p in (mod.weight, mod.bias)]
    for p in M.enc.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr); tgt50 = torch.full((50,), 1 / 50, device=dev)
    for k in range(steps):
        if k % every == 0:
            with torch.no_grad():
                lo, _ = fwd_all(segs, A); Ptar = sinkhorn_p(F.softmax(lo, 1), T, tgt50)
            perm = rng.permutation(len(segs)); pos = 0
        b = [segs[j] for j in perm[pos:pos + 24]]; pos = (pos + 24) % max(len(segs) - 24, 1)
        idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, A, gi, pad, rel); lo = ce - tau * M.log_prior
        loss = F.kl_div(F.log_softmax(lo, 1), Ptar[torch.from_numpy(ROW[np.concatenate(b)]).to(dev)], reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
    for p in M.enc.parameters(): p.requires_grad_(True)


def self_train(segs, A_init, T, steps, every=25, lr=3e-4, lam=1e-2):
    A_s = nn.Parameter(A_init.clone()); opt = torch.optim.Adam([A_s], lr=lr); tgt50 = torch.full((50,), 1 / 50, device=dev)
    for k in range(steps):
        if k % every == 0:
            with torch.no_grad():
                lo, _ = fwd_all(segs, A_s.detach()); Ptar = sinkhorn_p(F.softmax(lo, 1), T, tgt50)
            perm = rng.permutation(len(segs)); pos = 0
        b = [segs[j] for j in perm[pos:pos + 24]]; pos = (pos + 24) % max(len(segs) - 24, 1)
        idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        # rows of this batch in the subject's global order
        _, ce, _ = M.forward(x, v, A_s, gi, pad, rel); lo = ce - tau * M.log_prior
        rowid = torch.from_numpy(ROW[np.concatenate(b)]).to(dev)
        loss = F.kl_div(F.log_softmax(lo, 1), Ptar[rowid], reduction="batchmean") + lam * ((A_s - A_init) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach()


res = {}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), maxw)
        if not segs: continue
        ROW = np.full(W.n, -1, np.int64); ROW[np.concatenate(segs)] = np.arange(sum(map(len, segs)))   # word id -> row in fwd_all order
        A0 = M.A[M.sid[0]] if s == 0 else I; T = 3.0 if s == 0 else 2.0; r = {}
        def ev(name, A):
            lo, y = fwd_all(segs, A); k = y >= 0; lo, y = lo[k].cpu(), y[k]
            pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
            r[name] = bacc_null(lo, y)[0]; r[name + "+sk"] = bacc_null(sinkhorn_p(F.softmax(lo, 1), T, tgt).log(), y)[0]
        ev("base", A0)
        for st in (50, 150):
            ev(f"st{st}", self_train(segs, A0, T, st))
        saved = bn_recal(segs, A0); ev("bn", A0); ev("bn+st150", self_train(segs, A0, T, 150)); M.enc.load_state_dict(saved)
        saved = bn_recal(segs, A0, dropout_off=True); ev("bnD", A0)
        bn_affine_train(segs, A0, T, 100); ev("bnD+aff100", A0); M.enc.load_state_dict(saved)
        res[f"{blk}/{s}"] = r; print(blk, s, json.dumps({k: round(v, 3) for k, v in r.items()}), round(time.time() - t0), "s", flush=True)

out = {}
for grp, members in (("broad", BROAD), ("deep", [0])):
    rows = [v for k, v in res.items() if int(k.split("/")[1]) in members]
    keys = sorted({k for v in rows for k in v}); out[grp] = {k: round(float(np.mean([v[k] for v in rows if k in v])), 4) for k in keys}
    print("MEAN", grp, json.dumps(out[grp]), flush=True)
json.dump({"per": res, "mean": out, "ckpt": sys.argv[1]}, open("runs/tta_map.json", "w"), indent=1)
