"""v3e: subject adaptation on top of a train_v3 run, split v2 protocol.
For each fold (select block -> test block), using the fold's selected checkpoint/readout/tau:
  seen subjects  (deployment: labelled subjects 1-32): per-subject fine-tuning of A_s + a private CE head on the
                  subject's own labelled training words (Q1-Q2); steps chosen on the select block.
  unseen subjects (deployment: 33-39; locally VAL/EVAL zero-shot): TTA on the subject's OWN unlabelled words of the
                  block being scored: CORAL covariance alignment (blend alpha) then entropy minimisation of A_s.
                  Hyper-parameters chosen on the select block (TTA there uses select-block MEG), reported on the test block.
No cross-subject correspondence of rows is used anywhere."""
import argparse, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words, normalize
from src.v3lib import Model, segments, pack, predict, group, dev

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--tag", default=None)
ap.add_argument("--ft_steps", default="0,100,300"); ap.add_argument("--ft_lr", type=float, default=3e-4)
ap.add_argument("--coral", default="0,0.5,1.0"); ap.add_argument("--ent_steps", default="0,30"); ap.add_argument("--ent_lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); tag = a.tag or f"{a.run}_adapt"
R = json.load(open(f"runs/{a.run}.json")); maxw = R["args"]["maxw"]; t0 = time.time()
W = Words(load_meg=True, load_t5=False); q = W.quarters()
scored = set(S2.EVAL + S2.VAL + S2.SEEN)
trainable = ~np.isin(W.sub, S2.EVAL + S2.VAL) & np.isin(q, ["other", "Q1", "Q2"])
SEG = {blk: segments(W, (q == blk) & np.isin(W.sub, list(scored)), maxw) for blk in ("Q3", "Q4")}
SEGSUB = {blk: {s: [g for g in SEG[blk] if W.sub[g[0]] == s] for s in scored} for blk in SEG}
FT = {s: segments(W, trainable & (W.sub == s), maxw) for s in S2.SEEN}
print("load", round(time.time() - t0), "s; ft sentences", {s: len(v) for s, v in FT.items()}, flush=True)


def score(M, blk, Amap, heads=None, ro="ce", tau=0.0):
    """Per-subject scores on a block. heads: optional dict sub -> private CE head."""
    out = {}
    for s, segs in SEGSUB[blk].items():
        if not segs: continue
        P = predict(M, W, segs, {s: Amap.get(s)})
        ce = P["ce"]
        if heads and s in heads:
            e = []
            for i in range(0, len(segs), 40):
                idx, gi, pad, rel = pack(W, segs[i:i + 40], rel_on=bool(M.a.get('rel', 1))); x, v = W.windows(idx)
                with torch.no_grad(): e.append(M.forward(x, v, Amap[s], gi, pad, rel)[0])
            ce = heads[s](torch.cat(e)).float().cpu()
        lo = M.readout(ce.to(dev), P["zt"].to(dev), ro, tau).cpu()
        keep = P["y"] >= 0
        out[s] = group(lo[keep], P["y"][keep], P["sub"][keep], [s])
    return out


def agg(sc, members):
    r = [sc[m] for m in members if m in sc]; return (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r])))


# ---------- adaptation primitives ----------
def finetune_seen(M, s, steps, ro, tau):
    """Fine-tune A_s and a private copy of head_ce on subject s's labelled training segments."""
    A_s = nn.Parameter(M.A[M.sid[s]].clone()); head = copy.deepcopy(M.head_ce); A0 = M.A[M.sid[s]].clone()
    opt = torch.optim.Adam([A_s] + list(head.parameters()), lr=a.ft_lr); segs = FT[s]
    for k in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]
        idx, gi, pad, rel = pack(W, b, rel_on=bool(M.a.get('rel', 1))); x, v = W.windows(idx)
        e, _, _ = M.forward(x, v, A_s, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); has = y >= 0
        loss = F.cross_entropy(head(e)[has], y[has], label_smoothing=0.1) + 1e-3 * ((A_s - A0) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach(), head


def pool_cov(M, n=3000):
    """Covariance of normalised training windows of Broad subjects (what the trunk sees before Amean)."""
    idx = np.flatnonzero(trainable & (W.sub != 0)); idx = rng.choice(idx, min(n, len(idx)), replace=False)
    x, v = W.windows(idx); x = normalize(x.to(dev), v.to(dev)); v = v.to(dev)
    X = x.permute(1, 0, 2).reshape(306, -1)[:, v.reshape(-1)]
    return X @ X.T / X.shape[1]


def subj_cov(M, segs):
    idx = np.concatenate(segs); x, v = W.windows(idx); x = normalize(x.to(dev), v.to(dev)); v = v.to(dev)
    X = x.permute(1, 0, 2).reshape(306, -1)[:, v.reshape(-1)]
    return X @ X.T / X.shape[1]


def msqrt(C, power, eps=1e-3):
    C = C + eps * C.diagonal().mean() * torch.eye(306, device=dev)
    e, U = torch.linalg.eigh(C); return (U * e.clamp_min(1e-8) ** power) @ U.T


def coral_A(M, segs, Cp_half, alpha):
    Cs = subj_cov(M, segs); Mx = Cp_half @ msqrt(Cs, -0.5)
    return M.Amean @ ((1 - alpha) * torch.eye(306, device=dev) + alpha * Mx)


def entropy_tta(M, segs, A_init, steps, tau):
    A_s = nn.Parameter(A_init.clone()); opt = torch.optim.Adam([A_s], lr=a.ent_lr)
    for k in range(steps):
        b = [segs[j] for j in rng.integers(0, len(segs), 24)]
        idx, gi, pad, rel = pack(W, b, rel_on=bool(M.a.get('rel', 1))); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, A_s, gi, pad, rel); p = F.softmax(ce - tau * M.log_prior, 1)
        ent = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
        pm = p.mean(0); div = (pm * pm.clamp_min(1e-8).log()).sum()           # keep the marginal spread (avoid collapse)
        loss = ent + div + 1e-2 * ((A_s - A_init) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach()


# ---------- per fold ----------
res = {}
for f, fd in R["folds"].items():
    sel, test, ro, tau = fd["select"], fd["test"], fd["readout"], fd["tau"]
    M = Model(f"runs/{a.run}_ep{fd['epoch']}.pt"); M.eval()
    base = {blk: score(M, blk, {s: M.A[M.sid[s]] for s in S2.SEEN}, ro=ro, tau=tau) for blk in (sel, test)}
    out = {"base": {blk: {"seen": agg(base[blk], S2.SEEN), "val_zs": agg(base[blk], S2.VAL), "eval_zs": agg(base[blk], S2.EVAL)} for blk in base}}
    print(f, "base", json.dumps(out["base"]), round(time.time() - t0), flush=True)
    # seen fine-tuning: choose steps on select block
    ft = {}
    for steps in [int(x) for x in a.ft_steps.split(",")]:
        Amap, heads = {}, {}
        for s in S2.SEEN:
            if steps == 0: Amap[s] = M.A[M.sid[s]]; continue
            Amap[s], heads[s] = finetune_seen(M, s, steps, ro, tau)
        ft[steps] = {blk: agg(score(M, blk, Amap, heads, ro, tau), S2.SEEN) for blk in (sel, test)}
        print(f, "ft", steps, json.dumps(ft[steps]), round(time.time() - t0), flush=True)
    best_ft = max(ft, key=lambda k: ft[k][sel][0]); out["ft"] = {"grid": {str(k): v for k, v in ft.items()}, "chosen": best_ft}
    # unseen TTA: grid on select block (TTA uses that block's own MEG), report on test block
    Cp = pool_cov(M); torch.save(Cp.cpu(), "runs/pool_cov.pt"); Cp_half = msqrt(Cp, 0.5)
    tta = {}
    for alpha in [float(x) for x in a.coral.split(",")]:
        for steps in [int(x) for x in a.ent_steps.split(",")]:
            r = {}
            for blk in (sel, test):
                Amap = {}
                for s in S2.VAL + S2.EVAL:
                    segs = SEGSUB[blk][s]
                    if not segs: continue
                    A0 = coral_A(M, segs, Cp_half, alpha) if alpha > 0 else M.Amean
                    Amap[s] = entropy_tta(M, segs, A0, steps, tau) if steps > 0 else A0
                sc = score(M, blk, Amap, ro=ro, tau=tau); r[blk] = {"val_zs": agg(sc, S2.VAL), "eval_zs": agg(sc, S2.EVAL)}
            tta[(alpha, steps)] = r; print(f, "tta", alpha, steps, json.dumps(r), round(time.time() - t0), flush=True)
    best_tta = max(tta, key=lambda k: tta[k][sel]["val_zs"][0]); out["tta"] = {"grid": {f"{k[0]},{k[1]}": v for k, v in tta.items()}, "chosen": list(best_tta)}
    seen_t = ft[best_ft][test]; zs_t = tta[best_tta][test]["eval_zs"]
    out["test"] = {"seen": seen_t, "eval_zs": zs_t, "deploy": S2.deploy(seen_t[0], zs_t[0]), "deploy_null": S2.deploy(seen_t[1], zs_t[1]),
                   "deploy_base": S2.deploy(out["base"][test]["seen"][0], out["base"][test]["eval_zs"][0])}
    res[f] = out; print(f, "TEST", json.dumps(out["test"]), flush=True)
summ = {k: float(np.mean([res[f]["test"][k] for f in res])) for k in ("deploy", "deploy_null", "deploy_base")}
summ["deploy_lift"] = summ["deploy"] - summ["deploy_null"]
json.dump({"args": vars(a), "folds": res, "summary": summ}, open(f"runs/{tag}.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ))
