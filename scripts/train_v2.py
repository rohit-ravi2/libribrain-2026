"""Trainer on split v2 (src/split2.py). B5a-style multi-subject trunk with per-subject input layers.
Per epoch, both stimulus blocks (Q3, Q4) are scored for: SEEN subjects with their own layer, VAL and EVAL zero-shot
(mean Broad layer). Fold A selects (epoch, tau) on Q3 and reports Q4; fold B the reverse.
Selection target = deployment mixture (32*seen + 7*zero-shot)/39. Logit adjustment uses the SAMPLER prior
((1-bf)*p_subj0 + bf*p_Broad), not the row-weighted prior. Every score carries its label-permutation null."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import parse
from src.data import load_sets, normalize
from src.models import WordCNN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--broad_frac", type=float, default=0.5); ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--p", type=float, default=0.5); ap.add_argument("--wd", type=float, default=0.3)
ap.add_argument("--aug", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)
scored = set(S2.EVAL + S2.VAL + S2.SEEN)
D = load_sets({"tr": lambda b, s, ses, t: S2.train_mask(b, t),
               "Q3": lambda b, s, ses, t: S2.block_mask(b, t, "Q3") if s in scored else None,
               "Q4": lambda b, s, ses, t: S2.block_mask(b, t, "Q4") if s in scored else None})
Xtr, ytr, st, _ = D["tr"]; blocks = {k: D[k][:3] for k in ("Q3", "Q4")}; del D
subs = sorted(set(st.tolist())); sid = {s: i for i, s in enumerate(subs)}; st_i = torch.tensor([sid[s] for s in st.tolist()])
i0 = torch.nonzero(st == 0).squeeze(1); ib = torch.nonzero(st != 0).squeeze(1)
p0 = torch.bincount(ytr[i0], minlength=50).float() / len(i0); pb = torch.bincount(ytr[ib], minlength=50).float() / len(ib)
log_prior = torch.log(((1 - a.broad_frac) * p0 + a.broad_frac * pb).clamp_min(1e-8)).to(dev)   # sampler prior
print(f"train subj0 {len(i0)} broad {len(ib)} ({len(subs) - 1} subj); Q3 {len(blocks['Q3'][0])} Q4 {len(blocks['Q4'][0])}", flush=True)

trunk = WordCNN(d=a.d, p=a.p).to(dev)
A = nn.Parameter(torch.eye(306, device=dev).repeat(len(subs), 1, 1)); I = torch.eye(306, device=dev)
opt = torch.optim.AdamW([{"params": trunk.parameters(), "weight_decay": a.wd}, {"params": [A], "weight_decay": 0.0}], lr=a.lr)
n0 = int(a.bs * (1 - a.broad_frac)); n1 = a.bs - n0; nb = len(i0) // n0
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)


def augment(x):
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 306, 1, device=dev)) * (torch.rand(B, 306, 1, device=dev) > 0.1)
    x = torch.bmm(I + 0.05 * torch.randn(B, 306, 306, device=dev) / 306 ** 0.5, x)
    return torch.roll(x, int(np.random.randint(-2, 3)), -1)


@torch.no_grad()
def logits(X, Am):
    trunk.eval(); out = []
    for i in range(0, len(X), 2048):
        Ab = Am if Am.dim() == 2 else Am[i:i + 2048]
        x = normalize(X[i:i + 2048].to(dev))
        x = torch.einsum("ij,bjt->bit", Ab, x) if Ab.dim() == 2 else torch.einsum("bij,bjt->bit", Ab, x)
        with torch.autocast("cuda", torch.bfloat16):
            out.append(trunk(x).float())
    return torch.cat(out)


def bacc_null(lo, y):
    """Balanced recall@10 over present classes, and its expected value under within-subject label permutation."""
    top = lo.topk(10, 1).indices; pres = torch.unique(y)
    hit = (top == y[:, None]).any(1).float()
    sc = torch.stack([hit[y == c].mean() for c in pres]).mean()
    nu = torch.stack([(top == c).any(1).float().mean() for c in pres]).mean()
    return float(sc), float(nu)


def group(lo_all, y, s, members):
    r = [bacc_null(lo_all[s == m], y[s == m]) for m in members if (s == m).any()]
    return float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))


def evaluate():
    Ab = A[[sid[s] for s in subs if s != 0]].detach().mean(0); res = {}
    for blk, (X, y, s) in blocks.items():
        seen_m = torch.tensor([v in S2.SEEN for v in s.tolist()]); unseen = ~seen_m
        lo_u = logits(X[unseen], Ab).cpu(); yu, su = y[unseen], s[unseen]
        Aown = A.detach()[[sid[v] for v in s[seen_m].tolist()]]
        lo_s = logits(X[seen_m], Aown).cpu(); ys, ss = y[seen_m], s[seen_m]
        for tau in TAUS:
            adj = tau * log_prior.cpu()
            res[(blk, tau)] = {"seen": group(lo_s - adj, ys, ss, S2.SEEN), "val_zs": group(lo_u - adj, yu, su, S2.VAL),
                               "eval_zs": group(lo_u - adj, yu, su, S2.EVAL)}
    return res


hist = []; t0 = time.time(); os.makedirs("runs", exist_ok=True)
for ep in range(a.epochs):
    trunk.train(); p0i = i0[torch.randperm(len(i0))]; tl = 0
    for k in range(nb):
        b = torch.cat([p0i[k * n0:(k + 1) * n0], ib[torch.randint(len(ib), (n1,))]])
        x = torch.einsum("bij,bjt->bit", A[st_i[b].to(dev)], normalize(Xtr[b].to(dev, non_blocking=True)))
        if a.aug: x = augment(x)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(trunk(x), ytr[b].to(dev), label_smoothing=0.1)
        loss = loss + a.lam * ((A - I) ** 2).sum((1, 2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    ev = evaluate(); hist.append({"epoch": ep, "loss": tl / nb, "eval": {f"{k[0]}|{k[1]}": v for k, v in ev.items()}, "t": round(time.time() - t0)})
    torch.save({"model": trunk.state_dict(), "A": A.detach().cpu(), "subs": subs, "log_prior": log_prior.cpu(), "args": vars(a), "epoch": ep},
               f"runs/{a.tag}_ep{ep}.pt")
    q = {blk: max((S2.deploy(ev[(blk, t)]["seen"][0], ev[(blk, t)]["val_zs"][0]), t) for t in TAUS) for blk in ("Q3", "Q4")}
    print(json.dumps({"epoch": ep, "loss": round(tl / nb, 3), "sel_Q3": q["Q3"], "sel_Q4": q["Q4"], "t": round(time.time() - t0)}), flush=True)

folds = {}
for f, (sel, test) in S2.FOLDS.items():
    best = max(((S2.deploy(h["eval"][f"{sel}|{t}"]["seen"][0], h["eval"][f"{sel}|{t}"]["val_zs"][0]), h["epoch"], t) for h in hist for t in TAUS))
    _, ep, tau = best; e = hist[ep]["eval"][f"{test}|{tau}"]
    folds[f] = {"select": sel, "test": test, "epoch": ep, "tau": tau, "sel_score": best[0],
                "seen": e["seen"], "eval_zs": e["eval_zs"], "deploy": S2.deploy(e["seen"][0], e["eval_zs"][0]),
                "deploy_null": S2.deploy(e["seen"][1], e["eval_zs"][1])}
summ = {k: float(np.mean([folds[f][k] if not isinstance(folds[f][k], list) else folds[f][k][0] for f in folds])) for k in ("deploy", "deploy_null")}
summ["deploy_lift"] = summ["deploy"] - summ["deploy_null"]
summ["seen"] = float(np.mean([folds[f]["seen"][0] for f in folds])); summ["eval_zs"] = float(np.mean([folds[f]["eval_zs"][0] for f in folds]))
json.dump({"args": vars(a), "folds": folds, "summary": summ, "history": hist}, open(f"runs/{a.tag}.json", "w"), indent=1)
print("FOLDS", json.dumps(folds)); print("SUMMARY", json.dumps(summ))
