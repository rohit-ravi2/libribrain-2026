"""Text-target decoder (d'Ascoli-style adaptation, 1 s word windows): MEG -> T5-large layer-12 contextual word embedding,
SigLIP loss with duplicate-aware positives (identical targets = same token heard by different subjects), all words.
Readout = cosine to each vocab word's prototype (mean normalised training target). Optional CE head on vocab rows.
Split v2 selection/test exactly as scripts/train_v2.py (folds A/B, seen + zero-shot, label-permutation nulls)."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.common import CACHE
from src.data import load_sets, normalize
from src.models import WordCNN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--broad_frac", type=float, default=0.5); ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--d", type=int, default=256); ap.add_argument("--p", type=float, default=0.3); ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--ce", type=float, default=0.0, help="weight of an auxiliary 50-way CE head on vocab rows")
ap.add_argument("--dup", type=float, default=0.999, help="cosine threshold above which two targets count as the same positive")
ap.add_argument("--aug", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_steps", type=int, default=0, help="smoke test: stop after this many steps per epoch")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"

M = np.load(f"{CACHE}/mm/meta.npz"); bases = list(M["bases"]); sub = M["sub"].astype(int); lab = M["label"].astype(int)
X = np.load(f"{CACHE}/mm/X.npy", mmap_mode="r"); T = np.load(f"{CACHE}/mm/T.npy", mmap_mode="r")
tm = np.zeros(len(sub), bool)
for r, b in enumerate(bases):
    idx = np.flatnonzero(M["rid"] == r); tm[idx] = S2.train_mask(b, M["timechapter"][idx])
tr = np.flatnonzero(tm & ~np.isnan(T[:, 0]))
subs = sorted(set(sub[tr].tolist())); sid = {s: i for i, s in enumerate(subs)}
i0 = tr[sub[tr] == 0]; ib = tr[sub[tr] != 0]
print(f"train rows subj0 {len(i0)} broad {len(ib)} ({len(subs) - 1} subj)", flush=True)

# vocab prototypes from TRAINING targets only
Tn_tr = lambda idx: F.normalize(torch.from_numpy(np.asarray(T[np.sort(idx)])).float(), dim=1)
proto = torch.zeros(50, 1024)
for c in range(50):
    ix = tr[lab[tr] == c]
    if len(ix): proto[c] = F.normalize(Tn_tr(ix[np.random.permutation(len(ix))[:5000]]).mean(0), dim=0)
proto = proto.to(dev)

scored = set(S2.EVAL + S2.VAL + S2.SEEN)
D = load_sets({"Q3": lambda b, s, ses, t: S2.block_mask(b, t, "Q3") if s in scored else None,
               "Q4": lambda b, s, ses, t: S2.block_mask(b, t, "Q4") if s in scored else None})
blocks = {k: D[k][:3] for k in ("Q3", "Q4")}; del D

trunk = WordCNN(d=a.d, p=a.p).to(dev)
F_DIM = a.d // 2 * 25
head = nn.Sequential(nn.Flatten(), nn.Dropout(a.p), nn.Linear(F_DIM, 1024)).to(dev)
cehead = nn.Linear(1024, 50).to(dev)
logit_s = nn.Parameter(torch.tensor(np.log(10.0), device=dev)); logit_b = nn.Parameter(torch.tensor(-10.0, device=dev))
A = nn.Parameter(torch.eye(306, device=dev).repeat(len(subs), 1, 1)); I = torch.eye(306, device=dev)
params = list(trunk.parameters()) + list(head.parameters()) + list(cehead.parameters())
opt = torch.optim.AdamW([{"params": params, "weight_decay": a.wd}, {"params": [A, logit_s, logit_b], "weight_decay": 0.0}], lr=a.lr)
n0 = int(a.bs * (1 - a.broad_frac)); n1 = a.bs - n0; nb = len(i0) // n0 if not a.max_steps else a.max_steps
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1)


def embed(x):
    return head(trunk.features(x))


def augment(x):
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 306, 1, device=dev)) * (torch.rand(B, 306, 1, device=dev) > 0.1)
    x = torch.bmm(I + 0.05 * torch.randn(B, 306, 306, device=dev) / 306 ** 0.5, x)
    return torch.roll(x, int(np.random.randint(-2, 3)), -1)


@torch.no_grad()
def scores_for(Xb, Am):
    trunk.eval(); head.eval(); out = []
    for i in range(0, len(Xb), 2048):
        Ab = Am if Am.dim() == 2 else Am[i:i + 2048]
        x = normalize(Xb[i:i + 2048].to(dev))
        x = torch.einsum("ij,bjt->bit", Ab, x) if Ab.dim() == 2 else torch.einsum("bij,bjt->bit", Ab, x)
        with torch.autocast("cuda", torch.bfloat16):
            e = embed(x).float()
        out.append(torch.cat([F.normalize(e, dim=1) @ proto.T, cehead(e)], 1))   # [:50] retrieval, [50:] CE logits
    return torch.cat(out).cpu()


def bacc_null(lo, y):
    top = lo.topk(10, 1).indices; pres = torch.unique(y); hit = (top == y[:, None]).any(1).float()
    return (float(torch.stack([hit[y == c].mean() for c in pres]).mean()),
            float(torch.stack([(top == c).any(1).float().mean() for c in pres]).mean()))


def group(lo, y, s, members):
    r = [bacc_null(lo[s == m], y[s == m]) for m in members if (s == m).any()]
    return float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))


READOUTS = ("retr", "ce", "sum")


def evaluate():
    Ab = A[[sid[s] for s in subs if s != 0]].detach().mean(0); res = {}
    for blk, (Xb, y, s) in blocks.items():
        seen_m = torch.tensor([v in S2.SEEN for v in s.tolist()])
        su_ = scores_for(Xb[~seen_m], Ab); ss_ = scores_for(Xb[seen_m], A.detach()[[sid[v] for v in s[seen_m].tolist()]])
        for ro in READOUTS:
            pick = (lambda z: z[:, :50]) if ro == "retr" else (lambda z: z[:, 50:]) if ro == "ce" else \
                   (lambda z: F.log_softmax(z[:, :50] * 20, 1) + F.log_softmax(z[:, 50:], 1))
            u, v = pick(su_), pick(ss_)
            res[(blk, ro)] = {"seen": group(v, y[seen_m], s[seen_m], S2.SEEN), "val_zs": group(u, y[~seen_m], s[~seen_m], S2.VAL),
                              "eval_zs": group(u, y[~seen_m], s[~seen_m], S2.EVAL)}
    return res


hist = []; t0 = time.time()
for ep in range(a.epochs):
    trunk.train(); head.train(); p0 = np.random.permutation(i0); tl = np.zeros(2)
    for k in range(nb):
        b = np.sort(np.r_[p0[k * n0:(k + 1) * n0], ib[np.random.randint(len(ib), size=n1)]])
        xb = torch.from_numpy(np.asarray(X[b])).to(dev, non_blocking=True)
        tb = F.normalize(torch.from_numpy(np.asarray(T[b])).to(dev).float(), dim=1)
        x = torch.einsum("bij,bjt->bit", A[torch.as_tensor([sid[v] for v in sub[b]], device=dev)], normalize(xb))
        if a.aug: x = augment(x)
        with torch.autocast("cuda", torch.bfloat16):
            e = embed(x).float()
        en = F.normalize(e, dim=1)
        pos = (tb @ tb.T) > a.dup                                                  # duplicate-aware positives
        lg = en @ tb.T * logit_s.exp() + logit_b
        loss = -F.logsigmoid(torch.where(pos, lg, -lg)).sum(1).mean()
        tl[0] += loss.item()
        if a.ce > 0:
            yb = torch.from_numpy(lab[b]).to(dev); ok = yb >= 0
            if ok.any():
                lce = F.cross_entropy(cehead(e[ok]), yb[ok], label_smoothing=0.1); loss = loss + a.ce * lce; tl[1] += lce.item()
        loss = loss + a.lam * ((A - I) ** 2).sum((1, 2)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    ev = evaluate(); key = lambda blk, ro: S2.deploy(ev[(blk, ro)]["seen"][0], ev[(blk, ro)]["val_zs"][0])
    hist.append({"epoch": ep, "siglip": tl[0] / nb, "ce": tl[1] / nb, "eval": {f"{k[0]}|{k[1]}": v for k, v in ev.items()}, "t": round(time.time() - t0)})
    torch.save({"trunk": trunk.state_dict(), "head": head.state_dict(), "cehead": cehead.state_dict(), "A": A.detach().cpu(),
                "subs": subs, "proto": proto.cpu(), "args": vars(a), "epoch": ep}, f"runs/{a.tag}_ep{ep}.pt")
    print(json.dumps({"epoch": ep, "siglip": round(tl[0] / nb, 3), "ce": round(tl[1] / nb, 3),
                      **{f"sel_{blk}_{ro}": round(key(blk, ro), 4) for blk in ("Q3", "Q4") for ro in READOUTS}, "t": round(time.time() - t0)}), flush=True)

folds = {}
for f, (sel, test) in S2.FOLDS.items():
    ros = READOUTS if a.ce > 0 else ("retr",)
    best = max((S2.deploy(h["eval"][f"{sel}|{ro}"]["seen"][0], h["eval"][f"{sel}|{ro}"]["val_zs"][0]), h["epoch"], ro) for h in hist for ro in ros)
    _, ep, ro = best; e = hist[ep]["eval"][f"{test}|{ro}"]
    folds[f] = {"select": sel, "test": test, "epoch": ep, "readout": ro, "sel_score": best[0], "seen": e["seen"], "eval_zs": e["eval_zs"],
                "deploy": S2.deploy(e["seen"][0], e["eval_zs"][0]), "deploy_null": S2.deploy(e["seen"][1], e["eval_zs"][1])}
summ = {k: float(np.mean([folds[f][k] for f in folds])) for k in ("deploy", "deploy_null")}
summ["deploy_lift"] = summ["deploy"] - summ["deploy_null"]
summ["seen"] = float(np.mean([folds[f]["seen"][0] for f in folds])); summ["eval_zs"] = float(np.mean([folds[f]["eval_zs"][0] for f in folds]))
json.dump({"args": vars(a), "folds": folds, "summary": summ, "history": hist}, open(f"runs/{a.tag}.json", "w"), indent=1)
print("FOLDS", json.dumps(folds)); print("SUMMARY", json.dumps(summ))
