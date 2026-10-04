"""B1: multi-subject decoder. Shared WordCNN trunk trained on subj0 + training-pool Broad subjects (split train mask),
per-subject 306x306 input layer (identity init, ||A-I|| penalty), cross-subject augmentation.
Zero-shot = mean subject layer over Broad pool subjects. Few-shot = fit a new subject layer (trunk frozen) on fs_q data
(+ KD to subj0's paired window, as in P1a). Early stopping on subj0 Sherlock1 ses-10 as in B0."""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="b1"); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--broad_frac", type=float, default=0.3, help="fraction of each batch drawn from Broad pool subjects")
ap.add_argument("--aug", type=int, default=1); ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--fs_epochs", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--con", type=float, default=0.0, help="weight of cross-subject InfoNCE on exact token pairs (all words)")
ap.add_argument("--con_bs", type=int, default=128); ap.add_argument("--con_temp", type=float, default=0.1)
ap.add_argument("--noA", action="store_true", help="keep every subject's input layer frozen at identity during trunk training")
ap.add_argument("--select", default="es", choices=["es", "val"], help="es: subj0 ses-10 (B1); val: VAL_SUBJECTS block B zero-shot")
ap.add_argument("--final", action="store_true", help="train on ALL windows (all subjects, block B included) for --epochs, save last epoch; no local eval")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
ESVAL = "sub-0_ses-10_task-Sherlock1_run-1"

sel_ev = lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None
D = load_sets({
    "tr": (lambda b, s, ses, t: np.ones(len(t), bool)) if a.final else (lambda b, s, ses, t: S.train_mask(b, t, val=(a.select == "val")) if b != ESVAL else None),
    "es": lambda b, s, ses, t: np.ones(len(t), bool) if b == ESVAL else None,
    "0B": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s == 0 else None,
    "ev": sel_ev,
    "va": lambda b, s, ses, t: S.block_masks(b, t)["B"] if (s in S.VAL_SUBJECTS and a.select == "val") else None})
Xtr, ytr, st, _ = D["tr"]; Xes, yes, _, _ = D["es"]; X0B, y0B, _, _ = D["0B"]; Xe, ye, se, be = D["ev"]
Xva, yva, sva, _ = D["va"] if a.select == "val" else (None, None, None, None); del D
subs = sorted(set(st.tolist())); sid = {s: i for i, s in enumerate(subs)}; st_i = torch.tensor([sid[s] for s in st.tolist()])
i0 = torch.nonzero(st == 0).squeeze(1); ib = torch.nonzero(st != 0).squeeze(1)
print(f"train subj0 {len(i0)} broad {len(ib)} ({len(subs) - 1} subjects); es {len(Xes)}; eval {len(Xe)}", flush=True)
log_prior = torch.log(torch.bincount(ytr, minlength=50).float().clamp_min(1) / len(ytr)).to(dev)

if a.con > 0:
    # All-word windows on the shared stimulus: pool Broad subjects (train mask) paired to subj0 (same token, pre75).
    Pa = {}
    for f in sorted(glob.glob(f"{S.CACHE}/runs_all/*.npz")):
        b_ = os.path.basename(f)[:-4]; s_, ses_ = S.parse(b_); d_ = np.load(f); tcc = d_["timechapter"]
        if a.final: m_ = np.ones(len(tcc), bool)
        elif s_ > 0: m_ = S.train_mask(b_, tcc, val=(a.select == "val"))
        else: m_ = S.block_masks(b_, tcc)["pre75"]
        Pa[b_] = (s_, ses_, torch.from_numpy(d_["X"][m_]), np.round(tcc[m_], 2))
    z = {(ses_, tc_): (b_, i) for b_, (s_, ses_, X_, tcs_) in Pa.items() if s_ == 0 for i, tc_ in enumerate(tcs_)}
    zX = {b_: v[2] for b_, v in Pa.items() if v[0] == 0}
    PS, P0, Psub = [], [], []
    for b_, (s_, ses_, X_, tcs_) in Pa.items():
        if s_ == 0 or s_ not in sid: continue
        for i, tc_ in enumerate(tcs_):
            k = z.get((ses_, tc_))
            if k is not None: PS.append(X_[i]); P0.append(zX[k[0]][k[1]]); Psub.append(sid[s_])
    PS = torch.stack(PS); P0 = torch.stack(P0); Psub = torch.tensor(Psub); del Pa, zX
    print(f"contrastive pairs {len(PS)}", flush=True)

trunk = WordCNN().to(dev)
proj = nn.Linear(128 * 25, 256).to(dev)
A = nn.Parameter(torch.eye(306, device=dev).repeat(len(subs), 1, 1), requires_grad=not a.noA)
opt = torch.optim.AdamW([{"params": list(trunk.parameters()) + list(proj.parameters()), "weight_decay": 0.05}, {"params": [A], "weight_decay": 0.0}], lr=a.lr)
nb = len(i0) // int(a.bs * (1 - a.broad_frac)); steps = a.epochs * nb
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
I = torch.eye(306, device=dev)


def augment(x):
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 306, 1, device=dev))                       # channel gain jitter
    x = x * (torch.rand(B, 306, 1, device=dev) > 0.1)                            # sensor dropout 10%
    M = I + 0.05 * torch.randn(B, 306, 306, device=dev) / 306 ** 0.5              # small random spatial mixing
    x = torch.bmm(M, x)
    sh = int(np.random.randint(-2, 3)); x = torch.roll(x, sh, -1)                # +-40 ms shift at 50 Hz
    return x


def apply_A(Am, x):
    return torch.einsum("bij,bjt->bit", Am, x) if Am.dim() == 3 else torch.einsum("ij,bjt->bit", Am, x)


@torch.no_grad()
def logits(X, Am):
    trunk.eval(); out = []
    for i in range(0, len(X), 2048):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(trunk(apply_A(Am, normalize(X[i:i + 2048].to(dev)))).float())
    return torch.cat(out)


TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)
t0 = time.time(); hist = []; best = -1
for ep in range(a.epochs):
    trunk.train(); p0 = i0[torch.randperm(len(i0))]; tl = 0
    n0 = int(a.bs * (1 - a.broad_frac)); n1 = a.bs - n0
    for k in range(nb):
        b = torch.cat([p0[k * n0:(k + 1) * n0], ib[torch.randint(len(ib), (n1,))]])
        x = normalize(Xtr[b].to(dev, non_blocking=True)); y = ytr[b].to(dev); si = st_i[b].to(dev)
        x = apply_A(A[si], x)
        if a.aug: x = augment(x)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(trunk(x), y, label_smoothing=0.1)
        loss = loss + a.lam * ((A - I) ** 2).sum((1, 2)).mean()
        if a.con > 0:
            j = torch.randint(len(PS), (a.con_bs,))
            xs = apply_A(A[Psub[j].to(dev)], normalize(PS[j].to(dev))); x0 = apply_A(A[sid[0]], normalize(P0[j].to(dev)))
            if a.aug: xs, x0 = augment(xs), augment(x0)
            with torch.autocast("cuda", torch.bfloat16):
                e = F.normalize(proj(trunk.features(torch.cat([xs, x0])).flatten(1)).float(), dim=1)
            sim = e[:a.con_bs] @ e[a.con_bs:].T / a.con_temp; lab = torch.arange(a.con_bs, device=dev)
            loss = loss + a.con * 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    les = logits(Xes, A[sid[0]].detach())
    Abroad = A[[sid[s] for s in subs if s != 0]].detach().mean(0)
    if a.select == "val":
        lva = logits(Xva, Abroad)
        es = {tau: float(np.mean([bacc_at_k((lva[sva == s] - tau * log_prior).cpu(), yva[sva == s]) for s in S.VAL_SUBJECTS])) for tau in TAUS}
    else:
        es = {tau: bacc_at_k((les - tau * log_prior).cpu(), yes) for tau in TAUS}
    tau = max(es, key=es.get)
    le = (logits(Xe, Abroad) - tau * log_prior).cpu()
    Bm = torch.zeros(len(Xe), dtype=torch.bool)
    for b_ in set(be):
        idx = [i for i, x in enumerate(be) if x == b_]
        Bm[idx] = torch.from_numpy(S.block_masks(b_, np.load(f"{S.CACHE}/runs/{b_}.npz")["timechapter"])["B"])
    zs = {s: bacc_at_k(le[(se == s) & Bm], ye[(se == s) & Bm]) for s in S.EVAL_SUBJECTS}
    r = {"epoch": ep, "es": es[tau], "tau": tau, "subj0_B": bacc_at_k((logits(X0B, A[sid[0]].detach()) - tau * log_prior).cpu(), y0B),
         "zs_meanA_B": float(np.mean(list(zs.values()))), "zs_per": zs, "loss": tl / nb, "t": round(time.time() - t0)}
    hist.append(r); print(json.dumps(r), flush=True)
    if a.final:
        torch.save({"model": trunk.state_dict(), "A": A.detach().cpu(), "subs": subs, "log_prior": log_prior.cpu(), "tau": tau, "epoch": ep, "args": vars(a)}, f"runs/{a.tag}.pt")
        continue
    if r["es"] > best:
        best = r["es"]; torch.save({"model": trunk.state_dict(), "A": A.detach().cpu(), "subs": subs, "log_prior": log_prior.cpu(), "tau": tau, "epoch": ep, "args": vars(a)}, f"runs/{a.tag}.pt")

if a.final:
    json.dump({"args": vars(a), "history": hist}, open(f"runs/{a.tag}.json", "w"), indent=1); sys.exit(0)
# Few-shot on the selected checkpoint: fit a new subject layer per eval subject per condition (trunk frozen).
ck = torch.load(f"runs/{a.tag}.pt", map_location=dev); trunk.load_state_dict(ck["model"]); trunk.eval()
for p in trunk.parameters(): p.requires_grad_(False)
tau = ck["tau"]; adj = tau * log_prior; Ab = ck["A"][[ck["subs"].index(s) for s in ck["subs"] if s != 0]].to(dev).mean(0)
masks = {k: np.zeros(len(Xe), bool) for k in ["B", "fs_s25", "fs_s50", "fs_s100"]}
for b_ in set(be):
    idx = np.array([i for i, x in enumerate(be) if x == b_]); m = S.block_masks(b_, np.load(f"{S.CACHE}/runs/{b_}.npz")["timechapter"])
    for k in masks: masks[k][idx] = m[k]


def fit(X, y, A0):
    Am = A0.clone().requires_grad_(True); o = torch.optim.Adam([Am], lr=1e-3); Xn = normalize(X.to(dev)); y = y.to(dev)
    for _ in range(a.fs_epochs):
        perm = torch.randperm(len(Xn), device=dev)
        for i in range(0, len(Xn), 128):
            bb = perm[i:i + 128]
            with torch.autocast("cuda", torch.bfloat16):
                lo = trunk(apply_A(Am, Xn[bb])).float() - adj
            loss = F.cross_entropy(lo, y[bb]) + a.lam * ((Am - A0) ** 2).sum()
            o.zero_grad(); loss.backward(); o.step()
    return Am.detach()


res = {"s0_meanA": {}, "s25": {}, "s50": {}, "s100": {}}
for s in S.EVAL_SUBJECTS:
    ms = (se == s).numpy(); Bm = torch.from_numpy(ms & masks["B"])
    res["s0_meanA"][s] = bacc_at_k((logits(Xe[Bm], Ab) - adj).cpu(), ye[Bm])
    for g in ["s25", "s50", "s100"]:
        fm = torch.from_numpy(ms & masks["fs_" + g])
        res[g][s] = bacc_at_k((logits(Xe[Bm], fit(Xe[fm], ye[fm], Ab)) - adj).cpu(), ye[Bm])
summ = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
summ["composite"] = S.composite({"s100": summ["s100"], "s50": summ["s50"], "s25": summ["s25"], "s0": summ["s0_meanA"]})
sel = max(hist, key=lambda h: h["es"])
json.dump({"args": vars(a), "selected_by": a.select, "selected": sel, "fewshot": summ, "fewshot_per": res, "history": hist}, open(f"runs/{a.tag}.json", "w"), indent=1)
print("SELECTED", json.dumps(sel)); print("FEWSHOT", json.dumps(summ))
