"""P1a: per-subject linear sensor map A_s into subj0's input space, frozen subj0 decoder (PREDICTIONS.md P1a).
Loss = CE(label) + kd * KL(teacher || student) + lam * ||A - I||^2, teacher = frozen decoder on subj0's window for
the same stimulus token (exact pairing on Sherlock1 ses-11/12 timechapter). Zero-shot: mean A over training-pool subjects."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="runs/b0.pt"); ap.add_argument("--tag", default="p1a")
ap.add_argument("--kd", type=float, default=1.0); ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); dev = "cuda"

ck = torch.load(a.base, map_location=dev)
model = WordCNN().to(dev); model.load_state_dict(ck["model"]); model.eval()
for p in model.parameters(): p.requires_grad_(False)
adj = ck["tau"] * ck["log_prior"].to(dev)

shared = lambda ses: ses in S.SHARED
D = load_sets({
    "z": lambda b, s, ses, t: S.block_masks(b, t)["pre75"] if (s == 0 and shared(ses)) else None,
    "pool": lambda b, s, ses, t: S.train_mask(b, t) if (s > 0 and shared(ses)) else None,
    "ev": lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None})
import glob
def tcs(sel):  # timechapter + session for the windows selected, in load_sets order
    out = []
    for f in sorted(glob.glob(f"{S.CACHE}/runs/*.npz")):
        base = os.path.basename(f)[:-4]; sub, ses = S.parse(base); t = np.load(f)["timechapter"]
        m = sel(base, sub, ses, t)
        if m is not None and np.any(m): out += [(ses, round(float(x), 2), base) for x in t[m]]
    return out
X0, y0, _, _ = D["z"]
key0 = {(ses, tc): i for i, (ses, tc, _) in enumerate(tcs(lambda b, s, ses, t: S.block_masks(b, t)["pre75"] if (s == 0 and shared(ses)) else None))}
with torch.no_grad():
    T0 = torch.cat([model(normalize(X0[i:i + 2048].to(dev))).float() for i in range(0, len(X0), 2048)])
teach = torch.softmax(T0 - adj, 1)  # teacher distribution (balanced-adjusted, as used for ranking)


def pairs(meta):
    return torch.tensor([key0.get((ses, tc), -1) for ses, tc, _ in meta])


def fit_A(X, y, pidx):
    A = torch.eye(306, device=dev, requires_grad=True)
    opt = torch.optim.Adam([A], lr=a.lr); I = torch.eye(306, device=dev)
    Xn = normalize(X.to(dev)); y = y.to(dev); pidx = pidx.to(dev)
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xn), device=dev)
        for i in range(0, len(Xn), a.bs):
            b = perm[i:i + a.bs]
            with torch.autocast("cuda", torch.bfloat16):
                lo = model(torch.einsum("ij,bjt->bit", A, Xn[b])).float() - adj
            loss = F.cross_entropy(lo, y[b]) + a.lam * ((A - I) ** 2).sum()
            ok = pidx[b] >= 0
            if a.kd > 0 and ok.any():
                loss = loss + a.kd * F.kl_div(F.log_softmax(lo[ok], 1), teach[pidx[b][ok]], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    return A.detach()


@torch.no_grad()
def score(A, X, y):
    Xn = normalize(X.to(dev)); out = []
    for i in range(0, len(Xn), 2048):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(torch.einsum("ij,bjt->bit", A, Xn[i:i + 2048])).float() - adj)
    return bacc_at_k(torch.cat(out).cpu(), y)


t0 = time.time()
Xp, yp, sp, _ = D["pool"]
pmeta = tcs(lambda b, s, ses, t: S.train_mask(b, t) if (s > 0 and shared(ses)) else None)
pp = pairs(pmeta); print("pool windows", len(Xp), "paired", float((pp >= 0).float().mean()), flush=True)
As = {}
for s in sorted(set(sp.tolist())):
    m = sp == s; As[s] = fit_A(Xp[m], yp[m], pp[m])
print(f"fit {len(As)} pool subjects in {time.time() - t0:.0f}s", flush=True)
Amean = torch.stack(list(As.values())).mean(0)

Xe, ye, se, _ = D["ev"]
emeta = tcs(lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None)
tc_e = np.array([tc for _, tc, _ in emeta]); bases_e = [b for _, _, b in emeta]
masks = {k: np.zeros(len(emeta), bool) for k in ["B", "fs_s25", "fs_s50", "fs_s100"]}
for b in set(bases_e):
    idx = np.array([i for i, x in enumerate(bases_e) if x == b])
    m = S.block_masks(b, tc_e[idx])
    for k in masks: masks[k][idx] = m[k]
pe = pairs(emeta)
res = {"identity": {}, "s0_meanA": {}, "s25": {}, "s50": {}, "s100": {}}
I = torch.eye(306, device=dev)
for s in S.EVAL_SUBJECTS:
    ms = (se == s).numpy(); Bm = torch.from_numpy(ms & masks["B"])
    res["identity"][s] = score(I, Xe[Bm], ye[Bm]); res["s0_meanA"][s] = score(Amean, Xe[Bm], ye[Bm])
    for g in ["s25", "s50", "s100"]:
        fm = torch.from_numpy(ms & masks["fs_" + g])
        res[g][s] = score(fit_A(Xe[fm], ye[fm], pe[fm]), Xe[Bm], ye[Bm])
summ = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
summ["composite"] = S.composite({"s100": summ["s100"], "s50": summ["s50"], "s25": summ["s25"], "s0": summ["s0_meanA"]})
out = {"args": vars(a), "summary": summ, "per_subject": res, "t": round(time.time() - t0)}
os.makedirs("runs", exist_ok=True); json.dump(out, open(f"runs/{a.tag}.json", "w"), indent=1)
torch.save({"As": {k: v.cpu() for k, v in As.items()}, "Amean": Amean.cpu()}, f"runs/{a.tag}_A.pt")
print(json.dumps(summ))
