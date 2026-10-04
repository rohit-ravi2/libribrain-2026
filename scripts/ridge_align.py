"""P1b: closed-form ridge sensor map X_s -> X_0 on exact token pairs (no labels), frozen B0 decoder. CPU-friendly."""
import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN
torch.set_num_threads(8)
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
ck = torch.load("runs/b0.pt", map_location=dev); model = WordCNN().to(dev); model.load_state_dict(ck["model"]); model.eval()
adj = ck["tau"] * ck["log_prior"].to(dev)
shared = lambda ses: ses in S.SHARED


def meta(sel):
    out = []
    for f in sorted(glob.glob(f"{S.CACHE}/runs/*.npz")):
        base = os.path.basename(f)[:-4]; sub, ses = S.parse(base); t = np.load(f)["timechapter"]
        m = sel(base, sub, ses, t)
        if m is not None and np.any(m): out += [(ses, round(float(x), 2), base, float(x)) for x in t[m]]
    return out


sel0 = lambda b, s, ses, t: S.block_masks(b, t)["pre75"] if (s == 0 and shared(ses)) else None
selp = lambda b, s, ses, t: S.train_mask(b, t) if (s > 0 and shared(ses)) else None
sele = lambda b, s, ses, t: np.ones(len(t), bool) if s in S.EVAL_SUBJECTS else None
D = load_sets({"z": sel0, "p": selp, "e": sele})
X0 = normalize(D["z"][0]); key0 = {(ses, tc): i for i, (ses, tc, _, _) in enumerate(meta(sel0))}
Xp = normalize(D["p"][0]); sp = D["p"][2]; mp = meta(selp)
Xe = normalize(D["e"][0]); ye = D["e"][1]; se = D["e"][2]; me = meta(sele)
pp = np.array([key0.get((ses, tc), -1) for ses, tc, _, _ in mp]); pe = np.array([key0.get((ses, tc), -1) for ses, tc, _, _ in me])


def ridge(Xs, Xt, rel):
    xs = Xs.permute(1, 0, 2).reshape(306, -1).double(); xt = Xt.permute(1, 0, 2).reshape(306, -1).double()
    C = xs @ xs.T; al = rel * torch.trace(C) / 306; I = torch.eye(306, dtype=torch.float64)
    return ((xt @ xs.T + al * I) @ torch.linalg.inv(C + al * I)).float()


@torch.no_grad()
def score(A, X, y):
    out = []
    for i in range(0, len(X), 1024):
        x = normalize(torch.einsum("ij,bjt->bit", A, X[i:i + 1024]).to(dev))
        out.append(model(x).float() - adj)
    return bacc_at_k(torch.cat(out).cpu(), y)


tc_e = np.array([x[3] for x in me]); be = [x[2] for x in me]
masks = {k: np.zeros(len(me), bool) for k in ["B", "fs_s25", "fs_s50", "fs_s100"]}
for b in set(be):
    idx = np.array([i for i, x in enumerate(be) if x == b]); m = S.block_masks(b, tc_e[idx])
    for k in masks: masks[k][idx] = m[k]
results = {}
t0 = time.time()
for rel in (1.0, 0.1, 10.0):
    As = []
    for s in sorted(set(sp.tolist())):
        m = (sp == s).numpy() & (pp >= 0); As.append(ridge(Xp[m], X0[pp[m]], rel))
    Am = torch.stack(As).mean(0)
    res = {"s0_meanA": {}, "s25": {}, "s50": {}, "s100": {}}
    for s in S.EVAL_SUBJECTS:
        ms = (se == s).numpy(); Bm = torch.from_numpy(ms & masks["B"])
        res["s0_meanA"][s] = score(Am, Xe[Bm], ye[Bm])
        for g in ["s25", "s50", "s100"]:
            fm = ms & masks["fs_" + g] & (pe >= 0)
            res[g][s] = score(ridge(Xe[fm], X0[pe[fm]], rel), Xe[Bm], ye[Bm])
    summ = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    summ["composite"] = S.composite({"s100": summ["s100"], "s50": summ["s50"], "s25": summ["s25"], "s0": summ["s0_meanA"]})
    results[rel] = {"summary": summ, "per_subject": res}
    print(rel, json.dumps(summ), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(results, open("runs/p1b_ridge.json", "w"), indent=1)
