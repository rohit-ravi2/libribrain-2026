"""B0: compact CNN on subj0 only (train mask of src/split.py), no alignment, no TTA.
Early stopping on subj0 Sherlock1 ses-10 (held out of training; a different stimulus from block B).
Local scores: subj0 block B (within-subject, new stimulus) and eval subjects' block B (cross-subject, new stimulus).
Logit adjustment (subtract log class prior) at inference, since the metric is class-balanced."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="b0"); ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
ESVAL = "sub-0_ses-10_task-Sherlock1_run-1"

D = load_sets({
    "tr": lambda b, s, ses, t: S.train_mask(b, t) if (s == 0 and b != ESVAL) else None,
    "es": lambda b, s, ses, t: np.ones(len(t), bool) if b == ESVAL else None,
    "0B": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s == 0 else None,
    "eB": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s in S.EVAL_SUBJECTS else None})
(Xtr, ytr, _, _), (Xes, yes, _, _), (X0B, y0B, _, _), (XeB, yeB, seB, _) = D["tr"], D["es"], D["0B"], D["eB"]
del D
if a.frac < 1:
    idx = torch.randperm(len(Xtr))[: int(a.frac * len(Xtr))]; Xtr, ytr = Xtr[idx], ytr[idx]
print(f"train {len(Xtr)} es {len(Xes)} subj0-B {len(X0B)} eval-B {len(XeB)}", flush=True)
log_prior = torch.log(torch.bincount(ytr, minlength=50).float().clamp_min(1) / len(ytr)).to(dev)

model = WordCNN().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
steps = a.epochs * (len(Xtr) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)


@torch.no_grad()
def predict(X):
    model.eval(); out = []
    for i in range(0, len(X), 2048):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(model(normalize(X[i:i + 2048].to(dev))).float())
    return torch.cat(out)


TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)   # logit-adjustment strength, chosen on the early-stop set only


def scores():
    les = predict(Xes)
    es = {tau: bacc_at_k((les - tau * log_prior).cpu(), yes) for tau in TAUS}
    tau = max(es, key=es.get); adj = lambda l: (l - tau * log_prior).cpu()
    r = {"es": es[tau], "tau": tau, "es_raw": es[0.0], "subj0_B": bacc_at_k(adj(predict(X0B)), y0B)}
    le = adj(predict(XeB))
    per = {int(s): bacc_at_k(le[seB == s], yeB[seB == s]) for s in S.EVAL_SUBJECTS}
    r["eval_B_per_subj"] = per; r["eval_B_mean"] = float(np.mean(list(per.values())))
    lp = log_prior.cpu()
    r["prior_floor_eval_B"] = float(np.mean([bacc_at_k(lp.expand(int((seB == s).sum()), 50), yeB[seB == s]) for s in S.EVAL_SUBJECTS]))
    r["prior_floor_subj0_B"] = bacc_at_k(lp.expand(len(y0B), 50), y0B)
    r["bacc1_eval_B_mean"] = float(np.mean([bacc_at_k(le[seB == s], yeB[seB == s], k=1) for s in S.EVAL_SUBJECTS]))
    return r


os.makedirs("runs", exist_ok=True); best, hist = -1, []
t0 = time.time()
for ep in range(a.epochs):
    model.train(); perm = torch.randperm(len(Xtr)); tl = 0
    for i in range(len(Xtr) // a.bs):
        b = perm[i * a.bs:(i + 1) * a.bs]
        x, y = normalize(Xtr[b].to(dev, non_blocking=True)), ytr[b].to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
    r = scores(); r.update(epoch=ep, train_loss=tl / (len(Xtr) // a.bs), t=round(time.time() - t0)); hist.append(r)
    print(json.dumps(r), flush=True)
    if r["es"] > best:
        best = r["es"]; torch.save({"model": model.state_dict(), "log_prior": log_prior.cpu(), "tau": r["tau"], "args": vars(a), "epoch": ep}, f"runs/{a.tag}.pt")
sel = max(hist, key=lambda h: h["es"])
json.dump({"selected_by": "es (subj0 Sherlock1 ses-10)", "selected": sel, "history": hist, "args": vars(a)},
          open(f"runs/{a.tag}.json", "w"), indent=1)
print("SELECTED", json.dumps(sel))
