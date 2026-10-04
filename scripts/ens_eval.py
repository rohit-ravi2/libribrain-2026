"""Zero-shot (mean Broad A) ensemble evaluation on eval and VAL subjects' block B: mean of softmaxes over checkpoints."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split as S
from src.data import load_sets, normalize, bacc_at_k
from src.models import WordCNN
cks = sys.argv[1].split(","); out = sys.argv[2]; dev = "cuda"
D = load_sets({"ev": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s in S.EVAL_SUBJECTS else None,
               "va": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s in S.VAL_SUBJECTS else None})
res = {}
for name, (X, y, ss, _) in D.items():
    P = []
    for c in cks:
        ck = torch.load(c, map_location=dev); m = WordCNN().to(dev); m.load_state_dict(ck["model"]); m.eval()
        subs = ck["subs"]; Ab = ck["A"][[subs.index(s) for s in subs if s != 0]].to(dev).mean(0); adj = ck["tau"] * ck["log_prior"].to(dev)
        with torch.no_grad():
            lo = torch.cat([m(torch.einsum("ij,bjt->bit", Ab, normalize(X[i:i + 2048].to(dev)))).float() - adj for i in range(0, len(X), 2048)])
        P.append(torch.softmax(lo, 1).cpu())
    subj = S.EVAL_SUBJECTS if name == "ev" else S.VAL_SUBJECTS
    single = [float(np.mean([bacc_at_k(p[ss == s], y[ss == s]) for s in subj])) for p in P]
    ens = torch.stack(P).mean(0)
    res[name] = {"single": single, "ensemble": float(np.mean([bacc_at_k(ens[ss == s], y[ss == s]) for s in subj]))}
    print(name, json.dumps(res[name]), flush=True)
json.dump({"ckpts": cks, **res}, open(out, "w"), indent=1)
