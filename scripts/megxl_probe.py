"""MEGXL-P: is there complementary evidence in frozen MEG-XL features? Linear probes (top-10 bacc, chance 0.20) on
split-v2 vocab words, train Q1-Q2 / test Q3+Q4: (a) our frozen deep_c ep6 word embedding (256-d, sentence context),
(b) MEG-XL features (channel-mean, post-onset tokens, PCA 256), (c) concatenation. subj0 and pooled Broad subjects 3/6/9."""
import sys, os, json, glob
import numpy as np, torch, torch.nn.functional as F
sys.argv = ["megxl_feats.py"]; src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "megxl_feats.py")).read().split("if a.memtest")[0]
exec(compile(src, "megxl_feats.py", "exec"))
from src.split import SHARED
from src.cont import Words
from src.v3lib import Model, segments, predict
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
M = Model("runs/deep_c_ep6.pt"); I = torch.eye(306, device=dev); SUBS = (0, 3, 6, 9)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); D = {}
for s in SUBS:
    segs = segments(W, np.isin(q, ["Q1", "Q2", "Q3", "Q4"]) & (W.sub == s), M.a["maxw"])
    P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = (P["y"] >= 0).numpy(); idx = P["idx"][k]; e = P["e"][torch.from_numpy(k)].numpy()
    X = []
    for i in range(0, len(idx), 16):
        x, v = W.windows(idx[i:i + 16])
        with torch.autocast("cuda", torch.bfloat16): f = feats(x)
        X.append(f.float().mean(1)[:, 4:10].reshape(len(x), -1).cpu().numpy())
    D[s] = dict(e=e, x=np.concatenate(X), y=W.label[idx], tr=np.isin(q[idx], ["Q1", "Q2"]))
    print("subject", s, len(idx), flush=True)
def bacc10(P, y): top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))
def probe(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    P = np.zeros((len(yte), 50)); P[:, clf.classes_] = clf.predict_proba((Xte - mu) / sd); return round(bacc10(P, yte), 4)
out = {}
for grp, ss in (("deep", [0]), ("broad", [3, 6, 9])):
    cat = lambda key, m: np.concatenate([D[s][key][D[s]["tr"] == m] for s in ss])
    pca = PCA(256, random_state=0).fit(cat("x", True)); xtr, xte = pca.transform(cat("x", True)), pca.transform(cat("x", False))
    etr, ete, ytr, yte = cat("e", True), cat("e", False), cat("y", True), cat("y", False)
    out[grp] = dict(ours=probe(etr, ytr, ete, yte), megxl=probe(xtr, ytr, xte, yte), concat=probe(np.hstack([etr, xtr]), ytr, np.hstack([ete, xte]), yte), n_tr=int(len(ytr)))
    print(grp, out[grp], flush=True)
json.dump(out, open("runs/megxl_probe.json", "w"), indent=1)
