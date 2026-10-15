"""REP-1: within-sentence repetition coupling. For word pairs inside one sentence (same brain state, same day),
does MEG similarity tell 'same word' apart from 'different word' BEYOND the decoder posteriors?
AUC of same-vs-diff from (a) p_i.p_j (posterior agreement), (b) embedding cosine, (c) raw post-onset cosine,
and logistic (a)+(b)+(c) vs (a) alone, fit on Q3 pairs -> scored on Q4 pairs. Broad subjects (identity A) + subj0."""
import sys, os, glob, json
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize, ON, FS
from src.common import CACHE
from src.v3lib import Model, segments, predict, dev
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(6)
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")); I = torch.eye(306, device=dev)
SUBS = [0, 1, 2, 3, 4, 5, 6]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); rows = {"Q3": [], "Q4": []}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = (P["y"] >= 0).numpy()
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu().numpy()[k]
        e = P["e"].numpy()[k]; e /= np.linalg.norm(e, axis=1, keepdims=True); idx = P["idx"][k]; y = W.label[idx]
        x, v = W.windows(idx); r = normalize(x, v).numpy()[:, :, ON:ON + int(0.6 * FS)].reshape(len(idx), -1)
        r -= r.mean(0); r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
        sent = W.sent[idx]; on = W.onset[idx]
        for sid in np.unique(sent):
            m = np.flatnonzero(sent == sid)
            for a in range(len(m)):
                for b in range(a + 1, len(m)):
                    i, j = m[a], m[b]
                    if abs(on[i] - on[j]) < 0.6: continue          # avoid overlapping post-onset windows
                    rows[blk].append((float(p[i] @ p[j]), float(e[i] @ e[j]), float(r[i] @ r[j]), int(y[i] == y[j]), s))
        print(blk, s, len(rows[blk]), flush=True)
out = {}
A3, A4 = np.array(rows["Q3"]), np.array(rows["Q4"])
for g, sel in (("broad", lambda A: A[:, 4] != 0), ("deep", lambda A: A[:, 4] == 0)):
    a3, a4 = A3[sel(A3)], A4[sel(A4)]; y3, y4 = a3[:, 3], a4[:, 3]
    o = {"n_pairs_Q4": int(len(y4)), "same_frac": round(float(y4.mean()), 3)}
    for nm, c in (("post_agree", 0), ("emb_cos", 1), ("raw_cos", 2)): o[nm] = round(roc_auc_score(y4, a4[:, c]), 4)
    for nm, cols in (("lr_post", [0]), ("lr_all", [0, 1, 2]), ("lr_post_raw", [0, 2])):
        lr = LogisticRegression(max_iter=2000).fit(a3[:, cols], y3); o[nm] = round(roc_auc_score(y4, lr.predict_proba(a4[:, cols])[:, 1]), 4)
    out[g] = o; print(g, o, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rep_pair.json"), "w"), indent=1)
