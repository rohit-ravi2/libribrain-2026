"""ABS-1 stage 2: fuse the v3 decoder with the linear encoding model (runs/abs_trf.npz) on split-v2 Q3/Q4.

Per sentence epoch [first onset, last onset + dur + 1 s] (normalised as in the holdout, PCA + whitening of abs_fit):
  fixed part  = generic onset response of every word + first-word response + 'other' response of non-vocab words
  slot i      = vocab word, class unknown; neighbours j != i explained away by their expected response under the decoder
                posterior q_j (one mean-field step, independent of the fusion weight)
  gen_i(c)    = <r_i, R_c(t - o_i)> - 1/2 ||R_c||^2   (whitened, truncated at the epoch end)
  fused_i(c)  = log q_dec_i(c) + lam * gen_i(c); lam chosen on the select block (fold A: Q3 -> Q4, fold B: Q4 -> Q3); then Sinkhorn.
Also reports gen alone and gen with ORACLE neighbours (diagnostic). Only each subject's own rows are used.
Usage: abs_eval.py runs/<ckpt>.pt"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize, FS
from src.v3lib import Model, segments, sentences, predict, bacc_null, dev
from scripts.abs_fit import design, NC, GEN, OTHER, FIRST

t0 = time.time()
Z = np.load("runs/abs_trf.npz"); U, beta, Wh, L, K = Z["U"], Z["beta"], Z["Wh"], int(Z["L"]), int(Z["K"])
BW = beta @ Wh; Rc = BW.reshape(NC, L, K)                                          # whitened responses per class
M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = BROAD + [0]
import glob
from src.common import CACHE
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev)
print("runs", len(runs), "words", W.n, "load", round(time.time() - t0), "s", flush=True)
LAMS = [0.0] + list(np.logspace(-4, 0, 17))


def sinkhorn(lo, T, tgt, iters=10):
    P = F.softmax(lo / T, 1); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-9) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P.log()


def gen_scores(s, blk, logq):
    """Returns dict word idx -> (gen (50,), gen_oracle (50,)) for vocab words of subject s in block blk."""
    out = {}
    for sent in sentences(W, (q == blk) & (W.sub == s)):
        r = int(W.run[sent[0]]); meg = W.meg[r]
        a = int(round(W.s0[sent[0]] * FS)); b = int(round((W.s1[sent[0]] + 1.0) * FS))
        if a < 0 or b > meg.shape[1] or b - a < FS or b - a > 8 * FS: continue
        x = torch.from_numpy(np.ascontiguousarray(meg[:, a:b]).astype(np.float32))[None]
        yw = (normalize(x, torch.ones(1, b - a, dtype=torch.bool))[0].T.numpy() @ U) @ Wh; T = len(yw)
        o = np.round(W.onset[sent] * FS).astype(int) - a; lab = W.label[sent]; voc = lab >= 0
        cfix = np.where(voc, -1, OTHER)                                             # vocab slots: class part left out of the fixed design
        res = yw - design(T, o, cfix) @ BW
        res_true = yw - design(T, o, np.where(voc, lab + 1, OTHER)) @ BW           # oracle: all classes at truth
        E = {}
        for j in np.flatnonzero(voc):
            p = np.exp(logq[sent[j]]); E[j] = np.tensordot(p, Rc[1:51], 1)       # expected class response (L, K)
        for i in np.flatnonzero(voc):
            ri = res.copy(); rt = res_true.copy()
            for j in np.flatnonzero(voc):
                if j == i: continue
                t = o[j] + np.arange(L); ok = (t >= 0) & (t < T); ri[t[ok]] -= E[j][ok]
            t = o[i] + np.arange(L); ok = (t >= 0) & (t < T); tt = t[ok]
            rt[tt] += Rc[lab[i] + 1, ok]                                            # oracle residual with slot i's class added back
            Rk = Rc[1:51, ok]; nrm = 0.5 * (Rk ** 2).sum((1, 2))
            g = np.einsum("tk,ctk->c", ri[tt], Rk) - nrm; go = np.einsum("tk,ctk->c", rt[tt], Rk) - nrm
            out[int(sent[i])] = (g, go)
    return out


rows = {}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), maxw)
        if not segs: continue
        A = M.A[M.sid[0]] if s == 0 else I
        P = predict(M, W, segs, {s: A}); lo = F.log_softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau), 1).cpu().numpy()
        logq = {int(i): lo[k] for k, i in enumerate(P["idx"])}
        G = gen_scores(s, blk, logq)
        ids = [i for i in P["idx"] if W.label[i] >= 0 and int(i) in G]
        rows[(blk, s)] = dict(y=torch.from_numpy(W.label[ids]).long(), lq=torch.tensor(np.stack([logq[int(i)] for i in ids])),
                              g=torch.tensor(np.stack([G[int(i)][0] for i in ids])), go=torch.tensor(np.stack([G[int(i)][1] for i in ids])))
        print(blk, s, "rows", len(ids), round(time.time() - t0), "s", flush=True)


def score(blk, members, lam, kind="fused", sk=True):
    out = []
    for s in members:
        if (blk, s) not in rows: continue
        d = rows[(blk, s)]; y = d["y"]
        lo = {"fused": d["lq"] + lam * d["g"], "gen": d["g"], "oracle": d["go"], "dec": d["lq"]}[kind]
        if sk:
            pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres); lo = sinkhorn(lo, 3.0 if s == 0 else 2.0, tgt)
        out.append(bacc_null(lo, y)[0])
    return float(np.mean(out))


res = {}
for grp, members in (("broad", BROAD), ("deep", [0])):
    r = {k: {b: score(b, members, 0, k, sk=False) for b in ("Q3", "Q4")} for k in ("dec", "gen", "oracle")}
    r["dec+sk"] = {b: score(b, members, 0, "dec") for b in ("Q3", "Q4")}
    curve = {b: [score(b, members, lam) for lam in LAMS] for b in ("Q3", "Q4")}
    folds = {}
    for f, (sel, test) in S2.FOLDS.items():
        k = int(np.argmax(curve[sel])); folds[f] = dict(lam=LAMS[k], sel=curve[sel][k], test=curve[test][k], test_base=curve[test][0])
    r["folds"] = folds; r["curve"] = curve
    r["fused_test_mean"] = float(np.mean([v["test"] for v in folds.values()])); r["base_test_mean"] = float(np.mean([v["test_base"] for v in folds.values()]))
    res[grp] = r
    print("RESULT", grp, json.dumps({"dec": r["dec"], "gen": r["gen"], "oracle": r["oracle"], "dec+sk": r["dec+sk"], "folds": folds,
                                     "fused_mean": round(r["fused_test_mean"], 4), "base_mean": round(r["base_test_mean"], 4)}), flush=True)
json.dump({"res": res, "lams": LAMS, "ckpt": sys.argv[1], "trf_alpha": float(Z["alpha"])}, open("runs/abs_eval.json", "w"), indent=1)
