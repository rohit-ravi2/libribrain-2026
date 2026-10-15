"""PACKARD-1: pockets of predictability (Packard 1990, Complex Systems 4: conditions C on covariates scored by
F(C) = KL(P_C(y) || P_0(y)) - a/N_C, significance from a shuffle null, diversity across conditions).
mode test12 (frozen deep_c ep6, split-v2 Q3/Q4, 12 Broad subjects identity A + subj0 A0):
  T1 brain-state pocket: top-10 hit vs pre-onset band power (theta/alpha/beta, relative) and window artifact level.
  T2 exhaustive 1-2-variable interval conditions on quantile-binned covariates, fitness on y = hit, a chosen so the best
     shuffled condition has F <= 0; fit on Q3, replication on Q4; then condition-dependent Sinkhorn temperature
     (learned on Q3 per tercile of the best non-model covariate) scored on Q4 vs global T.
mode cart (T3): dataset cartography over deep_c ep0..ep7 on a subsample of subj0 TRAINING words; per word mean and std
  of p(true); Packard search for training-data conditions (task, recording day, time in run, position, sentence length,
  word frequency, pre-onset alpha, artifact) whose words are systematically clean or noisy.
Usage: packard_tests.py test12|cart"""
import sys, os, json, glob, time, csv
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, ON, FS
from src.common import CACHE, IS_MAG
from src.v3lib import Model, segments, predict, bacc_null, dev
MODE = sys.argv[1]; rng = np.random.default_rng(0); t0 = time.time(); GRAD = ~IS_MAG


def band_feats(W, idx):
    """Pre-onset 1 s (if inside the sentence epoch) relative theta/alpha/beta power (gradiometers) + window artifact level."""
    x, v = W.windows(idx); x = x.float().numpy(); v = v.numpy(); out = np.full((len(idx), 4), np.nan)
    f = np.fft.rfftfreq(ON, 1 / FS)
    for i in range(len(idx)):
        pre = x[i][GRAD][:, :ON]; ok = v[i, :ON]
        if ok.all():
            P = (np.abs(np.fft.rfft(pre - pre.mean(1, keepdims=True), axis=1)) ** 2).mean(0); tot = P[(f >= 1) & (f <= 24)].sum()
            out[i, 0] = P[(f >= 4) & (f < 8)].sum() / tot; out[i, 1] = P[(f >= 8) & (f < 13)].sum() / tot; out[i, 2] = P[(f >= 13) & (f <= 24)].sum() / tot
        seg = x[i][:, v[i]]; out[i, 3] = np.abs(seg - seg.mean(1, keepdims=True)).max() / (seg.std() + 1e-9) if seg.size else np.nan
    return out


def qbins(v, k=5):
    b = np.full(len(v), -1); ok = ~np.isnan(v)
    if ok.sum() > k: b[ok] = np.searchsorted(np.quantile(v[ok], np.linspace(0, 1, k + 1)[1:-1]), v[ok])
    return b


def kl_fit(y, m, nbin, a):
    """Packard fitness of the subset m for a (binned) outcome y with nbin states."""
    n = m.sum()
    if n < 30: return -9.0
    p0 = np.bincount(y, minlength=nbin) / len(y) + 1e-9; pc = np.bincount(y[m], minlength=nbin) / n + 1e-9
    return float((pc * np.log(pc / p0)).sum() - a / n)


def search(B, names, y, nbin, a, k=5):
    """All 1- and 2-variable conditions 'var in [lo, hi]' on binned covariates (contiguous intervals, not the full range)."""
    ivs = [(lo, hi) for lo in range(k) for hi in range(lo, k) if not (lo == 0 and hi == k - 1)]
    res = []
    for i in range(B.shape[1]):
        for (lo, hi) in ivs:
            mi = (B[:, i] >= lo) & (B[:, i] <= hi); res.append((kl_fit(y, mi, nbin, a), f"{names[i]}[{lo}-{hi}]", mi))
            for j in range(i + 1, B.shape[1]):
                for (lo2, hi2) in ivs:
                    mj = mi & (B[:, j] >= lo2) & (B[:, j] <= hi2); res.append((kl_fit(y, mj, nbin, a), f"{names[i]}[{lo}-{hi}]&{names[j]}[{lo2}-{hi2}]", mj))
    res.sort(key=lambda r: -r[0]); return res


def sinkhorn_p(P, T, tgt):
    P = P.clamp_min(1e-12) ** (1 / T[:, None] if torch.is_tensor(T) else 1 / T); P = P / P.sum(1, keepdim=True); n = P.shape[0]
    for _ in range(10): P = P / P.sum(0, keepdim=True).clamp_min(1e-12) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P


if MODE == "test12":
    M = Model("runs/deep_c_ep6.pt"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)
    runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
    runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
    W = Words(runs=runs, load_meg=True); q = W.quarters(); rows = {}
    for blk in ("Q3", "Q4"):
        for s in BROAD + [0]:
            segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
            if not segs: continue
            P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
            p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; idx = P["idx"][k.numpy()]; y = P["y"][k]
            pos = np.array([np.sum(W.onset[W.sent == W.sent[w]] < W.onset[w]) for w in idx]); slen = np.array([np.sum(W.sent == W.sent[w]) for w in idx])
            ent = -(p * p.clamp_min(1e-12).log()).sum(1).numpy(); hit = (p.topk(10, 1).indices == y[:, None]).any(1).numpy().astype(int)
            rows[(blk, s)] = dict(p=p, y=y, hit=hit, cov=np.column_stack([band_feats(W, idx), pos, slen, ent]))
    names = ["theta", "alpha", "beta", "artifact", "position", "sent_len", "entropy"]
    out = {}
    # T1: hit rate by covariate quintile (pooled over subject-blocks, binned within subject-block), permutation p-value
    for blk in ("Q3", "Q4"):
        R = [r for (b, s), r in rows.items() if b == blk]
        B = np.concatenate([np.column_stack([qbins(r["cov"][:, j]) for j in range(7)]) for r in R]); H = np.concatenate([r["hit"] for r in R])
        t1 = {}
        for j, nm in enumerate(names):
            ok = B[:, j] >= 0; rates = [float(H[ok & (B[:, j] == b)].mean()) for b in range(5)]
            obs = rates[4] - rates[0]; null = []
            for _ in range(200):
                Hs = H[ok].copy(); rng.shuffle(Hs); bb = B[ok, j]; null.append(Hs[bb == 4].mean() - Hs[bb == 0].mean())
            t1[nm] = dict(rates=[round(x, 3) for x in rates], q5_minus_q1=round(obs, 4), p=round(float(np.mean(np.abs(null) >= abs(obs))), 3), n=int(ok.sum()))
        out[f"T1_{blk}"] = t1
    # T2: Packard search on Q3 (non-model covariates only, then with entropy), a from a shuffle null; replicate on Q4
    R3 = [r for (b, s), r in rows.items() if b == "Q3"]; R4 = [r for (b, s), r in rows.items() if b == "Q4"]
    def stack(R): return np.concatenate([np.column_stack([qbins(r["cov"][:, j]) for j in range(7)]) for r in R]), np.concatenate([r["hit"] for r in R])
    B3, H3 = stack(R3); B4, H4 = stack(R4)
    for tag, cols in (("state", [0, 1, 2, 3, 4, 5]), ("with_entropy", list(range(7)))):
        nullbest = []
        for _ in range(10):
            Hs = H3.copy(); rng.shuffle(Hs); nullbest.append(search(B3[:, cols], [names[c] for c in cols], Hs, 2, 0.0)[0][0] if True else 0)
        a = float(np.max(nullbest)) * 300                                   # a/N_C at N_C=300 cancels the best shuffled KL
        top = search(B3[:, cols], [names[c] for c in cols], H3, 2, a)[:8]
        rep = []
        for f_, nm_, _ in top:
            m4 = np.ones(len(H4), bool)
            for term in nm_.split("&"):
                v, iv = term[:-1].split("["); lo, hi = map(int, iv.split("-")); m4 &= (B4[:, names.index(v)] >= lo) & (B4[:, names.index(v)] <= hi)
            m3 = np.ones(len(H3), bool)
            for term in nm_.split("&"):
                v, iv = term[:-1].split("["); lo, hi = map(int, iv.split("-")); m3 &= (B3[:, names.index(v)] >= lo) & (B3[:, names.index(v)] <= hi)
            rep.append(dict(cond=nm_, F=round(f_, 4), Q3_hit=round(float(H3[m3].mean()), 3), Q3_n=int(m3.sum()), Q4_hit=round(float(H4[m4].mean()), 3), Q4_n=int(m4.sum())))
        out[f"T2_{tag}"] = dict(a=round(a, 3), base_hit_Q3=round(float(H3.mean()), 3), base_hit_Q4=round(float(H4.mean()), 3), top=rep)
    # T2b: condition-dependent Sinkhorn temperature by tercile of each non-model covariate (learned on Q3, scored on Q4)
    Ts = (1.0, 1.5, 2.0, 3.0, 4.0); t2b = {}
    def bacc_blocks(R, Tfun):
        sc = []
        for r in R:
            y = r["y"]; pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
            sc.append(bacc_null(sinkhorn_p(r["p"], Tfun(r), tgt).log(), y)[0])
        return float(np.mean(sc))
    for j, nm in enumerate(names):
        def terc(r): return np.nan_to_num(qbins(r["cov"][:, j], 3), nan=1).clip(0, 2)
        g3 = bacc_blocks(R3, lambda r: 2.0); best = (2.0, 2.0, 2.0)
        for t_lo in Ts:
            for t_mid in Ts:
                for t_hi in Ts:
                    sc = bacc_blocks(R3, lambda r: torch.tensor(np.array([t_lo, t_mid, t_hi])[np.where(terc(r) < 0, 1, terc(r))], dtype=torch.float32))
                    if sc > g3 + 1e-9: g3, best = sc, (t_lo, t_mid, t_hi)
        q4 = bacc_blocks(R4, lambda r: torch.tensor(np.array(best)[np.where(terc(r) < 0, 1, terc(r))], dtype=torch.float32))
        t2b[nm] = dict(T=best, Q4=round(q4, 4))
    t2b["global_T2_Q4"] = round(bacc_blocks(R4, lambda r: 2.0), 4); out["T2b_condT"] = t2b
    json.dump(out, open("runs/packard_test12.json", "w"), indent=1, default=str)
    for k_, v_ in out.items(): print(k_, json.dumps(v_)[:1500], flush=True)

if MODE == "cart":
    os.environ.setdefault("LB_MMAP", "1")
    days = {}
    for f in glob.glob("/mnt/ssd4tb/data/LibriBrain100/*/sub-0/ses-*/sub-0_ses-*_scans.tsv"):
        for row in csv.reader(open(f, encoding="utf-8-sig"), delimiter="\t"):
            if row[0].startswith("meg/"): days[os.path.basename(row[0]).replace("_meg.fif", "")] = row[1][:10]
    runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz")]
    runs = [b for b in runs if not any(s in b for s in SHARED)]
    W = Words(runs=runs, load_meg=True); M0 = Model("runs/deep_c_ep0.pt")
    sents = segments(W, W.label > -2, M0.a["maxw"]); pick = [sents[i] for i in rng.choice(len(sents), 3000, replace=False)]
    idx_all = np.concatenate(pick); voc = W.label[idx_all] >= 0
    PT = []
    for ep in range(8):
        M = Model(f"runs/deep_c_ep{ep}.pt"); P = predict(M, W, pick, {0: M.A[M.sid[0]]})
        assert np.array_equal(P["idx"], idx_all)
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu().numpy()
        PT.append(p[np.arange(len(p)), np.clip(W.label[idx_all], 0, 49)]); print("epoch", ep, round(time.time() - t0), "s", flush=True)
    PT = np.array(PT)[:, voc]; idx = idx_all[voc]; conf = PT[4:].mean(0); var = PT[4:].std(0)
    task = np.array([W.bases[r].split("_task-")[1].split("_")[0] for r in W.run[idx]]); tasks = sorted(set(task))
    dayk = np.array([days.get(W.bases[r], "na") for r in W.run[idx]]); dlist = sorted(set(dayk))
    run_len = np.array([W.meg[r].shape[1] / FS for r in W.run[idx]]); trun = W.onset[idx] / run_len
    pos = np.array([np.sum(W.onset[W.sent == W.sent[w]] < W.onset[w]) for w in idx]); slen = np.array([np.sum(W.sent == W.sent[w]) for w in idx])
    freq = np.bincount(W.label[idx], minlength=50)[W.label[idx]]; bf = band_feats(W, idx)
    cov = {"task": np.array([tasks.index(t) for t in task]), "day_rank": qbins(np.array([dlist.index(d) for d in dayk], float)),
           "time_in_run": qbins(trun), "position": qbins(pos.astype(float)), "sent_len": qbins(slen.astype(float)),
           "word_freq": qbins(freq.astype(float)), "alpha": qbins(bf[:, 1]), "artifact": qbins(bf[:, 3])}
    yb = qbins(conf, 4).clip(0); out = {"n_words": int(len(idx)), "tasks": tasks, "conf_mean": round(float(conf.mean()), 4)}
    # per-variable mean confidence by level (task is categorical)
    for nm, b in cov.items():
        out[f"conf_by_{nm}"] = {int(l): [round(float(conf[b == l].mean()), 4), int((b == l).sum())] for l in np.unique(b) if l >= 0}
    names = [n for n in cov if n != "task"]; B = np.column_stack([cov[n] for n in names])
    nullbest = []
    for _ in range(5):
        ys = yb.copy(); rng.shuffle(ys); nullbest.append(search(B, names, ys, 4, 0.0)[0][0])
    a = float(np.max(nullbest)) * 300; top = search(B, names, yb, 4, a)[:10]
    out["search_a"] = round(a, 3); out["top"] = [dict(cond=n_, F=round(f_, 4), n=int(m_.sum()), conf=round(float(conf[m_].mean()), 4), var=round(float(var[m_].mean()), 4)) for f_, n_, m_ in top]
    json.dump(out, open("runs/packard_cart.json", "w"), indent=1, default=str)
    for k_, v_ in out.items(): print(k_, json.dumps(v_)[:1200], flush=True)
