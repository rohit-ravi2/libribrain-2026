"""v3 trainer + THIRD training stream from an external MEG corpus (Bet C, docs/extmeg/PLAN.md): scripts/train_v3.py
(copied 2026-10-13 10:40 EDT, incl. --vocab_runs) with --ext_frac f (segments per batch drawn from ext sentences, ADDED to the
LibriBrain segments by default, SigLIP-to-T5 loss only; ext words carry label -1 so they never reach the CE loss or any score)
and --ext_subjects N (first N ext subjects, own A each). With --ext_frac 0 the code path is train_v3.py exactly. Evaluation
(split v2, --deep scoring) is untouched: only LibriBrain words are ever scored. Word ids >= W.n index the ext table
(scripts/extmeg/ext_words.ExtWords, subject ids 1000+).

Original header:
 (src/cont.py), split v2 blocks with window-aware guards.
Rungs (PREDICTIONS.md "V3 ladder"): --win 1s|3s, --ctx 0|1 (sentence transformer), --t5 W (SigLIP to T5 targets, all words).
Per epoch, blocks Q3/Q4 are scored for SEEN (own A), VAL and EVAL (mean Broad A), with context windows and with
1 s-only windows; selection (epoch, readout, tau) on the fold's select block; deploy = (32 seen + 7 zs)/39;
deploy_mix = 0.9 ctx + 0.1 one-second. Every score carries its label-permutation null."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.cont import Words, normalize, WIN, FS
from src.models import CtxEncoder, SentCtx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "extmeg"))
from ext_words import ExtWords

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--bs_seg", type=int, default=32); ap.add_argument("--maxw", type=int, default=12)
ap.add_argument("--win", default="3s", choices=["1s", "3s"]); ap.add_argument("--ctx", type=int, default=1)
ap.add_argument("--t5", type=float, default=0.0); ap.add_argument("--ce", type=float, default=1.0)
ap.add_argument("--w2v", type=float, default=0.0, help="weight of a second SigLIP head to wav2vec2-base L8 acoustic targets (rows with audio only)")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--lam", type=float, default=1e-3)
ap.add_argument("--d", type=int, default=128); ap.add_argument("--D", type=int, default=256); ap.add_argument("--p", type=float, default=0.3)
ap.add_argument("--broad_frac", type=float, default=0.5); ap.add_argument("--aug", type=int, default=1)
ap.add_argument("--p_iso", type=float, default=0.15); ap.add_argument("--p_short", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", default=None)
ap.add_argument("--upweight", default=None, help="regex over run base names: sentences from matching runs are oversampled by --upw_factor")
ap.add_argument("--upw_factor", type=float, default=4.0)
ap.add_argument("--vocab_runs", type=float, default=0.0, help="fraction of subj0 segments drawn from natural runs of >=4 consecutive vocab words (holdout-like contexts)")
ap.add_argument("--exclude", default=None, help="regex over run base names: excluded from training AND scored as the PROXY block (vocab words, sentence segments, subj0 own A)")
ap.add_argument("--deep", type=int, default=0, help="1 = Deep track: also score subj0 (own A) on Q3/Q4 and SELECT on subj0")
ap.add_argument("--ctx_layers", type=int, default=2)
ap.add_argument("--rel", type=int, default=0, help="1 = feed relative word-onset timing to the sentence transformer (stimulus metadata; OFF by default since 2026-10-10)")
ap.add_argument("--ext_frac", type=float, default=0.0, help="ext segments per batch = round(ext_frac * bs_seg), ADDED to the batch (LibriBrain streams unchanged); SigLIP-only")
ap.add_argument("--ext_subjects", type=int, default=58, help="use the first N ext-corpus subjects")
ap.add_argument("--ext_corpus", default="lpp_listen")
ap.add_argument("--ext_mode", default="add", choices=["add", "replace"], help="add: batch = bs_seg LB + n_ext ext; replace: n_ext ext segments displace subj0 segments (bs_seg total)")
ap.add_argument("--final", type=int, default=0, help="1 = train on ALL subjects and ALL stimulus (no local eval blocks); saves per-epoch ckpts only")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"
a.lb_win = os.environ.get("LB_WIN", "1.0,2.0")   # recorded so inference can set the same window
a.lb_fs = int(os.environ.get("LB_FS", 50))
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0); READOUTS = ("ce", "cos", "ce+cos") if a.t5 > 0 else ("ce",)
os.makedirs("runs", exist_ok=True); t0 = time.time()

W = Words(load_meg=True, load_t5=a.t5 > 0, load_w2v=a.w2v > 0); q = W.quarters()
scored = set(S2.EVAL + S2.VAL + S2.SEEN) | ({0} if a.deep else set())
trainable = ~np.isin(W.sub, S2.EVAL + S2.VAL) & np.isin(q, ["other", "Q1", "Q2"])
if a.final: trainable = q != ""
import re as _re
excl = np.zeros(W.n, bool)
if a.exclude:
    for r_, b_ in enumerate(W.bases):
        if _re.search(a.exclude, b_): excl[W.run == r_] = True
    trainable &= ~excl; print("excluded runs:", [b for b in W.bases if _re.search(a.exclude, b)], flush=True)                      # everything (guard-dropped boundary words excluded only)
if a.t5 > 0:
    has_t5 = np.zeros(W.n, bool)
    for r in range(len(W.bases)):
        m = W.run == r; has_t5[m] = np.abs(W.t5[r]).sum(1) > 0
    trainable &= has_t5
subs = sorted(set(W.sub[trainable].tolist())); sid = {s: i for i, s in enumerate(subs)}
vocab_tr = trainable & (W.label >= 0)
i0 = np.flatnonzero(vocab_tr & (W.sub == 0)); ib = np.flatnonzero(vocab_tr & (W.sub != 0))
p0 = np.bincount(W.label[i0], minlength=50) / len(i0); pb = (np.bincount(W.label[ib], minlength=50) / len(ib)) if len(ib) else p0
log_prior = torch.log(torch.tensor((1 - a.broad_frac) * p0 + a.broad_frac * pb).float().clamp_min(1e-8)).to(dev)


def sentences(mask):
    """List of word-id arrays (onset-ordered) per sentence restricted to mask."""
    idx = np.flatnonzero(mask); order = np.lexsort((W.onset[idx], W.sent[idx])); idx = idx[order]
    cuts = np.flatnonzero(np.diff(W.sent[idx])) + 1
    return np.split(idx, cuts)


S0 = [s for s in sentences(trainable & (W.sub == 0))]; SB = [s for s in sentences(trainable & (W.sub != 0))]
# holdout-like oversampling: (a) whole runs matching --upweight (e.g. TIMIT/MOCHA read sentences); (b) natural vocab-only runs
if a.upweight:
    up = [s for s in S0 if _re.search(a.upweight, W.bases[int(W.run[s[0]])])]
    S0 = S0 + up * (int(a.upw_factor) - 1); print("upweighted sentences", len(up), "x", int(a.upw_factor), flush=True)
VR = []
if a.vocab_runs > 0:
    for s in S0:
        lab = W.label[s]; i = 0
        while i < len(s):
            j = i
            while j < len(s) and lab[j] >= 0: j += 1
            if j - i >= 4: VR.append(s[max(0, i - 1):min(len(s), j + 1)])   # the vocab run plus one neighbour each side
            i = j if j > i else i + 1
    print("vocab-only runs (>=4 words) for oversampling:", len(VR), flush=True)
print(f"train words subj0 {len(i0)} vocab / {sum(map(len, S0))} all; broad {len(ib)} vocab / {sum(map(len, SB))} all ({len(subs) - 1} subj); "
      f"sentences {len(S0)} / {len(SB)}; load {round(time.time() - t0)}s", flush=True)

# ---- external third stream (ext ids are offset by W.n) ----
n_ext = int(round(a.bs_seg * a.ext_frac)) if a.ext_frac > 0 else 0
if n_ext > 0:
    assert a.t5 > 0, "--ext_frac needs --t5 > 0 (ext words are SigLIP-only)"
    EXT = ExtWords(a.ext_corpus, subjects=a.ext_subjects, load_meg=True, load_t5=True, mmap=True)
    ext_ok = EXT.has_t5(); ext_subs = sorted(set(EXT.sub.tolist()))
    for k_, s_ in enumerate(ext_subs): sid[s_] = len(subs) + k_
    idx_e = np.flatnonzero(ext_ok); order_e = np.lexsort((EXT.onset[idx_e], EXT.sent[idx_e])); idx_e = idx_e[order_e]
    SE = [c + W.n for c in np.split(idx_e, np.flatnonzero(np.diff(EXT.sent[idx_e])) + 1)]
    print(f"ext {a.ext_corpus}: {len(EXT.bases)} runs / {len(ext_subs)} subjects (ids {ext_subs[0]}..{ext_subs[-1]}), {len(idx_e)} words with T5, "
          f"{len(SE)} sentences, {EXT.hours():.1f} h, n_ext {n_ext}/batch ({a.ext_mode}); load {round(time.time() - t0)}s", flush=True)
else:
    EXT = None; ext_subs = []; SE = []


def g_split(idx):
    """global word ids -> (mask of LB ids, LB ids, ext ids)"""
    lb = idx < W.n; return lb, idx[lb], idx[~lb] - W.n


def g_windows(idx, iso=None, clip=None):
    lb, il, ie = g_split(idx)
    if lb.all(): return W.windows(idx, iso=iso, clip=clip)
    sub_ = lambda arr, m: None if arr is None else arr[m]
    xl, vl = W.windows(il, iso=sub_(iso, lb), clip=None if clip is None else (clip[0][lb], clip[1][lb]))
    xe, ve = EXT.windows(ie, iso=sub_(iso, ~lb), clip=None if clip is None else (clip[0][~lb], clip[1][~lb]))
    x = torch.empty(len(idx), 306, WIN, dtype=torch.float16); v = torch.zeros(len(idx), WIN, dtype=torch.bool)
    lbt = torch.from_numpy(lb); x[lbt] = xl; v[lbt] = vl; x[~lbt] = xe; v[~lbt] = ve
    return x, v


def g_t5_rows(idx):
    lb, il, ie = g_split(idx)
    if lb.all(): return W.t5_rows(idx)
    out = np.zeros((len(idx), 1024), np.float32); out[lb] = W.t5_rows(il); out[~lb] = EXT.t5_rows(ie); return out


def g_label(idx):
    lb, il, _ = g_split(idx); out = np.full(len(idx), -1, np.int64); out[lb] = W.label[il]; return out


def g_sub(idx):
    lb, il, ie = g_split(idx); out = np.zeros(len(idx), np.int64); out[lb] = W.sub[il]
    if (~lb).any(): out[~lb] = EXT.sub[ie]
    return out


def g_onset(idx):
    lb, il, ie = g_split(idx); out = np.zeros(len(idx), np.float64); out[lb] = W.onset[il]
    if (~lb).any(): out[~lb] = EXT.onset[ie]
    return out


# T5 prototypes (train vocab words only) and target centering
if a.t5 > 0:
    samp = np.sort(np.concatenate([rng.choice(i0, min(len(i0), 20000), replace=False), rng.choice(ib, min(len(ib), 20000), replace=False)]))
    Tt = torch.from_numpy(W.t5_rows(samp)); t_mean = Tt.mean(0)
    Tn = F.normalize(Tt - t_mean, dim=1); proto = torch.zeros(50, 1024)
    for c in range(50):
        m = W.label[samp] == c
        if m.any(): proto[c] = F.normalize(Tn[m].mean(0), dim=0)
    proto, t_mean = proto.to(dev), t_mean.to(dev)
if a.w2v > 0:
    samp = np.sort(rng.choice(np.flatnonzero(trainable), 20000, replace=False)); Wt = torch.from_numpy(W.w2v_rows(samp))
    w2v_mean = Wt[torch.isfinite(Wt).all(1)].mean(0).to(dev)

enc = CtxEncoder(d=a.d, D=a.D, T=WIN, p=a.p).to(dev)
ctx = SentCtx(D=a.D, maxw=a.maxw, layers=a.ctx_layers).to(dev) if a.ctx else None
head_ce = nn.Linear(a.D, 50).to(dev); head_t5 = nn.Linear(a.D, 1024).to(dev); head_w2v = nn.Linear(a.D, 768).to(dev)
logit_s2 = nn.Parameter(torch.tensor(np.log(10.0), device=dev)); logit_b2 = nn.Parameter(torch.tensor(-10.0, device=dev))
logit_s = nn.Parameter(torch.tensor(np.log(10.0), device=dev)); logit_b = nn.Parameter(torch.tensor(-10.0, device=dev))
A = nn.Parameter(torch.eye(306, device=dev).repeat(len(subs) + len(ext_subs), 1, 1)); I = torch.eye(306, device=dev)
mods = [enc, head_ce, head_t5, head_w2v] + ([ctx] if ctx else [])
if a.init:
    ck = torch.load(a.init, map_location=dev); enc.load_state_dict(ck["enc"]); head_ce.load_state_dict(ck["head_ce"])
    if ctx and ck.get("ctx"): ctx.load_state_dict(ck["ctx"])
params = [p for m in mods for p in m.parameters()]
opt = torch.optim.AdamW([{"params": params, "weight_decay": a.wd}, {"params": [A, logit_s, logit_b, logit_s2, logit_b2], "weight_decay": 0.0}], lr=a.lr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
n0 = int(a.bs_seg * (1 - a.broad_frac)); n1 = a.bs_seg - n0
if n_ext > 0 and a.ext_mode == "replace": n0 = max(n0 - n_ext, 1)


def segment(sent, maxw, train=True):
    """Chunk a sentence (word ids) into consecutive segments of <= maxw words; training takes one random chunk."""
    if len(sent) <= maxw: return [sent]
    if train:
        s = rng.integers(0, len(sent) - maxw + 1); return [sent[s:s + maxw]]
    return [sent[i:i + maxw] for i in range(0, len(sent), maxw)]


def pack(segs):
    """segs: list of word-id arrays -> flat idx, seg index per word, (S,L) gather index, pad mask, rel onset."""
    idx = np.concatenate(segs); L = max(map(len, segs)); S = len(segs)
    gi = np.zeros((S, L), np.int64); pad = np.ones((S, L), bool); rel = np.zeros((S, L), np.float32); o = 0
    for k, s in enumerate(segs):
        gi[k, :len(s)] = np.arange(o, o + len(s)); pad[k, :len(s)] = False
        if a.rel: on_ = g_onset(s); rel[k, :len(s)] = on_ - on_[0]
        o += len(s)
    return idx, gi, pad, rel


def augment(x, v):
    B = x.shape[0]
    x = x * (1 + 0.1 * torch.randn(B, 306, 1, device=dev)) * (torch.rand(B, 306, 1, device=dev) > 0.1)
    x = torch.bmm(I + 0.05 * torch.randn(B, 306, 306, device=dev) / 306 ** 0.5, x)
    return x * v[:, None, :]


def forward(idx, gi, pad, rel, Am, iso=None, clip=None, train=False):
    """Returns per-word (emb, ce logits, t5 proj) for the flat word list."""
    x, v = g_windows(idx, iso=iso, clip=clip); x, v = x.to(dev, non_blocking=True), v.to(dev, non_blocking=True)
    x = normalize(x, v)
    x = torch.einsum("bij,bjt->bit", Am, x) if Am.dim() == 3 else torch.einsum("ij,bjt->bit", Am, x)
    if train and a.aug: x = augment(x, v)
    x = torch.cat([x, v[:, None, :].float()], 1)
    with torch.autocast("cuda", torch.bfloat16):
        e = enc(x)
        if ctx is not None:
            gi_t = torch.as_tensor(gi, device=dev); pad_t = torch.as_tensor(pad, device=dev); rel_t = torch.as_tensor(rel, device=dev)
            E = e[gi_t] * (~pad_t)[..., None]
            E = ctx(E, rel_t, pad_t)
            e = E[~pad_t]                                   # back to flat order (row-major over segments = pack order)
        return e.float(), head_ce(e).float(), head_t5(e).float()


def forward_w2v(e):
    return head_w2v(e).float()


def siglip(z, idx):
    t = torch.from_numpy(g_t5_rows(idx)).to(dev); ok = t.abs().sum(1) > 0
    z, t = F.normalize(z[ok], dim=1), F.normalize(t[ok] - t_mean, dim=1)
    pos = (t @ t.T) > 0.999                                  # duplicate-aware positives (same token, other subject)
    lg = logit_s.exp() * (z @ t.T) + logit_b
    return -F.logsigmoid(torch.where(pos, lg, -lg)).sum() / len(z)


def siglip_w2v(z, idx):
    t = torch.from_numpy(W.w2v_rows(idx)).to(dev); ok = torch.isfinite(t).all(1)
    if ok.sum() < 8: return torch.zeros((), device=dev)
    z, t = F.normalize(z[ok], dim=1), F.normalize(t[ok] - w2v_mean, dim=1)
    pos = (t @ t.T) > 0.999
    lg = logit_s2.exp() * (z @ t.T) + logit_b2
    return -F.logsigmoid(torch.where(pos, lg, -lg)).sum() / len(z)


def bacc_null(lo, y):
    top = lo.topk(10, 1).indices; pres = torch.unique(y)
    hit = (top == y[:, None]).any(1).float()
    sc = torch.stack([hit[y == c].mean() for c in pres]).mean(); nu = torch.stack([(top == c).any(1).float().mean() for c in pres]).mean()
    return float(sc), float(nu)


def group(lo, y, s, members):
    r = [bacc_null(lo[s == m], y[s == m]) for m in members if (s == m).any()]
    return (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))) if r else (float("nan"), float("nan"))


# ---- eval sets: per block, all words of scored subjects' sentences in that block (context), scored = vocab words ----
EV = {}
BLOCKS = ("Q3", "Q4") + (("PX",) if a.exclude else ())
for blk in BLOCKS:
    m = (q == blk) & np.isin(W.sub, list(scored)) if blk != "PX" else excl
    segs = [c for s in sentences(m) for c in segment(s, a.maxw, train=False)]
    EV[blk] = segs
    print(blk, "segments", len(segs), "words", sum(map(len, segs)), "vocab", int((W.label[np.concatenate(segs)] >= 0).sum()), flush=True)


@torch.no_grad()
def evaluate():
    for m in mods: m.eval()
    Amean = A[[sid[s] for s in subs if s != 0]].detach().mean(0); res = {}
    for blk, segs in EV.items():
        outs = {}
        for mode in ("ctx", "1s"):
            CE, COS, Y, SUB = [], [], [], []
            for i in range(0, len(segs), 40):
                idx, gi, pad, rel = pack(segs[i:i + 40])
                own = np.array([(v in S2.SEEN) or ((a.deep or blk == "PX") and v == 0) for v in W.sub[idx]])
                Am = torch.stack([A[sid[int(v)]].detach() if o else Amean for v, o in zip(W.sub[idx], own)])
                iso = np.ones(len(idx), bool) if mode == "1s" else None
                e, ce, zt = forward(idx, gi, pad, rel, Am, iso=iso)
                keep = W.label[idx] >= 0
                CE.append(ce[keep].cpu()); Y.append(torch.from_numpy(W.label[idx][keep])); SUB.append(torch.from_numpy(W.sub[idx][keep]))
                if a.t5 > 0: COS.append((logit_s.exp() * (F.normalize(zt[keep], dim=1) @ proto.T)).cpu())
            CE, Y, SUB = torch.cat(CE), torch.cat(Y), torch.cat(SUB); COS = torch.cat(COS) if COS else None
            for ro in READOUTS:
                for tau in TAUS:
                    if ro != "ce" and tau > 0: continue
                    lo = CE - tau * log_prior.cpu() if ro == "ce" else COS if ro == "cos" else F.log_softmax(CE - 0.5 * log_prior.cpu(), 1) + F.log_softmax(COS, 1)
                    outs[(mode, ro, tau)] = {"seen": group(lo, Y, SUB, S2.SEEN), "val_zs": group(lo, Y, SUB, S2.VAL), "eval_zs": group(lo, Y, SUB, S2.EVAL),
                                             "sub0": group(lo, Y, SUB, (0,)) if (a.deep or blk == "PX") else (float("nan"), float("nan"))}
        res[blk] = outs
    for m in mods: m.train()
    return res


def dep(r): return r["sub0"][0] if a.deep else S2.deploy(r["seen"][0], r["val_zs"][0])


hist = []
for ep in range(a.epochs):
    for m in mods: m.train()
    tl = tce = tt5 = 0.0
    for k in range(a.steps):
        nv = int(round(n0 * a.vocab_runs)) if VR else 0
        segs = [segment(S0[j], a.maxw)[0] for j in rng.integers(0, len(S0), n0 - nv)] + [segment(VR[j], a.maxw)[0] for j in rng.integers(0, len(VR), nv)] + ([segment(SB[j], a.maxw)[0] for j in rng.integers(0, len(SB), n1)] if n1 > 0 else [])
        if n_ext > 0: segs += [segment(SE[j], a.maxw)[0] for j in rng.integers(0, len(SE), n_ext)]
        idx, gi, pad, rel = pack(segs); N = len(idx)
        iso = rng.random(N) < a.p_iso
        short = rng.random(N) < a.p_short
        clip = (np.where(short, rng.uniform(0, 1.0, N), 9.0), np.where(short, rng.uniform(1.0, 2.0, N), 9.0))
        if a.win == "1s": iso[:] = True
        Am = A[torch.as_tensor([sid[int(s)] for s in g_sub(idx)], device=dev)]
        e, ce, zt = forward(idx, gi, pad, rel, Am, iso=iso, clip=clip, train=True)
        y = torch.from_numpy(g_label(idx)).to(dev); has = y >= 0
        pen = ((A - I) ** 2).sum((1, 2)); loss = a.lam * pen[:len(subs)].mean()          # LB-subject penalty identical to train_v3
        if ext_subs: loss = loss + a.lam * pen[len(subs):].mean()                             # ext subjects: same per-subject weight, separate mean
        if has.any() and a.ce > 0:
            l_ce = F.cross_entropy(ce[has], y[has], label_smoothing=0.1); loss = loss + a.ce * l_ce; tce += l_ce.item()
        if a.t5 > 0:
            l_t5 = siglip(zt, idx); loss = loss + a.t5 * l_t5; tt5 += l_t5.item()
        if a.w2v > 0:
            lb_ = idx < W.n; loss = loss + a.w2v * siglip_w2v(forward_w2v(e[torch.from_numpy(lb_).to(dev)]), idx[lb_])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step(); tl += loss.item()
        if k % 200 == 0: print(f"  ep{ep} step {k} loss {tl / (k + 1):.3f} ce {tce / (k + 1):.3f} t5 {tt5 / (k + 1):.3f} t {round(time.time() - t0)}s", flush=True)
    if a.final:
        torch.save({"enc": enc.state_dict(), "ctx": ctx.state_dict() if ctx else None, "head_ce": head_ce.state_dict(), "head_t5": head_t5.state_dict(),
                    "logit_s": logit_s.detach().cpu(), "logit_b": logit_b.detach().cpu(), "A": A.detach().cpu(), "subs": subs, "ext_subs": ext_subs,
                    "proto": proto.cpu() if a.t5 > 0 else None, "t_mean": t_mean.cpu() if a.t5 > 0 else None,
                    "log_prior": log_prior.cpu(), "args": vars(a), "epoch": ep}, f"runs/{a.tag}_ep{ep}.pt")
        print(json.dumps({"epoch": ep, "loss": round(tl / a.steps, 3), "ce": round(tce / a.steps, 3), "t5": round(tt5 / a.steps, 3), "t": round(time.time() - t0)}), flush=True)
        continue
    ev = evaluate()
    hist.append({"epoch": ep, "loss": tl / a.steps, "ce": tce / a.steps, "t5": tt5 / a.steps, "t": round(time.time() - t0),
                 "eval": {blk: {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in o.items()} for blk, o in ev.items()}})
    torch.save({"enc": enc.state_dict(), "ctx": ctx.state_dict() if ctx else None, "head_ce": head_ce.state_dict(), "head_t5": head_t5.state_dict(),
                "logit_s": logit_s.detach().cpu(), "logit_b": logit_b.detach().cpu(), "A": A.detach().cpu(), "subs": subs, "ext_subs": ext_subs,
                "proto": proto.cpu() if a.t5 > 0 else None, "t_mean": t_mean.cpu() if a.t5 > 0 else None,
                "log_prior": log_prior.cpu(), "args": vars(a), "epoch": ep}, f"runs/{a.tag}_ep{ep}.pt")
    best = {blk: max((dep(v), k) for k, v in ev[blk].items() if k[0] == "ctx") for blk in ev if blk != "PX"}
    if "PX" in ev: print(json.dumps({"epoch": ep, "PROXY": {f"{k[1]}|{k[2]}": round(v["sub0"][0], 4) for k, v in ev["PX"].items() if k[0] == "ctx"}}), flush=True)
    print(json.dumps({"epoch": ep, "loss": round(tl / a.steps, 3), "ce": round(tce / a.steps, 3), "t5": round(tt5 / a.steps, 3),
                      "sel_Q3": [round(best["Q3"][0], 4), best["Q3"][1]], "sel_Q4": [round(best["Q4"][0], 4), best["Q4"][1]],
                      "eval_zs_Q3_ctx": round(ev["Q3"][("ctx", "ce", 0.5)]["eval_zs"][0], 4), "sub0_Q3": round(ev["Q3"][("ctx", "ce", 0.5)]["sub0"][0], 4) if a.deep else None, "t": round(time.time() - t0)}), flush=True)

if a.final: sys.exit(0)
folds = {}
for f, (sel, test) in S2.FOLDS.items():
    best = max(((dep(h["eval"][sel][key]), h["epoch"], key) for h in hist for key in h["eval"][sel] if key.startswith("ctx|")))
    _, ep, key = best; mode, ro, tau = key.split("|"); e = hist[ep]["eval"][test][key]; e1 = hist[ep]["eval"][test][f"1s|{ro}|{tau}"]
    d_ctx = e["sub0"][0] if a.deep else S2.deploy(e["seen"][0], e["eval_zs"][0]); d_1s = e1["sub0"][0] if a.deep else S2.deploy(e1["seen"][0], e1["eval_zs"][0])
    folds[f] = {"select": sel, "test": test, "epoch": ep, "readout": ro, "tau": float(tau), "sel_score": best[0],
                "seen": e["seen"], "eval_zs": e["eval_zs"], "sub0": e["sub0"], "deploy": d_ctx, "deploy_null": (e["sub0"][1] if a.deep else S2.deploy(e["seen"][1], e["eval_zs"][1])),
                "deploy_1s": d_1s, "deploy_mix": 0.9 * d_ctx + 0.1 * d_1s}
summ = {k: float(np.mean([folds[f][k] for f in folds])) for k in ("deploy", "deploy_null", "deploy_1s", "deploy_mix")}
summ["deploy_lift"] = summ["deploy"] - summ["deploy_null"]
summ["seen"] = float(np.mean([folds[f]["seen"][0] for f in folds])); summ["eval_zs"] = float(np.mean([folds[f]["eval_zs"][0] for f in folds]))
json.dump({"args": vars(a), "folds": folds, "summary": summ, "history": hist}, open(f"runs/{a.tag}.json", "w"), indent=1)
print("FOLDS", json.dumps(folds)); print("SUMMARY", json.dumps(summ))
