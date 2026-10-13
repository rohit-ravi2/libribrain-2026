"""Shared v3 inference pieces: load a train_v3 checkpoint, build sentence segments, run the forward pass,
score with label-permutation nulls. Used by adapt_v3.py and predict_holdout_v3.py (train_v3.py keeps its own copy)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from . import split2 as S2
from .cont import Words, normalize, WIN
from .models import CtxEncoder, SentCtx

dev = "cuda"


class Model:
    def __init__(self, ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False); a = ck["args"]; self.a = a; self.ck = ck
        self.enc = CtxEncoder(d=a["d"], D=a["D"], T=WIN, p=a["p"]).to(dev); self.enc.load_state_dict(ck["enc"])
        self.ctx = None
        if ck.get("ctx"):
            self.ctx = SentCtx(D=a["D"], maxw=a["maxw"], layers=a.get("ctx_layers", 2)).to(dev); self.ctx.load_state_dict(ck["ctx"])
        self.head_ce = nn.Linear(a["D"], 50).to(dev); self.head_ce.load_state_dict(ck["head_ce"])
        self.head_t5 = nn.Linear(a["D"], 1024).to(dev); self.head_t5.load_state_dict(ck["head_t5"])
        self.logit_s = ck["logit_s"].to(dev); self.A = ck["A"].to(dev); self.subs = ck["subs"]; self.sid = {s: i for i, s in enumerate(self.subs)}
        self.proto = ck["proto"].to(dev) if ck.get("proto") is not None else None
        self.log_prior = ck["log_prior"].to(dev)
        self.Amean = self.A[[self.sid[s] for s in self.subs if s != 0]].mean(0)
        self.mods = [self.enc, self.head_ce, self.head_t5] + ([self.ctx] if self.ctx else [])
        self.eval()

    def eval(self):
        for m in self.mods: m.eval()

    def train(self):
        for m in self.mods: m.train()

    def forward(self, x, v, Am, gi, pad, rel):
        """x (B,306,WIN) fp16/32 raw, v (B,WIN) bool, Am (B,306,306) or (306,306); returns (emb, ce, t5proj)."""
        x = normalize(x.to(dev), v.to(dev)); v = v.to(dev)
        x = torch.einsum("bij,bjt->bit", Am, x) if Am.dim() == 3 else torch.einsum("ij,bjt->bit", Am, x)
        x = torch.cat([x, v[:, None, :].float()], 1)
        with torch.autocast("cuda", torch.bfloat16):
            e = self.enc(x)
            if self.ctx is not None:
                gi_t = torch.as_tensor(gi, device=dev); pad_t = torch.as_tensor(pad, device=dev); rel_t = torch.as_tensor(rel, device=dev)
                E = self.ctx(e[gi_t] * (~pad_t)[..., None], rel_t, pad_t); e = E[~pad_t]
            return e.float(), self.head_ce(e).float(), self.head_t5(e).float()

    def readout(self, ce, zt, ro, tau):
        if ro == "ce": return ce - tau * self.log_prior
        cos = self.logit_s.exp() * (F.normalize(zt, dim=1) @ self.proto.T)
        if ro == "cos": return cos
        return F.log_softmax(ce - 0.5 * self.log_prior, 1) + F.log_softmax(cos, 1)


def sentences(W, mask):
    idx = np.flatnonzero(mask); order = np.lexsort((W.onset[idx], W.sent[idx])); idx = idx[order]
    return np.split(idx, np.flatnonzero(np.diff(W.sent[idx])) + 1)


def segments(W, mask, maxw):
    return [s[i:i + maxw] for s in sentences(W, mask) for i in range(0, len(s), maxw)]


def pack(W, segs, rel_on=False):
    idx = np.concatenate(segs); L = max(map(len, segs)); S = len(segs)
    gi = np.zeros((S, L), np.int64); pad = np.ones((S, L), bool); rel = np.zeros((S, L), np.float32); o = 0
    for k, s in enumerate(segs):
        gi[k, :len(s)] = np.arange(o, o + len(s)); pad[k, :len(s)] = False
        if rel_on: rel[k, :len(s)] = W.onset[s] - W.onset[s[0]]
        o += len(s)
    return idx, gi, pad, rel


def bacc_null(lo, y):
    top = lo.topk(10, 1).indices; pres = torch.unique(y)
    hit = (top == y[:, None]).any(1).float()
    sc = torch.stack([hit[y == c].mean() for c in pres]).mean(); nu = torch.stack([(top == c).any(1).float().mean() for c in pres]).mean()
    return float(sc), float(nu)


def group(lo, y, s, members):
    r = [bacc_null(lo[s == m], y[s == m]) for m in members if (s == m).any()]
    return (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))) if r else (float("nan"), float("nan"))


@torch.no_grad()
def predict(M, W, segs, Amap, iso=False, bs=40):
    """Amap: dict subject -> (306,306) tensor (or None -> Amean). Returns dict of per-word tensors for ALL words in segs
    (ce, zt, label, sub, idx) in pack order."""
    CE, ZT, IDX, EE = [], [], [], []
    for i in range(0, len(segs), bs):
        idx, gi, pad, rel = pack(W, segs[i:i + bs], rel_on=bool(M.a.get("rel", 1)))
        Am = torch.stack([Amap.get(int(s), M.Amean) if Amap.get(int(s), None) is not None else M.Amean for s in W.sub[idx]])
        x, v = W.windows(idx, iso=np.ones(len(idx), bool) if iso else None)
        e, ce, zt = M.forward(x, v, Am, gi, pad, rel); CE.append(ce.cpu()); ZT.append(zt.cpu()); EE.append(e.cpu()); IDX.append(idx)
    idx = np.concatenate(IDX)
    return {"ce": torch.cat(CE), "zt": torch.cat(ZT), "e": torch.cat(EE), "idx": idx, "y": torch.from_numpy(W.label[idx]), "sub": torch.from_numpy(W.sub[idx])}
