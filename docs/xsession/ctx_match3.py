"""CTXM-1: contextual target matching (analysis-by-synthesis with the training text model). For each vocab row, candidate
c in the 50-word vocab is placed in the word's +-4-word context window; T5-large block-12 embedding at c's first subword
(same recipe as the training targets); score_c = exp(logit_s) * cos(zt, normalize(t(c|ctx) - t_mean)).
ORACLE context = the true neighbouring words. Compare (all + Sinkhorn all-50): ce (deployed), cos-proto, cos-ctx, ce+cos-ctx.
Frozen deep_c ep6, split-v2 Q4, 12 Broad (identity A) + subj0 (A0)."""
import sys, os, glob, json
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE, VOCAB
from src.v3lib import Model, segments, predict, bacc_null, dev
torch.set_num_threads(12)
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")); tmean = M.ck["t_mean"].float(); S = float(M.logit_s.exp())
tok = AutoTokenizer.from_pretrained("t5-large"); enc = T5EncoderModel.from_pretrained("t5-large", torch_dtype=torch.float32).eval()
enc.encoder.block = enc.encoder.block[:12]; cap = {}; enc.encoder.block[11].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))
memo = {}
@torch.no_grad()
def emb_slot(items):
    """items: list of (word tuple, slot) -> (n,1024) embedding at the slot word's first subword."""
    todo = [it for it in dict.fromkeys(items) if it not in memo]
    for i in range(0, len(todo), 128):
        bt = todo[i:i + 128]; b = tok([list(w) for w, _ in bt], is_split_into_words=True, return_tensors="pt", padding=True)
        enc(input_ids=b.input_ids, attention_mask=b.attention_mask); H = cap["h"].float()
        for j, (w, sl) in enumerate(bt):
            k = b.word_ids(j).index(sl); memo[(w, sl)] = H[j, k].clone()
    return torch.stack([memo[it] for it in items])
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = [0] + BROAD[:4]; gperm = torch.Generator().manual_seed(0)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
W = Words(runs=[b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS], load_meg=True); q = W.quarters()
TXT = {}
def text(r):
    if r not in TXT:
        d = np.load(f"{CACHE}/cont/{W.bases[r]}_ev.npz", allow_pickle=True); TXT[r] = ([str(w) if str(w) else "." for w in d["word"]], d["sentenceidx"])
    return TXT[r]
R = {}
for s in SUBS:
    segs = segments(W, (q == "Q4") & (W.sub == s), M.a["maxw"])
    if not segs: continue
    P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
    ce = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5).cpu()[k]; zt = F.normalize(P["zt"][k].float(), dim=1); y = P["y"][k]; idx = P["idx"][k.numpy()]
    cosp = S * zt @ M.proto.float().cpu().T
    T = 3.0 if s == 0 else 2.0
    pred1 = sk(F.softmax(ce, 1), T).argmax(1).numpy(); PRED = {int(w): VOCAB[int(c)] for w, c in zip(idx, pred1)}
    def ctx_items(mode):
        items = []
        for w in idx:
            r = int(W.run[w]); words, sid = text(r); li = int(W.li[w]); same = np.flatnonzero(sid == sid[li]); p = int(np.searchsorted(same, li))
            lo, hi = max(0, p - 4), min(len(same), p + 5); ctx = [words[i] for i in same[lo:hi]]; sl = p - lo
            if mode == "pred":
                base0 = int(w) - li
                for jj, i in enumerate(same[lo:hi]):
                    if jj != sl: ctx[jj] = PRED.get(base0 + int(i), "<extra_id_0>")
            for c in VOCAB: cc = list(ctx); cc[sl] = c; items.append((tuple(cc), sl))
        return items
    def cos_of(items, z): E = emb_slot(items).view(len(idx), 50, 1024); U = F.normalize(E - tmean, dim=-1); return S * torch.einsum("nd,ncd->nc", z, U)
    it_pred = ctx_items("pred"); zsh = zt[torch.randperm(len(zt), generator=gperm)]
    cpred = cos_of(it_pred, zt); cpsh = cos_of(it_pred, zsh)
    g = "deep" if s == 0 else "broad"
    V = {"ce": ce}
    for lam in (0.25, 0.5, 1.0):
        V[f"ce+predM{lam}"] = F.log_softmax(ce, 1) + lam * F.log_softmax(cpred, 1)
        V[f"ce+predM_SHUF{lam}"] = F.log_softmax(ce, 1) + lam * F.log_softmax(cpsh, 1)
    for nm, L in V.items(): R.setdefault(g, {}).setdefault(nm, []).append(bacc_null(sk(F.softmax(L, 1), T).log(), y)[0])
    print(s, len(idx), "memo", len(memo), {nm: round(v[-1], 3) for nm, v in R[g].items()}, flush=True)
out = {g: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for g, d in R.items()}
print("CTXM3", json.dumps(out)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctx_match3.json"), "w"), indent=1)
