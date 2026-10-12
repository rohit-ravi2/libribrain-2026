"""Bet C (c): contextual T5 word targets for an external corpus cache — scripts/text_targets.py with the directory
parameterised. t5-large, raw output of encoder block 12/24 at each word's first subword token, the word's sentence
(sentenceidx group; orphans = own sentence) as context. Aligned to ext_<corpus>/<base>_ev.npz row order.
-> ext_<corpus>/t5/<base>.npy (N, 1024) fp32. Identical sentences are memoised (all LPP subjects hear the same text).
Usage: python scripts/extmeg/text_targets_ext.py lpp_listen [cuda|cpu]   (NT=threads for cpu)"""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoTokenizer, T5EncoderModel
from src.common import CACHE
corpus = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"; torch.set_num_threads(int(os.environ.get("NT", 8)))
SRC = f"{CACHE}/ext_{corpus}"; DST = f"{SRC}/t5"; os.makedirs(DST, exist_ok=True)
only = sys.argv[3] if len(sys.argv) > 3 else None                                 # optional substring filter (test on one run)
tok = AutoTokenizer.from_pretrained("t5-large")
enc = T5EncoderModel.from_pretrained("t5-large", torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
enc.encoder.block = enc.encoder.block[:12]
cap = {}
enc.encoder.block[11].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))
memo = {}


@torch.no_grad()
def embed(sents):
    todo = sorted({tuple(s) for s in sents if tuple(s) not in memo}, key=len)
    for i in range(0, len(todo), 48):
        batch = [list(s) for s in todo[i:i + 48]]
        b = tok(batch, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc(input_ids=b.input_ids.to(dev), attention_mask=b.attention_mask.to(dev))
        H = cap["h"].float().cpu().numpy()
        for j, s in enumerate(batch):
            wid = b.word_ids(j); first = {}
            for k, w in enumerate(wid):
                if w is not None and w not in first: first[w] = k
            memo[tuple(s)] = np.stack([H[j, first[w]] if w in first else np.zeros(1024, np.float32) for w in range(len(s))])
    return [memo[tuple(s)] for s in sents]


files = sorted(glob.glob(f"{SRC}/*_ev.npz"))
if only: files = [f for f in files if only in f]
for f in files:
    base = os.path.basename(f)[:-7]; dst = f"{DST}/{base}.npy"
    if os.path.exists(dst): continue
    d = np.load(f, allow_pickle=True); words = [str(w) if str(w) else "." for w in d["word"]]; sidx = d["sentenceidx"].astype(np.int64)
    sidx = np.where(sidx < 0, np.arange(len(words)) + 10 ** 6, sidx)
    out = np.zeros((len(words), 1024), np.float32); sents, rows = [], []
    order = np.argsort(sidx, kind="stable"); cur = None
    for i in order:
        if sidx[i] != cur: sents.append([]); rows.append([]); cur = sidx[i]
        sents[-1].append(words[i]); rows[-1].append(i)
    for e, r in zip(embed(sents), rows): out[r] = e
    np.save(dst, out); print(base, len(words), "sents", len(sents), "memo", len(memo), flush=True)
print("ALL DONE", flush=True)
