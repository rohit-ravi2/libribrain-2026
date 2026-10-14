"""Contextual T5 word targets (d'Ascoli recipe): t5-large, raw output of encoder block 12/24 ("layers": 0.5),
at each word's first subword token, with the word's sentence as context. Aligned to cont/<base>_ev.npz row order.
-> runs_t5/<base>.npy (N, 1024) fp32 (T5 outlier dims overflow fp16). Identical sentences are memoised."""
import glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, T5EncoderModel
from src.common import CACHE
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(int(os.environ.get("NT", 8)))
MODEL = os.environ.get("T5_MODEL", "t5-large"); OUTD = os.environ.get("T5_OUT", "runs_t5"); LAYER = int(os.environ.get("T5_LAYER", 12))
os.makedirs(f"{CACHE}/{OUTD}", exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
enc = T5EncoderModel.from_pretrained(MODEL, torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
enc.encoder.block = enc.encoder.block[:LAYER]                               # only blocks 1..LAYER are needed
cap = {}
enc.encoder.block[LAYER - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))   # raw block-LAYER output (pre final norm)
memo = {}


@torch.no_grad()
def embed(sents):
    """sents: list of word lists -> list of (n_words, 1024) arrays (first subword of each word)."""
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


while True:
    files = sorted(glob.glob(f"{CACHE}/cont/*_ev.npz"))
    todo = [f for f in files if not os.path.exists(f"{CACHE}/{OUTD}/{os.path.basename(f)[:-7]}.npy")]
    if not todo:
        if len(files) >= 226: break
        import time; time.sleep(20); continue            # cache still being built
    f = todo[0]; base = os.path.basename(f)[:-7]; dst = f"{CACHE}/{OUTD}/{base}.npy"
    d = np.load(f, allow_pickle=True); words = [str(w) if str(w) else "." for w in d["word"]]; sidx = d["sentenceidx"]
    out = np.zeros((len(words), 1024), np.float32); sents, rows = [], []
    order = np.argsort(sidx, kind="stable"); cur = None
    for i in order:
        if sidx[i] != cur: sents.append([]); rows.append([]); cur = sidx[i]
        sents[-1].append(words[i]); rows[-1].append(i)
    for e, r in zip(embed(sents), rows): out[r] = e
    np.save(dst, out); print(base, len(words), "sents", len(sents), "memo", len(memo), flush=True)
print("ALL DONE", flush=True)
