"""Contextual T5 word targets (d'Ascoli recipe): t5-large, raw output of encoder block 12/24 ("layers": 0.5),
at each word's first subword token, with the word's sentence as context. Aligned to runs_allw/<base>.npz via evidx.
-> runs_t5/<base>.npy (N, 1024) fp32 (T5 outlier dims overflow fp16). Identical sentences are memoised (Broad subjects share the Sherlock1 ses-11/12 text)."""
import glob, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, T5EncoderModel
from src.common import CACHE
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"; torch.set_num_threads(int(os.environ.get("NT", 8)))
os.makedirs(f"{CACHE}/runs_t5", exist_ok=True)
tok = AutoTokenizer.from_pretrained("t5-large")
enc = T5EncoderModel.from_pretrained("t5-large", torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16).to(dev).eval()
enc.encoder.block = enc.encoder.block[:12]                                  # only blocks 1..12 are needed
cap = {}
enc.encoder.block[11].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))   # raw block-12 output (pre final norm)
memo = {}


@torch.no_grad()
def embed(sents):
    """sents: list of word lists -> list of (n_words, 1024) arrays (first subword of each word)."""
    todo = [s for s in sents if tuple(s) not in memo]
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        b = tok(batch, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc(input_ids=b.input_ids.to(dev), attention_mask=b.attention_mask.to(dev))
        H = cap["h"].float().cpu().numpy()
        for j, s in enumerate(batch):
            wid = b.word_ids(j); first = {}
            for k, w in enumerate(wid):
                if w is not None and w not in first: first[w] = k
            memo[tuple(s)] = np.stack([H[j, first[w]] if w in first else np.zeros(1024, np.float32) for w in range(len(s))])
    return [memo[tuple(s)] for s in sents]


for f in sorted(glob.glob(f"{CACHE}/runs_allw/*.npz")):
    base = os.path.basename(f)[:-4]; dst = f"{CACHE}/runs_t5/{base}.npy"
    if os.path.exists(dst): continue
    task = base.split("task-")[1].split("_")[0]
    ev = pd.read_csv(f"/mnt/ssd4tb/data/LibriBrain100/{task}/derivatives/events/{base}_events.tsv", sep="\t")
    w = ev[ev.kind == "word"].copy(); w["segment"] = w.segment.astype(str)
    sents, rows = [], []
    for _, g in w.groupby(w.sentenceidx.fillna(-1), sort=False):
        sents.append(g.segment.str.strip().tolist()); rows.append(g.idx.values)
    E = embed(sents)
    lut = {int(r): e for rr, ee in zip(rows, E) for r, e in zip(rr, ee)}
    evidx = np.load(f)["evidx"]
    out = np.stack([lut.get(int(i), np.full(1024, np.nan, np.float32)) for i in evidx]).astype(np.float32)
    np.save(dst, out); print(base, len(evidx), "nan", int(np.isnan(out).any(1).sum()), "memo", len(memo), flush=True)
