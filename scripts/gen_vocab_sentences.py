"""Generate a corpus of grammatical 5-9-word sentences built ONLY from the 50 competition words, as a prior for the
holdout-style sentences. Sampling: GPT-2-medium with the output vocabulary masked to the 50 words (+ end), temperature 1.0,
then rescored by the unmasked GPT-2-medium; keep the best-scoring half. Output: results/vocab_sentences.txt (one per line)."""
import os, sys, math, re
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.common import VOCAB, norm_word
dev = "cuda"; N = int(os.environ.get("NSENT", 40000)); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("gpt2-medium"); lm = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(dev).eval()
words = [norm_word(w) for w in VOCAB]
# token ids for " word" (with leading space); words that split into several tokens are handled by allowing their first token
# and then forcing the continuation -- simpler: keep only single-token forms (check coverage).
ids, multi = {}, []
for w in words:
    t = tok.encode(" " + w)
    (ids.__setitem__(w, t[0]) if len(t) == 1 else multi.append((w, t)))
print("single-token words", len(ids), "multi-token", [m[0] for m in multi], flush=True)
allowed = torch.tensor(sorted(ids.values()), device=dev); id2w = {v: k for k, v in ids.items()}
eos = tok.eos_token_id
out = []
with torch.no_grad():
    B = 256
    while len(out) < N:
        L = np.random.randint(5, 10, size=B)                                   # target lengths 5..9
        seq = torch.full((B, 1), eos, device=dev); done = torch.zeros(B, dtype=torch.bool, device=dev)
        for step in range(9):
            logits = lm(seq).logits[:, -1]
            mask = torch.full_like(logits, -1e9); mask[:, allowed] = logits[:, allowed]
            probs = torch.softmax(mask / 1.0, -1); nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], 1)
        for b in range(B):
            s = [id2w[int(t)] for t in seq[b, 1:1 + L[b]]]
            out.append(" ".join(s))
        print(len(out), flush=True)
# rescore with unmasked LM, keep the most natural half
def score(batch):
    enc = tok([tok.eos_token + " " + s for s in batch], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad(): lg = lm(**enc).logits[:, :-1]
    tgt = enc.input_ids[:, 1:]; lp = torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0]
    m = enc.attention_mask[:, 1:].float(); return ((lp * m).sum(1) / m.sum(1)).cpu().numpy()
tok.pad_token = tok.eos_token
sc = np.concatenate([score(out[i:i + 256]) for i in range(0, len(out), 256)])
keep = [s for s, v in zip(out, sc) if v >= np.median(sc)]
os.makedirs("results", exist_ok=True); open("results/vocab_sentences.txt", "w").write("\n".join(keep) + "\n")
print("kept", len(keep), "examples:", keep[:8])
