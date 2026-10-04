"""Write a Broad-track submission CSV from one or more checkpoints (comma-separated; softmaxes averaged).
Row-local inference only: each holdout window is predicted from that subject's own window alone (pnpl canonical
windowing; no timing/shuffle metadata is read here). Per-subject input layer (B1-style ckpts): trained subjects use
their own A; unseen subjects (33-39) use the mean Broad A. The subject id is the row's own subject."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import downsample
from src.data import normalize
from src.models import WordCNN

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--track", default="broad"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
dev = a.device
H = LibriBrainCompetitionHoldout(track=a.track, data_path="/mnt/ssd4tb/data/LibriBrain-Competition-2026", download=False)
print(H, H.counts(), flush=True)
models = []
for c in a.ckpt.split(","):
    ck = torch.load(c, map_location=dev)
    m = WordCNN(d=ck.get("args", {}).get("d", 256), p=ck.get("args", {}).get("p", 0.2)).to(dev); m.load_state_dict(ck["model"]); m.eval()
    Amap = None
    if "A" in ck:
        subs = ck["subs"]; Ab = ck["A"][[subs.index(s) for s in subs if s != 0]].mean(0)
        Amap = {s: (ck["A"][subs.index(s)] if s in subs else Ab).to(dev) for s in H.subjects}
        print(c, "own A:", [s for s in H.subjects if s in subs], "| mean A:", [s for s in H.subjects if s not in subs], flush=True)
    models.append((m, Amap, ck.get("tau", 0.0) * ck["log_prior"].to(dev)))
probs = []
with torch.no_grad():
    for meg, metas in H.iter_windows(batch_size=1024):
        x0 = normalize(torch.from_numpy(downsample(meg)).to(dev)); p = 0
        for m, Amap, adj in models:
            x = x0 if Amap is None else torch.einsum("bij,bjt->bit", torch.stack([Amap[mt["subject"]] for mt in metas]), x0)
            with torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
                l = m(x).float() - adj
            p = p + torch.softmax(l, 1) / len(models)
        probs.append(p.cpu().numpy())
probs = np.concatenate(probs)
assert probs.shape == (len(H), 50) and np.isfinite(probs).all()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
write_submission(path=a.out, indices=H.indices, primary_probs=probs)
top = probs.argmax(1); print("rows", len(probs), "distinct top-1 words", len(np.unique(top)), "max top-1 share", np.bincount(top).max() / len(top))
