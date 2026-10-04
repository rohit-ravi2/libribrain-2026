"""Write a Broad-track submission CSV from a checkpoint. Row-local inference only: each holdout window is
predicted from that subject's own window alone (pnpl canonical windowing; no timing/shuffle metadata is read here)."""
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
ck = torch.load(a.ckpt, map_location=dev)
model = WordCNN().to(dev); model.load_state_dict(ck["model"]); model.eval()
adj = ck.get("tau", 0.0) * ck["log_prior"].to(dev)
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname("/mnt/ssd4tb/data/LibriBrain-Competition-2026/COMPETITION_HOLDOUT"), download=False)
print(H, H.counts(), flush=True)
# Per-subject input layer (B1-style ckpts): trained subjects use their own A; unseen subjects (33-39) use the mean
# Broad A. The subject id is the row's own subject (pnpl metadata), never another subject's rows.
Amap = None
if "A" in ck:
    subs = ck["subs"]; Ab = ck["A"][[subs.index(s) for s in subs if s != 0]].mean(0)
    Amap = {s: (ck["A"][subs.index(s)] if s in subs else Ab).to(dev) for s in H.subjects}
    print("subjects with own A:", [s for s in H.subjects if s in subs], "| mean A:", [s for s in H.subjects if s not in subs])
probs = []
with torch.no_grad():
    for meg, metas in H.iter_windows(batch_size=1024):
        x = normalize(torch.from_numpy(downsample(meg)).to(dev))
        if Amap is not None:
            Ab_ = torch.stack([Amap[m["subject"]] for m in metas]); x = torch.einsum("bij,bjt->bit", Ab_, x)
        with torch.autocast(dev, torch.bfloat16, enabled=(dev == "cuda")):
            l = model(x).float() - adj
        probs.append(torch.softmax(l, 1).cpu().numpy())
probs = np.concatenate(probs)
assert probs.shape == (len(H), 50) and np.isfinite(probs).all()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
write_submission(path=a.out, indices=H.indices, primary_probs=probs)
top = probs.argmax(1); print("rows", len(probs), "distinct top-1 words", len(np.unique(top)), "max top-1 share", np.bincount(top).max() / len(top))
