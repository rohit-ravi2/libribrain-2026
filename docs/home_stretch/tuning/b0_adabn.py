"""B0 (1 s subj0 WordCNN) on the Deep holdout with AdaBN: BatchNorm running statistics re-estimated on subj0's own 960 canonical
1 s holdout windows (within-subject TTA, as the promoted Deep AdaBN), then eval-mode prediction. Same preprocessing as
scripts/predict_holdout.py (downsample + src.data.normalize). CPU. Usage: b0_adabn.py out.csv"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import downsample
from src.data import normalize
from src.models import WordCNN
torch.set_num_threads(4)
H = LibriBrainCompetitionHoldout(track="deep", data_path="/mnt/ssd4tb/data/LibriBrain-Competition-2026", download=False)
ck = torch.load(os.path.expanduser("~/projects/libribrain-2026/runs/b0.pt"), map_location="cpu", weights_only=False)
m = WordCNN(d=ck.get("args", {}).get("d", 256), p=ck.get("args", {}).get("p", 0.2)); m.load_state_dict(ck["model"]); adj = ck.get("tau", 0.75) * ck["log_prior"].float()
X = torch.cat([normalize(torch.from_numpy(downsample(meg))) for meg, _ in H.iter_windows(batch_size=1024)])
bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm1d)]
for b in bns: b.reset_running_stats(); b.momentum = None
m.train()
for x in m.modules():
    if isinstance(x, nn.Dropout): x.eval()
with torch.no_grad():
    for i in range(0, len(X), 128): m(X[i:i + 128])
m.eval()
with torch.no_grad(): P = torch.softmax(m(X).float() - adj, 1).numpy()
write_submission(path=sys.argv[1], indices=H.indices, primary_probs=P)
print("BN layers", len(bns), "rows", len(P), "max top-1 share", np.bincount(P.argmax(1), minlength=50).max() / len(P))
