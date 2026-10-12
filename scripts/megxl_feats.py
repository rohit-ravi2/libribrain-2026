"""MEG-XL frozen features for short context windows (feasibility + extraction).
Input per word: our 50 Hz continuous cache sliced to a 3 s sub-segment [onset-1, onset+2) (MEG-XL was trained with 3 s
sub-segment preprocessing: baseline = first 0.5 s, RobustScaler per sensor type, clip ±5). Output: features [C=306, T', 512]
mean-pooled over time -> (306*512) per word is too big to store for 1M words; we store the channel-mean [T', 512] and the
time-mean [306, 512] projected... (see --mode). Sensor geometry: --sensors (MNE sample info, standard VectorView) or
approximate (mags radial, grads tangential)."""
import argparse, os, sys, time, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ext/MEG-XL"))
from src.common import CACHE, IS_MAG
ap = argparse.ArgumentParser(); ap.add_argument("--memtest", action="store_true"); ap.add_argument("--sensors", default="approx")
ap.add_argument("--bs", type=int, default=16); a = ap.parse_args(); dev = "cuda"
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.data.utils import norm_sensor_positions
R = "ext/MEG-XL"
tokzr = BioCodecModel._get_optimized_model()
ck = torch.load(f"{R}/brainstorm/neuro_tokenizers/biocodec_ckpt.pt", map_location=dev, weights_only=False)
tokzr.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}); tokzr.eval()
ck = torch.load(f"{R}/checkpoints/meg-xl-med-v2.ckpt", map_location=dev, weights_only=False)
import inspect; _ok = set(inspect.signature(CrissCrossTransformerModule.__init__).parameters)
model = CrissCrossTransformerModule(tokenizer=tokzr, **{k: v for k, v in ck["hyper_parameters"].items() if k in _ok}).to(dev)
sd = {k: v for k, v in ck["state_dict"].items() if "rope_embedding_layer.rotate" not in k}
miss, unexp = model.load_state_dict(sd, strict=False); model.eval(); print("missing", len(miss), "unexpected", len(unexp), flush=True)

# ---- sensor geometry ----
xyz = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")), np.float64)   # (306,3) device coords
if a.sensors == "approx":
    c = xyz.mean(0); rad = xyz - c; rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    abc = rad.copy()
    for i in range(0, 306, 3):                          # grads: two orthogonal tangents
        n = rad[i]; t1 = np.cross(n, [0, 0, 1.0]); t1 /= np.linalg.norm(t1) + 1e-9; t2 = np.cross(n, t1)
        abc[i + 1], abc[i + 2] = t1, t2
else:
    import mne
    info = mne.io.read_info(a.sensors, verbose=False); chs = {c["ch_name"]: c for c in info["chs"]}
    names = open("/mnt/ssd4tb/data/LibriBrain100/metadata/channels.tsv").read()
    import pandas as pd; ch = pd.read_csv("/mnt/ssd4tb/data/LibriBrain100/metadata/channels.tsv", sep="\t"); meg = [n for n in ch.name if n.startswith("MEG")][:306]
    abc = np.zeros((306, 3))
    for i, n in enumerate(meg):
        loc = chs[n]["loc"]; abc[i] = loc[3:6] if IS_MAG[i] else loc[3:6]    # first direction vector (grad: dir_idx=1 per MEG-XL loader)
        if not IS_MAG[i]: abc[i] = loc[3:6]
xyzdir = norm_sensor_positions(np.concatenate([xyz, abc], 1))
S_xyz = torch.tensor(xyzdir[:, :3], dtype=torch.float32, device=dev); S_abc = torch.tensor(xyzdir[:, 3:], dtype=torch.float32, device=dev)
S_type = torch.tensor(IS_MAG.astype(np.int64), device=dev); S_mask = torch.ones(306, device=dev)


def prep(x):
    """x (B,306,T) scaled 50 Hz window -> MEG-XL preprocessing per sub-segment: baseline first 0.5 s, robust scale per type, clip ±5."""
    x = x.float() - x[:, :, :25].float().mean(-1, keepdim=True)
    for m in (IS_MAG, ~IS_MAG):
        mt = torch.as_tensor(m, device=x.device); v = x[:, mt]
        med = v.median(-1, keepdim=True).values; q = torch.quantile(v.float(), torch.tensor([0.25, 0.75], device=x.device), dim=-1)
        x[:, mt] = (v - med) / (q[1] - q[0]).unsqueeze(-1).clamp_min(1e-6)
    return x.clamp(-5, 5)


@torch.no_grad()
def feats(x):
    B = x.shape[0]
    out = model(prep(x.to(dev)), S_xyz[None].expand(B, -1, -1), S_abc[None].expand(B, -1, -1), S_type[None].expand(B, -1), S_mask[None].expand(B, -1), apply_mask=False)
    return out["features"]                                   # [B, C, T', 512]


if a.memtest:
    from src.cont import Words
    W = Words(runs=["sub-0_ses-10_task-Sherlock1_run-1"], load_meg=True)
    idx = np.flatnonzero(W.label >= 0)[:a.bs]; x, v = W.windows(idx)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with torch.autocast("cuda", torch.bfloat16):
        f = feats(x)
    torch.cuda.synchronize(); print("features", tuple(f.shape), f.dtype, "time/batch %.2fs" % (time.time() - t0), "peak GPU %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("feature stats: mean %.3f std %.3f, nan %s" % (f.float().mean(), f.float().std(), bool(torch.isnan(f).any())))
