"""Per-window wav2vec2 speech-embedding targets aligned to the window cache (runs/*.npz) -> runs_emb/<base>.npy (N,768) fp16, NaN where no audio."""
import glob, os, sys, wave
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import Wav2Vec2Model
from src import split as S
from src.common import CACHE
AUD = "/mnt/ssd4tb/data/LibriBrain100_audio"; dev = "cuda"; SR = 16000; LAYER = 8; WIN = 0.4
os.makedirs(f"{CACHE}/runs_emb", exist_ok=True)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(dev).eval()


def load_wav(p):
    w = wave.open(p); sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    x = x.reshape(-1, ch).mean(1)
    if sr != SR:
        import scipy.signal as ss; x = ss.resample_poly(x, SR, sr)
    return (x - x.mean()) / (x.std() + 1e-8)


@torch.no_grad()
def frames(x):   # contextual layer-LAYER frames at 50 Hz over the whole chapter, 30 s chunks with 2 s context each side
    out = []; C, P = 30 * SR, 2 * SR
    for s in range(0, len(x), C):
        a, b = max(0, s - P), min(len(x), s + C + P)
        h = model(torch.from_numpy(x[a:b])[None].to(dev), output_hidden_states=True).hidden_states[LAYER][0].float()
        off = (s - a) // 320; n = (min(len(x), s + C) - s) // 320
        out.append(h[off:off + n].cpu())
    return torch.cat(out)


import json
AMAP = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/audio_map.json")))  # duration-verified session -> wav
chap = {}
for f in sorted(glob.glob(f"{CACHE}/runs/*.npz")):
    base = os.path.basename(f)[:-4]; dst = f"{CACHE}/runs_emb/{base}.npy"
    if os.path.exists(dst): continue
    sub, ses = S.parse(base); task = base.split("task-")[1].split("_")[0]; n = int(ses.split("_")[0].split("-")[1])
    tc = np.load(f)["timechapter"]; E = np.full((len(tc), 768), np.nan, np.float16)
    key = AMAP.get(f"{task}/{ses.split('_')[0]}")
    if key is not None:
        if key not in chap:
            chap.clear(); chap[key] = frames(load_wav(key))
        F_ = chap[key]
        for i, t in enumerate(tc):
            a = int(t * 50); b = max(a + 1, int((t + WIN) * 50))
            if 0 <= a < len(F_): E[i] = F_[a:b].mean(0).numpy()
    np.save(dst, E); print(base, "ok" if not np.isnan(E).all() else "no-audio", len(tc), flush=True)
