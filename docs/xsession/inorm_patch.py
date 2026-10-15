"""Import first: replaces src.cont.normalize with per-channel per-window instance normalisation (MEGConformer 2025),
keeping mean removal over valid samples, masking and the +-CLIP clip. Must be imported before src.v3lib / train_v3."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import torch, src.cont as C
def normalize_inst(x, valid):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = valid[:, None, :].float(); n = v.sum(-1).clamp_min(1.0)
    x = (x - (x * v).sum(-1, keepdim=True) / n[..., None]) * v
    sd = ((x ** 2).sum(-1) / n).sqrt().clamp_min(1e-6)
    return (x / sd[..., None]).clamp(-C.CLIP, C.CLIP) * v
C.normalize = normalize_inst
