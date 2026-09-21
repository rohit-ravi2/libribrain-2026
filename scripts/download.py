"""Download the LibriBrain100 serialised h5 + events and the 2026 holdout (no raw FIF/wav).
Order: holdout -> broad subjects (sub-1..32) -> sub-0. Resumable (re-run to continue)."""
import os
from huggingface_hub import snapshot_download

D = "/mnt/ssd4tb/data/LibriBrain100"
H = "/mnt/ssd4tb/data/LibriBrain-Competition-2026"
common = dict(repo_type="dataset", max_workers=8)
steps = [
    ("holdout", dict(repo_id="pnpl/LibriBrain-Competition-2026", local_dir=H, **common)),
    ("broad subjects", dict(repo_id="pnpl/LibriBrain2", local_dir=D,
        allow_patterns=[f"*sub-{s}_*.h5" for s in range(1, 33)] + [f"*sub-{s}_*.tsv" for s in range(1, 33)] + ["*.json", "*.md", "*.csv"],
        ignore_patterns=["*.fif", "*.wav"], **common)),
    ("sub-0 LibriBrain2", dict(repo_id="pnpl/LibriBrain2", local_dir=D,
        allow_patterns=["*.h5", "*.tsv", "*.json", "*.csv"], ignore_patterns=["*.fif", "*.wav"], **common)),
    ("sub-0 LibriBrain", dict(repo_id="pnpl/LibriBrain", local_dir=D,
        allow_patterns=["*.h5", "*.tsv", "*.json", "*.csv"], ignore_patterns=["*.fif", "*.wav"], **common)),
]
for name, kw in steps:
    print(f"== {name}", flush=True)
    snapshot_download(**kw)
    print(f"== done {name}", flush=True)
os.system('~/bin/notify "LibriBrain100 + holdout download complete" "LibriBrain"')
