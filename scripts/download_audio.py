"""Download stimulus audio (wav) for LibriBrain100 into /mnt/ssd4tb/data/LibriBrain100_audio (for speech-embedding targets)."""
from huggingface_hub import snapshot_download
for r in ["pnpl/LibriBrain", "pnpl/LibriBrain2"]:
    snapshot_download(r, repo_type="dataset", local_dir="/mnt/ssd4tb/data/LibriBrain100_audio", allow_patterns=["*/stimuli/audio/*"], max_workers=4)
    print("== done", r, flush=True)
