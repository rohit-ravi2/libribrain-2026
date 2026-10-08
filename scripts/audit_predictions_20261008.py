"""CPU-only diagnostics of saved checkpoints on existing local test data.

Not model selection: compares fixed prior corrections and estimates a per-subject
label-permutation null analytically from prediction frequencies. Writes separate results.
"""
import json
import pathlib
import sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import split as S
from src.data import load_sets, normalize
from src.models import WordCNN

torch.set_num_threads(4)
D = load_sets({"ev": lambda b, s, ses, t: S.block_masks(b, t)["B"] if s in S.EVAL_SUBJECTS else None})
X, y, subj, _ = D["ev"]
# Recover actual mixture prior from metadata (avoid loading training MEG).
import glob
counts = np.zeros((2, 50))
for f in sorted(glob.glob(f"{S.CACHE}/runs/*.npz")):
    b = pathlib.Path(f).stem; s, _ = S.parse(b)
    with np.load(f) as d:
        m = S.train_mask(b, d['timechapter'], val=True)
        if b == 'sub-0_ses-10_task-Sherlock1_run-1': m[:] = False
        counts[int(s != 0)] += np.bincount(d['label'][m], minlength=50)
prior = (counts / counts.sum(1, keepdims=True)).mean(0)

def score(p):
    top = p.topk(10, dim=1).indices.numpy()
    yy, ss = y.numpy(), subj.numpy()
    result = []
    for s in S.EVAL_SUBJECTS:
        m = ss == s; t, lab = top[m], yy[m]; present = np.unique(lab)
        hit = (t == lab[:, None]).any(1)
        actual = np.mean([hit[lab == c].mean() for c in present])
        # Under random permutation of labels, every class sees a random sample
        # of these prediction rows. Its expected recall is inclusion frequency.
        null = np.mean([(t == c).any(1).mean() for c in present])
        result.append([actual, null])
    a, n = np.mean(result, axis=0)
    return {'score': float(a), 'label_permutation_expected': float(n), 'lift': float(a-n)}

out, ensemble = {}, []
for name in ['b5a', 'e1_s0', 'e1_s1', 'e1_s2']:
    ck = torch.load(ROOT / f'runs/{name}.pt', map_location='cpu', weights_only=False)
    args = ck['args']; m = WordCNN(d=args.get('d', 256), p=args.get('p', .2))
    m.load_state_dict(ck['model']); m.eval()
    ids = [i for i, s in enumerate(ck['subs']) if s != 0]
    A = ck['A'][ids].mean(0)
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            logits.append(m(torch.einsum('ij,bjt->bit', A, normalize(X[i:i+128]))))
    logits = torch.cat(logits)
    deployed = logits - ck['tau'] * ck['log_prior']
    out[name] = {
        'saved_tau': ck['tau'], 'saved_prior_and_tau': score(deployed),
        'full_stored_prior_removal': score(logits - ck['log_prior']),
        'full_sampler_prior_removal': score(logits - torch.tensor(prior).clamp_min(1e-12).log()),
    }
    if name.startswith('e1'): ensemble.append(deployed.softmax(1))
    print(name, json.dumps(out[name]), flush=True)
out['e1_ensemble'] = score(torch.stack(ensemble).mean(0))
print('e1_ensemble', out['e1_ensemble'], flush=True)
(ROOT / 'results/audit_predictions_20261008.json').write_text(json.dumps(out, indent=2) + '\n')
