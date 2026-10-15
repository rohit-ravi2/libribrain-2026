"""MOON-1: moonshot feature families on split-v2 (train Q1-Q2 / test Q3+Q4 vocab words), subj0 and pooled Broad 3/6/9.
For each family F: linear probe on F alone, on [our frozen embedding e, F], and late fusion of the F-probe with the
model's own posterior (w 0.1/0.2/0.3, best reported); all top-10 balanced accuracy, chance 0.20.
Families: reservoir (echo-state network on 10 spatial PCs), PAC (theta-phase x beta/alpha-amplitude modulation index),
TDA (0-dim sublevel/superlevel persistence per PC), neural field (functa-style auto-decoder: shared SIREN over
(PC index, t) modulated by a 64-d per-window latent; latents = features), HDC (bipolar hypervectors of e bound with
sentence position and bundled with neighbours), hyperbolic head (Poincare-ball prototypes on e).
Lead-field (idea 4, hard constraint): input projected onto the top-k span of a template MEG lead field (MNE sample),
applied to the frozen model zero-shot, k = 30/50/70; scored on Q3/Q4 + Sinkhorn."""
import sys, os, json, glob, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import hilbert, butter, sosfiltfilt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.split import SHARED
from src.cont import Words, normalize, ON, FS
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
torch.manual_seed(0); rng = np.random.default_rng(0)
M = Model("runs/deep_c_ep6.pt"); I = torch.eye(306, device=dev); SUBS = (0, 3, 6, 9)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); D = {}
for s in SUBS:
    segs = segments(W, np.isin(q, ["Q1", "Q2", "Q3", "Q4"]) & (W.sub == s), M.a["maxw"])
    P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = (P["y"] >= 0).numpy(); idx = P["idx"][k]
    p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu().numpy()[k]
    pos = np.array([np.sum((W.sent == W.sent[w]) & (W.onset < W.onset[w])) for w in idx])
    x, v = W.windows(idx); x = normalize(x, v).numpy(); tr = np.isin(q[idx], ["Q1", "Q2"])
    seg = x[tr][:, :, ON:ON + 50]; C = np.einsum("nct,ndt->cd", seg, seg); U = np.linalg.eigh(C)[1][:, ::-1][:, :10]
    D[s] = dict(e=P["e"][torch.from_numpy(k)].numpy(), p=p, y=W.label[idx], tr=tr, pos=pos, z=np.einsum("ck,nct->nkt", U, x).astype(np.float32), idx=idx)
    print("subject", s, len(idx), flush=True)

def bacc10(P, y): top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))
def probe(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
    P = np.full((len(Xte), 50), 1e-6); P[:, clf.classes_] = clf.predict_proba((Xte - mu) / sd); return P / P.sum(1, keepdims=True)

# ---------------- feature families ----------------
def f_reservoir(z, n=300, leak=0.3, rho=0.9):
    g = np.random.default_rng(1); Win = g.normal(0, 0.5, (n, 10)); Wr = g.normal(0, 1, (n, n)); Wr *= rho / np.max(np.abs(np.linalg.eigvals(Wr)))
    h = np.zeros((len(z), n)); out = []
    for t in range(z.shape[2]):
        h = (1 - leak) * h + leak * np.tanh(z[:, :, t] @ Win.T + h @ Wr.T)
        if ON <= t < ON + 50 and (t - ON) % 10 == 9: out.append(h.copy())
    return np.concatenate(out, 1)
def bandpass(z, lo, hi): return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=FS, output="sos"), z, axis=-1)
def f_pac(z):
    ph = np.angle(hilbert(bandpass(z, 4, 8), axis=-1))[..., ON:ON + 50]; out = []
    for lo, hi in ((13, 24), (8, 12)):
        amp = np.abs(hilbert(bandpass(z, lo, hi), axis=-1))[..., ON:ON + 50]
        out.append(np.abs((amp * np.exp(1j * ph)).mean(-1)) / (amp.mean(-1) + 1e-9))
    return np.concatenate(out, 1)
def persistence0(sig):
    """0-dim sublevel-set persistence of a 1-D signal (union-find); returns sorted lifetimes."""
    order = np.argsort(sig); parent = -np.ones(len(sig), int); birth = {}; life = []
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in order:
        parent[i] = i; birth[i] = sig[i]
        for j in (i - 1, i + 1):
            if 0 <= j < len(sig) and parent[j] >= 0:
                a, b = find(i), find(j)
                if a != b:
                    young, old = (a, b) if birth[a] > birth[b] else (b, a); life.append(sig[i] - birth[young]); parent[young] = old
    return np.sort(life)[::-1]
def f_tda(z):
    out = np.zeros((len(z), 10, 2, 5))
    for n in range(len(z)):
        for c in range(10):
            s = z[n, c, ON:ON + 50]
            for d, sg in enumerate((s, -s)):
                l = persistence0(sg)[:5]; out[n, c, d, :len(l)] = l
    return out.reshape(len(z), -1)
class Field(nn.Module):
    def __init__(s, L=64, H=128):
        super().__init__(); s.inp = nn.Linear(2, H); s.mod = nn.Linear(L, 3 * H); s.l2 = nn.Linear(H, H); s.l3 = nn.Linear(H, H); s.out = nn.Linear(H, 1)
    def forward(s, coords, lat):
        m = s.mod(lat)[:, None]; b1, b2, b3 = m.chunk(3, -1)
        h = torch.sin(30 * s.inp(coords)[None] + b1); h = torch.sin(s.l2(h) + b2); h = torch.sin(s.l3(h) + b3); return s.out(h)[..., 0]
def f_field(z_tr, z_te, steps=1500, inner=200):
    T = 50; cc, tt = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, T), indexing="ij")
    coords = torch.tensor(np.stack([cc.ravel(), tt.ravel()], 1), dtype=torch.float32, device=dev)
    tgt = lambda z: torch.tensor(z[:, :, ON:ON + T].reshape(len(z), -1), dtype=torch.float32, device=dev)
    Ytr = tgt(z_tr); Ytr = Ytr / (Ytr.std() + 1e-6); net = Field().to(dev); lat = nn.Parameter(torch.zeros(len(z_tr), 64, device=dev))
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 1e-4}, {"params": [lat], "lr": 1e-2}])
    for it in range(steps):
        b = torch.randint(0, len(z_tr), (256,), device=dev); loss = ((net(coords, lat[b]) - Ytr[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    Yte = tgt(z_te) / (tgt(z_tr).std() + 1e-6); lte = nn.Parameter(torch.zeros(len(z_te), 64, device=dev)); o2 = torch.optim.Adam([lte], lr=1e-2)
    for p_ in net.parameters(): p_.requires_grad_(False)
    for it in range(inner):
        loss = ((net(coords, lte) - Yte) ** 2).mean(); o2.zero_grad(); loss.backward(); o2.step()
    return lat.detach().cpu().numpy(), lte.detach().cpu().numpy(), float(loss)
def f_hdc(e, pos, Dh=4096):
    g = np.random.default_rng(2); R = g.normal(size=(e.shape[1], Dh)); P = g.choice([-1, 1], size=(16, Dh))
    hv = np.sign(e @ R); return hv * P[np.clip(pos, 0, 15)]
class Hyp(nn.Module):
    def __init__(s, d=256, k=32):
        super().__init__(); s.proj = nn.Linear(d, k); s.proto = nn.Parameter(0.01 * torch.randn(50, k))
    @staticmethod
    def expmap0(v): n = v.norm(dim=-1, keepdim=True).clamp_min(1e-6); return torch.tanh(n) * v / n * 0.95
    def forward(s, x):
        a, b = s.expmap0(s.proj(x)), s.expmap0(s.proto); d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
        den = (1 - (a ** 2).sum(-1))[:, None] * (1 - (b ** 2).sum(-1))[None]; return -torch.acosh(1 + 2 * d2 / den.clamp_min(1e-6))
def hyp_probe(etr, ytr, ete):
    m = Hyp().to(dev); opt = torch.optim.Adam(m.parameters(), 3e-3); X = torch.tensor(etr, device=dev); Y = torch.tensor(ytr, device=dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6; w = torch.tensor(1 / np.maximum(np.bincount(ytr, minlength=50), 1), dtype=torch.float32, device=dev)
    for _ in range(600):
        b = torch.randint(0, len(X), (512,), device=dev); loss = F.cross_entropy(5 * m((X[b] - mu) / sd), Y[b], weight=w); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return F.softmax(5 * m((torch.tensor(ete, device=dev) - mu) / sd), 1).cpu().numpy()

out = {}
for grp, ss in (("deep", [0]), ("broad", [3, 6, 9])):
    cat = lambda key, m: np.concatenate([D[s][key][D[s]["tr"] == m] for s in ss])
    etr, ete, ytr, yte, pte = cat("e", True), cat("e", False), cat("y", True), cat("y", False), cat("p", False)
    ztr, zte = cat("z", True), cat("z", False); ptr_pos, pte_pos = cat("pos", True), cat("pos", False)
    base = probe(etr, ytr, ete); r = {"model": round(bacc10(pte, yte), 4), "probe_e": round(bacc10(base, yte), 4)}
    fams = {}
    rz = f_reservoir(np.concatenate([ztr, zte])); fams["reservoir"] = (rz[:len(ztr)], rz[len(ztr):])
    pz = f_pac(np.concatenate([ztr, zte])); fams["pac"] = (pz[:len(ztr)], pz[len(ztr):])
    tz = f_tda(np.concatenate([ztr, zte])); fams["tda"] = (tz[:len(ztr)], tz[len(ztr):])
    ftr, fte, fl = f_field(ztr, zte); fams["field"] = (ftr, fte); r["field_fit_mse"] = round(fl, 3)
    hz = f_hdc(np.concatenate([etr, ete]), np.concatenate([ptr_pos, pte_pos])); fams["hdc"] = (hz[:len(etr)], hz[len(etr):])
    for nm, (Ftr, Fte) in fams.items():
        if Ftr.shape[1] > 256: pca = PCA(256, random_state=0).fit(Ftr); Ftr, Fte = pca.transform(Ftr), pca.transform(Fte)
        alone = probe(Ftr, ytr, Fte); both = probe(np.hstack([etr, Ftr]), ytr, np.hstack([ete, Fte]))
        fus = max(bacc10(np.exp((1 - w) * np.log(pte + 1e-9) + w * np.log(alone)), yte) for w in (0.1, 0.2, 0.3))
        r[nm] = dict(alone=round(bacc10(alone, yte), 4), concat_e=round(bacc10(both, yte), 4), fused_model=round(fus, 4))
    hp = hyp_probe(etr, ytr, ete); r["hyperbolic"] = dict(alone=round(bacc10(hp, yte), 4), fused_model=round(max(bacc10(np.exp((1 - w) * np.log(pte + 1e-9) + w * np.log(hp + 1e-9)), yte) for w in (0.1, 0.2, 0.3)), 4))
    out[grp] = r; print(grp, json.dumps(r), flush=True)
# ---------------- lead-field hard constraint (zero-shot, frozen model) ----------------
import mne
fwd = mne.read_forward_solution(os.path.expanduser("~/mne_data/MNE-sample-data/MEG/sample/sample_audvis-meg-oct-6-fwd.fif"), verbose=False)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False); names = fwd["sol"]["row_names"]; G = fwd["sol"]["data"]
import pandas as pd
ch = [n for n in pd.read_csv("/mnt/ssd4tb/data/LibriBrain100/metadata/channels.tsv", sep="\t").name if n.startswith("MEG")][:306]
rowmap = {n.replace(" ", ""): i for i, n in enumerate(names)}; G = np.stack([G[rowmap[n.replace(" ", "")]] if n.replace(" ", "") in rowmap else np.zeros(G.shape[1]) for n in ch])
G = G / np.where(np.arange(306) % 3 == 0, 1e-13, 1e-12)[:, None]      # same unit scaling as the cache (mags 1e13, grads 1e12 -> O(1))
Ug, sg, _ = np.linalg.svd(G, full_matrices=False); lf = {"n_missing_rows": int(sum(n.replace(" ", "") not in rowmap for n in ch))}
u = torch.full((50,), 1 / 50)
for kk in (30, 50, 70):
    Pk = torch.tensor(Ug[:, :kk] @ Ug[:, :kk].T, dtype=torch.float32, device=dev); sc = {"broad": [], "deep": []}
    for blk in ("Q3", "Q4"):
        for s in SUBS:
            segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"]); A = (M.A[M.sid[0]] if s == 0 else I) @ Pk
            P = predict(M, W, segs, {s: A}); k = P["y"] >= 0; pp = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]
            sc["deep" if s == 0 else "broad"].append(bacc_null(sinkhorn_p(pp, 3.0 if s == 0 else 2.0, u).log(), P["y"][k])[0])
    lf[f"k{kk}"] = {g: round(float(np.mean(v)), 4) for g, v in sc.items()}
sc = {"broad": [], "deep": []}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"]); P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
        pp = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; sc["deep" if s == 0 else "broad"].append(bacc_null(sinkhorn_p(pp, 3.0 if s == 0 else 2.0, u).log(), P["y"][k])[0])
lf["identity"] = {g: round(float(np.mean(v)), 4) for g, v in sc.items()}; out["leadfield"] = lf; print("leadfield", json.dumps(lf), flush=True)
json.dump(out, open("runs/moonshot_probe.json", "w"), indent=1)
