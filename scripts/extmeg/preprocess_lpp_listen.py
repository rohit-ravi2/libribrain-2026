"""Bet C (a): convert Le Petit Prince MEG listen (OpenNeuro ds007523, Neuromag TRIUX 306, raw MaxShield fif) to the
v3 continuous-cache format, in LibriBrain channel order and scale:

  ext_lpp_listen/<base>.npy      (306, T50) float16, x SCALE, 1000 Hz -> SSS -> hp 0.1 -> 250 Hz (MNE) -> 50 Hz (resample_poly 1:5)
  ext_lpp_listen/<base>_ev.npz   build_cont keys (onset_s, label=-1, timechapter, sentenceidx, sent_start_s, sent_end_s,
                                 word, evidx, n_times50) + pos, content_word, is_last_word, sub, run, corpus, lang
  ext_lpp_listen/sensors.npz     xyz, ori, kind (0 mag / 1 grad), names  (LibriBrain order, device coordinates)
  ext_lpp_listen/log/<base>.json bads, alignment offset, timings

Chain mirrors LibriBrain's "bads+headpos+sss+notch+bp+ds" as far as the shipped files allow (no cHPI -> no headpos;
50 Hz notch is moot after the 25 Hz anti-alias). base = sub-XX_ses-01_task-listen_run-0R.
Usage: python scripts/extmeg/preprocess_lpp_listen.py [--sub 03[,04]] [--nproc 4] [--skip_bads 0] [--overwrite 0]
Skips runs whose fif is missing / still a .partial download, and runs already cached."""
import argparse, difflib, glob, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import CACHE, SCALE, DATA

ROOT = "/mnt/ssd4tb/data/extmeg/lpp_listen"
OUT = f"{CACHE}/ext_lpp_listen"
LB_NAMES = [n.replace(" ", "") for n in pd.read_csv(f"{DATA}/metadata/channels.tsv", sep="\t")["name"] if str(n).startswith("MEG")]
assert len(LB_NAMES) == 306, len(LB_NAMES)


def norm(w):
    return re.sub(r"[^a-zà-ÿœ'\-]", "", str(w).lower().replace("’", "'"))


def align_sentences(ev_words, ex):
    """Map events.tsv words -> extra_info rows (same word sequence, different onset origin) with difflib.
    Returns sequence_id per event word (-1 if unmatched), per-word pos/content/last flags, and the row map."""
    a = [norm(w) for w in ev_words]; b = [norm(w) for w in ex["word"]]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False); m = np.full(len(a), -1)
    for i, j, n in sm.get_matching_blocks(): m[i:i + n] = np.arange(j, j + n)
    return m


def process(args):
    base, fif, a = args
    dst = f"{OUT}/{base}.npy"
    if not a.overwrite and os.path.exists(dst) and os.path.exists(f"{OUT}/{base}_ev.npz"): return base, "cached"
    import mne; mne.set_log_level("ERROR")
    d = os.path.dirname(fif); sub = base.split("_")[0]; run = int(base.split("run-")[1])
    cal, ctc = f"{d}/{sub}_ses-01_acq-calibration_meg.dat", f"{d}/{sub}_ses-01_acq-crosstalk_meg.fif"
    ev_f = fif.replace("_meg.fif", "_events.tsv"); ex_f = f"{ROOT}/sourcedata/task-listen_run-{run:02d}_extra_info.tsv"
    for f in (cal, ctc, ev_f, ex_f):
        if not os.path.exists(f): return base, f"MISSING {os.path.basename(f)}"
    t0 = time.time(); log = {}
    raw = mne.io.read_raw_fif(fif, allow_maxshield=True, preload=True, verbose=False); raw.fix_mag_coil_types()
    if not a.skip_bads:
        bads, flats = mne.preprocessing.find_bad_channels_maxwell(raw, calibration=cal, cross_talk=ctc, coord_frame="head", verbose=False)
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bads) | set(flats))
    log["bads"] = list(raw.info["bads"]); log["t_bads"] = round(time.time() - t0, 1)
    sss = mne.preprocessing.maxwell_filter(raw, calibration=cal, cross_talk=ctc, coord_frame="head", verbose=False)
    del raw
    sss.pick("meg").filter(0.1, None, verbose=False).resample(250, verbose=False)
    names = [n.replace(" ", "") for n in sss.ch_names]; perm = [names.index(n) for n in LB_NAMES]
    x = sss.get_data()[perm].astype(np.float32) * SCALE
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    from scipy.signal import resample_poly
    x50 = resample_poly(x, 1, 5, axis=1).astype(np.float16)
    log["t_sss_filt"] = round(time.time() - t0, 1)
    if not os.path.exists(f"{OUT}/sensors.npz"):
        loc = np.array([sss.info["chs"][p]["loc"] for p in perm])
        kind = np.array([0 if sss.info["chs"][p]["coil_type"] == 3022 or sss.info["chs"][p]["coil_type"] == 3024 else 1 for p in perm])
        np.savez(f"{OUT}/sensors.npz", xyz=loc[:, :3], ori=loc[:, 9:12], kind=kind, names=np.array(LB_NAMES), coord_frame="device",
                 system="Neuromag TRIUX 306 (ds007523)")
    # ---- events ----
    e = pd.read_csv(ev_f, sep="\t", keep_default_na=False); e = e[e.trial_type.str.lower() == "word"].reset_index(drop=True)
    ex = pd.read_csv(ex_f, keep_default_na=False)                     # comma-separated despite the .tsv name
    m = align_sentences(e.stimulus.values, ex)
    ok = m >= 0; log["n_words"] = int(len(e)); log["n_aligned"] = int(ok.sum()); log["n_extra"] = int(len(ex))
    off = e.onset.values[ok] - ex.onset.values[m[ok]]
    log["offset_s_median"] = float(np.median(off)); log["offset_s_mad"] = float(np.median(np.abs(off - np.median(off))))
    sidx = np.where(ok, ex.sequence_id.values[np.maximum(m, 0)], -1).astype(np.int32)
    pos = np.where(ok, ex.pos.values[np.maximum(m, 0)].astype(str), ""); cont = np.where(ok, ex.content_word.values[np.maximum(m, 0)].astype(bool), False)
    last = np.where(ok, ex.is_last_word.values[np.maximum(m, 0)].astype(bool), False)
    dur = pd.to_numeric(e.duration, errors="coerce").fillna(0.3).values.astype(float); on = e.onset.values.astype(np.float64)
    g = pd.Series(on).groupby(sidx); s0 = g.transform("min").to_numpy(copy=True); s1 = pd.Series(on + dur).groupby(sidx).transform("max").to_numpy(copy=True)
    orphan = sidx < 0; s0[orphan] = on[orphan]; s1[orphan] = on[orphan] + dur[orphan]
    np.save(dst, x50)
    np.savez(f"{OUT}/{base}_ev.npz", onset_s=on, label=np.full(len(e), -1, np.int16), timechapter=on, sentenceidx=sidx,
             sent_start_s=s0, sent_end_s=s1, word=e.stimulus.values.astype(str), evidx=np.arange(len(e)), n_times50=x50.shape[1],
             pos=pos.astype(str), content_word=cont, is_last_word=last, sub=int(sub.split("-")[1]), run=run, corpus="lpp_listen", lang="fr")
    log["n_times50"] = int(x50.shape[1]); log["t_total"] = round(time.time() - t0, 1)
    json.dump(log, open(f"{OUT}/log/{base}.json", "w"))
    return base, f"{len(e)} words ({ok.sum()} aligned, offset {log['offset_s_median']:.3f}±{log['offset_s_mad']:.3f}s), {x50.shape[1]} samples, bads {log['bads']}, {log['t_total']}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sub", default=None); ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--skip_bads", type=int, default=0); ap.add_argument("--overwrite", type=int, default=0)
    a = ap.parse_args(); os.makedirs(f"{OUT}/log", exist_ok=True)
    subs = None if a.sub is None else {f"sub-{s}" for s in a.sub.split(",")}
    jobs = []
    for fif in sorted(glob.glob(f"{ROOT}/sub-*/ses-01/meg/*_task-listen_run-*_meg.fif")):
        base = os.path.basename(fif)[:-8]
        if subs and base.split("_")[0] not in subs: continue
        jobs.append((base, fif, a))
    print(len(jobs), "runs", flush=True)
    with Pool(a.nproc) as p:
        for i, (b, msg) in enumerate(p.imap_unordered(process, jobs)):
            print(i, b, msg, flush=True)
