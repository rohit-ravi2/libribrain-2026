## Appendix A: Reproducibility

All code is in the public repository github.com/rohit-ravi2/libribrain-2026. Paths are relative to its root, with
`A/` for `docs/writeups/analyses/`, `X/` for `docs/xsession/` and `H/` for `docs/home_stretch/`; outputs without a
directory sit beside their script. The commit is the last one that changed the script. LibriBrain data are not
redistributed. Experiments were logged with their predictions in `PREDICTIONS.md` (to 15 October) and then in
`docs/XSESSION_TESTS_2026-10-15.md` and `H/LEDGER.md`; runs and their scores are in `RUNS.md`.

| Claim (section) | Script | Output / data | Commit |
|--------------------|----------------------------|--------------------------|-----------|
| Timing ladder and its ablation, one seed (4.1) | `scripts/train_v3.py` (`--rel 0/1`), `scripts/ablate_v3.py` | `runs/v3{b,c,c0,d,d0}.json` | 57a4d05, 22e21ed |
| Timing ladder and ablation, three seeds (4.1) | `scripts/queue_ci_timing.sh`, `A/seeded_cis.py` | `runs/ci_v3*.json`, `logs/ci/`, `seeded_cis.json` | cb248d9 |
| Timing-only baseline, local blocks (4.1) | `A/timing_only_baseline.py` | `timing_only_baseline.json` | e2adffb |
| Timing-only in epoch format; timing + MEG fusion (4.1) | `A/timing_only_cv.py`, `A/timing_meg_fusion.py` | `timing_only_cv.json`, `timing_meg_fusion_*.json` | 9dc85b4, 3d5858c |
| Recording-day drift, day-novel instrument (4.3) | `scripts/proxy_calib.py` (`PX=`) | `runs/proxy_calib*.json`, `logs/` | 5bb2556 |
| Per-subject input layer map (4.4) | `scripts/fita_map.py` | `runs/fita_map_*.json` | 10d8d48 |
| Time-stretch augmentation (4.4) | `X/train_stretch.py` | `runs/stretch_s*.json`, `runs/ci_stretch_s2.json` | 14e10da |
| Transfer seeds (4.4, 4.7) | `scripts/queue_ci_transfer.sh`, `A/seeded_cis.py` | `runs/ci_*.json`, `seeded_cis.json` | abc8e79 |
| Conditional predictability and word control (4.5) | `scripts/packard_tests.py`, `X/packard_strong.py` | `PREDICTIONS.md` (PACKARD) | d03adbf |
| Self-learned sentence prior (4.6, 4.7) | `scripts/hmm_prior.py`, `balance_submission.py` | `submissions/slp*.csv` | de10a30, a70cdfa |
| Where the sentence prior acts (4.7) | `A/prior_topk_churn.py` | `prior_topk_churn.json` | 15b6eaf |
| Public and private scores of all 61 submissions (4.4, 4.7) | `A/private_lb_table.py` | `results/lb_private_*/` | bd9c50b |
| Figures 1 and 2 | `A/fig_local_vs_lb.py`, `A/fig_public_vs_private.py` | `docs/figures/` | c27ef29, 17c56d7 |
| Simulator (5) | `H/tuning/sim_sweep.py` | `H/LEDGER.md`, SIM-1 in the tests log | de10a30 |
| Oracle pooling and label-free purity (6) | `scripts/oracle_scan.py`, `scripts/purity_scan.py` | `runs/oracle_scan.json` | c9fdc9c, 5976b01 |

Hardware: one NVIDIA RTX 4060 Ti (8 GB), 64 GB RAM. Python environment: `requirements-lock.txt`.
