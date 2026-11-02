# Private leaderboard pull (2026-11-02)

Pulled 2026-11-02 ~12:29 UTC with `kaggle competitions submissions pnpl-competition-2026-{broad,deep} --csv --page-size 200`
and `kaggle competitions leaderboard <slug> --show --csv --page-size 200` (private ranking) / `--download` (public ranking).

- `subs_{broad,deep}.csv` — all 35 Broad / 26 Deep submissions with public and private score (raw CLI output).
- `submissions_public_private.csv`, `interventions_public_private.csv` — built by `docs/writeups/analyses/private_lb_table.py`.
- `lb_{broad,deep}/private_leaderboard_show_20261102.csv` — private ranking (teamId, name, date, score).
- `lb_{broad,deep}/pnpl-*-publicleaderboard-*.csv` — public ranking at close.

Final standing: Broad private 0.67846, rank 17/53 (public 0.72230, rank 8). Deep private 0.69704 (xr1_deep_bn_isoB0bn_k20.csv), rank 26/91 (public 0.72928, rank 15).
Best private files submitted but not selected: Broad bna05_broad_ens3_balT2.csv 0.68037; Deep bnD_deep_a1.0_balT3.csv / drift1_deep_bn_balT3.csv 0.71779.
Note: the private ranking may still change if the organisers' code review removes teams.
