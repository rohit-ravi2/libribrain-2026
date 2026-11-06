# Scheduler wrapper contract

`~/research/empirical_scheduler/acquire_lock.sh` and `release_lock.sh` were not on the machine that implemented this rung, so their interface was not verified. This repository does not call either script, and it does not vendor them.

The local coordinator holds the lock around one command. Suggested shape, to be checked against the real scripts before use:

1. `acquire_lock.sh` is executable, takes one argument `libribrain-sdr-out-of-day`, and exits 0 only when this process holds that lock. Any other exit status means the coordinator does not start the rung.
2. `release_lock.sh` is executable, takes the same argument, and exits 0 after releasing it.
3. The coordinator runs `release_lock.sh` from an `EXIT` trap so a failed or timed-out rung still releases the lock.
4. The locked command is exactly:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_out_of_day.sh
```

5. The coordinator does not start a second LibriBrain rung, and does not start the next ladder step, while this lock is held.

Until those two scripts are inspected locally, step 1–3 stay outside this repository.
