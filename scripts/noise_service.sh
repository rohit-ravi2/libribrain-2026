#!/bin/bash
# Runs inside systemd service lb-noise. The queue skips finished runs, so 3 passes redo anything that was killed.
for i in 1 2 3; do ~/projects/libribrain-2026/scripts/queue_noise_seeds_ablation.sh; done
