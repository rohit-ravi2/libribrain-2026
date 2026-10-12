#!/bin/bash
# Bet C: download OpenNeuro ds007523 (LittlePrince_MEG_French_Listen, CC0) MEG + events + stimuli, no anatomy.
# Anonymous S3 via rclone (no account). Resumable: re-running skips finished files.
DST=/mnt/ssd4tb/data/extmeg/lpp_listen
LOG=$HOME/projects/libribrain-2026/logs/extmeg_download_lpp_listen.log
mkdir -p "$DST"
exec rclone copy ":s3,provider=AWS,region=us-east-1:openneuro.org/ds007523/" "$DST" \
  --exclude "**/anat/**" --exclude ".datalad/**" --exclude ".git*" \
  --transfers 8 --multi-thread-streams 4 --checkers 16 --s3-chunk-size 64M \
  --stats 2m --stats-one-line -v --log-file "$LOG"
