#!/bin/bash
# Polls Kaggle until the account has joined the Broad competition, then submits $1 with message $2 once.
K=~/venvs/libribrain/bin/kaggle; C=pnpl-competition-2026-broad; F=$1; MSG=$2
while true; do
  if $K competitions list -s $C 2>/dev/null | grep -q "$C .* True "; then
    $K competitions submit $C -f "$F" -m "$MSG" && ~/bin/notify "LibriBrain: submitted $(basename $F) to Broad" "LibriBrain"
    sleep 600; $K competitions submissions $C 2>&1 | head -5
    exit 0
  fi
  sleep 600
done
