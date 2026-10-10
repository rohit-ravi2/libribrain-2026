#!/bin/bash
# v3e: adaptation on the best of v3c/v3d (by split-v2 deploy), full grid. Runs alone (RAM: one cache copy).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
BEST=$($PY -c "
import json; r={t: json.load(open(f'runs/{t}.json'))['summary']['deploy'] for t in ['v3c','v3d']}; print(max(r, key=r.get))")
echo "adapting $BEST"
$PY scripts/adapt_v3.py --run $BEST --tag ${BEST}_adapt > logs/${BEST}_adapt.log 2>&1
~/bin/notify "LibriBrain v3e adapt on $BEST: $(grep SUMMARY logs/${BEST}_adapt.log | cut -c1-120)" "LibriBrain"
