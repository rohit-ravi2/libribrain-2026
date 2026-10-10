#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/ens_v3.py v3d1_s0,v3d1_s1 ens_v3d1 > logs/ens_v3d1.log 2>&1
BEST=$($PY -c "
import json; r={t: json.load(open(f'runs/{t}.json'))['summary']['deploy'] for t in ['v3d1_s0','v3d1_s1']}; print(max(r, key=r.get))")
echo "adapting $BEST"
$PY scripts/adapt_v3.py --run $BEST --tag ${BEST}_adapt > logs/${BEST}_adapt.log 2>&1
~/bin/notify "LibriBrain: ens $(grep SUMMARY logs/ens_v3d1.log | cut -c1-80) | adapt $BEST $(grep SUMMARY logs/${BEST}_adapt.log | cut -c1-100)" "LibriBrain"
