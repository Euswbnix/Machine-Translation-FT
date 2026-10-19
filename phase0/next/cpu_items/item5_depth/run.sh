#!/usr/bin/env bash
# The order item 5 was run in on the Mac Studio (2026-10-19), from ~/cpu_items/item5_depth/.
# Inputs copied there first (inputs/): provenance_labels.npy and provenance_report.json from
# ~/mt_local/rebuild/prov_v2_fixed/, pool_mask_reused.npy from ~/mt_local/rebuild/rental_bundle_v2/,
# controls_manifest.json from ~/mt_local/phase0_results/ (the E0.3 build's manifest).
set -euo pipefail
cd ~/cpu_items/item5_depth
PY=~/e06env/venv-arm64/bin/python
nice -n 5 $PY -u depth.py textsha            > logs/textsha.log 2>&1 &   # 1 process
nice -n 5 $PY -u depth.py pass1 --workers 3  > logs/pass1.log 2>&1       # 3 workers
wait
nice -n 5 $PY -u depth.py rank               > logs/rank.log 2>&1
nice -n 5 $PY -u depth.py pass2 --workers 4  > logs/pass2.log 2>&1
nice -n 5 $PY -u depth.py reduce             > logs/reduce.log 2>&1
nice -n 5 $PY -u verify.py                   > logs/verify.log 2>&1     # direct recomputation on ft_topk prefixes
nice -n 5 $PY -u depth.py sized              > logs/sized.log 2>&1      # size-matched supplement
