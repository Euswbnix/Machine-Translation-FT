#!/usr/bin/env bash
# Independent check of section 3.2 item 4, run on the Mac Studio from ~/cpu_items/item4_heldout_check/.
# At most 3 python processes at a time (xargs -P 3), each under nice -n 5. Reads ~/e06data and
# ~/e06runs/feasibility (read only); writes only under ~/cpu_items/item4_heldout_check/.
# inputs/ is copied from the MacBook first: e06data_SHA256SUMS, controls_manifest.json (E0.3),
# provenance_labels.npy, provenance_report.json, pool_mask_reused.npy, bundle_SHA256SUMS
# (~/mt_local/rebuild/rental_bundle_v2/SHA256SUMS), feasibility_SHA256SUMS (repo), filtered_tests/heldout_un.*
set -euo pipefail
cd ~/cpu_items/item4_heldout_check
PY=~/e06env/venv-arm64/bin/python
N=6
mkdir -p logs work
{ echo pre; for k in $(seq 0 $((N - 1))); do echo "pool $k $N"; done; echo sets; echo inputs; echo evalcopy; } > work/tasks.txt
xargs -P 3 -L 1 sh -c 'nice -n 5 '"$PY"' -u hcheck.py "$@" > logs/$(echo "$*" | tr " " _).log 2>&1' _ < work/tasks.txt
nice -n 5 "$PY" -u hcheck.py reduce > logs/reduce.log 2>&1
echo "ALL_DONE" >> logs/reduce.log
