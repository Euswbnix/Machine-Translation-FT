#!/usr/bin/env bash
# The order item 4 is run in on the Mac Studio, from ~/cpu_items/item4_heldout/ (at most 6
# processes at a time, all under nice -n 5). Copied there first from the MacBook (inputs/):
#   e06data_SHA256SUMS            ~/mt_local/phase0_next/e06data_SHA256SUMS
#   controls_manifest.json        ~/mt_local/phase0_results/ (E0.3's build manifest)
#   provenance_labels.npy, provenance_report.json, pool_mask_reused.npy
#                                 ~/mt_local/rebuild/rental_bundle_v2/ (sha256 in its SHA256SUMS)
#   filtered_tests/heldout_un.*   ~/mt_local/phase0_results/filtered_tests/ (the copy the runs scored)
#   feasibility_SHA256SUMS        phase0/next/e06_feasibility/SHA256SUMS (the repo's record of the run)
set -euo pipefail
cd ~/cpu_items/item4_heldout
PY=~/e06env/venv-arm64/bin/python
O=~/cpu_items/item4_heldout
mkdir -p logs
rm -rf test_fixture
nice -n 5 $PY -u test_hygiene.py --tmp "$O/test_fixture" > logs/test.log 2>&1          # 1 process (+3 awk at its end)
# phase 1: 1 + 1 + 3 awk = 5 processes
nice -n 5 $PY -u heldout_hygiene.py inputs --out "$O" > logs/inputs.log 2>&1 & p1=$!
nice -n 5 $PY -u heldout_hygiene.py sets   --out "$O" > logs/sets.log 2>&1 & p2=$!
nice -n 5 bash check_awk.sh > logs/check_awk.log 2>&1 & p3=$!
rc=0
for p in $p1 $p2 $p3; do wait "$p" || rc=1; done
# phase 2 and 3: 6 workers each, one after the other
nice -n 5 $PY -u heldout_hygiene.py pool     --out "$O" --workers 6 > logs/pool.log 2>&1 || rc=1
nice -n 5 $PY -u heldout_hygiene.py pretrain --out "$O" --workers 6 > logs/pretrain.log 2>&1 || rc=1
nice -n 5 $PY -u heldout_hygiene.py reduce   --out "$O" > logs/reduce.log 2>&1 || rc=1
echo "EXIT=$rc" >> logs/reduce.log
exit $rc
