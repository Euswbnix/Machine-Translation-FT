#!/usr/bin/env bash
# Independent check of item 3, on the Mac Studio. At most 3 compute processes at a time, nice 5.
# Copy item3_check.py, nl_lid.swift, nl_lid_check.swift and e06data_SHA256SUMS here first.
set -euo pipefail
D=~/cpu_items/item3_contamination_check
cd "$D"
mkdir -p logs work tmp mcache
export TMPDIR="$D/tmp/"
PY=~/e06env/venv-arm64/bin/python
SUMS="$D/e06data_SHA256SUMS"
sw_vers > logs/env.log; xcrun swiftc --version >> logs/env.log 2>&1
xcrun swiftc -O -module-cache-path mcache nl_lid.swift -o nl_lid
xcrun swiftc -O -module-cache-path mcache nl_lid_check.swift -o nl_lid_check
nice -n 5 "$PY" -I -u item3_check.py prep --sums "$SUMS" > logs/prep.log 2>&1
pids=()
for k in 0 1 2; do
  nice -n 5 ./nl_lid "work/lid_in_$k.txt" "work/lid_out_$k.tsv" > "logs/lid_$k.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
nice -n 5 "$PY" -I -u item3_check.py sample --sums "$SUMS" > logs/sample.log 2>&1
nice -n 5 ./nl_lid_check work/xcheck_in.txt work/xcheck_out.tsv > logs/xcheck.log 2>&1
nice -n 5 "$PY" -I -u item3_check.py stats --sums "$SUMS" > logs/stats.log 2>&1
# second, separate count of the headline ratio rule (awk fields: blanks only, bytes, LC_ALL=C)
for s in ft_un_rep ft_un_div ft_un_random ft_topk_rep ft_topk_div ft_random_small heldout_un; do
  LC_ALL=C awk -v FR="$HOME/e06data/phase0_e05/$s.fr" -v S=$s '
    { if ((getline f < FR) <= 0) { print "short fr file"; exit 1 }
      ne = NF; nf = split(f, a, " "); n++
      for (t = 1; t <= 3; t++) { num = (t == 1 ? 13 : (t == 2 ? 15 : 20))
        if (ne == 0 || nf == 0 || 10 * nf > num * ne || 10 * ne > num * nf) c[t]++ } }
    END { printf "%s rows %d ratio_ws_1.3 %d ratio_ws_1.5 %d ratio_ws_2.0 %d\n", S, n, c[1], c[2], c[3] }' \
    "$HOME/e06data/phase0_e05/$s.en"
done > logs/awk_ratio.log
echo done > logs/DONE
