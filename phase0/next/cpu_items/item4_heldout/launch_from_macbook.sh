#!/usr/bin/env bash
# Run from the MacBook once the Mac Studio is reachable (over Tailscale only: alexs-mac-studio).
#   launch_from_macbook.sh start   copy scripts + inputs to ~/cpu_items/item4_heldout/ and start run.sh there
#   launch_from_macbook.sh status  tail the Studio's logs
#   launch_from_macbook.sh fetch   copy results.json, parts/, check/, logs/ back into this folder
# Writes on the Studio only under ~/cpu_items/item4_heldout/; reads ~/e06data and ~/e06runs there.
set -euo pipefail
R=/Users/euswbnic/Machine-Translation-SFT
HERE=$R/phase0/next/cpu_items/item4_heldout
OPT=(-o BatchMode=yes -o HostKeyAlias=alexs-mac-studio)
H=euswbnic@alexs-mac-studio   # was euswbnic@10.0.0.126 (LAN); the Studio is now reached only over Tailscale
W=cpu_items/item4_heldout
case "${1:-}" in
start)
  ssh "${OPT[@]}" "$H" "mkdir -p ~/$W/inputs/filtered_tests ~/$W/logs && test ! -e ~/$W/results.json"
  scp "${OPT[@]}" "$HERE"/{heldout_hygiene.py,check_awk.sh,test_hygiene.py,run.sh} "$H:$W/"
  scp "${OPT[@]}" ~/mt_local/phase0_next/e06data_SHA256SUMS ~/mt_local/phase0_results/controls_manifest.json \
      ~/mt_local/rebuild/rental_bundle_v2/{provenance_labels.npy,provenance_report.json,pool_mask_reused.npy} "$H:$W/inputs/"
  scp "${OPT[@]}" "$R/phase0/next/e06_feasibility/SHA256SUMS" "$H:$W/inputs/feasibility_SHA256SUMS"
  scp "${OPT[@]}" ~/mt_local/phase0_results/filtered_tests/heldout_un.{en,fr} "$H:$W/inputs/filtered_tests/"
  ssh "${OPT[@]}" "$H" "cd ~/$W && (nohup bash run.sh > logs/run.log 2>&1 &) && echo started"
  ;;
status)
  ssh "${OPT[@]}" "$H" "cd ~/$W && tail -n 3 logs/*.log; ls -la parts check 2>/dev/null"
  ;;
fetch)
  ssh "${OPT[@]}" "$H" "grep -q '^EXIT=0' ~/$W/logs/reduce.log"
  mkdir -p "$HERE/parts" "$HERE/check" "$HERE/logs"
  scp "${OPT[@]}" "$H:$W/results.json" "$HERE/"
  scp "${OPT[@]}" "$H:$W/parts/*.json" "$HERE/parts/"
  scp "${OPT[@]}" "$H:$W/check/*.txt" "$HERE/check/"
  scp "${OPT[@]}" "$H:$W/logs/*.log" "$HERE/logs/"
  ;;
*) echo "usage: $0 start|status|fetch"; exit 64 ;;
esac
