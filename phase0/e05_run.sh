#!/bin/bash
# E0.5: six sets, 3 seeds each, all on ONE machine in one session -- the two halves of the QE
# top band and a matched random control, once over every source and once inside UN text only.
# Pre-registered in PROTOCOL.md, "E0.5".
#
#   SRC=<dir with ft_topk.{en,fr,source} ft_random.{en,fr,source} heldout_un.{en,fr}> \
#     GPUS=<n> bash phase0/e05_run.sh
#
# 30 runs (six sets x five seeds) of about 15 minutes each on the author's own RTX 5090 and
# about 26 on the rented ones measured so far. One GPU per run, so the number of WAVES is
# what matters: 30 runs are 8 waves on four cards, 5 on six, 4 on eight, 3 on ten.
#
# Rerun the same command after a crash: finished runs are skipped. SRC is used only while the
# sets are not built. A directory counts as built only when its manifest says so and every
# file still has the sha256 recorded there; anything else is rebuilt from SRC, and rebuilt
# sets are retrained in full under a new checkpoint root (they are never resumed onto).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SFT="${SFT:-$(cd "$HERE/.." && pwd)}"
export PY="${PY:-$(command -v python || command -v python3 || echo python)}"
export PROBE=e05
export CONDITIONS=ft_topk_div,ft_topk_rep,ft_random_small,ft_un_div,ft_un_rep,ft_un_random
export SEEDS=42,1,2,3,4          # PROTOCOL E0.5: five, fixed before any run
export DATA="${DATA:-$SFT/data/phase0_e05}"
WORK="$(dirname "$SFT")"
mkdir -p "$SFT/logs" "$SFT/results"
CONSOLE="$SFT/logs/e05_console.log"
say() { printf '\n>>> %s %s\n' "$(date -Is)" "$*"; }

main() {
  local mt="${MT:-$WORK/Machine_translation}" free
  if ! "$PY" "$HERE/e05_build.py" --verify --dir "$DATA"; then
    : "${SRC:?SRC=<dir with ft_topk.*, ft_random.* (.en .fr .source) and heldout_un.*> is needed to build the sets}"
    # Nothing is built, so nothing is trained: all 30 runs are ahead, about 100 GB of
    # checkpoints. Find out now rather than after the build and the cache warm-up.
    # (probe_run.sh checks again on every pass, counting only the runs still to do.)
    free=$(df -Pk "$mt" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}')
    if [ -n "$free" ] && [ "$free" -lt 100 ]; then
      echo "NOT ENOUGH DISK: $free GB free under $mt; E0.5 needs about 100 GB (30 runs x 3 GB + 10)"
      return 1
    fi
    say "build the two halves, then the four E0.5 sets"
    mkdir -p "$DATA"
    # a cache left by an earlier build would be loaded in place of the rebuilt text
    rm -f "$DATA"/*.cached_*.npz
    "$PY" "$HERE/e04_split.py" "$SRC" "$DATA" || { echo "e04_split FAILED"; return 1; }
    "$PY" "$HERE/e05_build.py" --dir "$DATA" --src-dir "$SRC" || { echo "e05_build FAILED"; return 1; }
    "$PY" "$HERE/e05_build.py" --verify --dir "$DATA" || { echo "the build did not verify"; return 1; }
  fi

  # Which code, on which machine, produced the runs: appended on every invocation, so a
  # resumed session shows up as a second entry instead of overwriting the first.
  {
    echo "===== $(date -Is) on $(hostname) ====="
    echo "SFT  $(git -C "$SFT" rev-parse HEAD 2>&1)  tracked diff sha256: $(git -C "$SFT" diff HEAD 2>/dev/null | sha256sum 2>/dev/null | cut -c1-64)  status: $(git -C "$SFT" status --porcelain 2>/dev/null | tr '\n' ';')"
    echo "MT   $(git -C "$mt" rev-parse HEAD 2>&1)  diff sha256: $(git -C "$mt" diff 2>/dev/null | sha256sum 2>/dev/null | cut -c1-64)"
    "$PY" -c 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)' 2>&1
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1
    echo "--- pip freeze"
    "$PY" -m pip freeze 2>&1
  } >> "$SFT/results/e05_box.txt"

  bash "$HERE/probe_run.sh"
  local rc=$?
  say "apply the pre-stated outcomes"
  "$PY" "$HERE/e05_report.py" --results "$SFT/results/e05_probe.tsv" --logdir "$SFT/logs/phase0_e05" --expect-seeds 5 \
    | tee "$SFT/results/e05_report.txt"
  local rrc=${PIPESTATUS[0]}
  echo "E05 probe_run rc=$rc  report rc=$rrc"
  [ "$rc" -eq 0 ] && [ "$rrc" -eq 0 ]
}

main 2>&1 | tee -a "$CONSOLE"
status=${PIPESTATUS[0]}

# Everything that makes the result auditable lives in untracked or ignored paths on this box
# and dies with it. Pack it, whatever the status, and print the hash to check after copying.
cd "$SFT" || exit 1
items=()
for p in data/phase0_e05 logs/phase0_e05 logs/e05_console.log configs/phase0_e05 \
         results/e05_probe.tsv results/e05_probe.meta.json results/e05_probe.partial.tsv \
         results/e05_probe.partial.meta.json results/e05_report.txt results/e05_box.txt \
         results/.e03_collect_cache.json results/filtered_tests; do
  [ -e "$p" ] && items+=("$p")
done
extra=()
for f in sft_rev.txt mt_rev.txt pip_freeze_env.txt accept_base.log accept_valid_bleu.txt; do
  [ -e "$WORK/$f" ] && extra+=("$f")
done
ART="$SFT/results/e05_artifacts.tar.gz"
if [ "${#items[@]}" -gt 0 ] && tar -czf "$ART" --exclude='*.npz' --exclude='*.tmp' "${items[@]}" \
     ${extra[@]:+-C "$WORK" "${extra[@]}"}; then
  sum=$(sha256sum "$ART" 2>/dev/null || shasum -a 256 "$ART")
  printf '\n>>> %s artifacts: %s\n    sha256 %s\n    copy it off this box and check the hash BEFORE the box is destroyed\n' \
    "$(date -Is)" "$ART" "${sum%% *}"
else
  echo "PACKING FAILED: copy data/phase0_e05, logs/, configs/phase0_e05 and results/ by hand"
  [ "$status" -eq 0 ] && status=1
fi
exit "$status"
