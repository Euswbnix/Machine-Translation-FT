#!/bin/bash
# Fine-tune a set of conditions at one lr_scale, 3 seeds each, and evaluate them: the shared
# body of the explanatory probes (E0.4, E0.5). Not a gate, and not called directly -- use the
# wrapper named after the probe (e04_run.sh, e05_run.sh), which fixes PROBE and CONDITIONS.
#
#   PROBE       e04 | e05 ...   names configs/phase0_<PROBE>, logs/phase0_<PROBE>,
#                               results/<PROBE>_probe.tsv and the default DATA dir
#   CONDITIONS  comma list of sets; each needs <name>.en/.fr in DATA
#   SEEDS       comma list of seeds; unset = the gate's three (42,1,2), which is what E0.4 ran
#   SFT / MT / PY / DATA / CKPT / GPUS / EVAL_JOBS   paths and parallelism, see defaults
set -uo pipefail          # pipefail: `cmd | tail` must report cmd's status, not tail's
: "${PROBE:?set PROBE (use the e0N_run.sh wrapper)}"
: "${CONDITIONS:?set CONDITIONS (use the e0N_run.sh wrapper)}"
SFT="${SFT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MT="${MT:-$HOME/mt/Machine_translation}"
PY="${PY:-$(command -v python || command -v python3 || echo python)}"
DATA="${DATA:-$SFT/data/phase0_$PROBE}"
CKPT="${CKPT:-ckpt_hf/enfr_base_v1.1_averaged.pt}"
GPUS="${GPUS:-1}"
EVAL_JOBS="${EVAL_JOBS:-$GPUS}"
LR=1                      # lr_scale 1.0: the regime in which the effect under study exists
SEEDS="${SEEDS:-}"
CFG="$SFT/configs/phase0_$PROBE"
LOGS="$SFT/logs/phase0_$PROBE"
OUT="$SFT/results/${PROBE}_probe.tsv"

say() { printf '\n>>> %s %s\n' "$(date -Is)" "$*"; }
IFS=',' read -r -a CONDS <<< "$CONDITIONS"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS:-42,1,2}"
N_RUNS=$(( ${#CONDS[@]} * ${#SEED_LIST[@]} ))
for c in "${CONDS[@]}"; do
  for e in en fr; do
    [ -s "$DATA/$c.$e" ] || { echo "MISSING $DATA/$c.$e -- build the sets first"; exit 1; }
  done
done
for f in heldout_un.en heldout_un.fr manifest.json; do
  [ -s "$DATA/$f" ] || { echo "MISSING $DATA/$f -- build the sets first"; exit 1; }
done
# A set that is on disk but not listed in the manifest is outside the controls fingerprint:
# it could change without anything noticing. Refuse to train on one.
"$PY" -c '
import json, sys
written = set(json.load(open(sys.argv[1], encoding="utf-8")).get("written", []))
out = [f"{c}.{e}" for c in sys.argv[2:] for e in ("en", "fr") if f"{c}.{e}" not in written]
if out:
    print("NOT IN THE MANIFEST (so not in the controls fingerprint): " + ", ".join(out))
    raise SystemExit(1)
' "$DATA/manifest.json" "${CONDS[@]}" || exit 1
[ -d "$MT" ] || { echo "MISSING training repo: $MT (set MT=)"; exit 1; }
# A run or an evaluation leased to a GPU id that does not exist falls back to CPU without
# an error and holds the whole pass open. Refuse a count the box cannot back.
if command -v nvidia-smi >/dev/null 2>&1; then
  have=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  if [ "$GPUS" -gt "${have:-0}" ] || [ "$EVAL_JOBS" -gt "${have:-0}" ]; then
    echo "GPUS=$GPUS EVAL_JOBS=$EVAL_JOBS but this box has ${have:-0} GPU(s)"; exit 1
  fi
fi
# The token accounting these probes report (Tok(applied)) exists only in the trainer patch,
# which lives in the working tree of $MT and is never committed. Without it the runs still
# train, but the pre-registered secondary report cannot be produced afterwards.
grep -q applied_target_tokens "$MT/src/training/trainer.py" 2>/dev/null || {
  echo "trainer patch not applied in $MT (no applied_target_tokens) -- see phase0/trainer_token_accounting.patch"; exit 1; }
cd "$MT" || exit 1
[ -s "$CKPT" ] || { echo "MISSING base checkpoint: $MT/$CKPT"; exit 1; }
rc=0

# --spike-ratio 0 is decision D9, frozen in phase0/e03_decisions.json: with the guard on,
# an arm whose loss spikes more often drops more effective batches, and the sets compared
# here differ in loss variance BY CONSTRUCTION -- exactly the axis under test.
say "run matrix over $CONDITIONS (spike guard off, per D9)"
"$PY" "$SFT/phase0/e03_run_matrix.py" --base-config "$SFT/configs/sft_base_enfr.yaml" \
  --data-dir "$DATA" --out-dir "$CFG" \
  --ckpt "$CKPT" --keep-last 1 --spike-ratio 0 \
  --conditions "$CONDITIONS" ${SEEDS:+--seeds "$SEEDS"} 2>&1 | tail -12
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "run matrix FAILED"; exit 1; }

# Each run keeps final.pt, best.pt and emergency.pt beside its last step checkpoint: about
# 3 GB. A full disk in the second wave kills every run in flight, so check before anything
# is trained. Runs of THIS matrix that already finished have their space and are not counted.
root=$("$PY" -c 'import json, sys; print((json.load(open(sys.argv[1])).get("controls_sha") or "")[:12])' "$CFG/matrix.json" 2>/dev/null)
done_runs=0
[ -n "$root" ] && done_runs=$(find "checkpoints/phase0/$root" -name final.pt 2>/dev/null | wc -l | tr -d ' ')
need_gb=$(( (N_RUNS > done_runs ? N_RUNS - done_runs : 0) * 3 + 10 ))
free_gb=$(df -Pk "$MT" | awk 'NR==2 {print int($4 / 1048576)}')
[ "${free_gb:-0}" -ge "$need_gb" ] || { echo "NOT ENOUGH DISK: $free_gb GB free under $MT, $need_gb GB needed for the $(( N_RUNS - done_runs )) runs still to do"; exit 1; }

# Parallel runs of one condition race on the trainer's non-atomic tokenisation cache, and a
# crash mid-write poisons every later attempt; warm_cache builds every cache serially and
# deletes an unreadable one on a retry.
say "warm the tokenisation caches"
"$PY" "$SFT/phase0/warm_cache.py" --runner "$CFG/run_stage2.sh" \
  --lr-scale "$LR" --cwd "$MT" || exit 1

say "$N_RUNS runs at lr_scale $LR on $GPUS GPU(s)"
"$PY" "$SFT/phase0/run_parallel.py" --runner "$CFG/run_stage2.sh" \
  --lr-scale "$LR" --gpus "$GPUS" --cwd "$MT" --logdir "$LOGS" \
  || { say "run_parallel non-zero"; rc=1; }

# run_parallel counts any existing final.pt as a finished run, and e03_collect torch.loads
# it without a guard: one checkpoint truncated by a crash mid-save would abort the whole
# evaluation and never be retrained. Delete the unreadable ones so a rerun redoes them.
say "verify the checkpoints are loadable"
"$PY" - "$CFG/run_stage2.sh" "$LR" <<'PYEOF' || { say "checkpoint verification non-zero"; rc=1; }
import re, sys
from pathlib import Path
try:
    import torch
except ImportError:
    print("  torch not importable here; skipping (e03_collect will load them)"); raise SystemExit(0)
runner, lr = Path(sys.argv[1]), sys.argv[2]
bad = n = 0
for line in runner.read_text(encoding="utf-8").split("\n"):
    m = re.search(r"--config\s+(\S+).*?--suffix\s+(\S+)", line)
    if not m or "train.py" not in line:
        continue
    n += 1
    cfg = Path(m.group(1).replace("${LR}", lr))
    try:
        import yaml
        d = yaml.safe_load(open(cfg))["checkpoint"]["dir"]
    except Exception as e:
        print(f"  cannot read {cfg}: {e}"); bad += 1; continue
    p = Path(d + m.group(2)) / "final.pt"
    if not p.exists():
        print(f"  MISSING {p}"); bad += 1; continue
    try:
        torch.load(p, map_location="cpu", weights_only=False)
        print(f"  ok {p}")
    except Exception as e:
        print(f"  UNREADABLE {p}: {e} -- deleting so a rerun retrains it")
        p.unlink(); bad += 1
if not n:
    print(f"  no train.py lines in {runner}"); bad = 1
raise SystemExit(1 if bad else 0)
PYEOF

say "evaluate ($EVAL_JOBS at a time)"
PYTHONPATH="$MT" "$PY" "$SFT/phase0/e03_collect.py" --mt-root "$MT" \
  --runner "$CFG/run_stage2.sh" --lr-scale "$LR" \
  --baseline-ckpt "$CKPT" \
  --baseline-config "$SFT/configs/sft_base_enfr.yaml" \
  --controls-dir "$DATA" --out "$OUT" \
  --ft-ckpt final --max-src-tokens 254 --jobs "$EVAL_JOBS" || { say "collect non-zero"; rc=1; }

say "${PROBE} DONE rc=$rc"
exit $rc
