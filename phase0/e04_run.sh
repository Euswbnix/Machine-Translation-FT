#!/bin/bash
# E0.4: fine-tune the two halves of the QE top band at lr_scale 1.0, 3 seeds each.
# Pre-registered in PROTOCOL.md, "E0.4". Roughly 1h45m on one RTX 5090.
#
# Paths come from the environment so this runs on any box:
#   SFT  = this repo            MT = the training repo (Machine_translation)
#   PY   = python interpreter   DATA = the E0.4 split (phase0/e04_split.py output)
set -uo pipefail          # pipefail: `cmd | tail` must report cmd's status, not tail's
SFT="${SFT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MT="${MT:-$HOME/mt/Machine_translation}"
PY="${PY:-$(command -v python || command -v python3 || echo python)}"
DATA="${DATA:-$SFT/data/phase0_e04}"
CKPT="${CKPT:-ckpt_hf/enfr_base_v1.1_averaged.pt}"

say() { printf '\n>>> %s %s\n' "$(date -Is)" "$*"; }
for f in ft_topk_div.en ft_topk_div.fr ft_topk_rep.en ft_topk_rep.fr heldout_un.en heldout_un.fr; do
  [ -s "$DATA/$f" ] || { echo "MISSING $DATA/$f -- run phase0/e04_split.py first"; exit 1; }
done
[ -d "$MT" ] || { echo "MISSING training repo: $MT (set MT=)"; exit 1; }
# The token accounting this probe reports (Tok(applied)) exists only in the trainer patch,
# which lives in the working tree of $MT and is never committed. Without it the runs still
# train, but the pre-registered secondary report cannot be produced afterwards.
grep -q applied_target_tokens "$MT/src/training/trainer.py" 2>/dev/null || {
  echo "trainer patch not applied in $MT (no applied_target_tokens) -- see phase0/trainer_token_accounting.patch"; exit 1; }
cd "$MT"
rc=0

# --spike-ratio 0 is decision D9, frozen in phase0/e03_decisions.json: with the guard on,
# an arm whose loss spikes more often drops more effective batches, and these two halves
# differ in loss variance BY CONSTRUCTION -- exactly the axis under test.
say "run matrix over the two halves (spike guard off, per D9)"
"$PY" "$SFT/phase0/e03_run_matrix.py" --base-config "$SFT/configs/sft_base_enfr.yaml" \
  --data-dir "$DATA" --out-dir "$SFT/configs/phase0_e04" \
  --ckpt "$CKPT" --keep-last 1 --spike-ratio 0 \
  --conditions ft_topk_div,ft_topk_rep 2>&1 | tail -12
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "run matrix FAILED"; exit 1; }

# Parallel runs of one condition race on the trainer's non-atomic tokenisation cache, and a
# crash mid-write poisons every later attempt; warm_cache builds both serially and deletes
# an unreadable cache on a retry.
say "warm the tokenisation caches"
"$PY" "$SFT/phase0/warm_cache.py" --runner "$SFT/configs/phase0_e04/run_stage2.sh" \
  --lr-scale 1 --cwd "$MT" || exit 1

say "6 runs at lr_scale 1.0"
"$PY" "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0_e04/run_stage2.sh" \
  --lr-scale 1 --gpus "${GPUS:-1}" --cwd "$MT" --logdir "$SFT/logs/phase0_e04" \
  || { say "run_parallel non-zero"; rc=1; }

# run_parallel counts any existing final.pt as a finished run, and e03_collect torch.loads
# it without a guard: one checkpoint truncated by a crash mid-save would abort the whole
# evaluation and never be retrained. Delete the unreadable ones so a rerun redoes them.
say "verify the 6 checkpoints are loadable"
"$PY" - "$SFT/configs/phase0_e04/run_stage2.sh" <<'PYEOF' || { say "checkpoint verification non-zero"; rc=1; }
import re, sys
from pathlib import Path
try:
    import torch
except ImportError:
    print("  torch not importable here; skipping (e03_collect will load them)"); raise SystemExit(0)
runner = Path(sys.argv[1])
bad = 0
for line in runner.read_text(encoding="utf-8").split("\n"):
    m = re.search(r"--config\s+(\S+).*?--suffix\s+(\S+)", line)
    if not m or "train.py" not in line:
        continue
    cfg = Path(m.group(1).replace("${LR}", "1"))
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
raise SystemExit(1 if bad else 0)
PYEOF

say "evaluate"
PYTHONPATH="$MT" "$PY" "$SFT/phase0/e03_collect.py" --mt-root "$MT" \
  --runner "$SFT/configs/phase0_e04/run_stage2.sh" --lr-scale 1 \
  --baseline-ckpt "$CKPT" \
  --baseline-config "$SFT/configs/sft_base_enfr.yaml" \
  --controls-dir "$DATA" --out "$SFT/results/e04_probe.tsv" \
  --ft-ckpt final --max-src-tokens 254 --jobs 1 || { say "collect non-zero"; rc=1; }

say "E04 DONE rc=$rc"
exit $rc
