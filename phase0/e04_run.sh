#!/bin/bash
# E0.4: fine-tune the two halves of the QE top band at lr_scale 1.0, 3 seeds each.
# Pre-registered in PROTOCOL.md, "E0.4". Roughly 1h45m on one RTX 5090.
#
# Paths come from the environment so this runs on any box:
#   SFT  = this repo            MT = the training repo (Machine_translation)
#   PY   = python interpreter   DATA = the E0.4 split (phase0/e04_split.py output)
set -u
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
cd "$MT"

say "run matrix over the two halves"
"$PY" "$SFT/phase0/e03_run_matrix.py" --base-config "$SFT/configs/sft_base_enfr.yaml" \
  --data-dir "$DATA" --out-dir "$SFT/configs/phase0_e04" \
  --ckpt "$CKPT" --keep-last 1 \
  --conditions ft_topk_div,ft_topk_rep 2>&1 | tail -12
# the pipe's status is tail's, not the matrix's -- this repo has shipped that bug before
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "run matrix FAILED"; exit 1; }

say "6 runs at lr_scale 1.0"
"$PY" "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0_e04/run_stage2.sh" \
  --lr-scale 1 --gpus "${GPUS:-1}" --cwd "$MT" --logdir "$SFT/logs/phase0_e04" || say "run_parallel non-zero"

say "evaluate"
PYTHONPATH="$MT" "$PY" "$SFT/phase0/e03_collect.py" --mt-root "$MT" \
  --runner "$SFT/configs/phase0_e04/run_stage2.sh" --lr-scale 1 \
  --baseline-ckpt "$CKPT" \
  --baseline-config "$SFT/configs/sft_base_enfr.yaml" \
  --controls-dir "$DATA" --out "$SFT/results/e04_probe.tsv" \
  --ft-ckpt final --max-src-tokens 254 --jobs 1 || say "collect non-zero"
say "E04 DONE"
