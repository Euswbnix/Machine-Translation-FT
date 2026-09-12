#!/usr/bin/env bash
# Phase 0 on a rented GPU box, from nothing.
# USAGE-BEGIN
#   bash rental_setup.sh env          clone, install, apply trainer patch, self-test   (no GPU)
#   bash rental_setup.sh accept       rebuild Base v1.1 from HF, reproduce test BLEU    (1 GPU, minutes)
#   bash rental_setup.sh data         download WMT14 fr-en, rebuild the v2 corpus      (CPU, ~1 h)
#   bash rental_setup.sh score        CometKiwi-22 over v2 (needs YOUR hf login)        (all GPUs)
#   bash rental_setup.sh provenance   statmt constituents + e01 hash-join              (CPU; can run during score)
#   bash rental_setup.sh controls     E0.3 control sets + run matrix                   (CPU)
#   bash rental_setup.sh stage1       4-run LR sweep, one GPU per run                   (GPUs)
#   bash rental_setup.sh stage2 <lr>  3 conditions x 3 seeds at the chosen lr_scale     (GPUs)
#   bash rental_setup.sh gate <lr>    evaluate every run, apply the pre-registered rule (GPUs)
# USAGE-END
#
# UNTESTED until it has run once on a real box — written against the repos'
# CLIs, but no stage has been executed end to end yet.
#
# Why no original training machine is needed: the paper fine-tuned the averaged
# Base v1.1 / Big v1.1 checkpoints, which are exactly the public HF releases
# (steps and BLEU match paper_wmt_upload/sections/05_sft.tex digit for digit);
# phase0/hf_to_ckpt.py restores the keys the trainer needs. What the original box
# alone still holds is the training logs and the paper's exact scored file; both
# are optional for E0.3 (the log grep only confirms an LR already derived from code).
#
# Disk: plan on >= 150 GB (HF parquet cache + 40M raw + 30M clean + scored TSV).

set -euo pipefail
WORK="${WORK:-$HOME/mt}"
MT="$WORK/Machine_translation"
# The SFT configs hard-code ../Machine-Translation-SFT/... paths, so the clone
# directory must keep this name even though the GitHub repo was renamed.
SFT="$WORK/Machine-Translation-SFT"
BRANCH="${BRANCH:-wmt2027-phase0}"
EXPECTED_V2_ROWS=30129500          # paper_section_7.md: "v2's 30,129,500 pairs"
EXPECTED_BASE_TEST_BLEU=35.31      # HF config.json and 05_sft.tex:26
BLEU_TOL=0.15                      # decode batch size can move BLEU by a few hundredths

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

stage_env() {
  mkdir -p "$WORK"
  [ -d "$MT/.git" ]  || git clone https://github.com/Euswbnix/Machine_translation.git "$MT"
  [ -d "$SFT/.git" ] || git clone -b "$BRANCH" https://github.com/Euswbnix/Machine-Translation-FT.git "$SFT"
  git -C "$SFT" fetch -q && git -C "$SFT" checkout -q "$BRANCH" && git -C "$SFT" pull -q --ff-only
  nvidia-smi -L || die "no GPU visible"
  torch_before=$(python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo "none")
  log "installing python deps (image torch: $torch_before)"
  python -m pip install -q -r "$MT/requirements.txt"
  python -m pip install -q unbabel-comet huggingface_hub
  torch_after=$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')
  echo "torch before: $torch_before | after: $torch_after"
  # unbabel-comet / lightning can pull in a different torch. An RTX 5090 needs a
  # recent CUDA build; a silently replaced torch fails later, mid-experiment.
  python -c 'import torch, sys; torch.zeros(1).cuda(); sys.exit(0)' || die \
"CUDA is not usable after installing deps (torch now: $torch_after, was: $torch_before).
pip likely replaced the image's CUDA torch. Reinstall the CUDA build matching this driver, then rerun 'env'."
  log "trainer patch"
  if git -C "$MT" apply --check -p1 "$SFT/phase0/trainer_token_accounting.patch" 2>/dev/null; then
    git -C "$MT" apply -p1 "$SFT/phase0/trainer_token_accounting.patch"
    echo "applied"
  elif git -C "$MT" apply --check --reverse -p1 "$SFT/phase0/trainer_token_accounting.patch" 2>/dev/null; then
    echo "already applied"
  else
    die "trainer patch neither applies nor is already applied — Machine_translation has diverged"
  fi
  log "self-test"
  (cd "$SFT" && MT_REPO="$MT" python tests/regress.py) || die "regression failed on this box"
}

stage_accept() {
  cd "$MT"
  mkdir -p data_enfr_v1 ckpt_hf
  if [ ! -s data_enfr_v1/test.en ]; then
    log "WMT14 fr-en valid/test (the download script writes train too; that is reused by 'data')"
    python scripts/download_wmt_enfr.py --output-dir data_enfr_v2_raw
    cp data_enfr_v2_raw/valid.en data_enfr_v2_raw/valid.fr data_enfr_v2_raw/test.en data_enfr_v2_raw/test.fr data_enfr_v1/
  fi
  log "rebuild Base v1.1 averaged checkpoint from HF"
  python "$SFT/phase0/hf_to_ckpt.py" --repo euswbnix/transformer-wmt14-enfr-base \
      --mt-root "$MT" --train-config "$SFT/configs/sft_base_enfr.yaml" \
      --download-dir "$WORK/hf_cache" --out ckpt_hf/enfr_base_v1.1_averaged.pt
  # the SFT config points spm_model at data_enfr_v1/spm_enfr_v1_fixed.model
  cp ckpt_hf/enfr_base_v1.1_averaged.sentencepiece.model data_enfr_v1/spm_enfr_v1_fixed.model
  log "acceptance: reproduce the released test BLEU before any fine-tuning"
  python scripts/eval_bleu.py --ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
      --config "$SFT/configs/sft_base_enfr.yaml" \
      --src data_enfr_v1/test.en --ref data_enfr_v1/test.fr \
      --beam 5 --length-penalty 1.0 | tee "$WORK/accept_base.log"
  got=$(grep -oE "BLEU \([a-z]+, sacrebleu 13a\): [0-9.]+" "$WORK/accept_base.log" | awk '{print $NF}' || true)
  # without this, set -e would exit here silently when the BLEU line is missing
  [ -n "$got" ] || die "no BLEU line in $WORK/accept_base.log — eval_bleu.py failed; read the log"
  python - "$got" <<PY || die "test BLEU $got is not within $BLEU_TOL of $EXPECTED_BASE_TEST_BLEU — weights, tokenizer or test set do not match the paper. Do NOT run E0.3."
import sys; g=float(sys.argv[1]); sys.exit(0 if abs(g-$EXPECTED_BASE_TEST_BLEU) <= $BLEU_TOL else 1)
PY
  echo "ACCEPTED: test BLEU $got (released $EXPECTED_BASE_TEST_BLEU)"
}

stage_data() {
  cd "$MT"
  [ -s data_enfr_v2_raw/train.en ] || python scripts/download_wmt_enfr.py --output-dir data_enfr_v2_raw
  mkdir -p data_enfr_v2
  log "clean full stream (v2)"
  python scripts/clean_data_enfr.py --src data_enfr_v2_raw/train.en --tgt data_enfr_v2_raw/train.fr \
      --out-src data_enfr_v2/train.clean.en --out-tgt data_enfr_v2/train.clean.fr
  rows=$(wc -l < data_enfr_v2/train.clean.en)
  if [ "$rows" -ne "$EXPECTED_V2_ROWS" ]; then
    die "v2 clean corpus has $rows rows, paper says $EXPECTED_V2_ROWS. The pipeline has drifted \
(HF stream contents, datasets version, or cleaning defaults). Record the difference before using it."
  fi
  echo "v2 rows = $rows (matches paper)"
}

stage_score() {
  cd "$MT"
  hf auth whoami >/dev/null 2>&1 || huggingface-cli whoami >/dev/null 2>&1 || die \
"not logged in to Hugging Face. Unbabel/wmt22-cometkiwi-da is gated (auto-approve, CC-BY-NC-SA-4.0):
  1. accept the terms at https://huggingface.co/Unbabel/wmt22-cometkiwi-da
  2. run 'hf auth login' YOURSELF on this box
This script never handles your token."
  ngpu=$(nvidia-smi -L | wc -l)
  log "scoring 30M pairs on $ngpu GPU(s) (paper: 13.8 h on one RTX 5090); resumable"
  python "$SFT/scripts/score_with_comet.py" --src data_enfr_v2/train.clean.en --tgt data_enfr_v2/train.clean.fr \
      --out data_enfr_v2/v2_scored.tsv --gpus "$ngpu" --resume
}

statmt_url() {
  # URLs checked (HTTP 200) on 2026-09-12.
  case "$1" in
    europarl)        echo https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz ;;
    commoncrawl)     echo https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz ;;
    un)              echo https://www.statmt.org/wmt13/training-parallel-un.tgz ;;
    news-commentary) echo https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz ;;
    giga-fren)       echo https://www.statmt.org/wmt10/training-giga-fren.tar ;;
    *) die "unknown corpus $1" ;;
  esac
}

stage_provenance() {
  # Plain bash 3.2 on purpose (no associative arrays, no bare empty-array
  # expansion under set -u), so this stage can be exercised offline on a Mac.
  cd "$MT"
  [ -s data_enfr_v2/train.clean.en ] || die "run 'data' first"
  local R="$WORK/statmt" lab d f pat en fr n_en n_fr
  mkdir -p "$R"
  local args=()
  # This ORDER defines the integer codes in provenance_labels.npy;
  # e03_build_controls reads it back from the report rather than trusting a retype.
  for lab in europarl commoncrawl un news-commentary giga-fren; do
    d="$R/$lab"; mkdir -p "$d"
    f="$d/$(basename "$(statmt_url "$lab")")"
    if [ ! -s "$f" ]; then
      [ "${STATMT_OFFLINE:-0}" = 1 ] && die "$lab: $f missing and STATMT_OFFLINE=1"
      curl -fL --retry 3 -o "$f" "$(statmt_url "$lab")"
    fi
    if [ ! -f "$d/.extracted" ]; then
      tar -xf "$f" -C "$d"
      # giga-fren ships gzipped text inside the tar
      find "$d" -type f -name '*.gz' -exec gunzip -f {} \;
      touch "$d/.extracted"
    fi
    if [ "$lab" = giga-fren ]; then pat='*giga-fren*'; else pat='*fr-en*'; fi
    # Inner file names are DISCOVERED, not assumed: exactly one .en and one .fr
    # with equal line counts, or stop.
    en=$(find "$d" -type f -name "$pat" -name '*.en' | sort)
    fr=$(find "$d" -type f -name "$pat" -name '*.fr' | sort)
    n_en=$(printf '%s' "$en" | grep -c . || true)
    n_fr=$(printf '%s' "$fr" | grep -c . || true)
    if [ "$n_en" -ne 1 ] || [ "$n_fr" -ne 1 ]; then
      die "$lab: expected exactly one .en and one .fr under $d, found $n_en/$n_fr:
$en
$fr"
    fi
    [ "$(wc -l < "$en")" -eq "$(wc -l < "$fr")" ] || die "$lab: $en and $fr differ in line count"
    echo "  $lab: $(wc -l < "$en" | tr -d ' ') pairs  ($en)"
    args+=(--corpus "$lab:$en:$fr")
  done
  local qe=()
  [ -s data_enfr_v2/v2_scored.tsv ] && qe=(--qe-scores data_enfr_v2/v2_scored.tsv)
  python "$SFT/phase0/e01_provenance.py" \
      --clean-src data_enfr_v2/train.clean.en --clean-tgt data_enfr_v2/train.clean.fr \
      "${args[@]}" ${qe[@]+"${qe[@]}"} --norm exact --out "$SFT/phase0/provenance"
}

stage_controls() {
  cd "$MT"
  [ -s "$SFT/phase0/provenance_labels.npy" ] || die "run 'provenance' first. \
Without it there are no held-out domain sets and e03_decide cannot return GO."
  [ -s data_enfr_v2/v2_scored.tsv ] || die "run 'score' first"
  python "$SFT/phase0/e03_build_controls.py" --qe-scores data_enfr_v2/v2_scored.tsv \
      --provenance "$SFT/phase0/provenance_labels.npy" --out-dir "$SFT/data/phase0"
  python "$SFT/phase0/e03_run_matrix.py" --base-config "$SFT/configs/sft_base_enfr.yaml" \
      --data-dir "$SFT/data/phase0" --out-dir "$SFT/configs/phase0" \
      --ckpt ckpt_hf/enfr_base_v1.1_averaged.pt
  echo "next: bash $0 stage1"
}

stage_runs() {
  cd "$MT"
  local which="$1" lr="${2:-}" ngpu
  ngpu=$(nvidia-smi -L | wc -l | tr -d ' ')
  if [ "$which" = stage2 ]; then
    [ -n "$lr" ] || die "usage: $0 stage2 <lr_scale chosen from stage1>"
    python "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0/run_stage2.sh" \
        --lr-scale "$lr" --gpus "$ngpu" --cwd "$MT" --logdir "$SFT/logs/phase0/stage2"
    echo "next: bash $0 gate $lr"
  else
    python "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0/run_stage1.sh" \
        --gpus "$ngpu" --cwd "$MT" --logdir "$SFT/logs/phase0/stage1"
    echo
    echo "Choose the lr_scale by the PRE-REGISTERED rule: the largest value whose newstest2013"
    echo "BLEU does not decline monotonically from the first eval. Per-eval values:"
    grep -H "Valid BLEU" "$SFT"/logs/phase0/stage1/*.log || true
    echo "then: bash $0 stage2 <lr_scale>"
  fi
}

stage_gate() {
  cd "$MT"
  local lr="${1:-}"
  [ -n "$lr" ] || die "usage: $0 gate <lr_scale used for stage2>"
  python "$SFT/phase0/e03_collect.py" --mt-root "$MT" \
      --runner "$SFT/configs/phase0/run_stage2.sh" --lr-scale "$lr" \
      --baseline-ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
      --baseline-config "$SFT/configs/sft_base_enfr.yaml" \
      --controls-dir "$SFT/data/phase0" --out "$SFT/results/phase0_bleu.tsv" \
      --jobs "$(nvidia-smi -L | wc -l | tr -d ' ')" || die "collection incomplete — see above; do not apply the gate to partial results"
  # exit 0 = GO, 1 = NO-GO, 2 = cannot decide. The verdict is printed either way.
  python "$SFT/phase0/e03_decide.py" --results "$SFT/results/phase0_bleu.tsv"
}

case "${1:-}" in
  env) stage_env ;; accept) stage_accept ;; data) stage_data ;;
  score) stage_score ;; provenance) stage_provenance ;; controls) stage_controls ;;
  stage1) stage_runs stage1 ;; stage2) stage_runs stage2 "${2:-}" ;; gate) stage_gate "${2:-}" ;;
  *) sed -n '/^# USAGE-BEGIN/,/^# USAGE-END/p' "$0" | sed '1d;$d'; exit 2 ;;
esac
