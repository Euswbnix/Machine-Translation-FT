# Machine-Translation-SFT

SFT (Supervised Fine-Tuning) on top of the from-scratch Transformer
checkpoints from
[`Machine_translation`](https://github.com/Euswbnix/Machine_translation).

This is **Section 7** of the upcoming paper. It tests whether the noisy
30M v2 corpus can be made useful via quality filtering + short SFT,
closing the data-quality narrative: pretrain on scale, SFT on quality.

## Pipeline

```
Big v1.1 / Base v1.1 (pretrained on 9.3M strict v1)
        │
        ▼
score 30M v2 corpus with CometKiwi-22 (reference-free QE)
        │
        ▼
filter to top-1M (top 3.3% by quality score)
        │
        ▼
SFT 10K steps, low LR (peak 1e-4), fresh optimizer
        │
        ▼
eval on newstest2014 (sacrebleu + COMET-22)
```

## Hypothesis

- **Base v1.1 + SFT**: +0.5–1.0 BLEU over 35.31 baseline
- **Big v1.1 + SFT**: +0.3–0.8 BLEU over 35.87 baseline (smaller gain — capacity already saturating)

The Base/Big gap **after** SFT is itself an interesting data point for the paper's "capacity return" thesis.

## Layout

```
Machine-Translation-SFT/
├── configs/
│   ├── sft_base_enfr.yaml
│   └── sft_big_enfr.yaml
├── scripts/
│   ├── score_with_comet.py    # CometKiwi-22 scoring of v2 corpus
│   └── filter_by_score.py     # Take top-K by quality
├── data/                      # Filtered SFT data (gitignored)
└── checkpoints/               # SFT'd ckpts (gitignored)
```

No new training code — SFT reuses the main repo's `train.py` with
`--resume <averaged.pt> --reset-optimizer`. The Noam scheduler resumes
at late-decay (~1.5e-4 effective LR), which is in the standard SFT band.

## Setup

Place this repo as a sibling of `Machine_translation/` (configs use
`../Machine-Translation-SFT/...` relative paths). Then:

```bash
pip install -r requirements.txt
```

## Usage

All commands run from the **main repo root** (`Machine_translation/`):

```bash
# 1. Score 30M v2 corpus with CometKiwi-22 (~1-2h on 5090)
python ../Machine-Translation-SFT/scripts/score_with_comet.py \
    --src data/train.clean.en \
    --tgt data/train.clean.fr \
    --out ../Machine-Translation-SFT/data/v2_scored.tsv \
    --batch-size 64

# 2. Filter to top 1M
python ../Machine-Translation-SFT/scripts/filter_by_score.py \
    --in ../Machine-Translation-SFT/data/v2_scored.tsv \
    --src-out ../Machine-Translation-SFT/data/sft_train.en \
    --tgt-out ../Machine-Translation-SFT/data/sft_train.fr \
    --top-k 1000000

# 3. SFT — Base v1.1 (10K steps, ~25 min on 5090)
python train.py \
    --config ../Machine-Translation-SFT/configs/sft_base_enfr.yaml \
    --resume checkpoints_enfr/averaged.pt \
    --reset-optimizer

# 4. SFT — Big v1.1 (10K steps, ~1h on 5090)
python train.py \
    --config ../Machine-Translation-SFT/configs/sft_big_enfr.yaml \
    --resume checkpoints/big_enfr_v1_redo/averaged.pt \
    --reset-optimizer

# 5. Eval (after averaging last 5 SFT ckpts)
python scripts/average_checkpoints.py \
    --ckpt-dir ../Machine-Translation-SFT/checkpoints/sft_base_enfr --n 5 \
    --out ../Machine-Translation-SFT/checkpoints/sft_base_enfr/averaged.pt

python scripts/eval_bleu.py \
    --ckpt ../Machine-Translation-SFT/checkpoints/sft_base_enfr/averaged.pt \
    --config ../Machine-Translation-SFT/configs/sft_base_enfr.yaml \
    --src data_enfr_v1/test.en --ref data_enfr_v1/test.fr
```
