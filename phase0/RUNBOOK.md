# Phase 0 runbook — what to run on the server

Ordered by cost. Step 1 costs seconds and can settle a headline claim on its own;
do not skip ahead to GPU time.

Everything below assumes the training repo at `~/Machine_translation` and this
repo at `~/Machine-Translation-SFT`, adjust paths as needed.

---

## Primary path: run all of Phase 0 on a rented GPU box

**The original training machine is not required.** The paper fine-tuned the
averaged Base v1.1 / Big v1.1 checkpoints, and those are exactly the public HF
releases: `config.json` records `training.steps` 105000 / 210000 and BLEU
30.52/35.31 and 31.14/35.87, matching `paper_wmt_upload/sections/05_sft.tex:26,29`
digit for digit. `phase0/hf_to_ckpt.py` restores the keys the trainer needs. Every
dataset is public (HF `wmt/wmt14`, statmt constituents). Step 0 below — pulling
from the old box — is now optional, useful only for the original training logs
and the paper's exact scored file.

Everything is driven by one staged script, cloned with the rest of this branch:

```bash
git clone -b wmt2027-phase0 https://github.com/Euswbnix/Machine-Translation-FT.git ~/mt/Machine-Translation-SFT
```

Keep the directory name `Machine-Translation-SFT`: the SFT configs hard-code
`../Machine-Translation-SFT/...` paths even though the GitHub repo was renamed.

| stage | what | needs |
|---|---|---|
| `env` | clone training repo, install deps, apply trainer patch, run `tests/regress.py` | refuses to continue if pip replaced the image's CUDA torch |
| `accept` | rebuild Base v1.1 from HF, reproduce test BLEU 35.31 ± 0.15 | **stops** if it does not reproduce — do not run E0.3 |
| `data` | WMT14 fr-en → v2 cleaned corpus | **stops** unless it has exactly 30,129,500 rows, as in the paper |
| `score` | CometKiwi-22 over 30M pairs (paper: 13.8 h on one 5090) | your HF login (below) |
| `provenance` | statmt constituents → `e01` hash-join | archive contents discovered, never assumed |
| `controls` | E0.3 sets (dedup-matched, 931K-scale) + run matrix | |
| `stage1` | 4-run LR sweep, one GPU per run, resumable | |
| `stage2 <lr>` | 3 conditions × 3 seeds at the chosen `lr_scale` | |
| `gate <lr>` | `e03_collect` evaluates all 30 cells, then `e03_decide` | refuses to judge a partial collection |

```bash
bash ~/mt/Machine-Translation-SFT/phase0/rental_setup.sh env
```

**Things only you can do** (they involve your accounts; the scripts never touch
credentials):

1. Rent the box and give access. Suggested: **4× RTX 5090, ≥ 150 GB disk**. Scoring
   parallelises across GPUs, and `run_parallel.py` runs one fine-tuning run per GPU.
2. Before `score`: accept the terms of the gated
   [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
   (auto-approved, CC-BY-NC-SA-4.0), then on the box run `hf auth login` yourself.
3. After `stage1`: the script prints every per-eval validation BLEU. The lr_scale
   is chosen by the pre-registered rule — the largest value whose BLEU does not
   decline monotonically from the first eval — not by preference.

Honest status: the offline parts of this path (`provenance`, run parallelism,
collection, the gate, the patch-state logic) are exercised by `tests/regress.py`
against fixtures. `env`, `accept`, `data`, `score` and real training have **not
yet run on a GPU box**. Expect the first run to surface environment issues.

## 0. (Optional) Inventory the training machine, then pull what is irreplaceable (no GPU)

> **Done 2026-09-12.** Everything Phase 0 needs was pulled read-only and verified
> (sizes, line counts, sha256) into `~/mt_fetch/` on the Mac, mirroring the box's
> absolute paths: the paper's `v2_scored.tsv` (30,129,500 rows, sha256 match), the
> v2 / v1.1 / en-de cleaned corpora (30,129,500 / 9,312,233 / 4,174,104 rows), all
> tokenizer caches and SPM models, dev/test sets, `results/`, training reports, SFT
> logs, tensorboard events, swanlog, and every checkpoint's `history` + `config`
> (`~/mt_fetch/extracted/mt_histories.json.gz`). Not pulled, deliberately: checkpoint
> weights (~118 GB) — only needed to re-decode old runs. Findings are in
> `phase0/README.md`, "Findings from the original training machine". The sections
> below are kept for reference.

**Nothing needed for Phase 0 is on the Mac** (checked 2026-09-12): no data
directories, no checkpoints, no tokenizer caches, no QE scores, no training logs.
They live on the Linux side of the training machine.

The public HuggingFace releases (`euswbnix/transformer-wmt14-{enfr,ende}-{base,big}`)
are **not a substitute** for E0.3. `scripts/prepare_hf_release.py:256-265` saves
only the bare `state_dict`, while `trainer.load_checkpoint` reads `ckpt["model"]`
and `ckpt["global_step"]` — and `global_step` is what places the scheduler on its
decay curve at resume. Use the original `averaged.pt`.

Authentication is key-only; `fetch.py` runs every ssh/rsync with `BatchMode=yes`
and will fail rather than prompt for a password.

```bash
ssh-copy-id <host>
```

```bash
python3 phase0/fetch.py inventory --host <host> --inspect-ckpt
```

This streams `phase0/inventory.py` over ssh (nothing is copied to the box first),
lists every relevant file by priority, reads `global_step` / `accumulate_steps` /
`lr_scale` out of each averaged/best checkpoint, greps every log for the trainer's
own `LR = …` / `Resumed from step` lines, and reports whether the code on the box
matches GitHub (dirty files, unpushed commits). **Read it before pulling.**

```bash
python3 phase0/fetch.py pull --host <host> --prio P0
```

That is a dry run. It prints the plan and, separately, everything it will NOT pull.
Then add `--go`.

| priority | what | pull to the Mac? |
|---|---|---|
| P0 | training logs/reports, averaged/best checkpoints, eval traces | **yes** — irreplaceable |
| P1 | `*scored*.tsv` (≈12 GPU-hours to rebuild, per `score_with_comet.py:6`) | **confirm it exists**; pull if space allows |
| P1 | `.cached_*.npz`, SPM models, `sft_train.*`, the 30M cleaned corpus | leave on the box; run e01/e02 **there** |
| P2 | rotating `step_*.pt`, dev/test sets, other logs | no |

If the inventory shows the code on the box has uncommitted changes or unpushed
commits, those are what actually ran — capture them before anything else.

## 1. Confirm the fine-tuning LR from the log (seconds, no GPU)

> **Answered 2026-09-12, from the checkpoints rather than the log.** The paper's
> Base FT stdout was not captured (`logs/sft_base.log` is a later v1.0 re-run), but
> `sft_base_enfr/final.pt` records its own LR: 2.728e-4 at step 105,000 continuing to
> 2.653e-4 at 111,000; Big 2.046e-4 → 2.017e-4 (and `sft_big.log` prints
> `LR = 2.05e-04`). LR was continuous, as derived.

`trainer.py:580-581` prints the true LR at checkpoint load:

```bash
grep -rn "Optimizer/scheduler RESET" ~/Machine_translation/ ~/Machine-Translation-SFT/ 2>/dev/null
```

| printed value | meaning |
|---|---|
| `2.73e-04` | as derived. Note this is **not** an anomaly — it equals the LR pretraining ended at (see below) |
| `1.36e-04` | the `// 4` reconstruction was not applied; tell me, the analysis changes |
| `6.99e-04` | the scheduler restarted at warmup peak — that WOULD be the catastrophic case |
| no match | stdout was not captured; skip to step 2 |

**Read this before interpreting the result.** An earlier version of this runbook
said `2.73e-04` would "confirm" that fine-tuning ran at an anomalously high
restart LR. That was wrong and is retracted. `scheduler.step()` is called only at
accumulation boundaries (trainer.py:461,472), so pretraining's scheduler had
already reached ~26250 and 2.73e-4 is the **continuation** of the LR it was
already training at — a 0.5% change, not a 2x jump. What is genuinely wrong is
only the comment in `configs/sft_base_enfr.yaml:3-6` (the Big config's equivalent
comment is correct). Details in `phase0/README.md`, section "CORRECTED: the LR
framing was wrong".

Worth grepping regardless, since the real boundary discontinuity is the discarded
Adam state and the eval cadence:

```bash
grep -rn "Resumed from step" ~/Machine_translation/ 2>/dev/null | tail -20
```

## 2. Apply the cumulative token-accounting patch (minutes, no GPU)

Makes the compute budget a MEASURED quantity instead of one reconstructed from
`steps x batch_size`, which is what produced the 4x error. The patch now does four
things (16 hunks):

1. **Two** cumulative counters — `total_train_tokens` (processed, the MFU
   denominator) and `applied_target_tokens` (gradient actually applied, the
   learning-curve x-axis) — plus a real `optimizer_steps` count, since
   `global_step // accumulate_steps` overcounts whenever the spike guard drops a
   batch. All persist in checkpoints.
2. A **token-keyed training gate**. Set `max_target_tokens` and `max_steps` is no
   longer consulted: it counts micro-batches, so equal `max_steps` hands arms with
   different `accumulate_steps` different token budgets. A deliberate
   `max_micro_steps_backstop` can still stop a runaway, loudly.
3. A **token-keyed eval cadence** (`eval_every_tokens`), bypassing
   `_update_eval_interval`, whose interval is a function of training loss and
   therefore of capacity.
4. `_evaluate_ce()` — held-out cross-entropy in nats per non-pad target token,
   fp32, teacher-forced, label smoothing off — appended to
   `<ckpt_dir>/dev_ce_trace.tsv`. This is the file `phase1/converge.py` consumes.

All four default to the old behaviour when the new config keys are absent, so
existing runs are unaffected.

```bash
cd ~/Machine_translation
git apply --check -p1 ~/Machine-Translation-SFT/phase0/trainer_token_accounting.patch
```

If that prints nothing, it applies cleanly. Then:

```bash
cd ~/Machine_translation && git checkout -b phase0-token-accounting
git apply -p1 ~/Machine-Translation-SFT/phase0/trainer_token_accounting.patch
git diff --stat
```

Note: resuming from a checkpoint saved BEFORE this patch leaves the counter short
by everything prior. The patch prints a warning and marks the report `UNDERCOUNT`
rather than reporting a wrong number silently. Phase 1 runs start fresh, so they
are unaffected.

## 3. Re-measure the padding factor on the real cache (~10 min, no GPU)

The `/ 4` correction is exact and data-independent. The padding factor is NOT —
it was measured on synthetic lengths (0.68–0.72) and must be re-measured on the
real length arrays before any number goes in a paper.

```bash
cd ~/Machine_translation
python ~/Machine-Translation-SFT/phase0/e02_token_accounting.py \
    --cache data_enfr_v1/train.cached_256.npz \
    --seeds 3 \
    --json-out ~/Machine-Translation-SFT/phase0/e02_enfr.json
```

`--cache` is the `.cached_256.npz` that `TranslationDataset` writes; if it is not
on disk, pass `--src/--tgt/--spm` instead and it will tokenize. `--seeds 3` repeats
the sampler with different shuffles so the padding factor comes with a spread
rather than a single point estimate. Repeat for en-de.

## 4. Provenance hash-join (~1 hour, mostly download, no GPU)

**This gates E0.3 criterion 2.** Without provenance labels there are no in-domain
held-out sets, so the domain claim is untestable and `e03_decide.py` CANNOT return
GO — it will exit 2. Do this before spending any GPU time on E0.3.

Download the constituent corpora separately (Europarl v7, Common Crawl, UN, News
Commentary, Giga-FrEn for en-fr), then:

`--corpus` is repeated once per constituent, in `LABEL:SRC:TGT` form. The label
order defines the integer codes in the `.npy`, so **keep it identical** to the
`--sources` list you later pass to `e03_build_controls.py`:

```bash
cd ~/Machine_translation
python ~/Machine-Translation-SFT/phase0/e01_provenance.py \
    --clean-src data/v2_clean.en --clean-tgt data/v2_clean.fr \
    --corpus europarl:raw/europarl.en:raw/europarl.fr \
    --corpus commoncrawl:raw/commoncrawl.en:raw/commoncrawl.fr \
    --corpus un:raw/un.en:raw/un.fr \
    --corpus news-commentary:raw/nc.en:raw/nc.fr \
    --corpus giga-fren:raw/giga.en:raw/giga.fr \
    --qe-scores data/v2_scored.tsv \
    --norm exact \
    --out ~/Machine-Translation-SFT/phase0/provenance
```

Writes `provenance_labels.npy` (consumed by `e03_build_controls.py --provenance`)
and `provenance_report.json` (the table that replaces the tilde-hedged
`tab:qe_source` in the paper).

Start with `--norm exact`. `clean_data_enfr.py` is a pure filter — it writes the
ORIGINAL line and only strips for length/ratio computation — so exact matching
should already be high. If it is not, escalate to `strip` then `collapse_ws` and
**report which normalizer you needed**: needing a loose one is itself a finding
about the pipeline, not a detail to bury.

Below a 95% match rate the script says the labels are not trustworthy. If it lands
there, STOP and tell me — it would mean the cleaned corpus is not a pure subset of
what we think it is, which is paper-relevant on its own.

## 5. E0.3 controls and the gate (~7 GPU-hours)

Only after 4 succeeds.

```bash
python ~/Machine-Translation-SFT/phase0/e03_build_controls.py \
    --qe-scores data/v2_scored.tsv \
    --provenance phase0/provenance_labels.npy \
    --sources europarl,commoncrawl,un,news-commentary,giga-fren \
    --out-dir data/phase0

python ~/Machine-Translation-SFT/phase0/e03_run_matrix.py \
    --base-config ~/Machine-Translation-SFT/configs/sft_base_enfr.yaml \
    --data-dir data/phase0 --out-dir configs/phase0

bash configs/phase0/run_stage1.sh            # 4 runs, LR sweep on ft_topk
```

Pick the largest `lr_scale` whose newstest BLEU does NOT fall monotonically from
the first eval, then:

```bash
bash configs/phase0/run_stage2.sh 0.15       # 9 runs, 3 conditions x 3 seeds
```

Evaluate **every** run on `newstest2014` AND `heldout_un` AND `heldout_europarl`,
collect into a TSV (`condition seed testset bleu`, plus `baseline - <testset> <bleu>`
rows for the pre-FT checkpoint), then:

```bash
python ~/Machine-Translation-SFT/phase0/e03_decide.py --results results/phase0_bleu.tsv
echo "exit=$?"      # 0 = GO, 1 = NO-GO, 2 = cannot decide (missing inputs)
```

---

## The thing to hold on to

The gate can come back NO-GO. That is a real outcome, not a failure of execution:
it would mean the data-quality/domain angle does not carry a WMT 2027 submission
and the program rests on the compute-alignment work instead. `e03_decide.py` was
written before the runs specifically so that outcome cannot be argued away
afterwards. Do not reframe it post-hoc — that is the failure mode that produced
the rejected paper.
