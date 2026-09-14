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
git clone -b wmt2027-phase0 https://github.com/Euswbnix/Machine-Translation-SFT.git ~/mt/Machine-Translation-SFT
```

Keep the directory name `Machine-Translation-SFT`: the SFT configs hard-code
`../Machine-Translation-SFT/...` paths even though the GitHub repo was renamed.

| stage | what | needs / stops when |
|---|---|---|
| `env` | clone training repo; install deps with **pinned** scoring stack (unbabel-comet 2.2.7, pytorch-lightning 2.5.5, transformers 4.57.1, numpy 1.26.4, plus pyarrow); apply trainer patch; run `tests/regress.py` | Python must be 3.10-3.12 (numpy 1.26.4 wheels); stops if a pin did not take or pip replaced the image's CUDA torch |
| `accept` | newstest2013/2014 from the dev/test parquet pinned at `wmt/wmt14@b199e406` (no train dump); Base v1.1 from HF pinned at `8a58dcc4` with weights/spm sha256 checked; reproduce test BLEU 35.31 ± 0.15; record pre-FT newstest2013 BLEU | **stops** unless valid/test sha256 equal the paper's files and BLEU reproduces — do not run E0.3 |
| `data` | pinned train parquet → CR-safe v2 corpus AND the v1.1 pretraining corpus (first 10M HF rows), both via `rebuild_corpus.py --mode fixed` | writes `.sha_verified` only after both sides' on-disk sha256 equal the Mac rebuild (v2 38,275,284 rows `c8cc665c…`/`e4f5301a…`; v1.1 `7230adc6…`/`b0a6ba91…`); needs 30 GB free |
| `score` | by `score_mode` in `phase0/e03_decisions.json`: **none** = reused scores + literal `nan` (no GPU); **reuse** = score the 21,628,292 new pairs with `score_sharded.py` (one process per GPU), merge; **full** = score all 38,275,284 pairs, then `rescore_plan.py calibrate` against the reused scores | decisions file; bundle v2 `SHA256SUMS`; `.sha_verified`; HF login for reuse/full; 20 GB (none) / 50 GB (reuse, full) free; **full stops if calibration fails** |
| `provenance` | bundle labels (exact match 99.40%, SHA256SUMS-verified), else statmt constituents → `e01` | `.sha_verified`; label rows = corpus rows |
| `controls` | E0.3 sets from the frozen decisions (`--pool-mask`, `--heldout-domains`, `--n-ft`, `--pretrain-*`), then the run matrix (`keep_last` from `ft_checkpoint`, `--budget`, `--spike-ratio`); log tee'd to `logs/phase0/`; manifest, matrix and decisions sha copied to `results/phase0/` | scored rows = corpus rows = label rows (= pool-mask rows); ≥ 16 GiB MemAvailable; every held-out set the decisions name must exist and be non-empty |
| `stage1` | serial tokenisation-cache warm-up, 4-run LR sweep (one GPU per run, resumable), then `e03_select_lr.py` applies the frozen `lr_selection_rule` → `results/phase0/lr_selection.json` | decisions unchanged since `controls`; stops if no rung passes |
| `stage2 <lr>` | cache warm-up, 3 conditions × 3 seeds | refuses an `lr` other than the selected one (`LR_OVERRIDE=1` records a deviation); refuses runs trained on other control sets |
| `gate <lr>` | `e03_collect --ft-ckpt <decision>` evaluates every cell, then `e03_decide --indomain <decision>` | refuses a partial or stale collection |

```bash
PIN_SFT_REV=<pushed sha> bash ~/mt/Machine-Translation-SFT/phase0/rental_setup.sh env
```

**Things only you can do** (they involve your accounts; the scripts never touch
credentials):

0. **Make the E0.3 decisions before renting (first).** Copy `phase0/e03_decisions.example.json`
   to `phase0/e03_decisions.json`, replace every `"CHOOSE"` (13 entries; options and
   measured consequences in `PROTOCOL.md`, "E0.3 decisions required before controls";
   a recommendation for each is in the decisions brief), check it with
   `python phase0/e03_decisions.py check phase0/e03_decisions.json`, then **commit, tag and
   push both** (`git push origin wmt2027-phase0 <tag>`). Do this BEFORE step 1: the pin taken
   there must contain this commit. `score`, `controls`, `stage1`,
   `stage2` and `gate` refuse a decisions file that is untracked, differs from HEAD, has no
   tag containing its commit, or whose commit is on no remote branch
   (`DECISIONS_UNTAGGED=1` proceeds and writes a deviation). Which edits invalidate what:
   - `score_mode` or `calibration` changed after `score` completed: `score` stops and deletes
     nothing; `SCORE_REDO=1` rescores (a deviation). Any other key (n_ft, held-out domains,
     LR rule, ft_checkpoint, a `_comment`) does **not** invalidate the scores; it only needs
     freezing before `controls`.
   - any key changed after E0.3 results exist (`lr_selection.json`, stage logs, a
     `final.pt`, a BLEU TSV or cache): `score` and `controls` stop; `DECISIONS_AMEND=1`
     proceeds and records the old/new sha and changed keys in `results/phase0/deviations.txt`.
     Every `controls` appends to the append-only `results/phase0/decisions_history.tsv` and
     keeps `decisions.<sha12>.json`; `gate` prints both files before the verdict.
1. **Push the reviewed code, then take the pin (after step 0).** The box gets code only through `git clone`
   + checkout in `rental_setup.sh repos`/`env`. Run `python3 tests/regress.py` (must end
   `FAIL=0`), commit everything under `phase0/`, `scripts/`, `tests/` and `configs/` on
   `wmt2027-phase0`, then `git push origin wmt2027-phase0` and write down `git rev-parse HEAD`. That sha must be the decisions commit from step 0 or a
   later one: `git ls-tree --name-only <sha> phase0/e03_decisions.json` must print the path.
   `repos`/`env` warn when the pinned commit lacks the decisions file that origin has ("PIN_SFT_REV
   predates the decisions commit"): re-pin and rerun `repos` before `score`.
   Check `git fetch && git status -sb` shows no ahead/behind and
   `git ls-remote origin refs/heads/wmt2027-phase0` prints that sha. On the box, run
   `PIN_SFT_REV=<that sha> bash rental_setup.sh env`: it checks out exactly that commit and
   checks out Machine_translation at the pinned `MT_REV` (200f6c06…, the commit the trainer
   patch was checked against; changing `MT_REV` is a deliberate, logged change). If a
   checkout moves the running script, `env` stops and asks you to rerun it.
2. Rent the box and give access. Hardware:
   - **GPUs:** 4× RTX 5090 is fine for fine-tuning (one run per GPU). Scoring now shards one
     process per GPU; multi-GPU scoring throughput is **not measured**. Single-card
     estimates (derived from the paper's 13.8 h / 30.1M pairs, not measured on the pinned
     stack): `reuse` ~10 GPU-h, `full` ~17.5 GPU-h. `SCORE_SMOKE=1 bash rental_setup.sh
     score` first scores 3,000 rows on GPUs 0 and 1 and checks row order against a
     single-GPU run before the long job.
   - **RAM:** ≥ 32 GB of the offer's own allocation (vast.ai `cpu_ram`): inside the container
     `/proc/meminfo` shows the host. `controls` refuses below 16 GiB of headroom, taking the
     smaller of MemAvailable and the container cgroup limit minus usage (inactive file cache
     excluded; cgroup v2 `memory.max`/`memory.high` or v1 `memory.limit_in_bytes`). The only measurement is
     613,883,904 B peak RSS on a 2,014,489-row sample; ~11.7 GB at 38.3M rows is a linear
     extrapolation, not measured.
   - **Disk: ≥ 200 GB.** Measured: train parquet 7.8 GB, v2 corpus 13.8 GB, v1.1 corpus
     3.1 GB, to_score.* 7.9 GB. Derived: scoring peak ~46 GB (`reuse`, before its cleanup) or
     ~42 GB (`full`); merged scored TSV ~14.2 GB; checkpoints ~0.7 GB each (not measured) ×
     ~4 kept per run (final.pt, best.pt, emergency.pt, keep_last step files) × 13 runs ≈ 36 GB
     (`avg-last5`: 7 per stage-1 run, 8 per stage-2 run because `gate` writes avg_last5.pt,
     which keeps the optimizer state; ≈ 100 × 0.7 ≈ 70 GB, derived). Each stage checks `df`
     against its share (`stage2` under `avg-last5`: 55 GB; `gate` under `avg-last5`: 8 GB).
   - **Sessions:** run `score`, `stage1`, `stage2` and `gate` inside `tmux`. Do not kill only a
     parent process: `score_sharded.py` forwards SIGTERM/SIGINT/SIGHUP to its scorers, and a
     second `score` while one is running refuses on the work-dir lock instead of appending
     duplicate rows.
3. Copy the bundle from the Mac to `$WORK/rental_bundle_v2.tar.gz`:
   `~/mt_local/rebuild/rental_bundle_v2.tar.gz`, 86,607,565 B, sha256 `49cd6222…df6bdf6`, 7
   members at the archive root: `plan.json`, `missing_rows.npy`, `reuse_scores.npy`,
   `provenance_labels.npy`, `provenance_report.json`, `pool_mask_reused.npy` and `SHA256SUMS`.
   The script unpacks it into `$WORK/rescore/` and verifies every member against `SHA256SUMS`.
   Claude can do the copy once it has ssh access. (The older 5-member
   `rental_bundle.tar.gz` has no `SHA256SUMS` or pool mask and is refused.)
   `rental_setup.sh` also pins the sha256 of that `SHA256SUMS`
   (`6bbd4724…8ba501d4`) and unpacks again whenever the tarball's sha256 changes, so a stale
   unpack or a self-consistent but different bundle is refused. A future bundle must update
   `EXPECTED_BUNDLE_SUMS_SHA`. Every train/dev/test parquet is checked by size **and** LFS
   sha256 (`phase0/hf_wmt14_filelist.tsv`, third column); a wrong file is downloaded again
   once, fresh, then `data` stops naming it.
4. Only for `score_mode` `reuse` or `full`: accept the terms of the gated
   [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
   (auto-approved, CC-BY-NC-SA-4.0), then on the box run `hf auth login` yourself.
   The login check reads `whoami`'s output as well as its exit code (huggingface_hub < 1.0
   prints "Not logged in" and exits 0).
4b. Only for `score_mode` `full`, if calibration does not pass: `score` keeps the scores as
   `data_enfr_v2/v2_scored.calibration_failed.tsv` with a sidecar `.json` (TSV sha256,
   corpus sha, decisions sha, thresholds, scorer meta, report). **Whether thresholds may be
   changed after seeing that result is your decision** (PROTOCOL.md D7): either rescore after
   fixing the stack (plain `score`; ~17.5 GPU-h derived), or change the thresholds in the
   decisions (commit/tag/push) and run `RECALIBRATE=1 bash rental_setup.sh score`, which
   re-runs only `calibrate` on the kept TSV after checking its sha256 against the sidecar,
   refuses unchanged thresholds, and records the change in `results/phase0/deviations.txt`.
   A calibrate exit 3 (unreadable input) never discards anything: fix the input and rerun
   `score`; the verified shards are reused.
4c. **Scoring-stack changes and failed GPUs during `score`.** Each shard's meta records the
   stack (package versions, CUDA, GPU names, checkpoint path, and the pinned hub revisions of
   Unbabel/wmt22-cometkiwi-da `1ad78519…` and microsoft/infoxlm-large `d616d637…`). If a resumed
   or restarted shard finds a different stack, its scorer refuses (exit 3) and `score` stops
   with "refused to continue under a changed scoring stack"; a plain rerun fails the same way.
   Your options: (1) restore the stack the meta records; (2) delete that shard's
   `shard.NNN.tsv` and `shard.NNN.meta.json` under `…/new_scores.tsv.shards` (reuse) or
   `data_enfr_v2/v2_scored.partial.tsv.shards` (full) to rescore it under the new stack;
   (3) `SCORE_ALLOW_STACK_CHANGE=1 bash rental_setup.sh score` (writes a deviation; default is to
   refuse). After scoring, if shards differ from EACH OTHER (`stack_mixed` in the merged meta),
   `reuse` warns and `full` stops before calibration, keeping scores and shards (full has no
   override: delete the shards of the unwanted stack and rerun). A scorer that crashes for another
   reason is reported as soon as it exits ("shard k FAILED"), its GPU leaves the pool for that run,
   and the other shards continue; rerunning `score` resumes the unfinished shards on the GPUs now
   visible (a smaller GPU count is accepted; output is byte-identical).
5. After `stage1`: the script selects the lr_scale itself, by the rule frozen in the
   decisions file, and prints every per-eval validation BLEU with the verdict per rung.
   If no rung passes it stops; that is a result to report, not a prompt to pick one.
6. After `gate`: read the verdict, and the decisions history and deviations it prints
   first. `gate <lr>` refuses an lr other than the stage-1 selection unless `stage2` recorded
   an `LR_OVERRIDE` deviation for it. `results/phase0/phase0_bleu.tsv` and its `.meta.json`
   always belong to the most recent **complete** collection (both are deleted when a
   collection starts); the meta names `lr_scale`, `selected_lr` and `decisions_sha`.
   `e03_decide.py` exit codes are listed in section 5 below.

Honest status: `repos`, `accept` and every stage from `data` onward are exercised offline
by `tests/suites/rental.py`, which drives the real `rental_setup.sh` with fake GPUs,
downloads, scorer, builder, trainer and evaluator (plus `tests/regress.py` for the
individual scripts); `env`'s Python-version, pin and CUDA guards are exercised offline with a
fake python, while its real install steps are checked only by its own on-box self-test. On a
rented RTX 5090 (2026-09-13), the **previous** `env` and `accept` ran (test BLEU 35.31,
exact) and the patched trainer passed a smoke run; the pinned `env` and the pinned dev/test
download in `accept` have **not** run on a box. The corpus rebuild, the rescore plan and
e01 ran on the Mac. `data`, `score`, `controls` and real training have **not yet run on a
GPU box**.

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

> **Superseded (2026-09-12/13):** the HF releases are tensor-identical to `averaged.pt`
> (README "Findings from the original training machine" → Settled 1 and 4), and
> `phase0/hf_to_ckpt.py` restores `model` and `global_step`. E0.3 runs from HF weights
> through `rental_setup.sh`. The paragraph below is kept for history only.

(Historical.) The public HuggingFace releases (`euswbnix/transformer-wmt14-{enfr,ende}-{base,big}`)
are **not a substitute** for E0.3. `scripts/prepare_hf_release.py:256-265` saves
only the bare `state_dict`, while `trainer.load_checkpoint` reads `ckpt["model"]`
and `ckpt["global_step"]` — and `global_step` is what places the scheduler on its
decay curve at resume. (Historical: use the original `averaged.pt`.)

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
| P1 | `*scored*.tsv` (≈14 GPU-hours for 30.1M pairs, derived; see `PROTOCOL.md` D7) | **confirm it exists**; pull if space allows |
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

> **Superseded (2026-09-13): use `bash rental_setup.sh provenance`.** The paths below
> (`data/v2_clean.*`, `data/v2_scored.tsv`) are the paper's 30,129,500-row corpus, which is
> ~45% misaligned. The corrected corpus is `data_enfr_v2/train.clean.*`, and bundle v2
> already carries its labels. The report's `sources` order is authoritative; do not retype
> it as `--sources`. Kept for reference only.

**This gates E0.3 criterion 2.** Without provenance labels there are no in-domain
held-out sets: `e03_build_controls.py` now exits 1 instead of building FT sets without
them, and `e03_decide.py` exits 2 (cannot decide) when an `--indomain` set has no rows.
Do this before spending any GPU time on E0.3.

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

> **Superseded (2026-09-13): use `bash rental_setup.sh controls`, `stage1`, `stage2 <lr>`,
> `gate <lr>`.** The commands below use the misaligned 30.1M corpus paths, a hand-typed
> `--sources`, selection of the lr_scale by eye and a hand-assembled TSV. All of that is
> now replaced: the frozen decisions file drives the builder and matrix,
> `e03_select_lr.py` selects the rate, and `e03_collect.py` assembles the TSV.
> Kept for reference only; the exit codes at the end are current.

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

(Superseded: the lr_scale is selected by `phase0/e03_select_lr.py` under the rule
frozen in `phase0/e03_decisions.json`, see `PROTOCOL.md` D6.) Then:

```bash
bash configs/phase0/run_stage2.sh 0.15       # 9 runs, 3 conditions x 3 seeds
```

Evaluate every run on `newstest2014` and every held-out set in `manifest.json` with
`phase0/e03_collect.py` (do not assemble the TSV by hand), then:

```bash
python ~/Machine-Translation-SFT/phase0/e03_decide.py --results results/phase0/phase0_bleu.tsv --indomain <decision>
echo "exit=$?"
```

`e03_decide.py` exit codes, as the code behaves:

- **0 = GO:** both pre-registered criteria pass.
- **1 = NO-GO:** the rule was applied to complete data and at least one criterion
  failed. Nothing else exits 1.
- **2 = cannot decide:** any of
  - `baseline`, `ft_topk` or `ft_random` rows are absent, or there is no baseline row
    on newstest2014;
  - `--indomain` is empty, or a set it names has no baseline or no ft_topk rows;
  - an ft_topk cell (newstest2014 or an `--indomain` set) or the ft_random newstest2014
    cell does not have exactly `--expect-seeds` values (default 3);
  - an unexpected error.

It also prints a warning when the fine-tuned and baseline checkpoints are of different
kinds (`<results>.meta.json` from `e03_collect.py`), and repeats the collector's budget
warnings. `e03_collect.py` itself exits 1 on an incomplete collection (only
`.partial.tsv` is written) and 2 when the runs or control sets are stale.

---

## The thing to hold on to

The gate can come back NO-GO. That is a real outcome, not a failure of execution:
it would mean the data-quality/domain angle does not carry a WMT 2027 submission
and the program rests on the compute-alignment work instead. `e03_decide.py` was
written before the runs specifically so that outcome cannot be argued away
afterwards. Do not reframe it post-hoc — that is the failure mode that produced
the rejected paper.
