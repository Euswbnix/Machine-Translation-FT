# Phase 0 — Kill Gate (Month 1)

Decides whether the WMT 2027 program is worth 11 more months. Costs ~1 GPU-day
plus CPU. CPU steps run on the Mac; QE scoring and E0.3 need a rented GPU box (RUNBOOK
primary path, `rental_setup.sh`).

Plan: `../WMT2027_PLAN.md`

---

## Status

| Item | State |
|---|---|
| E0.2 token-accounting audit | ✅ script written; **corrected after audit** — see below |
| E0.1 exact provenance hash-join | ✅ script written + self-tested (100% match on fixtures) |
| E0.3 fine-tuning confound controls | ✅ written + self-tested; 6 fatal defects fixed after audit |
| Adversarial audit of all of the above | ✅ 2026-09-04, 40 agents, 6 lenses, every finding independently refuted. 12 survived, 8 fatal. **Both headline claims were damaged.** |
| Phase 1 alignment protocol | ✅ decided — run-to-overshoot + post-hoc argmin (`PROTOCOL.md`) |

> The audit found a second compute-arithmetic error inside the script written to
> catch the first one, and refuted the escalation of the LR finding. Both are
> corrected below. Treat every number here as provisional until re-measured on
> the real corpus — that failure mode has now materialized twice in this project.

---

## CRITICAL (2026-09-13): the v2 en-fr and en-de training corpora are misaligned by a pipeline bug

**Status: verified.** Found while running E0.1; then attacked by nine independent
refutation agents (one per claim plus an alternative-explanations critic and a
completeness critic). No claim was refuted; the corrections they made are
incorporated below.

### What is wrong

- **v2 en-fr (`data/train.clean.{en,fr}`, 30,129,500 rows, the full-stream corpus).**
  From 1-based line **16,575,777** to the end, English line *i* is paired with a French
  line that is not its translation. Up to about line 16,674,600 (inside
  news-commentary) the offset wanders and occasionally returns to zero; from there to
  the end it is a steady **+28 source lines** against the statmt originals.
  About **13.55M rows (45.0%)** are misaligned.
- **en-de (`data_ende/train.clean.{en,de}`, 4,174,104 rows).** From 1-based line
  **4,060,956** to the end: **113,149 rows (2.71%)**, offset drifting to about 100 lines.
- **v1.1 en-fr capped (`data_enfr_v1`, 9,312,233 rows): aligned.** 99.66% of its rows
  are byte-identical statmt pairs and a full offset scan finds k=0 everywhere. Isolated
  slips are also excluded: its first 10M raw pairs contain no CR, and the CR-safe rebuild
  is byte-identical to the published file (see "Rebuild results" below).

### Why

1. `download_wmt_enfr.py` / `download_wmt_ende.py` write each pair after `strip()` and
   `replace("\n", " ")`, so a carriage return **inside** a line survives into
   `train.en` / `train.fr`. The HF wmt14 fr-en stream has 244 English and 225 French
   pairs with an internal CR; de-en has 327 / 265.
2. `clean_data_enfr.py` / `clean_data_ende.py` read those files with `open()` in default
   text mode. **Universal newlines treat an internal CR as a line break**, so one side
   yields extra lines, `zip()` pairs line *i* with line *i+k*, and every later kept pair
   is shifted. The first event is a news-commentary sentence, "Encouraging weak
   countries … hope of a `\r` de facto`\r` bail-out", with two CRs in the English.
3. The QE scorer and the trainer (`dataset.py` builds `.cached_256.npz` once and reuses
   it) read the already-shifted, now CR-free files, so nothing downstream can see it.
   Cleaning ran per-pair length filters *after* the shift, so the damage cannot be
   repaired by shifting one file back.

### Evidence (independent lines)

| check | aligned region | misaligned region |
|---|---|---|
| exact (src,tgt) match to statmt originals | 97-100% per 1M rows | ~0% |
| CometKiwi-22 median | 0.875 | 0.402 (94.5% < 0.6, 0.0% > 0.85) |
| digit agreement across sides | 92% | 10% (statmt giga-fren's *own* pairs: 60-80%) |
| cache src/tgt length correlation | 0.96 | 0.54-0.66 |
| true translation location | row *i* | row *i*+28 in 286-300 of 300 rows per window |
| CR offset simulated from local statmt NC files | — | +28 fr-en, +147 de-en; matches the HF-stream scan exactly |

Fingerprints: orphan fragments with a leading space sit exactly where the CRs were
("␣bail-out would be very costly…" at v2 line 16,575,777; the English cut at "hope of
a" at en-de line 4,060,956). The "giga-fren is just noisy" alternative is refuted: every
sampled post-onset English and French line exists verbatim in statmt giga-fren, but
never next to its partner.

**This repo had the same bug.** The first `e01_provenance.py` read the statmt files in
text mode, split news-commentary identically, and reported shifted pairs as exact
matches, placing the onset ~99k rows too late (16,674,601). Fixed: every corpus read
uses `newline="\n"`; a regression test builds a CR-bearing reference corpus and requires
1 of 4 pairs to match, not 4 (mutation-tested).

### What this invalidates or weakens in the rejected paper

| claim | status |
|---|---|
| full-stream Base/Big en-fr cells (tab:two_by_two, seed tables, seed_variability figure) | **trained on a ~45% misaligned corpus**: all 8 runs, confirmed from configs, file mtimes and decoded cache pairs |
| "corpus expansion hurts" (−1.5 BLEU regime drop, ANOVA, Welch tests) | the regime axis is really *clean* vs *55% clean + 45% random pairs* |
| full-stream Big/Base variance 4.0× (F(3,3)=15.77, p=0.049) | rests on the corrupted runs |
| QE source composition (UN ~1.7×, Giga-fren eliminated) | largely QE rejecting misaligned rows. **Superseded (2026-09-13):** the "~1.13× UN enrichment over the aligned prefix" figure is not re-derived and may rest on the stale old-reader labels (`~/mt_local/Machine-Translation-SFT/phase0/provenance_exact_{labels.npy,report.json}`, news-commentary inflated to 85,173 by shifted pairs). Do not cite it; do not use those files |
| keep-rate 93% vs 74% as evidence of source noise | inflated: the length filters rejected shifted pairs |
| en-de Big < Base (p=0.007) | confounded by a 2.7% misaligned news-commentary tail |
| capped cells, v1.1 Base-vs-Big comparison | **unaffected** |
| QE-filtered fine-tuning result | **essentially unaffected**: starting checkpoints are v1.1; ≤215 of the top-1M rows (0.02%) come from the misaligned region |
| dev/test sets | **unaffected** (0 CRs, correct line counts) |

Separately surfaced: the two regimes also use **different SentencePiece models**
(`spm_enfr` vs `spm_enfr_v1_fixed`), an extra confound on the regime axis.

### What this does to Phase 0 / the WMT 2027 plan

- **E0.3 as built is uninterpretable.** From the whole v2 pool, `ft_random` would be ~47%
  misaligned and `ft_bottom` ~93%, so the gate would compare an aligned set against
  random pairs. The builder would also refuse to run (exact match rate 0.55 < 0.95).
- The e01 provenance labels must be regenerated with the fixed reader.
- The 30M CometKiwi-scored corpus and the planned provenance+QE index are ~45% invalid.
  Rows before the first CR-bearing pair are byte-identical after the rebuild, so only
  pairs absent from the old scored file need scoring: **21,628,292** of 38,275,284
  (about 10 GPU-hours on one RTX 5090 at the paper's 13.8 h / 30.1M rate). The earlier
  "~13.5M rows, 6 GPU-hours" estimate predates the rebuild, which recovers ~8.1M
  aligned pairs the length filters had rejected.
- The matched-step seed-variance check below uses the corrupted full-stream runs; its
  full-stream rows are not evidence.
- E0.2's aligned-token accounting: about 44% of full-stream target tokens carry no
  parallel signal.

### Required before anything downstream

1. Make the pipeline CR-safe (downloader replaces `\r`; cleaners read with
   `newline="\n"`), re-clean v2 and en-de from raw, and confirm the rows before each
   onset are byte-identical to the current files.
   Tool: `rebuild_corpus.py`, reading the WMT14 parquet pinned at `wmt/wmt14@b199e406`
   (`hf_wmt14_filelist.tsv`). `--mode legacy` must reproduce the published corpora's
   sha256 exactly, which proves the rebuild starts from the same rows the paper used.
   `--mode fixed` is then the corrected corpus.
2. Score the corrected v2 corpus as the frozen `score_mode` says (`none` / `reuse` /
   `full`; `PROTOCOL.md` D7). Tool: `rescore_plan.py plan` (done on the Mac, against the old
   `v2_scored.tsv`) → **bundle v2** `~/mt_local/rebuild/rental_bundle_v2.tar.gz`
   (86,607,565 B, sha256 `49cd62221730cb33ee9e6ddad99a8fa43f38f9cbcde72044317a11f71ab6bdf6`;
   members at the archive root: `plan.json`, `missing_rows.npy`, `reuse_scores.npy`,
   `provenance_labels.npy`, `provenance_report.json`, `pool_mask_reused.npy`, `SHA256SUMS`)
   copied to the rental's `$WORK/` → `rental_setup.sh data` then `score`. The script verifies
   `SHA256SUMS`, and extract/merge/calibrate refuse a corpus whose sha256 differs from the
   plan's. `merge` needs `reuse_scores.npy`; the old 5-file `rental_bundle.tar.gz` is refused.
3. Re-run `e01_provenance.py` (fixed) and require an exact match rate ≥ 95%.
4. Rebuild the E0.3 controls from the corrected corpus.

### Rebuild results (2026-09-13, Mac, `rebuild_corpus.py`)

Source: WMT14 parquet at `wmt/wmt14@b199e406`. **Legacy mode reproduces all three
published corpora byte for byte** (sha256), so the rebuild starts from exactly the rows
the paper used, and fixed mode differs from them only through CR handling.

| corpus | published (legacy rebuild, sha256 identical) | fixed (CR-safe) | first differing line |
|---|---|---|---|
| v2 en-fr full stream | 30,129,500 rows; src runs 28 lines ahead at EOF; `bad_ratio` drops 9,142,232 | **38,275,284** rows; offset 0; `bad_ratio` 1,067,052 | 16,573,212 |
| v1.1 (first 10M pairs) | 9,312,233 | identical to legacy (no CRs in range) | none |
| en-de | 4,174,104; offset +147 | **4,238,227**; offset 0 | 4,058,219 |

- The first differing line is a **CR merge, not a shift**, on both sides of both corpora
  (checked mechanically: the published line's text is contained in the fixed line). Example:
  the raw pair `'Work schedule.\r Design labor time…' / 'Horaires de travail.\r Planifier…'`
  has one CR per side. The old cleaner split both sides evenly, dropped the 2-token headings
  as too short and kept the rest aligned. Differences can therefore start a few thousand
  lines before the misalignment onset measured by e01 (16,575,777 / 4,060,956).
- Keep rate for v2 goes from 73.8% to **93.7%** of 40,836,715 raw pairs.
- **The fixed v2 corpus is pipeline-aligned throughout** (38,275,284 rows, 0 line offset,
  0 unpaired tail lines). In 1,069 giga-fren samples found from row 16.6M on, 1,061 match
  giga-fren's French on the same line and 0 match a neighbour; per-window digit agreement
  and length correlation never approach the published tail's (~1-2% digit agreement,
  corr ~0.65). It inherits upstream noise: giga-fren lines 6,258,774-6,258,776 are
  themselves shifted by one (fixed rows 22,745,675-22,745,677), which e01 labels as matched.
- The differences begin 2,565 (v2) and 2,737 (en-de) lines before the misalignment onset.
  The first CR pairs are split one CR per side, so fragments stay aligned; the first uneven
  split is raw row 18,152,848 (en 2 CRs, fr 0) for v2 and raw row 4,329,492 (en 2, de 1) for
  en-de. The duplicate-blocking sets are identical in both modes (4,661 fr-en, 126 de-en;
  symmetric difference 0; `rebuild_corpus.py` now checks this itself). "Rows before the
  first differing line are identical" does not mean every later row differs.
- **en-de alignment rests on statistics only.** The fixed en-de corpus (4,238,227 rows; the
  published one has 4,174,104) is byte-identical to the published one through line
  4,058,218. Digit agreement and length correlation in 20k windows stay at aligned levels
  to the end (≥ 88.8% / ≥ 0.970 from 4.04M on) while the published corpus drops to ~1% /
  ~0.56 after 4.06M. There is no statmt en-de copy here, so no exact provenance or
  adjacency check.
- **e01 on the fixed v2: exact match rate 99.40%** (38,045,508 / 38,275,284; was 55%).
  **99.40% is a lower bound on source-verified pairs, not a misalignment rate:** the 229,776
  unmatched rows are clustered (30,909 in Europarl rows 447k-801k; ~8-15k per 1M after row
  16M; longest run 356 rows at 0-based 22,745,320), and every run inspected is aligned rows
  that fail exact matching on normalisation (trailing U+2028 removed by the cleaner, leading
  "- " on Europarl FR). Measured with the data lane's e01 on the same corpus: `--norm strip`
  99.999255% (285 unmatched), `--norm collapse_ws` 100% (0 unmatched); labels under the
  looser normalisers were not compared row by row with exact.
- **Cross-source duplicate keys: 119,858, carried by 243,550 corpus rows (0.64%).** e01
  assigns such rows deterministically to the first-listed source (europarl < commoncrawl
  < un < news-commentary < giga-fren), so giga-fren never receives them. 240,457 of those
  rows also occur in giga-fren, so giga-fren's true share is 55.46-56.09%; news-commentary is
  the least robust figure (116,278 of its 234,407 rows are also giga-fren pairs): 0.31-0.61%.
  Rows by source pair: news-commentary+giga-fren 116,373; commoncrawl+giga-fren 90,443;
  un+giga-fren 33,815; europarl+un 1,934; commoncrawl+un 1,809; europarl+giga-fren 708;
  europarl+news-commentary 291; un+news-commentary 220; europarl+commoncrawl 39 (a row whose
  key is in k sources counts in each of its pairs). Source composition (exact, first-listed
  assignment):

  | source | rows | share |
  |---|---|---|
  | giga-fren | 21,226,420 | 55.46% |
  | un | 11,598,490 | 30.30% |
  | commoncrawl | 3,055,442 | 7.98% |
  | europarl | 1,930,749 | 5.04% |
  | news-commentary | 234,407 | 0.61% |
  | unmatched | 229,776 | 0.60% |

  giga-fren is the last constituent in the HF fr-en stream. Its bulk starts about 160k rows
  after the misalignment onset (first 10k window with > 50% giga-fren at fixed row
  16,732,820), after a news-commentary block in which the onset falls. Re-running the
  current CR-safe e01 on the **published** v2 corpus gives 4,380 exact giga-fren matches out
  of 30,129,500 rows (43 before line 16,573,212, 4,337 after); after line 16,573,212 only
  8,462 of 13.56M rows match any constituent exactly (0.06%). Published-corpus table:
  europarl 1,930,472; commoncrawl 3,010,114; un 11,584,891; news-commentary 20,212;
  giga-fren 4,380; unmatched 13,579,431; match rate 54.93%. The earlier figures of 4,355
  giga-fren and 85,173 news-commentary came from the old universal-newline reader
  (`~/mt_local/Machine-Translation-SFT/phase0/provenance_exact_*`: **stale, do not use**).
  Label arrays of the published and fixed corpora share row indices only up to 16,573,211.
  **The corrected full-stream corpus is majority giga-fren.** Any regime or QE-composition
  claim has to be re-derived on it, not patched.
- Re-score plan: 16,646,992 rows keep their old CometKiwi score: every row before 1-based
  line 16,573,212 (0-based row 16,573,211), plus 73,781 rows after it: 71,051 exact
  duplicates of pairs that occur earlier in the corpus (the cleaner keeps duplicate pairs),
  2,551 rows in the still-aligned stretch 16,573,218-16,575,775 (0-based) before the
  misalignment onset, and 179 short or boilerplate pairs that also occur later in the old
  file. Identical pairs scored twice in the old file differ by at most about 1e-6, and the
  plan takes the earliest occurrence. 21,628,292 rows need scoring (98.12% giga-fren), ~10
  GPU-h on one RTX 5090 at the paper's rate (derived; multi-GPU throughput not measured).
  The reused/new split is almost exactly by source, so mixing the two runs' scores is an
  open decision (`PROTOCOL.md` D7).

Artifacts (Mac, not committed): `~/mt_local/rebuild/{v2,v1,ende}_{legacy,fixed}/`,
`~/mt_local/rebuild/v2_rescore/` (`plan.json`, `missing_rows.npy`, `reuse_scores.npy`),
`~/mt_local/rebuild/prov_v2_fixed/provenance_{labels.npy,report.json}`,
`~/mt_local/rebuild/rental_bundle_v2{/,.tar.gz}` (bundle v2, above). **Do not use**
`~/mt_local/Machine-Translation-SFT/phase0/provenance_exact_*` (old reader) or the 5-file
`rental_bundle.tar.gz`.

### Reproduced on the training box (2026-09-18)

Phase 0 was brought up on the author's own RTX 5090 machine (RUNBOOK, "Alternative
path"). Three results are evidence, not logistics:

- **The corrected corpus reproduces across machines.** `rebuild_corpus.py --mode
  fixed` on that box produced train.clean.{en,fr} with the same sha256 as the Mac
  rebuild (`c8cc665c…` / `e4f5301a…`, 38,275,284 rows), and the v1.1 rebuild equals
  the published pretraining corpus. The rescore plan rebuilt there from the box's
  own `v2_scored.tsv` matches the Mac's plan exactly (16,646,992 reused /
  21,628,292 to score), and its `plan.json`, `missing_rows.npy`, `reuse_scores.npy`
  and `pool_mask_reused.npy` are byte-identical to the bundle's.
- **The release still reproduces.** Rebuilt from the HF Base v1.1 release, with
  newstest2014 regenerated from the pinned parquet: test BLEU **35.31**, the
  released figure, and newstest2013 30.52 as in the release's own training log.
- **Today's scoring stack matches the paper's.** On 3,000 aligned rows scored with
  the pinned CometKiwi-22/InfoXLM revisions, the new scores reproduce the paper's
  `v2_scored.tsv` to `max |diff| = 2e-6` (mean 0.000000), and a 2-shard run equals
  a single-process run to 1e-6 with identical row order. The old-vs-new-stack
  confound behind decision D7 is far below the pre-registered calibration
  thresholds (|mean| ≤ 2e-4, p99 ≤ 2e-3) on this sample; the full 16.6M-row
  calibration runs at the end of scoring.

### QE scoring inventory after the rental was released (2026-09-20)

The rented 4x RTX 5090 box was drained and destroyed. What its CometKiwi scoring
produced, and where it now lives (`~/mt_local/phase0_final/`):

| file | rows | state |
| --- | --- | --- |
| `v2_scored.tsv` (fixed en-fr full stream) | 38,275,284 | complete, `complete: true`, pulled (13.15 GB) |
| `v1_scored.tsv` (v1.1 capped pretrain) | 9,312,233 | complete, pulled (2.97 GB) |
| `ende_scored.tsv` (fixed en-de) | 4,238,227 | complete 2026-09-20 on the author's box (1.20 GB) |

The en-de run was launched on the rental as 8 shards over 4 GPUs and died when shard 5
failed on device 1 (`mt/logs/queue_ende.log`); the shard directory did not survive, so
there was nothing to resume and the whole 4.24M rows had to be scored again. That was the
only GPU work lost with the rental, and it was redone on the author's own RTX 5090 the
same day: 8 shards, 1h35m, `complete: true`, `stack_mixed: false`, mean score 0.7626 over
[0.0397, 0.9146], and the meta's `src_sha256`/`tgt_sha256` match the corpus
(`b976980a…` / `b662c96b…`).

One thing to carry into the write-up: these en-de scores come from a **different stack**
than the en-fr ones — python 3.10.18 / torch 2.11.0+cu128 / sentencepiece 0.2.1 here
against python 3.12.3 / torch 2.14.0 / sentencepiece 0.2.2 on the rental, with the same
`unbabel-comet` 2.2.7, the same `transformers` 4.57.1 and the same pinned model revision.
Nothing mixes *within* the en-de scores, and the stack difference was measured at ~2e-6
BLEU-equivalent on 3,000 aligned rows (see above), but en-fr and en-de numbers should not
be quoted as if they came off one run.

Nothing downstream is blocked by it: the en-de QE scores feed the source-composition
table, not E0.3 or E0.4.

E0.4 lost nothing with the rental either: its split was built on the author's own box, not
on the rental, and is still there (`data/phase0_e04`, 242,022 rows per half). The repo's
`phase0/e04_split.py` reproduces it byte for byte on that machine -- see PROTOCOL.md,
"E0.4", for what does and does not reproduce across machines.

## E0.3 RESULT (2026-09-19): NO-GO

Run on a rented 4x RTX 5090 from the frozen decisions (`phase0/e03_decisions.json`,
sha256 `e3b299b9…`, tagged `e03-decisions-v1`), on the CR-safe corpus, at the
lr_scale the frozen rule selected mechanically (0.15; rungs 0.15 and 0.05 passed,
1 and 0.5 failed). Artifacts kept: `~/mt_local/phase0_results/` on the Mac — the BLEU table, its meta
(lr_scale, decisions sha256, per-run applied tokens, dropped test rows), the decisions
file with its git record, the deviations history, the controls manifest, matrix.json
and lr_selection.json. NOT kept (the instance was released): `v2_scored.tsv` (38.3M
rows, ~22 GPU-h), the v1.1 and en-de scored corpora, the 13 fine-tuned checkpoints and
the full-pool control sets. Everything needed to state and defend the verdict survives;
regenerating the scored corpus would cost about 22 GPU-hours.

| condition | newstest2014 | vs baseline | heldout_un |
|---|---|---|---|
| baseline (Base v1.1 release) | 35.31 | — | — |
| ft_topk (top-1M by CometKiwi) | 35.17 ± 0.16 | **−0.14** | 46.85 ± 0.04 |
| ft_random (matched size) | 35.43 ± 0.07 | +0.12 | 47.34 ± 0.10 |
| ft_bottom (lowest-1M) | 32.23 ± 0.13 | −3.08 | 43.12 ± 0.10 |

- **Criterion 1 FAILS.** ft_topk does not degrade news beyond the seed-noise floor
  (−0.14 against sd 0.16), and its 0.26 BLEU gap to ft_random is not significant
  (Welch t=2.59, df=2.6, crit 4.30).
- **Criterion 2 FAILS.** The in-domain gain is +0.02 against that set's seed sd of
  0.04 — inside the noise (amendment X1's floor; it would have "passed" under the
  old `gain > 0`).
- **The paper's motivating observation does not reproduce** on the corrected corpus
  with a control and a mechanically chosen LR. Per the pre-registration the
  data-quality/domain angle does not carry the WMT 2027 submission, and must not be
  rescued post hoc.

Two by-products that do carry information:

- **QE signal lives at the bottom, not the top.** ft_bottom loses 3.08 BLEU on news
  and 3.71 in-domain, while top-1M is indistinguishable from random. The paper's own
  top-1M score band (0.8959–0.9145) is a dense plateau: selecting "better" inside it
  selects nothing.
- **Step-matched arms are not token-matched**: applied target tokens ranged
  61,981,430 to 74,458,476 across the nine runs (ratio 1.2013, flagged by the
  collector against its 2% threshold). Any statement about these cells has to carry
  that caveat — it is exactly the accounting the rejected paper lacked.

**Sensitivity (secondary, not pre-registered): the other pool cannot even be run.**
Rebuilding the controls from the full 38,275,284-row pool and repeating the stage-1
sweep, **no lr_scale passes the frozen rule** — slope statistics 1.082 / 0.788 /
0.483 / 0.376 at 1 / 0.5 / 0.15 / 0.05 against a 0.3 tolerance, i.e. dev BLEU
declines monotonically at every rate tried (the reused pool passed at 0.15 and 0.05).
`e03_select_lr` stopped rather than pick one by eye. Fine-tuning on a pool that is
55% giga-fren — data the baseline never saw — degrades the model at every LR, so the
pre-registered choice of the reused pool was not merely convenient. The forced-LR
completion of that arm was not run (the box was released first).

**The stage-1 sweeps themselves** (newstest2013 BLEU at ten evals; the pre-FT release
scores 30.52; "stat" is the frozen flat-slope statistic against a 0.3 tolerance). The
box was released before these logs were copied off it, so the traces are recorded here:

| pool | lr_scale | stat | trace |
|---|---|---|---|
| reused (primary) | 1 | 0.876 FAIL | 29.72 29.56 29.07 28.88 29.18 28.93 29.24 28.48 28.88 28.70 |
| reused | 0.5 | 0.489 FAIL | 30.02 29.96 29.62 29.46 29.75 29.51 29.81 29.41 29.64 29.30 |
| reused | **0.15** | **0.243 PASS** | 30.31 30.24 30.06 29.89 30.11 30.07 30.24 29.99 29.92 29.99 |
| reused | 0.05 | 0.221 PASS | 30.35 30.30 30.18 30.08 30.14 30.07 30.29 30.09 30.04 30.09 |
| full (secondary) | 1 | 1.082 FAIL | 29.22 28.83 28.85 28.88 28.49 28.83 28.27 28.57 28.17 27.85 |
| full | 0.5 | 0.788 FAIL | 29.53 29.25 29.60 29.44 29.27 29.28 28.81 28.83 29.06 28.71 |
| full | 0.15 | 0.483 FAIL | 29.92 29.80 29.86 29.75 29.56 29.50 29.54 29.50 29.47 29.47 |
| full | 0.05 | 0.376 FAIL | 30.05 29.99 29.99 29.96 29.85 29.89 29.81 29.68 29.71 29.72 |

The rule picks the largest passing rung, 0.15. Note that fine-tuning never beats the
pre-FT 30.52 on newstest2013 at any rate or either pool — it only costs less at the
small rates.

Scoring evidence collected on the way:

- **The paper's CometKiwi scores reproduce.** Calibrating the new full-corpus run
  against the paper's 16,646,992 reused scores: overall mean difference −2.7e-9, p99
  |diff| 1.02e-06, Spearman 1.0000000, and every source passes separately (UN's
  11.6M rows through giga-fren's 4,351).
- **Scoring is all but deterministic.** Re-scoring 5,000,000 rows on the same stack:
  99.10% of scores bit-identical, mean difference −1.5e-11, p99 exactly 0.
- One 649-token UN sentence (median in that set: 31) exceeds the model's 256-position
  encoding and took down all ten evaluations of heldout_un on the first gate attempt.
  `e03_collect --max-src-tokens` now drops such rows for every system alike and
  records the count; the numbers above are from the rerun.

### Follow-up: was the paper's effect the learning rate? (2026-09-19 evening)

E0.3 found nothing at the lr_scale its frozen rule selected (0.15). The rejected paper
fine-tuned at lr_scale **1.0** and compared against *no fine-tuning*, never against a
matched random set — so "top-k QE filtering degrades news BLEU" and "fine-tuning at this
rate degrades news BLEU" were never separated. The stage-1 sweep already hints at the
answer: ft_topk at lr 1.0 falls 29.72 → 28.70 on newstest2013 across the window, while at
0.15 it is flat.

This run puts **both** conditions at lr_scale 1.0 on the same FT sets (3 seeds each, plus
ft_bottom), on the author's own box:

- If ft_random falls with ft_topk, the paper's effect is the learning rate, not the data.
- If only ft_topk falls, the effect is real but only appears at a rate the LR rule rejects.

Not part of the pre-registered E0.3 (separate controls dir, configs dir and results file,
`results/lr1_probe.tsv`). Setup verified before running:

- The fine-tuning start is the same weights as the E0.3 runs: the box's own
  `base_enfr_v1_redo/averaged.pt` and the HF release have the same 261-tensor digest
  (`a96170fb55444afa`, global_step 105,000).
- The control sets were rebuilt from the paper's own reused scores (a `merge
  --allow-missing` TSV: real scores inside the reused pool, `nan` outside, which the
  pool mask excludes anyway). 87 of 95 manifest fields match the E0.3 build exactly,
  including n_unique_per_set 926,999, topk_duplicates_removed 73,001, the top-1M band
  0.8959360–0.9144790 and every held-out number. The 8 that differ are ties broken
  differently because E0.3 used the *rescored* values (which agree with the paper's to
  ~1e-6): ft_bottom's source composition moves by one row, ft_random∩ft_topk by six.

**Result at lr_scale 1.0** (3 seeds per condition, single final checkpoints,
`results/lr1_probe.tsv`; baseline is the averaged release, so criterion 1's baseline
comparison is biased toward "degrades" by the ~0.2-0.4 BLEU averaging is worth — the
condition-vs-condition comparison, which is the decisive one, is between like
checkpoints and unaffected):

| condition | newstest2014 | vs baseline | heldout_un | vs baseline |
|---|---|---|---|---|
| baseline | 35.31 | — | 46.83 | — |
| ft_topk | 33.62 ± 0.11 | **−1.69** | 45.27 ± 0.14 | **−1.56** |
| ft_random | 34.41 ± 0.07 | **−0.90** | 47.35 ± 0.02 | **+0.52** |
| ft_bottom | 29.49 ± 0.13 | −5.82 | 36.65 ± 0.53 | −10.18 |

Three things hold at once, and together they replace the paper's account:

1. **Half of the paper's effect is the learning rate.** At 1.0 even the matched random
   set loses 0.90 BLEU on news. The paper compared fine-tuning against *no* fine-tuning,
   so this part was attributed to quality filtering by construction.
2. **The other half is real.** ft_topk is 0.79 BLEU below ft_random on news (Welch
   t=10.62, df=3.5, crit 3.18 — significant), so top-k selection is genuinely worse than
   a random set of the same size. At the rule-selected 0.15 this difference disappears
   (0.26, not significant), so the effect exists only in a regime the LR rule rejects.
3. **The domain explanation is contradicted, not merely unsupported.** The paper's story
   was a shift toward UN/legislative text at the expense of news. Then heldout_un should
   rise. It *falls* 1.56 for ft_topk while ft_random *gains* 0.52 — and the top-1M set is
   78.3% UN by provenance. Training on overwhelmingly UN data makes the model worse on
   held-out UN.

What that leaves is a plausible and testable mechanism the rejected paper did not
consider: **CometKiwi's top band selects text that is easy to translate, not text that
teaches** — the top-1M band is 0.02 BLEU-points wide (0.8959-0.9145), deduplication
removes 73,001 rows from it, and its content is boilerplate-heavy UN prose. Fine-tuning
on it damages news and in-domain performance alike, and more than random data does. That
is a data-selection finding about QE metrics, not a domain-shift finding, and it survives
both the corpus bug and the LR confound. It is also the one thread of the rejected
paper's story that reproduces at all.

Not pre-registered, and not part of the E0.3 verdict: the gate ran at the LR its frozen
rule selected and returned NO-GO. This run exists to explain what the paper saw.

## E0.4 RESULT (2026-09-20): H HOLDS — the top band's damage is repetitiveness

Pre-registered in PROTOCOL.md, "E0.4", before the runs; applied by `phase0/e04_report.py`,
which knows the three outcomes in advance. **Outcome 1: H holds.**

Split `ft_topk` (926,999 rows) at its median within-set 8-gram repetitiveness, match the
halves' source-length histograms bucket by bucket (242,022 rows each), fine-tune both at
lr_scale 1.0 for the same 10,000 steps, 3 seeds each:

| condition | rows | newstest2014 | sd | vs base | heldout_un | sd | vs base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (v1.1 averaged) | — | 35.31 | — | — | 46.83 | — | — |
| ft_random (lr-1.0 control) | 926,999 | 34.41 | 0.07 | −0.90 | 47.35 | 0.02 | +0.52 |
| ft_topk (lr-1.0 control) | 926,999 | 33.62 | 0.11 | −1.69 | 45.27 | 0.14 | −1.56 |
| **ft_topk_div** (diverse half) | 242,022 | **32.77** | 0.20 | −2.54 | 44.59 | 0.12 | −2.24 |
| **ft_topk_rep** (repetitive half) | 242,022 | **30.05** | 0.28 | −5.26 | 44.49 | 0.17 | −2.34 |
| ft_bottom (lr-1.0 control) | 926,999 | 29.49 | 0.13 | −5.82 | 36.65 | 0.53 | −10.18 |

**The gap is 2.73 BLEU** against a seed-noise floor of 0.28 (Welch t = 13.96, df 3.6,
crit 3.18). Two things stand out beyond the verdict:

- **The repetitive half of the TOP band is within 0.55 BLEU of the BOTTOM band.** Text the
  QE model scores at the very top (qe_mean 0.899) damages newstest almost exactly as much
  as the text it scores at the bottom (0.450), once it is repetitive. Whatever CometKiwi
  is rewarding in that sub-band, it is not usefulness as training data.
- **Repetitiveness costs news BLEU specifically, not in-domain BLEU.** On heldout_un the
  two halves are 44.59 vs 44.49 — a 0.11 gap inside the seed noise, while the news gap is
  2.73. So this is not "the repetitive half is simply worse data"; it is narrower
  competence, bought at a much higher price outside the fine-tuning distribution.

**What this does NOT show, stated plainly.** Both halves are worse on news than the whole
top band (32.77 and 30.05 vs 33.62), and ft_topk_div is *further* from ft_random than the
whole set is. That part of prediction 1 cannot be read off these runs: the halves are a
quarter of the size at the same step count, so they make roughly four times as many passes
over their data. The half-vs-half contrast is size- and step-matched and is the only clean
comparison here; every comparison to the 926,999-row arms carries that epoch difference.

Secondary, as the protocol required: applied tokens 0.0727B (div) vs 0.0714B (rep), an
imbalance of **1.80%**, inside the 2% threshold but in the same direction as the effect.
Its cause is visible in the story itself — the split matched `\w+` word counts while the
trainer counts SentencePiece pieces, and repetitive text compresses into fewer pieces.
One over-long heldout_un row was dropped from every evaluation alike (`--max-src-tokens
254`), newstest2014 lost none.

The baseline was re-evaluated from scratch on this box's stack (the shared BLEU cache was
moved aside first, since its key carries no python/torch identity) and reproduced the
lr-1.0 control's 35.31 / 46.83 exactly, so the two experiments are directly comparable.

E0.4 is an explanatory probe. It does not revive E0.3's NO-GO verdict — it says what the
NO-GO was made of.

## Findings from the original training machine (2026-09-12)

Pulled read-only from the Linux box; mirror at `~/mt_fetch/` (not committed —
data and logs). Nothing below required loading or re-running a model except the
weight comparison.

### Settled

1. **The public HF releases ARE the paper's fine-tuning start, tensor for tensor.**
   sha256 over every tensor of `checkpoints/base_enfr_v1_redo/averaged.pt` and of
   `euswbnix/transformer-wmt14-enfr-base/pytorch_model.bin` are identical
   (`120aca446121d003…`); likewise Big v1.1 (`04ed81785ad86709…`). E0.3 on a
   rented box from HF weights is running the paper's exact starting point.
2. **The fine-tuning LR was continuous, confirmed from the checkpoints' own
   records.** `sft_base_enfr/final.pt` history: LR 2.728e-4 at pretraining step
   105,000, 2.653e-4 at FT step 111,000 (= 512^-0.5 · 27750^-0.5 exactly). Big:
   2.046e-4 → 2.017e-4. The retracted "restart LR" framing is now disproven by data,
   not only by code reading.
3. **`logs/sft_base.log` is not the paper's run.** It records a resume from step
   80,000 of the *v1.0* checkpoint at LR 3.13e-4, has no evaluations, and was
   written 2026-06-09 — after the paper. The paper's Base FT stdout was not
   captured; `final.pt` (step 111,000, best BLEU 29.8259 = v1.1's) is the record.
4. **The starting point reproduces on a fresh rented GPU.** On an RTX 5090 (vast.ai,
   torch 2.11.0+cu128), `phase0/hf_to_ckpt.py` rebuilt Base v1.1 from
   `euswbnix/transformer-wmt14-enfr-base` @ `8a58dcc4205991a5…` (strict load, 60.5M
   parameters, global_step 105,000; weights sha256 `4d7eb2bbfecd31cc…`); the release's
   SentencePiece model is byte-identical to the original machine's
   `spm_enfr_v1_fixed.model` (`d1c10ea80fb95984…`); and `scripts/eval_bleu.py` on
   newstest2014 (beam 5, lp 1.0, sacrebleu 13a, 3,003 sentences, 41 s) gives
   **35.31 — exactly the released value.** Requires the training repo root on
   PYTHONPATH (a fresh clone is not pip-installed).
5. **Per-seed translations were never saved.** `outputs/` holds only two files
   from 2026-04-19. Regenerating per-sentence outputs for the old runs needs the
   `best.pt` files, which exist only on that machine (≈36 GB; not pulled).

### New confound in the published 2×2 table — to verify at matched steps

The 2×2 means come from `results/summary.json` (the Base-capped cell reproduces the
paper's "35.11 ± 0.20" exactly). Each seed's number is the `best.pt` of its own
run, and **runs within a cell were trained to very different lengths.** The seed-42
runs are the originals; seeds 1–3 were re-runs with raised `max_steps` (the
uncommitted config edits on the box: `base_en_fr_v1_redo` 100K→150K,
`base_en_de` 100K→300K, `big_en_de` 300K→800K), and several runs of either kind
stopped early or were interrupted.

**In 5 of 6 cells the shortest run is also the lowest-BLEU run.** The shortest
run is not always seed 42 (Big capped: s3 at 181K; Base en-de: s1 at 168K), so
"drop seed 42" is the wrong cut — an earlier draft of this note used it and it
misattributes the effect.

| cell | best.pt steps | spread | shortest run | its BLEU | lowest BLEU | same run? | sd (4 runs) | sd (without shortest) |
|---|---|---|---|---|---|---|---|---|
| Base capped | 120,000–149,000 | 19% | s42 @ 120,000 | 34.84 | s42 (34.84) | yes | 0.20 | 0.10 |
| Big capped | 181,000–274,000 | 34% | s3 @ 181,000 | 34.18 | s3 (34.18) | yes | 0.62 | 0.27 |
| Base full-stream | 537,000–579,000 | 7% | s42 @ 537,000 | 33.45 | s3 (33.24) | no | 0.19 | 0.24 |
| Big full-stream | 374,000–775,000 | 52% | s42 @ 374,000 | 32.55 | s42 (32.55) | yes | 0.77 | 0.38 |
| Base en-de | 168,000–300,000 | 44% | s1 @ 168,000 | 23.44 | s1 (23.44) | yes | 0.26 | 0.21 |
| Big en-de | 458,000–784,000 | 42% | s42 @ 458,000 | 22.18 | s42 (22.18) | yes | 0.39 | 0.13 |

Big / Base seed-sd ratio (the paper's "Big has 3–4× higher seed variance"):

| regime | all 4 runs | without each cell's shortest run |
|---|---|---|
| capped | 3.1× | 2.7× |
| full-stream | 4.0× | 1.6× |
| en-de | 1.5× | 0.6× |

**Status: a lead, not a conclusion.** Removing the shortest run is also post hoc,
and sds from 3 values are noisy. Removing one run moves the full-stream ratio from
4.0× to 1.6× and the en-de ratio from 1.5× to 0.6×, which is itself the point: this
variance statistic is not stable enough to carry a headline. (Removing seed 42
instead, the cut an earlier draft used, would even raise the capped ratio to 7.5×.) What the data does establish is that the
paper's between-seed variance mixes seed noise with run-length differences of up to
~50%, and that `best.pt` selection over runs of different length adds a
winner's-curse term that also grows with length.

The clean test needs no GPU: `~/mt_fetch/extracted/mt_histories.json.gz` holds
every run's in-training validation BLEU at every eval step, so runs can be compared
at a common step. Not yet run. Either way this reinforces PROTOCOL.md's
run-to-overshoot design and its per-cell reporting of run length.

### Matched-step check (descriptive, 2026-09-12)

> **Warning (2026-09-13):** the full-stream en-fr runs in this table were trained on the
> misaligned v2 corpus (see the CRITICAL section above), and the en-de runs on a corpus
> with a 2.7% misaligned tail. Only the capped rows are clean evidence.

From the pulled checkpoint histories: for each cell, the common step C is the last
step every run reached; "best ≤ C" is each run's best in-training validation BLEU
up to C, i.e. the same `best.pt` selection the paper used, but at equal length.

| cell | common step C | seed sd, best overall (paper-like) | seed sd, best ≤ C |
|---|---|---|---|
| Base capped | 122,000 | 0.27 | 0.14 |
| Big capped | 211,000 | 0.28 | 0.11 |
| Base full-stream | 590,000 | 0.13 | 0.13 |
| Big full-stream | 416,000 | 0.47 | 0.15 |
| Base en-de | 208,000 | 0.18 | 0.05 |
| Big en-de | 458,000 | 0.46 | 0.11 |

| Big / Base seed-sd ratio | best overall | best ≤ C |
|---|---|---|
| capped | 1.0× | 0.8× |
| full-stream | 3.5× | 1.1× |
| en-de | 2.5× | 2.2× |

Reading: in full-stream, Big's excess seed variance **mostly disappears once runs
are compared at equal length**; in en-de part of it remains; in capped there is
little excess on this metric either way. Caveats, all material: this is in-training
validation BLEU on newstest2013, not the paper's newstest2014 test BLEU; each sd is
over 4 values; and truncating at C compares every run at the shortest run's length,
which is short of convergence for some. It supports treating the paper's variance
headline as unreliable; it is not a replacement estimate.

## E0.2 — FINDING: the reported token budgets are far too high, and EVERY cell is under-trained

> **Corrected 2026-09-04** after an adversarial audit found a second, independent
> error in this script's own arithmetic. The numbers below are the corrected ones.
> The superseded version claimed Base-capped sat "exactly on" the Vaswani
> reference at 33.9 tok/param. That was an artifact. It is ~13. See
> *What the audit changed* at the end of this section.

### Correction 1 — step semantics (exact, data-independent)

`src/training/trainer.py` increments `global_step` **once per micro-batch**
(line 441, inside `_train_step`, called per batch at line 321). The optimizer
only steps when `global_step % accumulate_steps == 0` (line 461), and
`max_steps` is compared against `global_step` (line 310).

So the reported step counts are **micro-batches**. The paper computed
`steps x 98,304` where `98,304 = 24,576 x accumulate_steps(4)`, which
**double-counts accumulation**.

### Correction 3 — the loss counts SHIFTED labels, not cached tokens

Found by a second audit, after the table above had already been corrected once.
`dataset.py:43` encodes with `add_bos=True, add_eos=True`, but `trainer.py:429`
takes `tgt_labels = tgt[:, 1:]` and `trainer.py:487` counts the non-pad entries of
*that*. A cached sequence of length L therefore yields **L−1** label tokens.
Counting L overcounts by exactly one per SENTENCE — about 3% at WMT subword
lengths, which is larger than the ±2% tolerance the batch calibration certifies
against. `e02_token_accounting.py` and `phase1/calibrate_batch.py` both had it, so
cross-checking one against the other would not have caught it.

The table above is post-fix. A unit assertion now guards it: three sentences of
cached length 7/9/11 must measure 24 label tokens, not 27.

### Correction 2 — `max_sentences` is the binding cap, not `max_tokens`

`TokenBatchSampler` closes a batch on **either** cap (dataset.py:265-267):

```python
if (len(batch) + 1) * new_max > max_tokens or len(batch) >= max_sentences:
```

The two cross at `max_tokens / max_sentences`, which is **128 for both arms**
(24576/192 = 8192/64 = 128). WMT subword sentences run ~30 tokens, and the
sampler length-sorts within chunks before batching, so `max_len` sits far below
128 and **the sentence cap binds ~96% of the time**. Batches reach only ~34% of
`max_tokens`.

Therefore `batch_size: 24576` is not the batch size in any usable sense. It is a
cap that almost never binds. The correct conversion is measured directly:

```
true tokens = steps x E[non-pad target tokens per micro-batch]
```

not `steps x max_tokens x (padding factor)` — which mixes a **nominal**
numerator with a **realized** denominator and inflates the result ~2-3x.

### The corrected table

Measured on synthetic WMT-like lengths (src mean 30.8, tgt mean 33.4).
**These must be re-measured on the real `.cached_256.npz` before publication.**

| cell | true non-pad tgt | tok/param | MFU |
|---|---|---|---|
| en-fr Base capped | 0.8B | **12.4** | 14.5% |
| en-fr Big capped | 0.5B | **2.6** | 14.8% |
| en-fr Base full | 3.7B | **61.0** | 12.7% |
| en-fr Big full | 0.9B | **4.1** | 15.8% |
| en-de Base | 1.4B | **23.4** | 9.8% |
| en-de Big | 1.0B | **4.6** | 19.9% |

Vaswani et al. 2017 reference: ~36-41 tok/param.

**Every cell is under-trained.** Base-capped at ~13 is roughly a third of the
reference; Big never exceeds 5. The earlier claim that Base-capped "sits exactly
on the reference" was wrong and pointed the diagnosis in the wrong direction — it
implied Base was fine and only Big was starved.

### The confound is ROBUST to both corrections

Big/Base true-token ratio: **0.71x / 0.23x / 0.68x** (capped / full / en-de),
against Vaswani's 3.00x. Both arms rescale identically, so the ratio barely moves
(it was 0.74/0.24/0.71 before Correction 2). Reviewers R1 and R3 were right; only
the absolute figures move — but they move a lot.

MFU now lands at 10-20%, low enough that several cells fail the script's own
plausibility band. That is a real open question, not a rounding detail: either the
wall-clock figures, the step counts, or the model FLOP estimate is off somewhere.
Resolve it on the real cache before quoting any MFU number.

> **Do not report a Big-vs-Base capacity claim in any form until this is fixed.**
> The corrected instrument must hold the effective non-pad token budget identical
> across models and absorb the difference into gradient accumulation.

### What the audit changed

A six-lens adversarial audit (40 agents, each finding independently refuted)
found `e02_token_accounting.py:179-180` used `reported = steps * mb` — the
**nominal** cap — while rescaling by a ratio whose denominator was the
**realized** padded budget. The script already computed the right quantity,
`mean_nonpad_tgt_per_microbatch` (line 133), and never used it.

Every absolute number in the superseded table was ~2.6x too high, and the
headline comparison to Vaswani inverted. Correction 1 (the 4x) was unaffected;
it is exact and data-independent.

This is the same class of error the paper was rejected for — an unverified
compute number — occurring inside the script written to catch that error. The
script now prints `sentence_capped_frac` and `nominal_fill_frac` so the binding
constraint is visible in its own output rather than inferable only from source.

## E0.3 — scope reduced: the "sorted curriculum" confound does not exist

The plan hypothesised that `scripts/filter_by_score.py` writing top-K sorted
high-to-low could have fed the SFT run a monotone quality curriculum.

**It cannot.** `TokenBatchSampler` (dataset.py) shuffles indices
(`np.random.shuffle`), re-sorts only *within* chunks by length for batching
efficiency, then shuffles batch order (`random.shuffle(batches)`); the trainer
constructs the training loader with `shuffle=True` (trainer.py:114). File order
does not survive.

**Drop control (d).** Remaining controls for the go/no-go gate:

- (a) random control at matched **unique** size, identical schedule
- (b) bottom-by-QE control at the same unique size

**Deduplication rule (added 2026-09-12).** The paper's FT set was not "top-1M": it
was top-1M by QE **then exact (src, tgt) dedup**, giving 931,366 unique pairs
(`paper_section_7.md:17`: "All SFT below uses this 931K-pair deduplicated set").
The first version of `e03_build_controls.py` did not dedup at all, so ft_random
and ft_bottom would have carried repeated pairs the top-k arm did not, making the
arms differ in effective data as well as in QE. Now: ft_topk reproduces the
paper's rule exactly; ft_bottom and ft_random are drawn from the deduplicated pool
at that same unique size; all three are asserted duplicate-free. The regression
suite reconstructs the paper's rule independently and a mutation that disables
dedup is caught.
- (c) LR sweep spanning >= 1 decade, including a rate low enough that BLEU is flat
- (e) a held-out UN/legislative test set, to show the in-domain *gain* the domain
      story predicts (Europarl was dropped on 2026-09-18: 2000/2000 candidate pairs
      are in the v1.1 pretraining corpus, so it cannot be held out)

### CORRECTED: the LR framing was wrong; the real discontinuity is in Adam

> This section previously claimed the fine-tuning LR was "2x what the config
> intended" and that "restart LR too high -> forgetting" was therefore a
> **confirmed** deviation. That escalation was wrong and is retracted. A reviewer
> holding the code release could have demonstrated it.

**What survives.** `configs/sft_base_enfr.yaml:3-6` says the scheduler resumes at
"~1.4e-4 effective peak ... 10-20% of pretraining peak 7e-4". That comment is
wrong: it forgot the `// 4`. The true value is

    1.0 * 512**-0.5 * (105000 // 4)**-0.5 = 2.728e-4   (39.0% of the 6.988e-4 peak)

It is one wrong comment, not a systematic error — `configs/sft_big_enfr.yaml:3-4`
says "global_step=210K ... lr_scale=1.5 gives effective ~2e-4", and
`1.5 * 1024**-0.5 * (210000 // 4)**-0.5 = 2.05e-4`. The Big comment **was**
computed with the `// 4` and is correct.

**What does not survive.** `scheduler.step()` has exactly one call site,
`trainer.py:472`, nested inside `if self.global_step % self.accumulate_steps == 0`
(trainer.py:461) and inside the non-spike branch. So the scheduler advances once
per *optimizer* step, and at the end of pretraining its true `_step` was already
`<= 105000 // 4 = 26250`.

That makes `sched_step = self.global_step // self.accumulate_steps`
(trainer.py:577) the **correct reconstruction, not a bug** — and it rounds
*downward*, never upward. So:

    LR at fine-tuning start  <=  LR at end of pretraining

There is **no upward LR discontinuity**. At global_step 104000 pretraining was
already running at `1.0 * 512**-0.5 * 26000**-0.5 = 2.741e-4`, versus 2.728e-4
after the reset — a 0.5% change. Across the whole 10K-micro-step SFT run the LR
moves 2.728e-4 -> 2.606e-4, i.e. -4.4%. This is LR **continuity**.

A hypothesis that requires an LR increase has no LR increase to point at. The
monotone BLEU decline is therefore **not** a "textbook too-large-restart-LR
signature", and it does not discriminate between forgetting and domain shift —
constant-LR continuation onto a new data distribution is exactly what the domain
story predicts too.

**The better hypothesis, which the audit surfaced.** The real discontinuity at the
FT boundary is that `reset_optimizer=True` **discards Adam's moments**
(trainer.py:572-579) while keeping the nominal LR. `torch.optim.Adam` is
bias-corrected (trainer.py:74-79, betas 0.9/0.98, eps 1e-9), so a fresh-moment
step has per-parameter magnitude ~= lr, whereas at pretraining convergence
`|m_hat| / (sqrt(v_hat) + eps) << 1`. **The effective update magnitude does jump
upward — at constant nominal LR.** The discontinuity is real; it lives in the
optimizer state, not the scheduler.

That reframing costs nothing operationally: the LR sweep is still the right
control (39% of peak is high for fine-tuning in absolute terms regardless), and
`e03_decide.py`'s two pre-registered criteria contain no LR term at all. What
changes is what E0.3 is allowed to *claim* it is testing.

**Also uncontrolled between arms, found in passing** — these are confounds in
their own right and must be fixed under any Phase 1 protocol:

| | Base | Big |
|---|---|---|
| effective batch (`batch_size x accumulate_steps`) | 98,304 | 32,768 |
| `warmup_steps` | 4,000 | 8,000 |
| `lr_scale` | 1.0 | 1.5 |

Big trained at **one third** of Base's effective batch. Vaswani held it constant.

### The Stage-1 LR grid

Effective peak = `lr_scale * 512**-0.5 * 26250**-0.5`:

| lr_scale | effective LR | % of pretrain peak | |
|---|---|---|---|
| 1.00 | 2.73e-4 | 39.0% | what the SFT run used, = the pretraining terminal LR |
| 0.50 | 1.36e-4 | 19.5% | what the Base config comment mistakenly claimed |
| 0.15 | 4.09e-5 | 5.9% | |
| 0.05 | 1.36e-5 | 2.0% | |

Range 20x. Over a 10K-micro-step run Noam decays only ~4%, and `min_lr` is pinned
to the same value, so each rung is flat.

### How to run E0.3

> **Superseded (2026-09-13): run `bash phase0/rental_setup.sh controls`, `stage1`,
> `stage2 <lr>`, `gate <lr>` (see `RUNBOOK.md`).** The block below points at the misaligned
> 30.1M-row `data/v2_scored.tsv`, passes a hand-typed `--sources`, and picks the lr_scale
> by eye. The builder now exits on that row-count mismatch. The design choices are frozen
> in `phase0/e03_decisions.json` (`PROTOCOL.md`, "E0.3 decisions required before
> controls"); the lr_scale is selected by `phase0/e03_select_lr.py`. Kept for reference.

Two stages, 13 runs (~7 GPU-hours), not the naive 3x4x3 = 36:

```bash
# 0. needs e01_provenance.py output to build the in-domain held-out sets;
#    without them criterion 2 is untestable and the gate cannot return GO.
python phase0/e03_build_controls.py \
    --qe-scores data/v2_scored.tsv \
    --provenance phase0/provenance_labels.npy \
    --sources europarl,commoncrawl,un,news-commentary,giga-fren \
    --out-dir data/phase0

python phase0/e03_run_matrix.py --base-config configs/sft_base_enfr.yaml \
    --data-dir data/phase0 --out-dir configs/phase0

bash configs/phase0/run_stage1.sh          # 4 runs: LR sweep on ft_topk
# (superseded) the lr_scale comes from phase0/e03_select_lr.py under the frozen rule, then:
bash configs/phase0/run_stage2.sh 0.15     # 9 runs: 3 conditions x 3 seeds

# evaluate EVERY run on newstest2014 + heldout_un + heldout_europarl, then:
python phase0/e03_decide.py --results results/phase0_bleu.tsv
```

`e03_decide.py` exits 0 = GO, 1 = NO-GO (rule applied to complete data), 2 = cannot
decide (missing conditions or gate sets, or a cell without exactly 3 seeds; full list in
`RUNBOOK.md` section 5).

### Pre-registered decision rule

Proceed to the full program **only if**, at the lr_scale selected from stage 1 by
phase0/e03_select_lr.py under the rule frozen in phase0/e03_decisions.json
(`flat-slope`, tolerance 0.3 BLEU; decision D6, frozen 2026-09-18), BOTH hold:

1. top-k-QE FT degrades news-domain eval **more than** the matched random
   control, by **more than** the seed-noise floor, **and**
2. top-k-QE FT **improves** the held-out UN/legislative eval by **more than that
   set's seed-noise floor** (the ft_topk seed sd on it). Frozen 2026-09-18:
   criterion 2 is UN-only (decision D5) and carries a noise floor (amendment X1),
   so it is no longer weaker than criterion 1 — a +0.01 BLEU gain used to pass
   while the seed sd is 0.05-0.2 BLEU. Europarl is not part of the gate: under
   `exclude_pretrain_from_heldout` every candidate Europarl pair (2000/2000) is
   already in the v1.1 pretraining corpus, so no honest Europarl set exists.

Otherwise the motivating observation is an optimization artifact — most likely
the discarded Adam moments (see the corrected LR section above), which raise the
*effective* update magnitude at the FT boundary even though the nominal LR is
continuous. In that case: publish a short correction note and rebuild the paper on
the compute-alignment work alone, which does not depend on it.
