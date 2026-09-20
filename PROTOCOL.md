# Phase 1 pre-registration — WMT 2027

**Status: DRAFT. Not yet frozen. No Phase 1 training run may start before this
file is git-tagged.**

### Implementation status

| § | Requirement | State |
|---|---|---|
| 2.1 | cumulative never-reset token counter | ✅ `phase0/trainer_token_accounting.patch` |
| 2.2 | token-keyed training gate + eval cadence | ✅ same patch. `max_steps` is **not consulted** in token-keyed runs (an audit found it silently pre-empting the budget, arm-dependently); regression I6/I6b in `phase0/test_token_counter.py` |
| 4.1 | dev cross-entropy evaluator | ✅ `_evaluate_ce()` in the same patch — fp32, teacher-forced, label smoothing off; writes `dev_ce_trace.tsv` (§7) |
| 2.3 | disable adaptive eval interval | ✅ same patch (`eval_every_tokens` bypasses `_update_eval_interval`) |
| 3.2 | effective-batch calibration | ✅ `phase1/calibrate_batch.py` — **must be re-run on the real cache** |
| 4.3 | post-hoc argmin decision script | ✅ `phase1/converge.py` (6 scenarios tested) |
| 4.4 | δ calibrated from a seed pilot | ⬜ **blocking** — needs GPU |
| 5.1 | LR probe sweep | ⬜ config generator not written |
| 6.x | reporting | partially — `converge.py` emits 6.2 and 6.3 |

**Two blockers remain before this file can be tagged:** the §3.2 calibration must
be run against the real length cache (the numbers below are from synthetic
lengths and are placeholders), and §4.4's δ must come from the seed pilot rather
than the 0.002 working prior.

Decided 2026-09-04 by a judge panel over three independently-argued protocols
(equal tokens / equal tokens-per-param / each-to-convergence), following a
six-lens adversarial audit of Phase 0. Every code claim below marked ✅ was
re-verified by hand against the source before being written here.

---

## The decision

**Primary alignment: run-to-overshoot with post-hoc argmin (RTO-PA).**

Not naive "train each cell until it stops improving" — that form was **rejected**,
and it was the session's prior working assumption. On this codebase it is
indefensible, for three verified reasons:

- ✅ `TransformerScheduler._get_lr` (optimizer.py:36-43) is open-ended inverse-sqrt
  floored at `min_lr`, with no horizon and no decay to zero. Validation loss creeps
  down asymptotically; a run never converges, it gets *stopped when patience
  expires*.
- ✅ `_update_eval_interval` (trainer.py:237-269) interpolates the eval interval
  from an EMA of **training loss**. Training loss is a function of capacity. So
  patience — counted in evaluations (trainer.py:376) — would differ between Base
  and Big by up to an order of magnitude, driven by the exact variable under study.
- ✅ The stopping signal is beam-5 corpus BLEU (trainer.py:356 → :508-522), noisy
  and non-monotone, whose argmax carries a winner's curse that grows with the
  number of evaluations, which differs by arm.

Under naive convergence the headline result would be an artifact of
trainer.py:237-269, and a reviewer holding the code release could prove it. That
is a worse outcome than the original rejection.

**What RTO-PA changes:** the run never stops. Every cell runs far past any
plausible stopping point, the full dev-CE-vs-cumulative-token trace is published,
and "convergence" is an argmin computed mechanically by a git-tagged script over
released data. *A stopping rule you never execute cannot bias anything.* The
paper's most attackable design choice becomes its most auditable artifact.

This move is available only because compute is not a constraint here.

### Why not the other two

- **Equal tokens** is the reviewer default and needs no defense — but at any single
  budget it under-trains the larger model, *in the same direction as the confound
  that got the paper rejected*, merely reduced from ~4.6x to ~3.5x. Its own
  advocate conceded this. There is no budget at which both models sit comparably
  on their own loss curves; that is arithmetic, not judgment.
- **Equal tokens/param** is the right *coordinate* but requires hard-coding a
  constant (R≈38) anchored to "Base capped sits exactly on the Vaswani reference"
  — which the audit showed was an artifact of the `e02:179` bug applied to
  synthetic lengths. **Hard-coding an unmeasured constant into the design of a
  paper whose rejection was caused by unverified arithmetic is the same error
  again.** Nothing inside this protocol could detect it if R were wrong.

**RTO-PA is the strict superset.** Run to overshoot and publish the curves, and
both other comparisons are vertical slices off the same data at zero marginal
cost. Neither can reciprocate. Under no compute constraint, choosing the subset
destroys information for no reason, and neither advocate supplied a positive
argument for doing so.

---

## Blocking corrections before any Phase 1 run

- **0.1** Correct `phase0/README.md`: Claim B's *consequence* is refuted. ✅
  `scheduler.step()` executes only inside the accumulation-boundary branch
  (trainer.py:461,472), so pretraining's true scheduler state at global_step
  105000 was ~26250 and trainer.py:577 is the **correct** reconstruction. 2.73e-4
  is LR **continuity** (pretraining was at 2.741e-4 immediately prior), not a 2x
  overshoot. ✅ Corroboration: `sft_big_enfr.yaml:3-4`'s "~2e-4" matches
  `1.5 * 1024**-0.5 * (210000//4)**-0.5 = 2.05e-4` — the Big comment used `//4`
  correctly; only the Base comment forgot it. "Restart LR too high → catastrophic
  forgetting" is **demoted to an unconfirmed rival hypothesis**. Live rivals: the
  Adam moment reset (trainer.py:572-579), the averaged checkpoint, and data shift.
  Claim A (the 4x) stands. *(done)*
- **0.2** Fix `e02_token_accounting.py` to `true_tok = steps *
  mean_nonpad_tgt_per_microbatch`. Re-run on the **real** cache. Publish the
  corrected tok/param table as descriptive only — it does **not** feed the Phase 1
  design. Registered in advance: corrected figures are expected LOWER than
  33.9/7.3/166.5/11.7/63.8/13.1, and "Base capped sits exactly on the Vaswani
  reference" is retracted. *(script fixed; real-cache re-run pending)*
- **0.3** Before asserting any effective-LR figure in print, grep the real SFT log
  for the value printed at ✅ trainer.py:580-581.

## 1. Protocol and unit of account

- **1.1** Primary alignment RTO-PA. No cell stopped early; every cell runs to the
  §4.2 ceiling; the full trace is released.
- **1.2** Unit of account: **cumulative non-pad target tokens**, the running sum of
  ✅ `(tgt_labels != PAD_ID).sum()` (trainer.py:487), over micro-batches whose
  gradient is actually applied. Never the sampler budget — ✅ dataset.py:265
  computes `(len(batch)+1) * new_max` over `max(src_len, tgt_len)`, a padded budget
  over the longer side.
- **1.3** Set `loss_spike_ratio: 0` in every Phase 1 config so no effective batch is
  dropped and the token counter cannot diverge from applied gradients. ✅ Verified
  that 0 *disables* the guard rather than tripping it: trainer.py:448 tests
  `self.loss_spike_ratio > 0` first. Report the (expected zero) skip count anyway.

## 2. Mandatory trainer patch, identical in all cells

- **2.1** Cumulative never-reset `cumulative_target_tokens`; ✅ the existing
  accumulator is zeroed every log interval at trainer.py:352 and is unusable.
  Checkpoint it; make it the x-axis of every published figure.
  *(implemented)* The patch tracks two counters, not one: `total_train_tokens`
  (processed — the MFU denominator) and `applied_target_tokens` (gradient
  actually applied — the learning-curve x-axis). They coincide when the spike
  guard is off per §1.3; tracking both makes any divergence visible instead of
  forcing a silent choice between two different quantities.
- **2.2** Replace the loop gate at trainer.py:308-310 with a **token-keyed** gate.
  Never key any control to `global_step`, which ✅ counts micro-batches
  (trainer.py:441) and therefore means different things across arms once
  `accumulate_steps` differs. *(implemented: `_should_continue` / `_eval_due`;
  `max_target_tokens` and `eval_every_tokens` default to 0, preserving the old
  step-keyed behaviour for non-Phase-1 runs.)*

  Concretely, `phase0/test_token_counter.py` demonstrates the failure the gate
  removes: **4,000 micro-steps gives 10.0M applied tokens on a Base-like arm and
  3.3M on a Big-like one.** Under a step-keyed gate the two arms were never
  running the same experiment.

  A sharper version (invariant I6b), illustrated with `accumulate_steps` 12 vs 17
  at 98,304 applied tokens per optimizer step: an inherited `max_steps` of 800,000
  micro-batches — the largest in any existing config — stops one arm at **6.55B**
  and the other at **4.63B** applied tokens. Ratio 17/12 = 1.417x, a 29% budget
  deficit, silent. The size of the gap depends on the calibrated
  `accumulate_steps`; the existence of the gap does not — any difference between
  the arms produces one.
  An audit found the first version of this gate consulted `max_steps`
  unconditionally, reintroducing exactly that; `max_steps` is now **not consulted
  at all** in token-keyed runs, and only an explicitly configured
  `max_micro_steps_backstop` can pre-empt the token budget — loudly, with the
  shortfall printed and the stop reason recorded as `step_backstop`.
- **2.3** Disable the adaptive eval interval: set `eval_interval_min ==
  eval_interval_max` so `_update_eval_interval` is a no-op. Publish the config
  diff. Methods-section rationale: cadence must not be a function of training
  loss, because training loss is a function of capacity.
- **2.4** Evaluate every 50M cumulative non-pad target tokens, identically in every
  cell.

## 3. Held fixed across all cells

- **3.1** One joint SentencePiece model, vocab 32000, fit once on the union of both
  corpus regimes, reused byte-identically. Non-negotiable: token counts are
  otherwise not commensurable.
- **3.2** Effective optimizer batch — defined as **measured** mean non-pad target
  tokens per optimizer step — target 98,304, matched across arms to within 2%.
  *(implemented: `phase1/calibrate_batch.py`)* It searches `max_sentences` —
  **not** `batch_size`, which is a `max_tokens` cap that almost never binds — and
  solves jointly: both arms inside tolerance of the target **and** their mutual
  spread inside tolerance, then maximise throughput among survivors. Selecting
  per-arm on |error| alone is wrong twice: it picks a tiny micro-batch with a huge
  `accumulate_steps`, and two arms each within tolerance can still be 2x tolerance
  apart from each other, which is the quantity that actually has to match.

  Placeholder output on synthetic lengths (Base `max_sentences` 192 /
  `accumulate_steps` 16, measured 6,153 tgt/micro; Big 384 / 17, measured 5,847;
  spread 0.97%). **Re-run on the real cache before writing any config** — these
  numbers are from a lognormal length model, not from the corpus. This repairs the ✅ 98,304 vs 32,768 mismatch between the
  two SFT configs — **Big trained at one third of Base's effective batch**, while
  Vaswani held it constant.
- **3.3** Warmup as a fixed **fraction** (4%) of planned optimizer steps in every
  cell. Repairs the uncontrolled ✅ 4000 vs 8000 difference.
- **3.4** `max_seq_len` 256, `data.max_tokens` 256, dropout 0.1, label_smoothing
  0.1, clip_grad_norm 1.0, bf16, Adam betas (0.9,0.98) eps 1e-9, beam 5 /
  length_penalty 1.0 at eval, identical frozen valid/test sets, identical corpus,
  identical data-order seed set.
- **3.5** Exactly 5 seeds per cell: {1,2,3,4,5}.

## 4. Stopping metric, ceiling, argmin

- **4.1** Convergence metric: held-out token-level **cross-entropy in nats per
  non-pad target token**, label smoothing off, teacher-forced, fp32, on the full
  frozen newstest2013. Beam BLEU is **retired as a stopping signal** and kept only
  for reporting.
- **4.2** Ceiling: compute a provisional point P = first evaluation at which dev CE
  has failed to improve on its running best by ≥ δ for a continuous window of 300M
  tokens. **The run does not stop at P.** It continues to
  `min(4 × tokens(P), 20B)` and never less than `P + 2B`.
- **4.3** Convergence is defined **post hoc** as the argmin of the 3-point
  median-smoothed dev-CE trace over the whole run, ties toward fewer tokens,
  computed by the tagged script with no human in the loop.
  *(implemented: `phase1/converge.py`.)* It emits all three §6.2 slices, applies
  the §6.3 null rule, and flags §4.5 non-convergence. Note what it exposes about
  slice (c): with a 3.5x parameter ratio, equal tok/param **cannot** put both
  cells at their own convergence points unless their convergence budgets happen
  to differ by that same 3.5x. The script prints each cell's budget as a fraction
  of its own convergence point so this cannot be glossed over.
- **4.4** δ is **calibrated, not guessed.** Pilot: 5 seeds of Base-capped to the
  ceiling; in the plateau region take the max between-seed |ΔCE| at matched token
  counts; set δ at that measured floor. Working prior 0.002 nats. The measured
  floor is recorded here **before** the main matrix runs; if it exceeds the prior,
  δ rises and that is recorded, not absorbed.
- **4.5** If any cell's argmin falls in the final 20% of its trace, that cell is
  declared **not converged**, the ceiling doubles, and the extension is reported as
  a documented deviation.

## 5. Learning rate

- **5.1** Symmetric per **architecture**, not per cell. Identical grid
  `lr_scale ∈ {0.25, 0.5, 1.0, 2.0}`, identical 500M-token probe, 1 seed,
  selection by dev CE at probe end. Full sweep published. No arm gets a hand-tuned
  schedule. Do **not** inherit lr_scale 1.0 vs 1.5 — that compensation for the
  Noam `d_model**-0.5` term was never validated.
- **5.2** If a selected value lands at a grid **endpoint**, extend the grid one step
  and re-probe. Committed here so it is not a post-hoc rescue.
- **5.3** Methods note: ✅ the optimizer is constructed with `lr=0.0` (trainer.py:76,
  comment "will be set by scheduler") and the scheduler overwrites
  `param_group["lr"]` every step, so the `lr: 0.0007` field in both configs is
  **dead** and `lr_scale` is the only real handle.

## 6. Reporting, frozen before results are seen

- **6.1** Per cell: total non-pad target tokens at convergence; tok/param against
  the re-measured reference; **optimizer** steps (not micro-batches); measured mean
  non-pad target tokens per optimizer step; **epochs over corpus**; wall clock; MFU
  from the corrected token count.
- **6.2** The headline contrast reported **three ways off the identical curves**:
  (a) post-hoc convergence argmin [primary]; (b) equal total non-pad target tokens
  at the minimum convergence-token count across cells [co-primary]; (c) equal
  tokens/param [third slice]. The paper states explicitly whether the **sign** of
  the Big−Base effect is invariant across all three, and if not, where it inverts.
- **6.3** Null rule, frozen: any Big−Base difference smaller than 2× the pooled
  within-cell seed standard deviation is **reported as null**, not as a trend.
  Per-seed values reported, never means alone.
- **6.4** Metrics: chrF2 and COMET-22 primary on newstest2014, BLEU secondary with
  the full sacreBLEU signature verbatim (✅ `src/evaluate.py:23` pins
  `tokenize="13a"` for non-zh). Validation cross-entropy in nats/token as the
  scaling-side quantity.
- **6.5** Checkpoint selection: report **both** the single checkpoint at argmin and
  the average of the 5 checkpoints centred on it. Both, regardless of which wins.
- **6.6** A section titled "Deviations from pre-registration" appears in the
  submitted paper whether or not it is empty.
- **6.7** Every pre-registered cell is reported regardless of outcome.

## 7. Release

Every dev-CE and dev-BLEU trace, every cell, every seed, against cumulative
non-pad target tokens, as CSVs; plus the tagged decision script, the trainer
patch, and all config diffs. The decision script consumes only released traces and
emits the convergence points and the Big-vs-Base verdict with no human input, so
any reviewer can re-derive the result **under their own preferred stopping rule**.

---

## E0.4 — what does the QE top band select? (pre-registered 2026-09-20, before the runs)

E0.3 returned NO-GO, and the lr-1.0 control showed that half the rejected paper's effect
is the learning rate while the remainder is real but contradicts the domain explanation
(`phase0/README.md`). The surviving hypothesis is about the selector, not the domain:

> **H.** CometKiwi's top band selects text that is *easy to translate* — short, formulaic,
> lexically thin — rather than text that teaches, and it is that property, not the domain
> mix, that makes ft_topk worse than a matched random set.

Measured on the E0.3 sets before any new run (926,999 rows each, 200,000-row sample for
the sampled statistics): type/token ratio 0.007 (top) vs 0.016 (random) vs 0.034 (bottom);
duplicate source rows 4.14% vs 0.97% vs 7.44%; repeated 8-grams 5.93% vs 4.66% vs 3.84%;
source-length p90 35 vs 47 vs 40 tokens. A UN-heading regex fired *less* often in the top
band (0.22% vs 0.63%), so "UN boilerplate" is not the right description; "low-diversity
short text" is.

**The test.** Split ft_topk in half on a statistic fixed here, hold size and length fixed,
and fine-tune both halves at lr_scale 1.0 (the regime where the effect exists), 3 seeds each.

- **Statistic (per row):** `rep(row)` = the fraction of the source's 8-gram tokens that
  occur in at least one other row of ft_topk. Computed over the whole set, lowercased,
  `\w+` tokens; rows shorter than 8 tokens take `rep = 0`.
- **Split:** rows above the median `rep` form `ft_topk_rep`; rows at or below form
  `ft_topk_div`.
- **Length matching, amended 2026-09-20 before any run of this probe.** The rule first
  written here — discard from the longer-tailed half until mean and median agree within 1%
  — does not converge, because repetitiveness and shortness are the same thing in this data
  (median `rep` is 0.0: more than half the top band contains no repeated 8-gram at all, and
  those rows average 28.6 tokens against 19.0 for the repetitive half). Trimming long rows
  shrank the diverse half to 117,338 rows while the gap stayed at 34%. The amended rule is
  **stratified matching**: bucket both halves by source token length and take
  `min(n_div, n_rep)` rows from each bucket on both sides (random within bucket, seed 42),
  which equalises the length histogram exactly rather than approximately. The resulting
  size is whatever the histograms overlap allows — reported, not tuned. Recorded here
  rather than silently applied: the change was forced by the data, not by a result, and no
  model had been trained on either half when it was made.
- **Runs:** `ft_topk_rep` and `ft_topk_div`, seeds 42/1/2, lr_scale 1.0, everything else as
  in E0.3 (same base checkpoint, same steps, same eval).

**Predictions, stated before the runs.**

1. If **H** holds: `ft_topk_rep` degrades newstest2014 more than `ft_topk_div`, by more
   than the seed-noise floor, and `ft_topk_div` is closer to (or better than) ft_random.
2. If the two halves are indistinguishable: **H is refuted** — the damage is a property of
   the whole top band (the 0.02-wide score plateau), not of repetitiveness within it.
3. If `ft_topk_div` is *worse*: **H is refuted the other way**, and the story is that QE
   rewards something else entirely; report that and stop this line.

Secondary, reported but not gating: the same two sets on heldout_un, and the applied-token
imbalance between the halves (length matching should keep it under the 2% threshold).

This is E0.4: an explanatory probe, not a gate. It cannot revive E0.3's NO-GO verdict.

**Implementation and provenance note (2026-09-20, not part of the rule).**

- `phase0/e04_split.py` keys 8-grams by a 64-bit BLAKE2b digest rather than builtin
  `hash()`, whose salt varies per process. Verified on the training box: the rewritten
  script reproduces the split that was already there **byte for byte** (`ft_topk_div.en`
  `a14b8b76…`, `ft_topk_rep.en` `725ef594…`), so the change costs nothing and removes a
  per-process dependence.
- The split is **not** reproducible across Python versions: the same script under Python
  3.14 on the Mac gives 242,022 rows per half with `len_mean` 26.32221 against 26.32213
  under the box's Python 3.10, because `\w+` follows the interpreter's Unicode database.
  The run of record is the box's, built before any model was trained on either half.
- The split's input is `data/phase0_lr1/ft_topk.*` — the control sets rebuilt on the box
  for the lr-1.0 probe, which is the experiment E0.4 is compared against. That set differs
  from the E0.3 primary arm's `ft_topk` by **exactly one row of 926,999** (a tie at the
  selection boundary: the primary arm scored the whole stream, the box's rebuild carried
  `nan` outside the reused pool). Every set statistic is identical to 16 digits —
  `qe_mean` 0.8991541266441345, and the same source counts — so the two are the same
  experiment; the one-row difference is recorded rather than hidden.
- **Deviation from decision D8 (`ft_checkpoint: avg-last5`), recorded before the runs.**
  E0.4 evaluates the *final* checkpoint with `--keep-last 1`, because it is read against
  the lr-1.0 control, which was run that way (`--ft-ckpt final`). Averaging here and not
  there would confound the comparison. D8 governs the E0.3 gate; E0.4 is not a gate.


## E0.3 decisions required before controls

**Status: FROZEN 2026-09-18**, before any stage-2 result existed. The chosen values are in
`phase0/e03_decisions.json` (sha256 `e3b299b9c3f805c3749cffdb1692a64a5d88f773f00bf5044c2446ef7b9c6f93`):
pool `reused`; n_ft 1,000,000; heldout_domains `["un"]`; indomain `["heldout_un"]`;
exclude_pretrain_from_heldout `true`; lr_selection_rule `flat-slope` with tolerance 0.3 BLEU;
score_mode `full` with the audit's calibration thresholds and `jaccard_min_k` 500;
ft_checkpoint `avg-last5`; budget `steps`; target_tokens `null`; loss_spike_ratio 0.
Amendment **X1**, made the same day and before any result: criterion 2 now requires the
in-domain gain to exceed that set's ft_topk seed sd, mirroring criterion 1's noise floor
(`e03_decide.py`, tests/suites/decide_noise_floor.py). Dropping Europarl from the gate is a
recorded deviation forced by D4: every candidate pair was in the pretraining corpus.
The sections below keep each option and its evidence as they stood when the choice was made.

These choices are frozen in
`phase0/e03_decisions.json` (template: `phase0/e03_decisions.example.json`, every value
`"CHOOSE"`), committed and git-tagged **before** `rental_setup.sh score`. `score`,
`controls`, `stage1`, `stage2` and `gate` refuse to run without a complete, valid file
(`python phase0/e03_decisions.py check phase0/e03_decisions.json` lists every missing or
invalid entry), and `stage1`/`stage2`/`gate` refuse a file whose sha256 differs from the
one `controls` recorded. Each choice changes the reserved held-out pool, the rng stream or
the evaluated checkpoint, so none can be changed after `controls` without rebuilding
everything downstream.

Enforcement (2026-09-13 fix round): the file must be tracked, identical to HEAD, contained
in a tag and on a remote branch of the box's clone (`DECISIONS_UNTAGGED=1` proceeds and writes
a deviation). The `score` done marker is keyed on `score_mode` + `calibration` only
(`D_SCORE_SHA`), so other keys can still be edited between `score` and `controls` without
rescoring; a scoring-key change after `score` stops unless `SCORE_REDO=1`. After E0.3 results
exist, any change stops `score`/`controls` unless `DECISIONS_AMEND=1`, which records the
old/new sha and changed keys in `results/phase0/deviations.txt`; `controls` appends every
frozen version to `results/phase0/decisions_history.tsv`.

Numbers are measured unless marked *derived* or *not measured*; sources are the
2026-09-13 audit (`wf_result.json` findings F2, F8-F11, F17 and the verified training-path
findings) and the implementation lanes. "Inside v1" percentages use the first 9,312,233
rows of the fixed v2 corpus as a proxy for the v1.1 pretraining corpus and are
**approximate**: v1 is not a byte prefix of v2_fixed (first difference at line 13,903).
Verbatim pair checks against the published v1.1 corpus are exact.

### D1. `pool` — which rows ft_topk / ft_random / ft_bottom and the held-out sets are drawn from

| option | what it is | measured consequence |
|---|---|---|
| `"reused"` | the 16,646,992 rows whose old CometKiwi score is reused (`pool_mask_reused.npy` in bundle v2) | top-1M score band 0.895937-0.914479, identical to the paper's 0.8959-0.9145; top-1M is UN 78.3%, Europarl 15.6%, commoncrawl 5.7%, news-commentary 0.2%, giga-fren 0.0%. Random 1M: UN 69.6%, commoncrawl 18.4%, Europarl 11.6%, giga-fren 0%. Bottom 1M: commoncrawl 69.0%. Inside v1 (approx.): top 49.8%, random 56.0%, bottom 81.9%. ft_random is a within-mix control, not a novel-domain arm. Needs no new scoring for E0.3 |
| `"full"` | all 38,275,284 rows of the corrected corpus | random 1M: giga-fren 55.5%, UN 30.4%; 24.3% inside v1 (approx.). The 21,628,292 unscored rows are 98.12% giga-fren and 0.96% news-commentary, so ft_topk's composition is unknown until they are scored. ft_random becomes majority data the baseline never saw (a domain AND novelty contrast). Requires `score_mode` reuse or full |

No pre-registered default exists: the builder used the whole scored file only because no
other file existed. Mechanism: `e03_build_controls.py --pool-mask`.

### D2. `n_ft` — FT set size before dedup

Builder default 1,000,000. The paper's set was top-1M then exact-pair dedup = 931,366 unique
pairs on the old corpus. top-1M is 2.61% of the full pool, 6.0% of the reused pool, and
3.3% in the paper. Unique size on the rebuilt corpus: **not measured** (recorded in
`manifest.json` when controls run). Options: 1,000,000 (the rule as written), or a fraction.

### D3. `heldout_domains` — which held-out sets are reserved (in draw order)

Pre-registered default `["un", "europarl"]`. Options: that, or add reported-only sets,
e.g. `"giga-fren"` (21,226,420 rows, 0 inside v1) or `"commoncrawl"` (only ~46,274-46,328
rows outside v1, approx.). Adding a set changes the reserved pool and the rng stream, so
every FT set changes. Names must match the e01 report exactly.

### D4. `exclude_pretrain_from_heldout` — draw held-out pools only from pairs/sources/targets absent from v1.1

Measured on the default picks (RandomState(42)): **heldout_europarl 2000/2000 pairs occur
verbatim in the baseline's pretraining corpus (2000/2000 source strings); heldout_un
793/2000 pairs (826/2000 sources).** Europarl: 1,930,016 of 1,930,749 rows inside v1
(approx.), leaving ~733, so with `true` heldout_europarl will most likely fail the
builder's pool-size exit (2,000 needed); UN has ~7.26M rows outside v1 (approx.).
Options: `false` (pre-registered behaviour; criterion 2 then partly measures re-exposure),
or `true` (needs `data` to have rebuilt v1.1, and an external Europarl-domain set or
dropping Europarl from D5). Mechanism: `--pretrain-src/--pretrain-tgt`.

### D5. `indomain` — criterion 2 wording

The documents disagree: `phase0/README.md` item 2 and WMT2027_PLAN.md say "improves the
UN/legislative eval" (UN only); `e03_decide.py`'s default `--indomain
heldout_un,heldout_europarl` requires a gain on **every** listed set. Options:
`["heldout_un"]` (Europarl printed as `[aux]`, not gating) or
`["heldout_un","heldout_europarl"]`. Each listed set must be in D3.
**Chosen: `["heldout_un"]`.** Criterion 2 was `gain > 0` with no noise floor in either
option; amendment X1 (2026-09-18, before any result) changed `e03_decide.py` so the gain
must exceed that set's ft_topk seed sd. That is a code change, not a value in this file.

### D6. `lr_selection_rule` and `lr_tolerance_bleu` — stage-1 LR selection

The rule was worded two ways and no code applied it: "largest lr_scale whose newstest
BLEU is flat" (README, e03_decide, plan) vs "largest value whose BLEU does not decline
monotonically from the first eval" (RUNBOOK, rental_setup.sh, run_stage1.sh). Stage 1 logs
10 validation evals (newstest2013) at 106K..115K (*derived* from trainer.py and the
matrix config; not run). A strictly monotone fall across 10 noisy beam-BLEU points is rare,
so the literal "no monotone decline" rule tends to accept lr_scale 1.0 even on a clear
downward trend. `phase0/e03_select_lr.py` applies exactly one of:

| rule | passes when | tolerance |
|---|---|---|
| `no-strict-monotone-decline-from-first-eval` | not every consecutive eval is lower (the wording the runner printed; script default) | `null` |
| `flat-endpoints` | \|last - first\| <= T | T BLEU |
| `flat-slope` | \|OLS slope\| x (step span) <= T | T BLEU |
| `flat-vs-baseline` | \|last - B\| <= T, B = pre-FT newstest2013 BLEU written by `accept` | T BLEU |

Selected = largest passing lr_scale; no rung passing stops the run. The regression
fixture 29.9, 29.6, 29.7, 29.4, 29.5, 29.3, 29.2, 29.25, 29.1, 29.0 selects 1.0 under the
first rule and a smaller rung under `flat-endpoints`, T = 0.3.

### D7. `score_mode` and `calibration`

| option | GPU | measured / derived consequence |
|---|---|---|
| `"none"` | none | v2_scored.tsv holds reused scores and the literal `nan` for 21,628,292 rows. Valid only with `pool` = `"reused"` |
| `"reuse"` | 21,628,292 pairs; ~10 GPU-h on one 5090 (*derived*: 9.9 h at the paper's 13.8 h / 30.1M, 10.4 h adjusting for 4.8% longer pairs) | reused vs new split almost exactly by source: europarl 1,930,749 / 0 new, commoncrawl 3,055,442 / 0, un 11,598,490 / 0, news-commentary 26,268 / 208,139, giga-fren 4,351 / 21,222,069. Any old-run vs new-run offset becomes a per-source shift in top-k selection; nothing can measure it in this mode |
| `"full"` | 38,275,284 pairs; ~17.5 GPU-h on one 5090 (*derived*, same rate) | uniform scores; `rescore_plan.py calibrate` compares new vs the 16,646,992 reused scores per source and **stops `score` unless it passes** |

`calibration` (object iff `"full"`; `null` otherwise) holds `max_mean_diff`,
`max_p99_diff`, `min_spearman`, `min_jaccard`, `min_rows`. The audit's proposals are
2e-4 / 2e-3 / 0.999 / 0.99 / 1000; they are not pre-registered. Density near cut-offs on
the reused scores (non-NaN `reuse_scores.npy`, 16,646,992 rows; threshold = value at descending
rank round(frac × 16,646,992); count of |score − threshold| <= 0.001): top-30% threshold
0.884624 with 812,089 rows within +/-0.001; top-5% threshold 0.896701 with 414,972 rows within
+/-0.001 (41,429 within +/-1e-4, audit measurement). With only 4,351 reused giga-fren rows
the top-5% set is ~218 rows, so two swaps already fail Jaccard 0.99. Multi-GPU throughput:
**not measured**. The paper's scoring stack is recorded only as "PyTorch 2.8 nightly (CUDA
12.8)"; the rental pins unbabel-comet 2.2.7, pytorch-lightning 2.5.5, transformers 4.57.1,
numpy 1.26.4 and writes per-shard metadata. The scorer downloads Unbabel/wmt22-cometkiwi-da at
the pinned commit 1ad785194e391eebc6c53e2d0776cada8f83179a and refuses unless
microsoft/infoxlm-large main (tokenizer/config) resolves to d616d637f0720deda963cebbfc630657d2b7d3ae
(both HF API, read 2026-09-13); both revisions are identity fields in the per-shard meta.
A stack change inside a shard (on resume) refuses unless `SCORE_ALLOW_STACK_CHANGE=1`
(deviation); a stack difference between shards (`stack_mixed`) stops `full` before
calibration (no override) and is a warning in `reuse`.

**Which threshold binds (audit F3; nothing chosen here).** At the proposed values the top-k
Jaccard is the binding check and p99 |diff| is roughly an order of magnitude looser: adding
Gaussian noise σ = 1e-4 to the reused scores gave top-5% Jaccard 0.980 (fails 0.99) but p99
|diff| 2.6e-4 (passes 2e-3) (audit measurement on `reuse_scores.npy`). Roughly,
Jaccard ≈ 1 − c·density·σ/k, with density = rows per unit score at the cut-off. In `full`
mode the selection uses only new scores, so a calibration failure says the new stack differs
from the paper's run, not that the new scores are internally non-uniform. The same-stack noise
floor is not measured yet: `SCORE_SMOKE=1` now writes its max and p99 |sharded − single| into
`v2_scored.meta.json` (`smoke`), so it can be known before thresholds are frozen. Options for
the user: loosen `min_jaccard`; gate Jaccard only where the top-k set has at least K rows
(optional `calibration.jaccard_min_k`, absent = 0 = every k gated, today's behaviour); or
report calibration without gating.

**If calibration fails after the full run (user decision, not pre-registered).** `score`
keeps the scores (`v2_scored.calibration_failed.tsv` + sidecar). The two options are: (a)
rescore after changing the stack (plain `score`, ~17.5 GPU-h derived); or (b) change the
thresholds and run `RECALIBRATE=1 score`, which re-runs only `calibrate` on the kept TSV (sha
checked against the sidecar; unchanged thresholds refused) and records "thresholds changed
after the first calibration" with old/new values in `results/phase0/deviations.txt`. Nothing
in the pipeline picks between them; the default path is (a). A calibrate exit 3 (invalid or
unreadable input) never discards scores or shards.

### D8. `ft_checkpoint` — which fine-tuned checkpoint the gate evaluates

`"final"` (the behaviour the gate was written with; keep_last 1) or `"avg-last5"`
(step_108000..114000 + final.pt averaged; keep_last 4). The baseline is the averaged HF
release in both. The paper itself puts averaging at +0.2-0.4 BLEU (05_sft.tex:15); the
seed sd is estimated at ~0.1-0.2 BLEU (not measured for E0.3). With `"final"`, criterion
1's baseline check leans toward "degrades" and criterion 2 against "improves";
`e03_decide.py` prints a warning. PROTOCOL 6.5 already requires reporting both for Phase 1.

### D9. `budget` / `target_tokens` and `loss_spike_ratio`

`budget`: `"steps"` (pre-registered "matched size and identical schedule": every arm stops at
115,000 micro-steps) or `"tokens"` (trainer-patch token gate at `target_tokens` applied
target tokens, evals every target/10, backstop 125,000 micro-steps). The sentence cap
(192) binds whenever a batch's longest sentence is under 128 tokens, so per-step target
tokens follow each condition's mean length; the size of the imbalance is **not measured**.
`e03_collect.py` records applied/dropped tokens per run and warns above a 2% max/min
applied-token ratio or 1% dropped. `loss_spike_ratio`: `"inherit"` (1.3 from the base
config; dropped batches still count as steps) or `0` (guard off, PROTOCOL 1.3's Phase 1
setting).

---

## Surviving attacks, to be conceded in the paper rather than left for a reviewer

1. **Regularization is confounded with capacity, and no protocol here fixes it.**
   ✅ Dropout is 0.1 in both configs across a 3.5x capacity difference, so Big
   overfits the capped corpus sooner and "trained to convergence" partly measures
   time-to-overfit. Holding dropout fixed is defensible but it *is* a choice.
   Mitigation: name it in Section 3, and add dropout ∈ {0.1, 0.3} as a secondary
   axis for Big on the capped cells. This is the one attack RTO-PA has no
   structural answer to — do not leave it unaddressed.
2. **The argmin is still a selection over a stochastic process.** Run lengths
   differ, so the number of evaluations differs, so winner's-curse magnitude
   differs by cell in a way correlated with capacity. Median smoothing and CE-over-
   BLEU shrink it; they do not remove it. Mitigation: also report dev CE at a fixed
   token grid common to all cells, and report the seed-spread of the argmin
   *location*.
3. **Fixed modest corpus + overshoot = heavy repetition.** Running to 4x the
   provisional point on a 1M-pair corpus means many epochs, and memorization
   confounds the late trace differently for 209M than 60M. §6.1 makes it visible;
   visibility is not a fix. Report train−dev CE gap alongside the CE trace.
4. **No precedent for the packaged protocol.** Pre-registration is close to absent
   from MT methodology. The literature supports the *components* — per-configuration
   budgets, early stopping on held-out loss, publishing learning curves — not the
   package. Present RTO-PA as a synthesis of standard components; do **not** claim
   it as an established named protocol. Overreach there is exactly what produced
   the last rejection.
5. **The LR probe length is a degree of freedom.** 500M tokens may be too short to
   rank schedules over a 20B-token run. Mitigation: re-probe the two best
   `lr_scale` values at 5B tokens on one cell and report whether the ranking held.
6. **Originality is not solved by any alignment protocol.** ← *most likely to be
   underweighted.* The prior scores were Originality 1/4/2: the paper was not
   rejected solely for the confound, it was rejected for being an uninteresting
   null *with* a confound. RTO-PA makes the null **defensible**; it does not make
   it **interesting**. The contribution has to be the **curve** — whether and where
   the Big−Base sign inverts as a function of budget, and whether that inversion
   point is invariant across the three alignments (§6.2). That is only visible
   under RTO-PA, which is a further reason to choose it, but **the paper must be
   framed around it rather than around the null.**

## A non-reviewer risk

Every literature citation produced during this design carries an explicit caveat
that the papers were **not opened** — including the Vaswani step counts and D/N
arithmetic. Combined with the two arithmetic errors confirmed in Phase 0, this
project's demonstrated failure mode is asserting unverified numbers and unverified
attributions. **Every citation must be opened and every number re-derived on the
real cache before submission.** That is not a hypothetical risk; it has already
materialized twice.
