# Phase 0 — Kill Gate (Month 1)

Decides whether the WMT 2027 program is worth 11 more months. Costs ~1 GPU-day
plus CPU. **Nothing here requires renting a machine.**

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
- (e) held-out UN/legislative **and** Europarl test sets, to show the in-domain
      *gain* the domain story predicts

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
# choose the largest lr_scale whose newstest BLEU is flat, then:
bash configs/phase0/run_stage2.sh 0.15     # 9 runs: 3 conditions x 3 seeds

# evaluate EVERY run on newstest2014 + heldout_un + heldout_europarl, then:
python phase0/e03_decide.py --results results/phase0_bleu.tsv
```

`e03_decide.py` exits 0 = GO, 1 = NO-GO, 2 = cannot decide (missing inputs).

### Pre-registered decision rule

Proceed to the full program **only if**, at the flat-LR setting:

1. top-k-QE FT degrades news-domain eval **more than** the matched random
   control, by **more than** the seed-noise floor, **and**
2. top-k-QE FT **improves** the UN/legislative eval.

Otherwise the motivating observation is an optimization artifact — most likely
the discarded Adam moments (see the corrected LR section above), which raise the
*effective* update magnitude at the FT boundary even though the nominal LR is
continuous. In that case: publish a short correction note and rebuild the paper on
the compute-alignment work alone, which does not depend on it.
