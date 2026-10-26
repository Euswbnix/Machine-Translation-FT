# Draft pre-registrations: E0.6, E0.7, E1.0

**Status: DRAFTS, written 2026-10-16. Nothing here is registered.** Each section is written in the
form of PROTOCOL.md "E0.5" so that, once the author has decided its OPEN DECISIONS, it can go into
PROTOCOL.md with each open item replaced by its choice. It is then committed and tagged before the
first run, as E0.5 was (tag `e05-prereg-v1`). Every number below comes from one of the sources
listed next. A number worked out from them is marked *derived*, and anything that could not be
verified is marked **to check**.

**This file is the only place** where the outcome rules of E0.6, E0.7 and E1.0, the reading of
each outcome, and the numbered OPEN DECISIONS are stated. `WMT2027_PLAN.md` (schedule, cost,
dependencies) and `eamt2027_outline.md` (what the paper says in each slot) point here by section
and decision number, and `reading_list.md` names the decision number for each of its items. A
change to a rule, a reading or a decision is made here only.

Sources:
- `phase0/README.md`, "E0.5 RESULT (2026-10-16)" and "E0.4 RESULT"
- `PROTOCOL.md`, "E0.4" and "E0.5" (also D3, D4, D8 and D9 of the E0.3 decisions, which E0.5 inherits)
- `phase0/PRIOR_ART_E04.md`
- `phase0/next/groundwork.json`: "notes" (10 papers), "vocab", "ende" and "llm" memos
- `phase0/next/e06_feasibility.md` (`vocab/` and `ende/` below mean `phase0/next/vocab/` and
  `phase0/next/ende/`; bulk files in `~/mt_local/phase0_next/`, sha256 in `phase0/next/SHA256SUMS`)
- the previous `WMT2027_PLAN.md` (in git history), keeping only what survives E0.3, and the new
  plan draft that replaces it
- Code that was read to pin down definitions: `phase0/e05_build.py`, `phase0/e04_split.py`,
  `phase0/e03_build_controls.py`, `phase0/rebuild_corpus.py`, `phase0/probe_stats.py`,
  `phase0/e05_run.sh`, the vocab memo's scratch scripts (`vocab/common.py`, `a1.py`, `a5.py`,
  `a8.py`, `a8_lib.py`, `a12.py`), and the ende memo's `ende/s1_labels.py`, `s1_report.json` and
  `s2_bands.py`
- a scratch rebuild of `rep_dedup` and `div_dd` to the definitions below, from the pinned
  `ft_topk` files (Mac, Python 3.14.3, 2026-10-16), made only to check the feasibility file; its
  values are marked "scratch rebuild" and are replaced by `e06_build.py`'s dry run

## What is fixed before these drafts

- **Venue.** EAMT 2027, research-technical track, up to 10 pages. Its 2027 call is not out yet.
  For EAMT 2026 the deadline was 18 March 2026 (extended to 27 March), and the research-technical
  track accepted 31 of 55 submissions (56%); these 2026 figures come from a web lookup during
  planning and are not in the sources listed below (**to check** against the EAMT 2026 call or
  proceedings). If 2027 keeps 2026's calendar, results are needed by mid-March 2027 (**to check**
  when the call appears). WMT 2027 (summer 2027) is the extended follow-up.
- **E0.5 result.** Q1 YES: inside UN text the repetitive half is 2.63 BLEU worse than the diverse
  half (t = 33.3). Q2 outcome B: the diverse half is 0.58 BLEU worse than matched random UN text
  (t = −7.0). E0.4 replicates (3.04 BLEU, against E0.4's 2.73). Secondary: on heldout_un both UN
  halves lose 2.1-2.7 BLEU against the baseline while random UN text gains 0.20.
- **What "C1" means here.** In these drafts C1 is the Q1 result: inside the QE top band, the
  repetitive half costs more out-of-domain translation quality than the diverse half (E0.5's Q1:
  en-fr UN text, CometKiwi-22, one base model, measured by newstest2014 BLEU). Each experiment
  below names its own measure. This is not `PRIOR_ART_E04.md`'s labelling, where C1 is the
  selection claim (QE top-k preferentially selects repetitive, lexically thin text) and the damage
  claim is C2.
- **Novelty.** The bare phenomenon (a QE top slice doing no better than random) is already
  published: Waldendorf et al. 2025, Chimoto et al. 2024, Tan et al. 2024 and Finkelstein et al.
  2024. NRC (Lo et al. 2018; Bernier-Colborne & Lo 2019) remarked that top-ranked sentences are
  redundant. No text produced from these experiments may claim "first to observe". The
  contribution is the controlled account of *why*.
- **Objections that must be answered by experiments, not by prose:**
  1. Saunders & DeNeefe 2024: forgetting is predicted by the vocabulary coverage of the
     fine-tuning set, so "repetitive" may only stand in for low coverage. E0.6 tests this for the
     coverage measure chosen in its OPEN DECISION 1, provided `div_narrow` can be built within ±2%
     (feasibility run, 2026-10-19, descriptive: it can under OD 1(a), (b) and (d), under (b) only
     with OPEN DECISION 2 = c2; (c) is not built by the script).
  2. "Just deduplicate" (Koehn et al. 2018; Waldendorf et al. 2025). E0.6 tests the
     exact-duplicate form. Near-duplicate removal (Waldendorf et al.'s BiCleaner
     `--aggressive_dedup`) is not in E0.6: the plan places it in W4 (WMT 2027), beside the NRC
     arms.
  3. Peter et al. 2023: QE filtering beats random when keeping 50% and training to convergence.
     Every claim is therefore scoped to an extreme top band (about the top 6% of the reused en-fr
     pool) and to fine-tuning. E0.7 and E1.0 extend the scope; none of the three experiments
     licenses a claim about QE filtering in general.
  4. The NRC redundancy re-ranking is the obvious baseline for any *remedy*. None of the three
     experiments tests a remedy, so no text may claim one until an experiment with that baseline
     has been registered and run. The plan places the remedy and its NRC arms in W4 (WMT 2027).
- **Order:** E0.6, then E0.7, then E1.0. GPU work runs on rented cards, and the author rents when
  told. No script in these designs handles credentials. Anything gated (FLORES+, Gemma) is accepted
  on the author's own Hugging Face account, and the author places any token on the box.
- **When to decide.** The dates are `WMT2027_PLAN.md` §4 and §7, not restated here. In order:
  E0.6's decisions at the end of the reading week (26 October to 1 November 2026), so that its
  coverage measure (OPEN DECISION 1) is chosen after Saunders & DeNeefe have been read; then the
  tag `e06-prereg-v1`, and E0.6 runs on R1, rented once the tag is pushed (plan §4: tag target
  Tue 3 November; if the tag slips, at the start of the R2 window), not on 24-25 October. E0.7's
  decisions come next and E1.0's last, because E1.0's vocabulary-matched set depends on E0.6's
  result.

**What survives of the previous `WMT2027_PLAN.md`.** Its thesis that QE filtering is covert
domain selection is refuted, not by the E0.3 gate verdict itself but by the in-domain results of
the lr-1.0 follow-up to E0.3 and of E0.5 (`phase0/README.md`; plan §0.3 item 1), and the
domain-shift statistic, the dose-response fit and the R/N/Q/D/C grid go with it.
Five things survive:
- multi-seed noise floors, reported beside every contrast (its E4.1);
- evaluation with a metric outside the COMET family, MetricX-24, alongside chrF and BLEU (its
  E2.5 and E3.2);
- an LLM arm that includes full-fine-tuning runs, so that "LoRA hides it" is answered with data
  (its E3.1, rescoped in E1.0 below to about 53 GPU-hours against its 100);
- a second language pair (its E2.6, which becomes E0.7);
- pre-registration with a timestamped tag and no arXiv.

One correction carries over. The previous plan cites "Zhou & Savova (arXiv 2503.07329)". The arXiv
page lists Bui, Savova and Wang (IJCNLP-AACL 2025), and its abstract does not mention LoRA. Nothing
below relies on that paper, and the new plan (§6.2) already makes the correction.

**Common to all three (inherited from E0.5 and restated in each section):**
- A gap counts only if it exceeds the seed-noise floor (the larger seed sd of the two sets) *and*
  is Welch-significant at 0.05, two-sided, using the exact critical value at the fractional
  Welch-Satterthwaite df (`phase0/probe_stats.py`, `contrast()`).
- A contrast that does not clear the rule is reported as **not detected**, with the smallest gap
  the rule could have called beside it (`bound` = max(noise, crit × se)). It is never reported as
  "equal".
- Every build is fail-closed and works from pinned inputs. The manifest is written last, and the
  run of record is built on the machine where it trains, with the interpreter recorded (the `\w+`
  tokens follow the interpreter's Unicode tables).
- All runs of one experiment are done on one machine in one session.
- Seeds, sets, step counts and statistics are fixed before any run and may not be changed after
  results exist.

---

## E0.6 — does the repetitive half cost beyond vocabulary coverage and beyond exact duplicates, and does a vocabulary-capped random set carry part of the gap to random text? (DRAFT 2026-10-16, not registered)

E0.5 showed, inside UN text and at matched size and length histogram, that the repetitive half of
the QE top band costs 2.63 BLEU more on newstest2014 than the diverse half, and that the diverse
half is still 0.58 BLEU worse than matched random UN text. `e05_build.py` matches only source, size
and length; exact-pair pretraining exposure (36.7 / 36.7%, PROTOCOL's dry run) and mean QE
(0.8990 / 0.8993, vocab memo) were measured nearly equal, not matched by design. Exposure is near-equal
as a share of rows; how often the base saw those pairs is not equal: the repetitive half's exposed pairs sit on more corpus lines (1.158 against 1.043), so per row the base presented its pairs 1.056 times against 0.951 (base-replay check, 18 October 2026); this belongs to the duplicate strand of the bundle. Three things it
left open are each a reviewer's first question:

1. **"Repetitive" is a bundle.** The two UN halves also differ in vocabulary (25,134 against 33,874
   source word types) and in duplicate source sentences (8.45% against 0%). Saunders & DeNeefe
   (EMNLP 2024) find that forgetting tracks the fine-tuning set's vocabulary coverage, with target
   coverage against COMET drop at Kendall tau 0.9 across the 5 de-en and 3 en-ja domains (pooled,
   n = 8; checked by the author against the paper, 2026-10-25). A reviewer will say repetitiveness is a proxy for low
   coverage.
2. **Duplicates.** Exact deduplication of the pool is standard (WMT18, Koehn et al. 2018), and
   Waldendorf et al. 2025 remove near-duplicates. The repetitive half's 8.45% duplicate sources may
   be what does the damage.
3. **Q2 is unexplained.** The diverse half has 58% of the word types of random UN text of the same
   size and length, so vocabulary is the obvious suspect. But measured since E0.5 (vocab memo), QE
   selection inside UN also changes other things: QE itself (0.8990 against 0.8629), word-for-word
   literalness of the French (0.783 against 0.714) and genre (narrative against procedural
   text). E0.5's random UN arm is **not QE-matched** to the halves (0.8629 against 0.8990 and
   0.8993). About a quarter of its rows also contain an 8-gram that recurs within its own parent,
   the same-size random set `ft_random` (`rep_rand` below, as `e05_build.py` computes it), so it
   never held repetition fixed.

E0.6 builds five new sets beside E0.5's three UN sets (six if OPEN DECISION 5(b) adds a ninth) so
that each strand can be varied with one other strand held fixed. Unless OPEN DECISION 5(b) adds
that ninth set, no pair holds coverage and duplicates fixed together;
neither C1 contrast matches French-side features such as literalness or recurring French 8-grams
(vocab memo); and the contrast S defined below does not separate QE level, literalness and genre.
The repetitive half stays a bundle that E0.6, as designed, does not fully separate.

**Design: eight sets (nine if OPEN DECISION 5(b)), five seeds, one machine.** Every set is UN text
only.

| set | built from (exact rules below) | rows | role |
| --- | --- | --- | --- |
| `ft_un_div` | E0.5's file, unchanged (sha256 pinned) | 214,489 | anchor |
| `ft_un_rep` | E0.5's file, unchanged | 214,489 | anchor |
| `ft_un_random` | E0.5's file, unchanged | 214,489 | anchor, bridge to E0.5's Q2 |
| `R0` | random UN rows of `ft_random` that contain no repeated 8-gram | 214,489 | clean random control |
| `A_narrow` | R0's pool, narrowed to the diverse half's vocabulary size | 214,489 | vocabulary-capped random set (narrower vocabulary, plus whatever else the caps change) |
| `div_narrow` | the diverse UN candidates, narrowed to `ft_un_rep`'s coverage | 214,489 | coverage control for `ft_un_rep` |
| `rep_dedup` | the repetitive UN candidates with duplicate sources removed | 198,333 | duplicates control |
| `div_dd` | the diverse side of `rep_dedup`'s length matching | 198,333 | partner of `rep_dedup` |
| `div_narrow_dd` (only if OPEN DECISION 5(b)) | the diverse UN candidates, narrowed to `rep_dedup`'s coverage at `rep_dedup`'s histogram | 198,333 | coverage-and-duplicates control for `rep_dedup` |

All 214,489-row sets share one exact source-length histogram, the one E0.5's two UN halves share.
`rep_dedup` and `div_dd` share a second histogram of their own. Their size, 198,333, is the
feasibility file's and a scratch rebuild's, with and without the build's step 2, which cannot
change it (see the build rule below); the dry run records the final size.

**Definitions shared by every build.**
- *Tokens:* lowercased `\w+` (`WORD = re.compile(r"\w+", re.UNICODE)`), as in `e04_split.py` and
  `e05_build.py`. Length means the source token count.
- *`rep` (top band):* E0.4's statistic as `e04_split.py`/`e05_build.py` compute it, over the whole
  of `ft_topk` (926,999 rows): per row, the fraction of its 8-grams (blake2b-64 keys) whose count
  over the set is above one. Rows under 8 tokens take `rep = 0`. The median is 0.0.
- *`rep_rand`:* the same statistic computed over the whole of `ft_random` (926,999 rows).
- *Diverse UN candidates:* UN-labelled rows of `ft_topk` with `rep = 0`, 486,475 rows.
  *Repetitive UN candidates:* UN-labelled rows of `ft_topk` with `rep > 0`, 227,406 rows. Source
  labels come from the pinned `ft_topk.source` and `ft_random.source`.
- *`maxrank(row)`:* the pretraining frequency rank of the row's rarest source word. Words are ranked
  by frequency in the v1.1 pretraining corpus (English side, lowercased `\w+`), 0 for the most
  frequent, with ties broken by first occurrence in corpus order. A word never seen in pretraining
  takes rank 10^7, and an empty row takes 0 (as `vocab/a5.py`).
- *Exact-histogram random draw* (E0.5's `match_random`): for every source length L, exactly as many
  rows as the target histogram has at L, drawn at random within the bucket with
  `random.Random(42)`, lengths in ascending order. **Any short bucket stops the build.**
- *Stratified length matching* (E0.4's rule, `e04_split.py`): bucket both sides by source length
  and take `min(n_A(L), n_B(L))` rows from each side, at random within the bucket, seed 42, built in
  ascending row order on both sides, as `e05_build.py` built E0.5's UN halves. `e04_split.py`
  orders the repetitive side by `rep`, and that order does not reproduce `ft_un_rep` (checked by
  rebuilding the UN split from the pinned inputs on the Mac, Python 3.14.3).
- *Source word types:* distinct source tokens in the set (`e05_build.py`, `lexicon()`).
  *French word types:* distinct French tokens in the first B French tokens of the set, with rows
  taken in a `random.Random(0)` shuffle (as `vocab/common.py`, `types_at()`); a set with fewer than
  B French tokens stops the build. For the 214,489-row sets B = 6,500,000; the smallest anchor,
  `ft_un_rep`, has 6,576,020 French tokens (counted on E0.5's file), a margin of about 1.2%. For
  the 198,333-row histogram (`rep_dedup`, `div_dd` and OPEN DECISION 5's `div_narrow_dd`) 6.5M is
  out of reach: the feasibility run counts 6,131,873 (`rep_dedup`) and 6,219,966 (`div_dd`) French
  tokens (descriptive; the earlier *derived* "about 6.08M" underestimated them, build spec A23). A lower
  B, proposed 6,000,000, is fixed before the dry run and applied to both sides of that pair. Both
  budgets are **to check in the dry run**, which records every set's French token count.
- *Duplicate sources:* `1 − (distinct source strings / rows)`, on the raw source line
  (`e05_build.py`, `duplicate_source_fraction`).

**How each new set is built.**

- **`R0`.** The pool is the UN-labelled rows of `ft_random` with `rep_rand = 0`: 450,709 rows with
  91,142 source word types. Draw `ft_un_div`'s exact histogram from it at random (seed 42). The
  tightest bucket holds 1.203x what is needed. R0 has no repeated source 8-gram and no duplicate
  source by construction: its histogram has no rows under 8 tokens, and a duplicate source of 8 or
  more tokens has `rep_rand > 0`.
- **`A_narrow`.**
  1. Count source and French word frequencies over R0's pool (450,709 rows, in ascending row order,
     ties broken by first occurrence).
  2. Keep only the pool rows whose source words are all among the 42,000 most frequent source words
     and whose French words are all among the 58,000 most frequent French words. This leaves 402,010
     rows.
  3. Within each source length L, every `ft_un_div` row takes the unused pool row with the nearest
     `log(1 + maxrank)`. The rule is `vocab/a8_lib.py`, `nn_match()`:
     - each pool row gets a tie-breaking key from `random.Random(42).random()`, drawn in ascending
       row order;
     - target rows are processed in decreasing distance from the median pool key at that length;
     - where the lower and upper neighbours are equally near, the lower one is taken;
     - a pool smaller than its targets at any length stops the build.
  4. Caps: 42,000 / 58,000 as built (OPEN DECISION 3; recommended: keep them). They were tuned once,
     before any run, to the diverse half's 33,874 source and 43,021 French types.
- **`div_narrow`.** The candidates are the diverse UN candidates (486,475 rows). The target is
  `ft_un_rep`'s exact length histogram and its coverage: 25,134 source word types and 32,684 French
  types at 6.5M French tokens, under the measure in OPEN DECISION 1. The rule is OPEN DECISION 2.
  Its recommendation was updated on 2026-10-19 to its rule c2, after the feasibility run showed
  that the rule first recommended, A_narrow's rule applied the other way round, cannot narrow at
  this histogram. That first rule:
  1. cap the candidates at the Ke most frequent source words and Kf most frequent French words of
     the candidate pool;
  2. nearest-neighbour match each `ft_un_rep` row on `log(1 + maxrank)` within its length (seed 42);
  3. take (Ke, Kf) from a grid fixed in the registered text (proposed: Ke and Kf in steps of 1,000).
     The pair chosen is the one that minimises the larger of the two relative type differences from
     the target. If no pair brings both within ±2%, the build stops.

  Measured feasibility (`e06_build.py --feasibility`, 2026-10-19, on the M5 Max reference
  interpreter from the pinned inputs; every option built for measurement only, not for training;
  `phase0/next/e06_feasibility/`): at `ft_un_rep`'s histogram, every bucket of 36 or more tokens
  holds no more diverse candidates than it needs, so any rule must take all 30,197 of those rows,
  and caps that keep them are no caps. Differences from the target (source / French types):
  - OD 2(a): +23.1% / +23.0% under any OD 1 measure; OD 2(b): +34.7% / +31.8%;
  - OD 2(c): with OD 1(b), −5.7% / +5.8% at its best point, so the build stops; with OD 1(a),
    +0.1% / +9.5% (French left free); with OD 1(d), −15.8% / −2.7%;
  - OD 2 c2: with OD 1(b), −0.3% / +0.3% (25,068 / 32,773; q = 0.725, λ = 0.55), the only rule
    within ±2% on both sides; with OD 1(a), +0.0% / −7.8%; with OD 1(d), −15.7% / −4.9%.

  Under OD 1(d) every rule passes its own test, because `ft_un_div` itself is within +1.9% of
  `ft_un_rep` on news-commentary coverage. The (c) family runs from 21,158 source types (q = 0) to
  33,858 (q = 1, equal to a random draw at the histogram); the earlier 20,284 and 34,070 had no
  script and are not reproduced. Every option keeps mean QE at 0.8990 (`ft_un_div`'s) and shares
  about 127,000 of its 214,489 rows with `ft_un_div` (126,844-128,334, the 30,197 forced rows
  among them). OD 1(c) is not built by the script (see OPEN DECISION 1). The dry run with the
  chosen decisions replaces these numbers.
  `div_narrow` has no repeated source 8-gram by construction, since every candidate has `rep = 0`
  over the whole band.
- **`rep_dedup` and `div_dd`.**
  1. Take the repetitive UN candidates (227,406 rows) in `ft_topk` file order and drop every row
     whose raw source string already occurred earlier in that list, so the first occurrence is kept.
     This leaves 207,561 rows (19,845 removed, *derived*).
  2. Apply the same rule to the diverse UN candidates, and record the count. It can remove only rows
     under 8 tokens: a duplicate source of 8 or more tokens shares all its 8-grams and so has
     `rep > 0`. The repetitive side has no rows under 8 tokens, so step 3 takes nothing from those
     buckets (`e04_split.py` skips an empty bucket without drawing), and step 2 cannot change
     step 3's result. (Scratch rebuild: step 2 removes 252 rows, all of 3-7 tokens.)
  3. Apply stratified length matching between the two lists, both in ascending row order (the
     definition above). This gives 198,333 rows per side:
     `rep_dedup` is the repetitive side and `div_dd` the diverse side. The scratch rebuild gives
     198,333 with and without step 2, as the argument in step 2 says; the dry run records the
     final size.
  4. `rep` is not recomputed after deduplication. The rows stay "repetitive" by the whole-band
     statistic, and what remains is measured as `rep_dedup`'s mean `rep` (the whole-band
     statistic, as in `ft_un_rep`'s 0.433). The feasibility file's 0.379 and 25,296 source word types are a
     feasibility draw, unrecorded (no script for it was found), and are not used. A scratch rebuild
     to the definition above (Mac, Python 3.14.3; 5,242,970 source words per side) gives mean
     `rep` 0.378 and 25,256 source word types for `rep_dedup`, and 33,098 for `div_dd`. These are
     scratch values, to be replaced by `e06_build.py`'s dry run.
- **`div_narrow_dd` (only if OPEN DECISION 5(b)).** `div_narrow`'s rule (OPEN DECISION 2), with
  `rep_dedup` in place of `ft_un_rep`: the diverse UN candidates, matched to `rep_dedup`'s exact
  length histogram and to its coverage under OPEN DECISION 1's measure (French at the lower
  budget B), within ±2%, or the build stops. Measured in the feasibility run: as for `div_narrow`,
  only OD 2 = c2 with OD 1(b) is within ±2% on both sides (OPEN DECISION 5).
- **Random draws keep top-band rows** at their base rate (E0.5's rule). The share of rows at or
  above the top-band threshold (QE ≥ 0.8959) is 9.1% in R0 and 11.7% in A_narrow. The manifest
  counts the exact overlap of every set with `ft_topk` and with each anchor. For example,
  A_narrow shares 136,916 rows with R0 (scratch: the intersection of the index files
  `vocab/R0.npy` and `vocab/A_narrow_42_58.npy`; the vocab memo prints it as "136,9k rows");
  `e06_build.py`'s dry run and manifest replace this count.

**Inputs are pinned.** The build refuses inputs whose sha256 differs from:
- `ft_topk.{en,fr,source}` and `ft_random.{en,fr,source}`: the six E0.5 pins in `e05_build.py`
  (`787339d0…`, `22263da1…`, `a4ad4586…`, `937d4d4a…`, `382e0de5…`, `fd675a19…`);
- `heldout_un.{en,fr}`: `f70e725e…`, `7daf654f…`;
- the three anchor files: the sha256 recorded in E0.5's manifest
  (`~/mt_local/phase0_final/e05_results/data/phase0_e05/manifest.json`);
- the v1.1 pretraining corpus `train.clean.en`, used only for `maxrank` (`7230adc6…`, as the
  feasibility run records it), and its French side `train.clean.fr` (`b0a6ba91…`), set-defining
  if OPEN DECISION 2 = c2.

**As a fail-closed check, the build recomputes E0.5's UN split from the pinned inputs and must
reproduce `ft_un_div` and `ft_un_rep` byte for byte.** It uses the stratified matching defined
above, in ascending row order; `e04_split.py`'s order reproduces `ft_un_div` but not `ft_un_rep`.
If the check fails, either the build departs from that definition or the interpreter's Unicode
tables differ from the box's 3.12.15, and the build stops. The build script must also reproduce
the scratch index files `vocab/R0.npy` and `vocab/A_narrow_42_58.npy` row for row on the
interpreter that wrote them, before it is moved to the box.

**Runs.** 8 sets x seeds **42, 1, 2, 3, 4** = 40 runs (45 if OPEN DECISION 5(b) adds the ninth
set), on one machine in one session. Every run uses:
- lr_scale 1.0 and spike guard off (D9);
- the final checkpoint with `--keep-last 1` (E0.4's recorded deviation from D8, kept for
  comparability);
- the same v1.1 averaged base checkpoint and the same 10,000 steps;
- the same evaluation: newstest2014 and heldout_un, `--max-src-tokens 254`.

At 10,000 steps the 214,489-row sets make 9.0 passes (PROTOCOL "E0.5"; the vocab memo's "about 11"
disagrees and is not used) and the 198,333-row pair about 9.7 (*derived*: 192 x 10,000 / 198,333).
The baseline is re-evaluated on the new box and recorded beside E0.4's and E0.5's 35.31 / 46.83. A
difference is reported, not a stop, because every contrast is within this one batch.

**Why five seeds.** E0.5's UN seed sds were 0.11-0.14 BLEU, and its rule could call gaps of 0.18
(Q1) and 0.19 (Q2). At a seed sd of about 0.13 the vocab memo puts the 80%-power gap at about 0.26
BLEU with five seeds and about 0.17 with ten. If narrowness is the cause, the vocab memo expects N
to be about the whole R0 − `ft_un_div` gap: at least E0.5's 0.58 BLEU, and probably more, because
R0 has no repetitive rows. A gap near 0.2 BLEU is about a coin flip, and gaps below about 0.26
BLEU have less than 80% power. Seeds may not be added after results exist.

**Measured on the sets before any run** (Mac, real data; the run of record is built where it
trains). The feasibility file is headed "measured 2026-10-17", but the file was written on
2026-10-16 (**to check** the date). "n.m." means not measured yet: the registered version must fill
those cells from a dry run of the fail-closed build. Cells marked "feasibility run" come from
`e06_build.py --feasibility` (2026-10-19, `phase0/next/e06_feasibility/`), which built every option
for measurement only and also rebuilt R0 and A_narrow at seed 42 with the sha256 of the scratch
index files; the dry run with the chosen decisions replaces them.

| set | rows | source word types | French types @6.5M | source repeated-8-gram share | duplicate sources | QE mean | literalness | source in pretraining | pair in pretraining |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ft_un_div` | 214,489 | 33,874 | 43,021 | 0.000 | 0.00% | 0.8990 | 0.783 | 36.8% | 36.7% |
| `ft_un_rep` | 214,489 | 25,134 | 32,684 | 0.433 | 8.45% | 0.8993 | 0.810 | 38.3% | 36.7% |
| `ft_un_random` | 214,489 | 58,441 | 69,320 | 0.087 | 0.23% | 0.8629 | 0.714 | 38.9% | 37.9% |
| `R0` | 214,489 | 62,000 | 72,574 | 0 | 0 | 0.8613 | 0.693 | 38.2% | 37.7% (feasibility run) |
| `A_narrow` | 214,489 | 33,562 | 43,703 | 0 | 0 | 0.866 | 0.698 | 38.3% (feasibility run; the vocab memo printed 38.2%) | 37.8% (feasibility run) |
| `div_narrow` | 214,489 | 25,068 (feasibility run, OD 2 = c2 with OD 1(b); target 25,134 ±2%) | 32,773 (same; target 32,684 ±2%) | 0 by construction | 0 by construction | 0.8990 (same) | 0.784 (same) | 37.1% (same) | 37.1% (same) |
| `rep_dedup` | 198,333 | 25,256 (scratch rebuild and feasibility run, OD 4(a); the unrecorded feasibility draw's 25,296 is not used) | 32,562 at B = 6,000,000 (feasibility run) | 0.378 (scratch rebuild and feasibility run; the unrecorded draw's 0.379 is not used) | 0.00% | 0.8993 (feasibility run) | 0.811 (feasibility run) | 37.3% (feasibility run) | 36.7% (feasibility run) |
| `div_dd` | 198,333 | 33,098 (scratch rebuild and feasibility run) | 41,813 at B = 6,000,000 (feasibility run) | 0 by construction | 0 by construction | 0.8990 (feasibility run) | 0.783 (feasibility run) | 36.9% (feasibility run) | 36.9% (feasibility run) |

Notes on the table:
- "Pair in pretraining" is E0.5's exact-pair measure and "source in pretraining" is the vocab memo's
  normalised-source measure. They are different statistics.
- "Literalness" is the vocab memo's word-for-word measure (a Dice dictionary from 1.5M pretraining
  pairs, 5-character prefix match). It is good for ranking and crude as an absolute value.
- `rep_dedup`'s and `div_dd`'s types are counted over 5,242,970 source words each, fewer than the
  214,489-row sets', so they are comparable with each other but not directly with `ft_un_rep`'s.

What follows from the table, and bears on how the outcomes may be read:
- **A_narrow narrows the vocabulary and leaves the measured QE-linked properties at the random
  level.** Its source types are within 0.9% of the diverse half's and its French types within 1.6%
  (*derived*). On per-row rarest-word rank it cannot be told apart from the diverse half (AUC
  0.505). Its QE (0.866 against 0.861), literalness (0.698 against 0.693), list-item rows (5.1%
  against 5.1%) and news-likeness of the words (−0.489 against −0.481) stay at R0's level, against
  the diverse half's 0.899, 0.783, 1.0% and −0.229. Not everything else is at R0's level: it holds
  more top-band rows (11.7% against 9.1%, below), and its French recurrence was never measured (the
  vocab memo's `a4.py` covers only the two halves, `ft_un_random` and R0); with 43,703 French types
  against R0's 72,574 its French may recur more than R0's.
- **S is a lump.** S lumps together QE level, more literal and more formulaic French, and the
  narrative-against-procedural genre shift. A detected S says QE selection costs something beyond
  source word-type count; it does not rule out narrowing, which S may partly absorb (next point),
  and it does not say which of these does the damage.
- **A_narrow does not lack the same words as the diverse half.** It loses random text's acronyms
  and document codes, while the diverse half lacks procedural words. Its coverage of news-commentary
  v9 tokens seen at least 5 times (95.29) lies between the diverse half's (94.98) and R0's (96.00).
  If narrowness only matters through news words, S may absorb part of a vocabulary effect.
- **Pretraining exposure is close but not equal.** On the source measure it is 38.3% for A_narrow
  and 38.2% for R0 (feasibility run) against 36.8% for the diverse half. This is reported, not matched (OPEN DECISION 7).
- **The `ft_un_rep` against `div_narrow` pair also differs in exact duplicates** (8.45% against 0%
  by construction), probably in French-side features (the repetitive half has a repeated French
  8-gram in 79.7% of rows against 23.0% for the diverse half, and literalness 0.810 against 0.783,
  vocab memo; for `div_narrow` under c2 with OD 1(b) the feasibility run gives literalness 0.784,
  and its French recurrence is not measured), and in whatever choosing rows by their rarest
  word's frequency shifts. The `rep_dedup` against `div_dd` pair is not coverage-matched (25,256
  against 33,098 source types, scratch rebuild and feasibility run) and differs in literalness
  (0.811 against 0.783, feasibility run); French recurrence is not measured for either set. No single pair in the eight sets holds both coverage and
  duplicates fixed while repetition varies (OPEN DECISION 5).
- **Two checks are still owed before the tag, both CPU work on the Mac** (OPEN DECISION 16; its
  status on 2026-10-19: piece types computed for the six E0.5 sets, R0 and A_narrow, still owed
  for the new sets; newstest2014 now on the M5 Max, its coverage check not yet run). The
  vocabulary was measured in `\w+` word types and the model sees SentencePiece pieces, so
  piece-type counts and piece-frequency profiles of every new set are re-checked. The vocab memo
  had no SentencePiece model, but the HF release `euswbnix/transformer-wmt14-enfr-base` ships it,
  byte-identical to the original machine's `spm_enfr_v1_fixed.model` (`d1c10ea80fb95984…`;
  `phase0/README.md`, "Findings from the original training machine", item 4). It is fetched from
  there and its sha256 checked, and the `sentencepiece` package is installed into
  `~/mt_local/venv` (not there yet when checked on 2026-10-16; the 2026-10-19 piece-type counts ran
  with sentencepiece 0.2.2 on the M5 Max); no rental is needed. News coverage above used news-commentary as a
  proxy because newstest2014 was not on the Mac when it was measured (only the WMT14 training shards were, in
  `~/mt_local/hf_wmt14/`; checked 2026-10-16). The pinned newstest2014 is the training-box copy,
  Machine_translation `data_enfr_v1/test.{en,fr}` (sha256 `967dca93…` / `82526d36…`, equal to
  E0.5's BLEU cache), copied to the M5 Max for the 2026-10-19 ForgetGenUse run; it replaces the planned fetch from
  `wmt/wmt14@b199e406`. newstest2014 may be measured
  descriptively, but **no test set may be used to define or tune any set** (OPEN DECISION 16 says
  what each check may change).

**The contrasts.** Each is a BLEU difference on newstest2014, oriented so that a positive value is
the registered direction:

| contrast | definition | positive means |
| --- | --- | --- |
| **N** | BLEU(`R0`) − BLEU(`A_narrow`) | the vocabulary-capped random set costs (narrower vocabulary, plus whatever else the caps change) |
| **S** | BLEU(`A_narrow`) − BLEU(`ft_un_div`) | the rest of QE selection costs |
| **C1-beyond-coverage** | BLEU(`div_narrow`) − BLEU(`ft_un_rep`) | the repetitive half costs beyond coverage (duplicates, French-side features and rarest-word selection's side effects left free) |
| **C1-beyond-duplicates** | BLEU(`div_dd`) − BLEU(`rep_dedup`) | the repetitive side costs beyond exact duplicates (vocabulary and French-side features left free) |
| C1-beyond-both (only if OPEN DECISION 5(b) adds `div_narrow_dd`) | BLEU(`div_narrow_dd`) − BLEU(`rep_dedup`) | the repetitive side costs beyond exact duplicates and the matched coverage measure together (French-side features and rarest-word selection's side effects left free) |
| precondition P | BLEU(`R0`) − BLEU(`ft_un_div`) (= N + S) | there is a gap to divide |
| bridge Q1′ | BLEU(`ft_un_div`) − BLEU(`ft_un_rep`) | E0.5's Q1 again |
| bridge Q2′ | BLEU(`ft_un_random`) − BLEU(`ft_un_div`) | E0.5's Q2 again |
| V (only if OPEN DECISION 18 registers it) | BLEU(`ft_un_div`) − BLEU(`div_narrow`) | narrowing the repetition-free top-band rows by rarest-word selection costs (smaller vocabulary, plus whatever that selection shifts, such as genre or difficulty) |

**The rule, for every contrast.** E0.5's rule, unchanged. A gap counts only if it exceeds the
seed-noise floor (the larger seed sd of the two sets) *and* is Welch-significant at 0.05, using the
exact critical value at the fractional df (`phase0/probe_stats.py`). Five seeds per set. A contrast
that does not clear the rule is **not detected**, with its bound beside it. The outcomes are applied
by `phase0/e06_report.py` to newstest2014. As in E0.5, there is no correction for multiple contrasts
(OPEN DECISION 9); E0.5's simulation put the rule's false-alarm rate at 3.4-4.5% per contrast.
Every outcome reading below is the only reading of that outcome, and the one to be registered
with the tag; the plan and the EAMT outline point here.

**Outcomes, stated before the runs.**

*Precondition P.* P = R0 − `ft_un_div` is registered as a contrast here, with the same rule. If R0
is not better than `ft_un_div` by the rule, then N and S are reported by their letters alone and no
reading about narrowness is offered: there is no gap to divide.

*N — does capping random text's vocabulary cost?*
- **N+, detected (A_narrow below R0).** The vocabulary-capped random set costs news BLEU: narrower
  vocabulary, plus whatever else the caps change. Its measured QE, literalness and genre stay at
  the random level, but its French recurrence is not measured and it holds more top-band rows
  than R0 (11.7% against 9.1%). Narrowness is therefore a candidate part of the gap P; this does
  not show it is the top band's mechanism (A_narrow lacks different words from the diverse half
  and is matched on `\w+` types rather than the pieces the model sees). This is consistent with
  Saunders & DeNeefe and is credited to them as the hypothesis. A detected N is one of two
  sub-outcomes, by a rule fixed now:
  - **N+ at the full gap:** S is not detected, and N's 95% Welch interval (N ± crit × se, with
    crit the exact critical value at the fractional df, `probe_stats.py`, `contrast()`) contains
    the observed P of the same batch. N + S = P by construction (*derived*), so this asks that S
    be small against N's precision. The capped set's cost is then consistent with the whole gap
    P; which words are missing, and what else the caps change, are not tested.
  - **N+ not at the full gap:** N is detected, and either S is detected or N's interval does not
    contain P. The capped set has a detected cost; the rest of P gets no reading from N, and no
    share wording is used.
- **N0, not detected.** Reported with its bound; no reading about narrowness.
- **N−, A_narrow is better.** Reported by direction alone; the cause of Q2 stays open.

*S — does the rest of QE selection cost?*
- **S+, detected.** QE selection costs something beyond source word-type count: QE level, literal
  or formulaic French, or genre, which this design does not separate. This does not rule out
  narrowing: A_narrow matches the type count, not which words are missing, so S may partly absorb a
  vocabulary effect. The follow-up is the vocab memo's Design C (the literal/free split of the top
  band, 107,227 rows per half), which separates only literalness (vocab memo risk 4); it is WMT 2027
  work, registered separately, not part of E0.6.
- **S0, not detected.** Reported with its bound; no reading.
- **S−, the diverse half is better than A_narrow.** Reported by direction alone; no reading.

*Joint readings* (only if P holds, and only if Q2′ holds; see *Bridges*):
- N+ at the full gap (which requires S0): the capped set's cost is consistent with the whole gap
  P; no remaining cost is detected (S's bound printed); which words are missing, and what else the
  caps change, are not tested.
- N+ not at the full gap and S0: the capped set has a detected cost; the rest of P has no reading,
  and no remaining cost is detected (S's bound printed).
- N0 and S+: the rest of QE selection has a detected cost; N is not detected (bound printed), and
  S may partly absorb a vocabulary effect.
- N+ and S+: both contribute. Each part is reported as a gap with its bound, never as a share.
- N0 and S0: the design could not divide the gap; both bounds are printed.
- N− or S−: each reversed contrast is reported by direction alone; the other is read by its own
  outcome above, and no joint reading is offered.

**No sentence of the form "narrowness explains x% of Q2"** is written. Q2 was measured against
`ft_un_random`, not R0, and the split is coarse at five seeds.

*C1-beyond-coverage — does the repetitive half cost beyond vocabulary coverage?*
- **YES.** The repetitive half is worse than a repetition-free set from the top band with the same
  size, length histogram, source and coverage (under the measure in OPEN DECISION 1). The
  repetitive half, with its exact duplicates and French-side features left free (and whatever
  choosing `div_narrow`'s rows by their rarest word's frequency shifts), costs beyond that coverage
  measure, which answers objection 1 for that measure only. It is not shown that repetition alone
  does; C1-beyond-duplicates addresses the duplicates.
- **NOT DETECTED.** E0.5's Q1 gap cannot be separated from coverage. C1 is then written as "the
  repetitive half, which is also lexically narrower", and coverage (Saunders & DeNeefe) is stated as
  an account this design cannot rule out. As PROTOCOL's E0.5 NO branch did, any text that reads
  Q1's gap as repetition rather than the bundle is corrected in `phase0/README.md` and wherever
  else it has been written.
- **REVERSED.** The diverse set at the repetitive half's vocabulary size is worse. Reported by
  direction alone; no reading in terms of repetition, coverage or duplicates.

*C1-beyond-duplicates — does the gap survive removing exact duplicates?*
- **YES.** The repetitive half without its duplicate sources is still worse than a diverse set
  length-matched to it, with vocabulary and French-side features left free (25,256 against 33,098
  source types in the scratch rebuild). "Just deduplicate", in its exact form, does not remove the
  effect. Near-duplicates are not addressed (they are in W4).
- **NOT DETECTED.** Not detected, with its bound. Repetition's cost cannot be shown once exact
  duplicate sources are removed; this does not show that deduplication suffices (remedies are the
  plan's W4).
- **REVERSED.** The diverse partner is worse. This is reported, and no reading about repetition is
  offered.

*C1-beyond-both (only if OPEN DECISION 5(b) adds `div_narrow_dd`) — does the gap survive removing
exact duplicates and matching coverage together?*
- **YES.** The repetitive side without its duplicate sources is still worse than a diverse set of
  the same size, length histogram, source and coverage (OPEN DECISION 1's measure). It costs
  beyond exact duplicates and that coverage measure together, with French-side features and
  whatever rarest-word selection shifts left free. It is not shown that repetition alone does.
- **NOT DETECTED.** Not detected, with its bound; no reading.
- **REVERSED.** Reported by direction alone; no reading.

If both C1 contrasts are YES, neither strand alone explains the gap: the repetitive side costs more
beyond coverage (with duplicates present) and beyond exact duplicates (with coverage not matched).
Unless OPEN DECISION 5(b) adds `div_narrow_dd`, no pair holds both fixed, and the paper says so. If
it is added, C1-beyond-both is reported beside the two, by its own outcome above, whatever they
show; it is the only contrast that holds both fixed, and no reading combines it with the other two
beyond what each states.

*Bridges.* Q1′ and Q2′ replicate E0.5 if they clear the rule in the same direction. They are never
pooled with E0.5's numbers. If Q1′ is not detected or reversed on the new box, the C1 contrasts
(and C1-beyond-both, if added) are reported by their letters alone. If Q2′ is not detected or reversed, N and S are likewise reported
by their letters alone: no Joint reading (not even as a division of P) and no reading about the
cause of E0.5's Q2, as PROTOCOL's E0.5 reports Q2 "by its letter alone" when Q1 is NO or
REVERSED. The stricter of the two possible rules is registered because the anchor that failed is
`ft_un_div`'s gap to random UN text, the gap N and S were built to explain.

*Reported and deciding nothing:*
- R0 against `ft_un_random`: does removing repeated rows from random text matter?
- A_narrow against `ft_un_random`.
- All eight sets on heldout_un. One reading is stated now, on one detected contrast only (E0.5:
  `ft_un_div` 44.72 against `ft_un_random` 47.03; R0 was not run): if `ft_un_div` is detected
  below A_narrow in domain, the in-domain loss is not explained by source word-type count alone,
  since the two sets match on it. In-domain coverage and genre or style both remain open, since
  A_narrow differs from `ft_un_div` in both (held-out UN tokens seen at least 5 times: A_narrow
  98.55%, R0 98.85%, `ft_un_div` 97.75%, vocab memo). Any other outcome of that contrast gets no
  reading. A_narrow against R0 in domain is reported by its verdict and bound, with no reading.
  The vocab memo's "genre or style, not vocabulary" is not used. This is offered only as a
  reading, never as a verdict.
- Any comparison across the two histograms (for example `rep_dedup` against `ft_un_rep`). These
  sets differ in rows and therefore in passes, so such comparisons are descriptive only.
- Each contrast's gap at the trainer's intermediate dev evaluations (the dev-BLEU curve the token
  rule below already uses; logged every 1,000 steps, ten points per run in E0.5's logs, so no
  extra checkpoint is kept). Waldendorf et al. (App. G.2) report filtering effects shrinking with
  more updates, so a reviewer will ask whether the 10,000-step gap is stable. The deciding
  checkpoint stays the final one.
- The secondary metrics of OPEN DECISION 11.

**Tokens.** Equal steps means equal sentence pairs, so the arms of each contrast make the same
number of passes over the same number of rows. Tokens differ because the texts differ (in E0.5 the
Q1 pair differed by 2.15% in applied tokens and the Q2 pair by 2.42%). The E0.5 rule applies
unchanged. For each contrast the report prints:
- the applied-token imbalance from exact counts;
- which set got more tokens;
- a bound on its worth: the imbalance as a share of the fine-tuning steps, times the steeper of the
  two sets' own dev-BLEU slopes over the second half of training.

A bound above the contrast's noise floor qualifies its verdict line. The piece-level imbalance of
the new pairs is **not measured** yet; it is computed on the Mac before the tag, with the release's
SentencePiece model (OPEN DECISION 16).

**Not to be done after seeing results:** more sets, more seeds, other step counts, other caps,
another coverage measure, tolerance or narrowing rule, a different statistic or threshold, another
deciding test set, or any pooling with E0.4's, E0.5's or the lr-1.0 control's numbers.

This is E0.6: an explanatory probe, not a gate.

**Cost and what to rent** (*derived*):
- 40 runs on four cards is 10 waves. E0.5's 30 runs took 8 waves, in 2 h 54 min including the
  25-minute cache warm-up (`phase0/README.md`, 21:24-00:18 UTC), so E0.6 needs roughly 3.5 h by
  waves, or 3.9 h at E0.5's 0.39 GPU-hours per run: about 3.5-4 h on a 4x RTX 5090 box, about
  14-16 GPU-hours. With OPEN DECISION 5(b), 45 runs: 2 more waves, about 2 GPU-hours and about
  0.5 h more.
- Disk is about 130 GB (40 runs x 3 GB + 10, by `e05_run.sh`'s estimate).
- At the llm memo's $0.60-0.80 per GPU-hour that is about $8-13.

### OPEN DECISIONS (E0.6)

This is the full list, and the only one. `WMT2027_PLAN.md` §7 gives its date and the inputs to
prepare for it, `eamt2027_outline.md` §9 names the decisions each paper slot depends on, and
`reading_list.md` §3 names the number of each decision it discusses.

1. **Coverage measure that `div_narrow` is matched on.** Options:
   - (a) source word types only, the measure the feasibility number used;
   - (b) source *and* French word types (French at a fixed 6.5M tokens);
   - (c) Saunders & DeNeefe's measure: the share of the base model's subword vocabulary that appears
     at least once in the set, source and target separately (target was their strongest correlate).
     It needs the base model's SentencePiece model, which the HF release ships, so it is CPU work
     on the Mac; their number is not comparable to word types until computed;
   - (d) coverage of a generic non-test text, for example news-commentary v9 tokens seen at least 5
     times.

   Measured on 2026-10-19 ("Measured feasibility" above; `phase0/next/cpu_items/item1_coverage/`,
   descriptive, an independent check agrees, its `REVIEW_NOTES.md` corrects some README wording):
   - under (b), only OPEN DECISION 2's rule c2 brings `div_narrow` within ±2% on both sides; under
     (a), (c) and c2 meet the source side and leave French types at +9.5% and −7.8%;
   - (d) does not separate the sets: `ft_un_div` is already within +1.9% of `ft_un_rep` on it, so
     every rule passes, and the sets it accepts are 15.7-34.7% off `ft_un_rep` in source types;
   - (c) was computed for the six E0.5 sets, R0 and A_narrow with the release's model (sha256
     checked). In piece types `ft_un_rep` is 6.8% (source) and 6.5% (target) below `ft_un_div`,
     against 25.8% and 24.0% in word types. A_narrow, within −0.9% / +1.6% of `ft_un_div` in word
     types, covers 4.5% more source and 9.7% more target pieces, outside ±2%: matching on word
     types did not match piece coverage. `div_narrow` under (c) is not built by the script and not
     measured; its target would be `ft_un_rep`'s 17,310 source and 18,979 target piece types.

   Whatever is chosen, a test set may not be used to define it. **Recommendation: (b), with ±2% on
   both sides**, which the feasibility run reaches only with OPEN DECISION 2 = c2. The ±2%
   tolerance is what A_narrow as built satisfies (−0.9% and +1.6%). Also compute (c) for all
   eight sets on the Mac before the tag and register it as a reported covariate. If (c) separates `div_narrow` from `ft_un_rep` clearly, the author decides before any
   run whether to re-match on (c).
2. **The narrowing rule for `div_narrow`.** Options:
   - (a) caps plus nearest-neighbour matching on `log(1 + maxrank)` to `ft_un_rep`'s rows, which is
     A_narrow's rule the other way round;
   - (b) caps plus a random draw at the histogram;
   - (c) a one-parameter family: within each length bucket, draw at random from the fraction q of
     candidates with the most frequent rarest word, and search q to hit the target. Its two ends
     measure 21,158 (q = 0) and 33,858 (q = 1, a random draw) source types in the feasibility run;
     the earlier 20,284 and 34,070 had no script and are not reproduced;
   - (c2) (added 2026-10-19, the PROPOSAL "c2" of `phase0/next/e06_build_spec.md` §6.5): as (c),
     but candidates are ordered by a blend, with weight λ, of the log pretraining ranks of the
     row's rarest source word and rarest French word; grid λ ∈ {0, 0.05, …, 1} × q in steps of
     0.005, ties to the larger q, then the smaller λ. It makes `train.clean.fr` (pinned
     `b0a6ba91…`) a set-defining input.

   Measured ("Measured feasibility" above): (a) and (b) cannot narrow at this histogram (30,197
   forced rows); under OD 1(b), (c) reaches −5.7% / +5.8% at best, so the build stops, and c2
   reaches −0.3% / +0.3%, the only rule within ±2% on both sides; the ninth set behaves the same
   way (OPEN DECISION 5). Whichever rule is chosen, every row of 36 or more tokens in `div_narrow`
   is a row of `ft_un_div`, so the narrowing acts only on shorter rows (the build spec, A24,
   proposes that the C1-beyond-coverage and V readings say so).

   **Recommendation (recommendation updated 2026-10-19 after the feasibility run): c2, with OD
   1(b).** The earlier recommendation, (a) for symmetry with A_narrow and (c) if (a) cannot reach
   ±2%, stops the build under OD 1(b). If OD 1(a) is chosen instead, (c) is feasible (+0.1% source
   types) with one parameter, and French types are left at +9.5%. The grid and the tie rule are
   fixed in the registered text.
3. **A_narrow's caps.** Keep 42,000 / 58,000 as built, or re-tune them to a stricter tolerance.
   Measured (feasibility run, `od3_cap_report`): with OD 6 = keep, 42,000 / 58,000 rebuilds
   `A_narrow_42_58.npy` (same sha256) at −0.9% / +1.6%, every pair on the grid 38,000-46,000 x
   54,000-62,000 (steps of 1,000) is feasible, and the closest, 43,000 / 55,000, gives −0.2% /
   +0.1%. With OD 6 = remove, no pair on that grid is feasible (2 to 8 length buckets short).
   **Recommendation: keep them.** They were set before any run, and the build must reproduce
   `A_narrow_42_58.npy`.
4. **What counts as a duplicate for `rep_dedup`.** Options:
   - (a) the exact raw source string, which is what was measured and is E0.5's duplicate measure;
   - (b) a normalised source (lowercased `\w+` sequence with digits masked, after Lo et al. 2018's
     number-aware deduplication).

   Near-duplicates as Waldendorf et al. 2025 remove them (BiCleaner `--aggressive_dedup`) are not an
   E0.6 option: the plan places them in W4 (WMT 2027). Also open: whether `rep` is recomputed after
   deduplication. **Recommendation: (a) as the registered arm, without recomputing `rep`. Measure
   the rates under (b) descriptively before registration.** Measured (feasibility run,
   descriptive): (b) removes 27,401 repetitive rows against (a)'s 19,845, and 1,017 diverse rows
   (356 of them with 8 or more tokens) against 252, leaving 191,718 rows per side against 198,333;
   `rep_dedup` under (b) then has 5,962,403 French tokens, below B = 6,000,000. If (b) removes far more rows than (a),
   the author decides before any run whether (b) replaces (a).
5. **A set that holds coverage and duplicates fixed together.** Options:
   - (a) keep eight sets and state the gap;
   - (b) add a ninth set, `div_narrow_dd`: the diverse UN candidates narrowed to `rep_dedup`'s
     coverage at `rep_dedup`'s histogram, compared with `rep_dedup`. The target is `rep_dedup`'s
     rebuilt type count (25,256 source types in the scratch rebuild and the feasibility run, and
     32,562 French types at B = 6,000,000 in the feasibility run; the dry run fixes them), not the
     feasibility file's unrecorded 25,296. That is 5 more runs and 2 more waves. Measured in the
     feasibility run: only OD 2 = c2 with OD 1(b) is within ±2% on both sides (−0.2% / +0.3%,
     q = 0.805, λ = 0.6); (a) gives +19.0% / +19.6%, (b) +31.0% / +28.2%, and (c) with OD 1(b)
     −5.6% / +5.1%. Its contrast is C1-beyond-both, with the outcomes stated above, and it counts
     in OPEN DECISION 9;
   - (c) replace `div_narrow` with `div_narrow_dd`, which loses the direct link to `ft_un_rep`.

   **Recommendation: (b) if its build passes before registration, otherwise (a).** By the
   feasibility run: under OD 1(b) it passes only with OD 2 = c2 (−0.2% / +0.3%); under OD 1(a),
   (c) passes (+0.0% source, French +8.6%; c2 picks λ = 0, the same set); under OD 1(d), every
   rule passes.
6. **Top-band rows inside the random draws** (9.1% of R0, 11.7% of A_narrow). Keep them (E0.5's
   rule), or remove them from every random arm to make S purer. Measured (feasibility run): with
   "remove", R0 still builds (pool 415,480 rows, tightest bucket 1.114x), but A_narrow at its caps
   42,000 / 58,000 does not (4 length buckets short: 28, 30, 32, 33), nor at any pair on OPEN
   DECISION 3's cap grid. **Recommendation: keep them**, for consistency with E0.5, and report the
   counts.
7. **The pretraining-exposure difference** (38.3% against 36.8% on the source measure, feasibility run). Accept and
   report it, or add exposure to A_narrow's matching. **Recommendation: accept and report it.** It is
   small, and adding a matching criterion now risks making the build infeasible.
8. **Step count for the 198,333-row pair.** Keep 10,000 steps (about 9.7 passes), or use about
   9,250-9,300 steps to match the 214,489-row sets' passes (*derived*: 8.95-9.0 x the final size /
   192, at the 198,333 the dry run is expected to confirm).
   **Recommendation: 10,000 steps.** It is the same recipe, the contrast is within the pair, and no
   claim is drawn across histograms.
9. **Multiplicity.** No correction (as E0.5), or Holm over the primary contrasts: N, S and the two
   C1 contrasts (four), plus V if OPEN DECISION 18 registers it and C1-beyond-both if OPEN
   DECISION 5(b) adds it (up to six). **Recommendation: no correction.** State the per-contrast
   false-alarm rate, and report all primary contrasts together, whatever they say.
10. **Where the run of record is built.** Fixed by the common rules, as in E0.5: on the training
    box with Python pinned to E0.5's 3.12.15, gated by reproducing E0.5's UN halves byte for byte.
    This is a check, not a choice.
11. **Secondary metrics and test sets.** Options include chrF and MetricX-24 reported but not
    deciding, and COMET-22 reported with a flag because it shares a family with the selector; a
    second out-of-domain en-fr test set, reported only; and Saunders & DeNeefe's token-level
    ForgetGenUse on the newstest2014 outputs. The candidate second set is FLORES+ devtest
    (sentence-level, 1,012 sentences; WMT24++ is paragraph-level and unsuited to a model trained on
    sentences). FLORES+ (`openlanguagedata/flores_plus`) is gated (llm memo): if chosen, the author
    accepts its terms on their own Hugging Face account and places the token (or the downloaded
    files) on the box before E0.6's rental; no script handles credentials. All of these need each
    system's decoded outputs, and `e03_collect.py` keeps only the BLEU score (E0.5's outputs were
    not kept; checked 2026-10-16), so the E0.6 evaluation must also write the outputs, with their
    sha256. Computing chrF and ForgetGenUse from them needs no GPU; MetricX-24 and COMET-22 do.
    heldout_un stays reported and deciding nothing for all sets, as in E0.5 ("Reported and
    deciding nothing" above); making any heldout_un contrast decide would change that section
    before the tag. **Recommendation: all three, reported only, with FLORES+ devtest as the second
    set if the author handles its gate; heldout_un deciding nothing.** BLEU on newstest2014 stays
    the deciding measure, as in E0.5.
12. **One draw per set, or a redraw per seed** for R0 and A_narrow (Waldendorf et al. redraw their
    random data per seed; the anchors stay E0.5's files either way). **Recommendation: one draw per
    set, as E0.5.** A redraw would put data-sampling variance into the seed sd, and it changes the
    build check: instead of one index file per set, the build writes five, the seed-42 draw must
    still reproduce `R0.npy` and `A_narrow_42_58.npy` row for row, and the other four, which have
    no scratch counterpart, are fixed by their sha256 in the dry run. The seed sds of R0 and
    A_narrow, and so the noise floors of N, S and P, would then include sampling variance and not
    be comparable with E0.5's.
13. **An NRC re-ranking arm** (Bernier-Colborne & Lo 2019: a 20% score discount for pairs with no
    new source bigram, then the same budget). **Not in E0.6.** E0.6 makes no remedy claim, and the
    NRC arms (bicov and the discount) are placed in the plan's W4 (WMT 2027), the only placement
    with a budget.
14. **R0 as the paper's Q2 reference.** The vocab memo suggests R0 replace `ft_un_random`, since the
    latter has repeated rows and Q2's 0.58 is probably an underestimate. **Recommendation: report
    both, with P as the registered precondition.** E0.5's Q2 stands as registered.
15. **What "narrowness" means for N** (vocab memo question 3). Type count, which A_narrow matches,
    or coverage of the news domain's vocabulary, which needs one more arm (random UN text capped to
    match news coverage as well; 5 runs). No test set may define it; a non-test news text such as
    news-commentary v9 could. **Recommendation: type count, as built**, with A_narrow's news
    coverage reported (95.29, between the diverse half's 94.98 and R0's 96.00, on news-commentary
    v9), since the S readings already state that S may absorb part of a vocabulary effect.
16. **When the piece-level and newstest2014 checks run, and what each may change.**
    **Recommendation: both on the Mac before the tag**, with the release's SentencePiece model and
    the pinned newstest2014 (the training-box copy, sha256 `967dca93…` / `82526d36…`, equal to
    E0.5's BLEU cache; see "Two checks are still owed"); no rental is needed for them. The two
    checks may change different things:
    - the piece-level check (piece-type counts and piece-frequency profiles of the training sets;
      no test data) may lead to a design change before the tag, for example re-matching on OPEN
      DECISION 1(c), and the registered text records it;
    - the newstest2014 coverage check is descriptive only. Whatever it shows (for example
      A_narrow's newstest2014 coverage leaving the range between `ft_un_div` and R0) is reported
      beside the results as a stated limitation, and it cannot change any set, cap, tolerance or
      rule, since no test set may define or tune a set.

    Status on 2026-10-19: the piece-type counts (with a token-weighted variant) were computed on
    the M5 Max for the six E0.5 sets, R0 and A_narrow with the release's model, sha256 checked
    (`phase0/next/cpu_items/item1_coverage/`; findings under OPEN DECISION 1). `div_narrow`,
    `rep_dedup` and `div_dd` are measured once they are built with the chosen decisions
    (`e06_build.py --measure` computes the same piece types), and the piece-level token imbalance
    of the new pairs is still owed. newstest2014 `test.{en,fr}` are now on the M5 Max (copied
    from the training box, sha256 equal to E0.5's BLEU cache; used for the descriptive
    ForgetGenUse run, `phase0/next/cpu_items/item6_forgetting/`); the newstest2014 coverage check
    itself has not been run.
17. **E0.6-LR** (the plan's risk 1): the UN triple at lr_scale 0.15, 5 seeds, 15 runs, about 6
    GPU-hours (*derived* in the plan). Yes or no. If yes, it is registered as its own section with
    its own tag, and it runs in the same rental; it is not part of E0.6's contrasts.
18. **Register V = BLEU(`ft_un_div`) − BLEU(`div_narrow`)** (the outline's proposal). It costs no
    runs: both sets are among the eight, drawn from the same repetition-free candidates at the
    same size and length histogram. They differ in vocabulary size and in whatever choosing rows by
    their rarest word's frequency shifts (genre or difficulty, for example), and they share about
    127,000 of their 214,489 rows (126,893 under c2 with OD 1(b); 126,844-128,334 across options;
    feasibility run), including all 30,197 rows of 36 or more tokens, so V compares the two sets
    only on their shorter rows (OPEN DECISION 2; build spec A24). Without V the design has no direct contrast of vocabulary size among
    repetition-free top-band rows, and a not-detected C1-beyond-coverage supports no vocabulary
    account (a null is never read as a finding). **Recommendation: register it, with E0.5's
    rule.** It then counts among the primary contrasts of OPEN DECISION 9. Its outcomes, if
    registered:
    - *detected* (`div_narrow` below `ft_un_div`): narrowing the non-repetitive half's vocabulary
      by rarest-word selection (which may also shift genre or difficulty) cost BLEU. It does not
      show that vocabulary size did. C1-beyond-coverage is reported beside it as detected or not
      detected, with its bound, and neither account is called the leading one;
    - *not detected*, with its bound and no reading;
    - *reversed*, by direction alone.

---

## E0.7 — does the within-source result replicate on en-de? (DRAFT 2026-10-16, not registered)

E0.5's within-source result rests on one pair (en-fr), one source (UN), one pool and one base model.
Prior-art risk 3 in `PRIOR_ART_E04.md` ("it is the pool, not the metric") needs a second pool or
pair. The fixed en-de corpus is scored (`ende_scored.tsv`, 4,238,227 rows, the same pinned
CometKiwi-22 revision, on a different software stack than en-fr) and labelled (ende memo). Copying
en-fr's band share does not give usable set sizes, so E0.7 is a scaled-down replication **inside
Europarl**.

This is a weaker test of C1 than en-fr, in four ways stated before any run:
- The repetitive half is about half as repetitive: Europarl rep_mean 0.197 against UN 0.433, and
  duplicate sources 1.03% against 8.45%.
- The vocabulary gaps are smaller: the repetitive half has 81% of the diverse half's types, against
  74% in en-fr.
- The base has seen every fine-tuning row (under OPEN DECISION 3(a)).
- The band is the top 23.6% of the pool, not about 6%.

It is stronger in four ways:
- Every row has an exact source label.
- Pretraining exposure is balanced in every triple (under the recommended base).
- Random length-matching never runs short.
- Before matching, the repetitive half is *longer* than the diverse half (Europarl at the 1M band:
  29.2 against 21.5 tokens, *derived* from the ende memo's rows.npz and band indices; all sources:
  26.6 against 20.4), the reverse of en-fr over all sources (19.0 against 28.6). Once 29.2 against
  21.5 is re-derived by the build, a gap here cannot come from short, formulaic text.

**Design: a held-out set, a retrained base, three Europarl sets, five seeds, one machine.**

*Source labels* (port of `ende/s1_labels.py` to `phase0/e07_source_labels.py`):
- Every corpus row is hashed after the corpus's own processing (`strip()`, then internal CR replaced
  by a space, as in `rebuild_corpus.py --mode fixed`), and matched against statmt de-en europarl-v7,
  commoncrawl and news-commentary-v9.
- Result: **all 4,238,227 rows matched**. Europarl 1,864,681 (44.0%), CommonCrawl 2,175,206
  (51.3%), News Commentary 198,340 (4.7%).
- 318 rows occur in more than one source and go to the first-listed source, Europarl.
- `ende_scored.tsv` matches the corpus row by row, except two rows (0-based 1,246,484 and
  1,246,485) where a tab in the German became a space.

*Held-out set, reserved first.* `heldout_europarl`, 2,000 rows, drawn before anything else, with
E0.3's mechanism (`e03_build_controls.py`):
- draw `RandomState(42).choice` over the eligible Europarl-labelled rows, as
  `e03_build_controls.py` does (not over deduplicated unique pairs);
- reserve, and remove from **both the base's training corpus and the fine-tuning pool**, every row
  sharing the exact pair, the normalised source or the normalised target of a held-out row (NFKC,
  casefold, whitespace collapsed).

Applying this reservation shifts the set sizes below by less than 0.1% (ende memo; the counts below
were made without it). Whether a second held-out set is wanted is OPEN DECISION 8; the memo's
details mention "2 x 2,000 rows", **to check**.

*Base, retrained.* The en-de base has to be retrained in any case: the earlier en-de cells were
trained on the misaligned corpus. Recommended (OPEN DECISION 3): the earlier en-de Base recipe on
the full fixed corpus minus the held-out rows. Every fine-tuning row is then 100% seen in
pretraining, exactly balanced across arms. The base config, step count and dev set are **to check**
against the earlier en-de Base cells and are fixed in the registered text. The base is the one
averaged checkpoint every fine-tuning run starts from. Its newstest2014 en→de and
`heldout_europarl` BLEU are recorded before any fine-tuning run, and a base below a threshold fixed
in the registered text (to set from the earlier en-de Base cells, **to check**) stops the
experiment.

*The band.* The top 1,000,000 rows by CometKiwi score (stable sort), then exact-pair dedup keeping
the first copy. This is E0.3's registered n_ft, the same absolute rule as en-fr.
- The top 1,000,000 rows (23.6% of the pool), 994,881 after exact-pair dedup (23.5%, *derived*:
  994,881 / 4,238,227; the ende memo's share_of_pool 0.2359 counts the 1,000,000 selected rows).
- QE 0.8599-0.9146, mean 0.8689 (en-fr band: 0.896-0.914).
- Composition: Europarl 71.0%, CommonCrawl 20.4%, News Commentary 8.7%.

*`ft_random`.* `RandomState(42).choice` over the unique pool, of size k = the band's unique size, as
in `e03_build_controls.py`. The order of the held-out draw and the `ft_random` draw on the RNG
stream mirrors that script (**to check**).

*`rep`.* E0.4's statistic over the whole band (lowercased `\w+`, blake2b-64 8-gram keys, count above
one over the band, rows under 8 tokens take 0). The median is 0.0, so the split is `rep > 0`
against `rep = 0`. 13.5% of band rows have `rep > 0` (en-fr 927k band: 27.0%).

**Sets.**

| set | built from | rows | role |
| --- | --- | --- | --- |
| `ende_ep_div` | Europarl rows of the band with `rep = 0` | 104,521 | diverse half |
| `ende_ep_rep` | Europarl rows of the band with `rep > 0` | 104,521 | repetitive half |
| `ende_ep_random` | Europarl rows of `ft_random`, the halves' exact length histogram | 104,521 | matched random |
| `ende_all_div`, `ende_all_rep`, `ende_all_random` (secondary, OPEN DECISION 5) | the same over all sources | 134,479 each | replication of E0.4's all-source contrast |

- The halves are paired by E0.4's stratified length matching (seed 42). The Europarl halves are
  built in the band's own order on both sides: score order with the first copy kept, as
  `e03_build_controls.py` writes `ft_topk` and as `ende/s2_bands.py` iterates it. Length buckets of
  the random arm fill in `ft_random`'s draw order, as in `e05_build.py`. The secondary all-source halves use `e04_split.py`'s order,
  which sorts the repetitive side by `rep`, as E0.4 did and as the memo measured them.
- The random arm is an exact-histogram draw, seed 42, **exact match or nothing**. The tightest
  bucket holds 2.78x what is needed (11.7x from the whole Europarl pool). Band rows are not removed
  from it: 38,673 of its 104,521 rows (37%) are band rows (*derived* from the ende memo's saved
  indices; the memo's 38,649 is its draw from the whole Europarl pool).
- The run of record is built on the training box. The ende memo ran under Python 3.14.3 / Unicode
  16.0, against en-fr's 3.12.15 / Unicode 15.0. The manifest records the sha256 of every output,
  as E0.5's did.

**Inputs are pinned.** The build refuses inputs whose sha256 differs from:
- the fixed en-de corpus `train.clean.{en,de}`: `b976980a…` / `b662c96b…` (the values
  `ende_scored.tsv`'s meta records, `phase0/README.md`);
- `ende_scored.tsv` itself (sha256 to record);
- the statmt de-en files used for the labels: europarl-v7, commoncrawl and news-commentary-v9,
  `.en` and `.de` each (sha256 to record).

Every fine-tuning run likewise refuses a base checkpoint or config other than the retrained one
(sha256 recorded when it is trained).

**Runs.** 3 sets x seeds **42, 1, 2, 3, 4** = 15 runs, or 30 with the secondary triple, on one
machine in one session. Every run uses:
- lr_scale 1.0, spike guard off, final checkpoint with `--keep-last 1`;
- the retrained en-de base;
- the step count of OPEN DECISION 2;
- evaluation on newstest2014 en→de (deciding) and `heldout_europarl` (in-domain, secondary), with
  the secondary metrics of OPEN DECISION 13.

The decoding settings follow the earlier en-de cells (**to check**). The pass counts below assume
the en-de config caps batches at 192 sentences, as en-fr's does (**to check**, OPEN DECISION 6).

**Steps against passes** (about 192 pairs per step, implied by en-fr):

| rows per set | passes at 10,000 steps | steps for 9.0 passes |
| --- | --- | --- |
| 104,521 (Europarl) | 18.4 | 4,896 (memo); 4,899 at exactly 192 pairs per step; 4,873 to match en-fr's exact 8.95 (*derived*) |
| 134,479 (all sources) | 14.3 | 6,300 |

**Measured on the sets before any run** (ende memo, Mac, without the held-out reservation). At
identical word counts (3,044,342 words, 1M band, Europarl):

| set | source word types | duplicate sources | rep_mean | rows with rep > 0 | QE mean | pretraining exposure (recommended base) |
| --- | --- | --- | --- | --- | --- | --- |
| `ende_ep_div` | 27,898 | 0 | 0 | 0 | 0.8686 | 100% |
| `ende_ep_rep` | 22,538 | 1.03% | 0.197 | all | 0.8692 | 100% |
| `ende_ep_random` | 32,860 | 0.008% | — | 12.8% | 0.8388 | 100% |
| en-fr UN, for reference (div / rep / random) | 33,874 / 25,134 / 58,441 | 0 / 8.45% / 0.23% | rep 0.433 | random 25.5% | — | 36.7 / 36.7 / 37.9% (pairs) |

What else was measured:
- **The matched Europarl sets are longer** than en-fr's: mean 29.1 tokens, against 26.1 for en-fr
  UN.
- **The source confound exists in en-de too, which is why only the single-source triple decides.**
  After matching at the 1M band, the diverse all-source half is 74.1% Europarl, 16.2% CommonCrawl
  and 9.7% News Commentary, and the repetitive half 77.7%, 20.7% and 1.5%. News Commentary is
  news-like and favours the diverse half on newstest2014.
- **The band over-selects repetition relative to random text, not relative to the bottom band.**
  - Against length-matched random text, the band has 13.5% of rows with `rep > 0` against 8.0%
    (1.7x), or 14.7% against 10.8% (1.4x) inside Europarl.
  - At the 6% band, the bottom-QE band of the same size has about the same `rep > 0` share as the
    top band (9.05% against 9.28%),
    but far more repetition per row (rep_mean 0.067 against 0.0245) and more duplicate sources
    (2.25% against 0.46%).
- **In Europarl, "repetitive" means shared parliamentary phrases** ("Member State of the European
  Union"). In CommonCrawl it means hotel and booking templates.
- **The leave-out base was costed (OPEN DECISION 3).** It removes 517,567 rows (12.2%) from
  pretraining, leaving 3,720,660 and removing 24.3% of Europarl. It would still leave 8-gram
  exposure of 0.6% (diverse), 9.7% (repetitive) and 2.8% (random). That asymmetry is built into
  the statistic; en-fr had it too, unmeasured.

**The rule, for every contrast.** E0.5's rule, unchanged (noise floor *and* exact Welch at 0.05,
`probe_stats.py`). Five seeds per set. A contrast that does not clear the rule is **not detected**,
with its bound. The outcomes are applied by `phase0/e07_report.py` to newstest2014 en→de.

**Outcomes, stated before the runs.**

*Q1-de: `ende_ep_rep` against `ende_ep_div`.*
- **YES, the repetitive half is worse.** C1's within-source result replicates on a second pair, a
  second source and at full pretraining exposure, with the halves' natural length difference
  running the other way. "It is the pool" is answered for one further pool and pair. It is still
  one QE model, and the en-de band is the top 23.6% rather than about 6%.
- **NOT DETECTED.** This is reported as a failed replication at a weaker dose, with its bound. It is
  not explained away, but it does not cleanly contradict en-fr, for the reasons stated before the
  runs: the treatment is about half as strong, and every row was seen in pretraining. The paper's
  C1 is then scoped to en-fr UN text.
- **REVERSED, the diverse half is worse.** C1 does not generalise. This is reported, and the paper's
  C1 is scoped to en-fr UN text.

*Q2-de: `ende_ep_div` against `ende_ep_random`.*
- **A, not detected.**
- **B, the diverse half is worse**, as in en-fr.
- **C, the diverse half is better.**

Every reading of Q2-de carries two facts. The random arm is 37% band rows (the like-for-like
en-fr arm, `ft_un_random`, has 8.8% of its rows as exact pairs of `ft_topk`: 18,947 of 214,489, a
scratch count on E0.5's files, re-derived in the build; E0.5's 2-3% is overlap with each half,
not with the band), so the contrast is diluted towards "not detected". The random arm's QE is
lower (0.8388 against 0.8686).
If Q1-de is NOT DETECTED or REVERSED, Q2-de is reported by its letter alone.

*Reported and deciding nothing:*
- all sets on `heldout_europarl`;
- the secondary metrics of OPEN DECISION 13;
- each contrast's gap at the trainer's intermediate dev evaluations, as in E0.6 (the deciding
  checkpoint stays the final one);
- the all-source triple, if run, with the same two contrasts. Its source mix differs (News
  Commentary 9.7% against 1.5%) and favours the diverse half on news, so it is a replication of
  E0.4's form, not a test;
- the size of any en-de gap against any en-fr gap. **No claim is drawn from it**: they differ in
  pair, source, rows, passes, exposure and dose.

**Tokens.** As in E0.5: for each deciding pair, the exact applied-token imbalance, which set got
more, and the bound on its worth, which qualifies the verdict if it is above the noise floor. In
en-de the repetitive half is longer before matching, so the direction of the imbalance is not known
in advance.

**Not to be done after seeing results:** another band, another step count, more seeds, another
random-arm rule, another base, another statistic or threshold, a second held-out set added
afterwards, or any pooling with en-fr numbers.

This is E0.7: an explanatory replication, not a gate.

**Cost and what to rent:**
- the en-de base retraining, whose cost is **to check** against the earlier en-de Base run;
- 15 fine-tuning runs (4 waves on four cards), or 30 with the secondary triple (8 waves, about the
  size of E0.5's session).

The base must be trained after the held-out reservation and, under the leave-out option, after the
sets are built. So the build comes first in either case.

### OPEN DECISIONS (E0.7)

1. **Band.** The 1M band (E0.3's absolute n_ft; Europarl sets of 104,521 rows), or the 6% share
   (sets of 18,217 rows, about 105 passes at 10,000 steps). The 6% share and an absolute 250k band
   are nearly the same set (252,756 against 248,198 unique rows), so that is not a real choice.
   Also open: whether the paper accepts a top-24% band as "the QE top band" for en-de.
   **Recommendation: the 1M band.** State its share of the pool beside it, and keep the 6% band as
   the descriptive anchor for over-selection (9.28% against 4.26% of rows with `rep > 0`).
2. **Step count.**
   - (a) 10,000 steps: the same recipe as every en-fr fine-tuning run, about 18.4 passes;
   - (b) about 4,900 steps (4,873-4,899 *derived*, 4,896 in the memo): the same passes as en-fr, 9.0.

   The trade-off, stated once: steps and passes cannot both match en-fr. (a) keeps the number of
   updates and the recipe and doubles the passes; (b) keeps en-fr's passes over each row (about 9)
   and changes the recipe, at about half the training time. Whichever is held fixed, the other is a
   reviewer's target. **Recommendation: (a) 10,000 steps.** It keeps one fine-tuning recipe across
   E0.4-E0.7; passes already differed between E0.5's triples (7.9 against 9.0), and the protocol
   draws no claim across sets with different passes. Against it: Waldendorf et al. (App. G.2)
   report filtering effects shrinking with more updates, so a null at 18.4 passes is harder to read
   against en-fr's 9. Register the pass count beside every result. If the author prefers (b), the
   exact step count must be fixed only after OPEN DECISION 6 is confirmed.
3. **Base.**
   - (a) the full fixed corpus minus the held-out rows: 100% exposure, balanced in every triple;
   - (b) leave-out: the corpus minus the fine-tuning arms (3,720,660 rows, 24.3% of Europarl
     removed): near-0% exposure, with the residual 8-gram asymmetry above.

   This must be decided before the base is trained. Under (a), a failed replication cannot be told
   apart from the exposure regime, though a successful one extends the claim across exposure
   levels. **Recommendation: (a)**, as the memo's recommended design. Write in the registered text
   that fine-tuning then reweights data the model has already seen.
4. **The random Europarl arm.** Keep en-fr's rule (random from `ft_random`, band rows included, 37%
   overlap), or exclude band rows: drawn from `ft_random`'s Europarl rows outside the band
   (269,361 rows), the tightest bucket holds 1.81x what is needed (*derived* from the ende memo's
   `rows.npz`, its band and an emulation of the `ft_random` draw), or 7.75x if drawn from the whole
   Europarl pool outside the band (the memo's `random_from_pool_same_source_excluding_band`), which
   also drops the `ft_random` parent. Either exclusion departs from en-fr's rule.
   **Recommendation: en-fr's rule**, for comparability, with the dilution stated in the Q2-de
   readings.
5. **The all-source secondary triple** (134,479 rows each, 15 more runs). Run it, or skip it.
   The trade-off, stated once: about 6 more GPU-hours (*derived*: 15 x E0.5's 0.39 GPU-hours per
   run) buy a replication of E0.4's form on a second pair, which can never decide anything because
   its source mix favours the diverse half on news (News Commentary 9.7% against 1.5%).
   **Recommendation: run it, secondary only, in the same session**, since a later run could not join
   the same batch; the plan's cut rule drops it first if time runs short.
6. **Sentence cap.** Confirm that the en-de fine-tuning config caps batches at 192 sentences like
   en-fr's. This is a check, not a choice: the pass counts depend on it.
7. **A CommonCrawl single-source arm** (27,886 rows per half; repetitive rep_mean 0.518, 5.4%
   duplicate sources) as a high-repetition comparison. **Recommendation: not in E0.7.** At 10,000
   steps it is about 69 passes (*derived*), a different regime.
8. **Held-out sets.** Only `heldout_europarl`, or also a second set (the memo's "2 x 2,000",
   **to check** what the second was meant to be). Also the eligibility rule: E0.3's mechanism over
   all eligible Europarl-labelled rows, or, as a stated departure from it, unique pairs only, or
   only pairs whose source and target occur once in the corpus.
   **Recommendation: one set, E0.3's mechanism**, for comparability with en-fr's heldout_un.
9. **Base recipe and stopping rule.** The earlier en-de Base config on the fixed corpus, with its
   stopping rule (steps or tokens), step count, dev set, which checkpoint or average becomes the
   base, and its SentencePiece model, all fixed before training (**to check** the values; the
   original Base en-de cells' best checkpoints lie at 168,000-300,000 steps, `phase0/README.md`).
   No source fixes
   them, and without them the en-de base cannot be called comparable to the en-fr v1.1 base (step
   105,000, averaged); README E0.2 and PROTOCOL.md §1-7 show how contested "convergence" is here.
   **Recommendation: as the earlier cells**, recorded in the registered text with the config's
   sha256, and answered before the base is trained (with OPEN DECISION 3).
10. **Further out-of-domain test sets**, reported only. Candidates are the WMT18 de-en multi-domain
    sets (Koehn et al. 2018), which would need reversing for en→de, and FLORES-200 (Waldendorf et
    al.). Availability is **to check**. **Recommendation: newstest2014 decides; add at most one
    reported set if it needs no further decisions.**
11. **Descriptive checks on the statistic.** Report the share of band rows under 8 tokens (they take
    `rep = 0` automatically; 5.8% of rows in both the 6% band and the 1M band, 5.6% of the 1M
    band's Europarl rows, ende memo's `s3_report.json`). Report a number-masked `rep`
    variant (after Lo et al. 2018). **Recommendation: both, descriptive only.**
12. **Scoring direction.** Confirm that the en-de QE scores were computed in the fine-tuning
    direction (en→de). Peter et al. note scoring one direction and reusing it as a bias. This is
    **to check**, not a choice.
13. **Secondary metrics**, as E0.6's OPEN DECISION 11: chrF and MetricX-24 reported, not deciding,
    and COMET-22 reported with a flag because it shares a family with the CometKiwi-22 that
    selected the band. **Recommendation: all three, reported only.** BLEU on newstest2014 en→de
    stays the deciding measure.

---

## E1.0 — does the repetitiveness gap hold when an LLM is fine-tuned? (DRAFT 2026-10-16, not registered)

E0.4-E0.6 fine-tune one released, pretrained 60M-parameter Transformer-base (v1.1, step 105,000,
averaged). Waldendorf et al. 2025
say in their Limitations that they did not fine-tune larger multilingual models, and Saunders &
DeNeefe leave LLMs to future work. Two papers point the other way on Q2 for LLMs. Pombal et al.
found CometKiwi top-k beating random for Gemma-2-2B (en→zh COMET 79.28 against 76.78). Fujita de
Mello & Takada 2025 (arXiv 2512.11388) report CometKiwi top-10k beating random with LoRA in BLEU in
8 of 8 cells and in COMET in 6 of 8, from single seeds. Their tables contain apparent copy errors
(**to check** against the PDF). E1.0 asks whether E0.5's within-UN contrasts hold when a base LLM
is fine-tuned on the same files. It replaces the previous `WMT2027_PLAN.md`'s E3.1, which planned R/2xN/2xQ/1xD/C
x 3 seeds at about 100 GPU-hours. A second model (a second family only if OPEN DECISION 1 picks
Gemma-3-4b-pt) is a bridge, not a general LLM claim.

**Design: a pilot gate, then four stages, 30 training runs.** The model is **Qwen/Qwen3-1.7B-Base**
(Apache-2.0, not gated; 1.7B total / 1.4B non-embedding parameters; 28 layers). It is a base model,
not an instruct one, and is not translation-tuned. The data are E0.5's three UN files, unchanged
(sha256 from E0.5's manifest; uploaded from
`/Users/Euswbnic/mt_local/phase0_final/e05_results/data/phase0_e05/`). No set is subsampled:
subsampling would thin out the repeated 8-grams and the 8.45% duplicates, which are the treatment.

**Revisions are pinned.** Every model, metric and dataset (Qwen/Qwen3-1.7B-Base, the second
model, google/metricx-24-hybrid-xl-v2p6, wmt22-comet-da, google/wmt24pp, and
openlanguagedata/flores_plus, v4.6 per the llm memo) is pinned by Hugging Face commit, recorded in
the registered text, and the run refuses a different revision, as D7 does for CometKiwi
(`1ad78519…`) and infoxlm-large (`d616d637…`). The commits are **to record**.

| stage | what | sets | seeds | runs |
| --- | --- | --- | --- | --- |
| 0, pilot gate | 8-shot base model, no fine-tuning | — | — | 0 |
| 1, learning rate | LoRA at 1e-4, 2e-4, 4e-4; chosen on FLORES+ dev only, then frozen | `ft_un_random` | 42 | 3 |
| 2, core | LoRA at the frozen rate | `ft_un_rep`, `ft_un_div`, `ft_un_random` (214,489 each) | 42, 1, 2, 3, 4 | 15 |
| 3, full fine-tuning | full fine-tuning, lr 2e-5, 8-bit AdamW, no sweep | `ft_un_rep`, `ft_un_div` | 42, 1, 2 | 6 |
| 4, second model (a second family only under OPEN DECISION 1(a)) | LoRA at the Qwen rate, no sweep (OPEN DECISION 1) | `ft_un_rep`, `ft_un_div` | 42, 1, 2 | 6 |
| by amendment, before E1.0's first run | the vocabulary-matched set(s) from E0.6, by an amendment registered after E0.6 reports, so these runs join stage 2's session | E0.6's files, unchanged | 42, 1, 2, 3, 4 | 5 (or 10) |

*Stage 0, the pilot gate.* It runs before any training run counts:
- **P1.** The base model, 8-shot on FLORES+ en→fr with exemplars from FLORES+ dev, must reproduce
  Shang et al. (arXiv 2602.11961v3, Table 5): 39.7 spBLEU / 83.8 COMET, within the tolerance of
  OPEN DECISION 2. **To check:** which FLORES+ split, spBLEU tokenizer and COMET model that number
  uses, and whether Shang's "Qwen3-1.7B" is the base or the instruct model. The passages read do not
  say.
- **P2.** The base model's off-target-language rate on FLORES+ devtest and WMT24++, below the
  threshold of OPEN DECISION 2.
- **P3.** Measure, without gating: tokens per example (estimated at about 100), throughput,
  over-length rows per set, and the decoding-noise floor (one checkpoint decoded twice).

If P1 or P2 fails, the fallback of OPEN DECISION 3 applies before any stage-1 run.

The 8 exemplars and the prompt are fixed after the pilot and listed by ID in the registered text
before stage 1. The pilot touches no treatment set, so it cannot reveal a contrast.

*Stage 1, learning rate.* One LoRA run per rate on `ft_un_random` only, so the tuning cannot favour
a condition. The rate chosen is the one with the best (lowest) MetricX-24 on FLORES+ dev at the end
of training. If the best two are within the decoding-noise floor measured in P3, the smaller rate
is taken. If the best rate is at an edge of the grid, the grid is not extended (OPEN DECISION 10).
The rate is then frozen, and stage-1 runs are not reused as seeds.

**Recipe** (all stages unless stated; following ALMA, Tan et al. 2024 and Shang et al.):
- *Prompt:* `Translate this from English to French:\nEnglish: {src}\nFrench:` followed by the target
  and EOS. There is no chat template. The separator after `French:` is fixed in the pilot. Loss is on
  target tokens only. Newlines inside sources are normalised the same way for every system.
- *LoRA:* all linear layers, r = 64 (the memo's judgment call, to stay close to full fine-tuning),
  alpha = 32.
- *Optimisation:*
  - one epoch over the full set: 1,676 optimizer steps at an effective batch of 128 sequences
    (*derived*: 214,489 / 128, the last batch partial);
  - 1% warmup then cosine decay (OPEN DECISION 4), weight decay 0.01, gradient clipping 1.0, bf16;
  - max length 512, with the same over-length rule for every set and dropped rows counted per set;
  - data order shuffled per seed; the seed also sets the LoRA initialisation.
- *Checkpoints:* at 25%, 50% and 100% of the epoch (steps 419, 838 and 1,676, *derived*). The
  primary result is at 100%, and the trajectory is secondary.
- *Full fine-tuning (stage 3):* the same, with lr 2e-5 and 8-bit AdamW. The memo's arithmetic puts
  memory at about 17 GB plus activations, which fits one card (the arm runs on 4x B200, about
  180 GB per card; plan §6.5).
- *Decoding:* greedy, stopping at newline or EOS, with a fixed inference batch size. The maximum
  number of new tokens is fixed in the pilot.

**Evaluation.**
- **Primary test set: WMT24++ en→fr_FR** (google/wmt24pp, Apache-2.0). It has 998 rows at paragraph
  level: literary 206, news 149, social 531, speech 111, plus 1 canary. The reference is the
  post-edited `target`, and rows flagged `is_bad_source` are dropped, which includes the canary.
  WMT24, WMT25 and WMT26 have no en-fr (checked). WMT15 has an en-fr test set of 1,500 sentences
  (Chimoto et al. evaluate on it as "WMT15 newstest"; what kind of text it is, **to confirm**);
  WMT16-23 are **to check**. The French references appeared in February 2025, so this is the set
  least likely to be in pretraining.
- **Primary metric: MetricX-24-hybrid-XL** (google/metricx-24-hybrid-xl-v2p6, reference-based,
  scores 0-25, lower is better).
- **Secondary sets:** FLORES+ devtest (1,012 sentences); `heldout_un` (2,000, in-domain);
  newstest2014 (3,003), kept for continuity only, since it is almost certainly in pretraining.
- **Secondary metrics:** chrF and BLEU (sacrebleu).
- **COMET-22** (wmt22-comet-da) is reported but never decisive. It comes from the same family as the
  CometKiwi-22 that selected the training data, and Pombal et al. show that metrics favour systems
  trained on data the same or related metrics selected. In Q2 that bias would flatter the diverse
  half.
- **Reference point:** the base model, 8-shot, on every set.
- **Mechanism checks on the outputs, at no GPU cost:**
  - E0.4's within-set 8-gram repetitiveness statistic;
  - distinct output word types;
  - length ratio, off-target rate and untranslated-copy rate;
  - per-domain WMT24++ scores (news has only 149 paragraphs, so these are noisy);
  - truncation per domain. The segments are paragraphs and training is sentences, so a fine-tuned
    model may stop after one sentence.

**Measured or known before any run.**
- *Data* (word counts by `wc` on the local files):

  | set | English words | French words |
  | --- | --- | --- |
  | `ft_un_rep` | 5,515,154 | 6,123,541 |
  | `ft_un_div` | 5,507,661 | 6,236,269 |
  | `ft_un_random` | 5,453,246 | 6,140,023 |

  The source/French properties of the three sets are E0.5's and E0.6's tables above.
- *Published reference points* (8-shot en→fr, Shang et al.):
  - FLORES+: Qwen3-1.7B 39.7 spBLEU / 83.8 COMET; Qwen3-4B 46.46 / 86.65; Gemma3-4B 51.56 / 87.42;
  - WMT24++: Qwen3-1.7B 68.21 XCOMET / 55.75 CometKiwi; Gemma3-4B 78.16 / 68.84.

  French needs about 1.58x as many tokens as English with Qwen's tokenizer.
- *Estimates, not measurements (the pilot measures them):*
  - about 100 tokens per example and about 22M tokens per epoch (±25%), of which about 11M are
    target tokens;
  - per run: about 1 h of training plus 0.4 h of evaluation for Qwen LoRA, 1.5 + 0.4 h for full
    fine-tuning, 2 + 0.4 h for Gemma-3-4b-pt LoRA;
  - in total about 53 GPU-hours (the memo also rounds this to 55), $32-42 at $0.60-0.80 per
    GPU-hour, under $90 with 2x contingency, and about 14 h of wall-clock on a 4x RTX 5090 box. These are RTX 5090 figures; the arm now runs
    on 4x B200 (plan §6.5), so wall-clock and cost are re-measured in the pilot.
    This covers the pilot (about 2 h) and the 30 registered runs. If the amendment adds E0.6's
    sets, add about 7-14 GPU-hours (*derived*: 5-10 Qwen LoRA runs x 1.4 h), $4-11, and about
    2-4 h of wall-clock (*derived*: 7-14 GPU-hours over the four cards).
    Prices are Vast.ai RTX 5090 at $0.602 per GPU-hour for a single card and $0.801 in 4-GPU
    offers, and Runpod at $0.69 (computeprices.com, updated 16 October 2026).
- *Power* (two-sided t, α = 0.05): with 5 seeds, 79% at a gap of 2 seed-sds and 93% at 2.5; with 3
  seeds, 46% at 2 sds. E0.5's Q1 gap was about 20 seed-sds. The danger is that the LLM's absolute
  gap shrinks, not that its noise grows. ALMA reports that large parallel sets wash out an LLM's
  prior, so all three sets may converge to a similar UN-style translator.

**The contrasts**, on WMT24++ with MetricX-24, oriented so that positive is the registered
direction (MetricX is lower-is-better):
- **Q1-LLM** = MetricX(`ft_un_rep`) − MetricX(`ft_un_div`), LoRA, stage 2.
- **Q2-LLM** = MetricX(`ft_un_div`) − MetricX(`ft_un_random`), LoRA, stage 2.
- **F-Q1** = the same as Q1-LLM under full fine-tuning, stage 3.
- **G-Q1** = the same as Q1-LLM on the second model, stage 4 (a second family only under OPEN
  DECISION 1(a)).

**The rule.**
- E0.5's rule on the seed means of MetricX-24 (noise floor *and* exact Welch at 0.05,
  `probe_stats.py`).
- **chrF must agree** in the sense fixed by OPEN DECISION 7 (recommended: a called contrast whose
  chrF gap has the opposite sign is reported as "metrics disagree" and given no reading).
- A called gap smaller than the smallest effect of interest (OPEN DECISION 6) is reported as
  "detected, below the smallest effect of interest".
- A segment-level paired bootstrap for test-set uncertainty is reported beside each contrast and
  decides nothing.
- Outcomes are applied by `phase0/e10_report.py` (name to fix).

**Outcomes, stated before the runs.** Every outcome, including "not detected" and a reversal, is
registered and reported; a not-detected contrast is reported with its bound and extends no claim.

*Q1-LLM.*
- **YES.** The repetitive half does more damage when this LLM is LoRA-fine-tuned too. C1 extends to
  one base LLM under LoRA, on WMT24++ with MetricX-24.
- **NOT DETECTED.** Reported with its bound; C1 is not extended to this model and regime. This is
  not evidence of equality or of a boundary, and no account of the non-detection is offered.
- **REVERSED.** This is reported, and C1 is scoped to the Transformer-base arm.

*Q2-LLM.*
- **A, not detected.**
- **B, the diverse half is worse**, as in E0.5.
- **C, the diverse half is better**, as Pombal et al. and Fujita de Mello & Takada would predict for
  LLMs. This is not by itself evidence that QE selection helps, and COMET-22 may not be used to
  support it.

If Q1-LLM is NOT DETECTED or REVERSED, Q2-LLM is reported by its letter alone.

*F-Q1, read against Q1-LLM.*
- Both YES: LoRA did not hide the effect.
- Q1-LLM not detected and F-Q1 YES: the gap is detected under full fine-tuning and not under LoRA;
  the LLM claim is stated for full fine-tuning only. No reason for the difference is offered: no
  contrast tests LoRA against full fine-tuning, and the two regimes also differ in rate,
  optimizer and seeds (5 against 3).
- Q1-LLM YES and F-Q1 not detected: reported with its bound. At 3 seeds the power is low, so no
  claim is made that full fine-tuning removes the effect.
- Both not detected: reported with both bounds; no LLM claim is made.

*G-Q1.* It replicates if the gap clears the rule in the same direction. Not detected (at 3 seeds)
and reversed are reported with their bounds.

*Reported and deciding nothing:*
- all sets on FLORES+ devtest, `heldout_un` and newstest2014;
- chrF, BLEU and COMET-22 on every set;
- the 25% and 50% checkpoints;
- the mechanism checks;
- the per-domain WMT24++ scores.

**Tokens and familiarity.** One epoch over equal rows means repetitive text, which compresses into
fewer tokens, gets fewer applied tokens. For each deciding pair the report prints the exact
applied-target-token imbalance and its bound under E0.5's 2% rule. Here the slope is the steeper
set's FLORES+ dev MetricX change between the 50% and 100% checkpoints, per step. The LLM's
pretraining corpus is not in hand, so exposure cannot be measured the way E0.5 measured it (exact
pairs found in the v1.1 corpus; measured, not matched by design). Before fine-tuning, the base
model's mean per-token loss on each training set's target tokens is measured and reported, as a
familiarity covariate (OPEN DECISION 9).

**Contamination checks** (descriptive; they never change a verdict, and **only contrasts between
conditions are claimed, never absolute levels**):
- (a) WMT24++ canary: probe the base model with the canary row and record whether it completes it
  (procedure per the dataset card, **to check**).
- (b) For the 8-shot base model on every test set: the exact-match rate against references, and the
  share of segments with sentence-level chrF ≥ 90. The threshold of 90 is a proposal.
- (c) Named-entity-swap sensitivity on a fixed sample, with the procedure fixed before the pilot
  (OPEN DECISION 13).
- (d) Recorded facts: newstest2014 is assumed contaminated. FLORES contamination is model-specific.
  The WMT24++ French references date from February 2025, close to the Qwen3 and Gemma 3 releases;
  their pretraining cutoffs were not verified (**to check**). The UN corpus is almost certainly in
  pretraining, and templated, duplicated sentences are the most likely to be memorised, which could
  make the repetitive set *less* damaging here. The familiarity covariate is the measurement of
  that.

**Not to be done after seeing results:** more seeds or sets than registered; another learning rate
after stage 1; another prompt, exemplar set, decoding rule or maximum length; another primary
metric or test set; dropping segments or domains; using COMET-22 as decisive; pooling across models
or with the Transformer-base arms.

This is E1.0: an explanatory replication in a second paradigm, not a gate.

### OPEN DECISIONS (E1.0)

1. **The second model's 6 runs.** Options:
   - (a) Gemma-3-4b-pt with LoRA. It is a different family, but gated: the author accepts the
     licence and places the token on the box. Its full fine-tuning does not fit on one card (about
     64 GB by the memo's arithmetic), so it gets LoRA only;
   - (b) Qwen3-4B-Base: ungated, same family, so it shows less generality;
   - (c) the all-source pair (242,022 rows), which replicates E0.4 but adds little after E0.5;
   - (d) random UN text under full fine-tuning, which answers Q2 under full fine-tuning.

   **Recommendation: (a)**, because a second family is what the generality sentence needs. Fall back
   to (b) if the author prefers not to handle a gated model. FLORES+ is gated too and is needed for
   stages 0 and 1 in every option, so a token is needed on the box whichever is chosen; the gating
   argument against Gemma concerns only a second licence, not whether a token is needed at all.
   Also add (d) (3 runs) if the budget allows, since Q2 under full fine-tuning is otherwise
   unanswered.
2. **Pilot-gate tolerances.** The gate needs a tolerance on P1 (spBLEU and COMET against 39.7 /
   83.8) and a threshold on P2 (off-target rate). **Recommendation (a proposal, not from a source):
   ±1.5 spBLEU and ±1.0 COMET, and off-target ≤ 2%, fixed before the pilot.** Shang et al.'s evaluation setup
   (base or instruct model, split, tokenizer, COMET model) is read before this decision is taken
   (plan §6.2, by Fri 20 Nov), so the gate is registered against a known reference. If the paper
   does not say, record that and state in the registration which reading the tolerance assumes.
3. **Fallback if the gate fails.** The memo says Gemma-3-4b-pt becomes primary. On a 32 GB card
   stage 3 could not run for Gemma; on the B200 cards now planned (about 180 GB each, plan §6.5) it
   can, so that constraint is gone. The options are Gemma-3-4b-pt or Qwen3-4B-Base as primary, or
   keeping Qwen3-1.7B-Base for stage 3 only. **Recommendation: Qwen3-4B-Base as primary if the
   failure is Qwen-specific; Gemma-3-4b-pt if the gate shows a pipeline problem that Gemma's
   published numbers can check.** Either way, it is re-registered before stage 1.
4. **Learning-rate schedule.** Warmup plus cosine (LLM practice), or a fixed rate to mirror the
   Transformer-base arm. **Recommendation: warmup plus cosine**, and state that the two arms differ in
   schedule.
5. **Paragraph-level scoring.** Accept WMT24++ paragraphs as the primary unit, or also register a
   sentence-split variant. **Recommendation: paragraphs primary**, with length ratio and truncation
   per domain reported. A sentence-split variant is reported only if its splitter is fixed before
   any run.
6. **Smallest effect of interest** on MetricX-24. The memo suggests something like 0.2 MetricX; it
   is the author's call. **Recommendation: fix it before the pilot** and use it only to label
   detected gaps, never to change a verdict.
7. **Metric agreement.** Options:
   - (a) MetricX decides, and a called contrast whose chrF gap has the opposite sign is reported as
     "metrics disagree";
   - (b) both MetricX and chrF must clear the rule, which costs power.

   **Recommendation: (a).**
8. **Seed-sd pilot and reallocation.** The memo suggests running 2 seeds of random UN text in the
   pilot, and moving the 6 second-model runs into extra core seeds if the sd is large. Its "8 seeds
   for the core triple" would need 9 runs, not 6 (**to check** what was meant). An sd from 2 runs is
   too noisy to steer by. **Recommendation: no adaptive reallocation; fix 5 seeds now.**
9. **Familiarity covariate.** Report it only, or use it in the analysis. **Recommendation: report
   it only.**
10. **Learning-rate grid edge.** If the best rate is 1e-4 or 4e-4, extend the grid once, or keep
    it. **Recommendation: keep it and record the fact.** An extension after seeing dev scores on
    the random set is allowed only if its rule is registered now.
11. **The vocabulary-matched set(s), after E0.6.** Options:
    - (a) A_narrow only (5 runs, as the memo says);
    - (b) R0 and A_narrow (10 runs), so that N and S can both be read in the LLM.

    The trade-off: (a) is the llm memo's proposal and costs about 7 GPU-hours; (b) costs about 14
    (*derived*: 1.4 h per Qwen LoRA run) but adds R0, without which only S, not N, has an LLM
    counterpart. **Recommendation: decide by an amendment registered after E0.6 reports and before
    E1.0's first run, so these runs join stage 2's session, using E0.6's files unchanged.** Take (b)
    if E0.6's precondition P holds and its bridge Q2′ replicates, the two conditions under which
    E0.6 gives N and S a reading (E0.6, *Bridges* and *Joint readings*); otherwise add nothing, so
    the LLM arm never adds sets for a reading the Transformer-base arm withholds. The budget line
    is therefore "+5 or +10, about 7-14 GPU-hours" (the plan's §3.1 and §3.5 carry it as a cost).
12. **Disjointness.** For the six E0.5 training sets this is already checked: E0.5's second
    pre-flight found that no row of the in-domain test set shares a source or a target with any of
    them (message of commit `0fb52a6`), and a scratch recount on the run-of-record files in
    `e05_results/data/phase0_e05/` (2026-10-17, exact lines) finds none either. The check is kept
    only for E0.6's new sets, if OPEN DECISION 11's amendment adds them to E1.0. This is a check,
    not a choice.
13. **Contamination-probe procedures.** The entity-swap sample size, which entities are swapped and
    how sensitivity is scored. **Recommendation: 100 FLORES+ devtest segments with a fixed list of
    person and place names, scored as the change in MetricX against the unswapped reference
    translation with the same swap applied.** This is a proposal, fixed before the pilot.
14. **Optional runs beyond the 30:** the all-source pair, 3 seeds each; random UN text under full
    fine-tuning, 3 runs. **Recommendation: only the latter** (see OPEN DECISION 1).
15. **Plan update.** Done in the new plan: the previous plan's E3.1 (100 GPU-hours, R/N/Q/D/C
    conditions) is replaced by this design at about 53-55 GPU-hours (§3.5), and the citation is
    corrected (§6.2). Nothing is left to decide.
