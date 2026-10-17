# Research plan: EAMT 2027, extended for WMT 2027

Version of 17 October 2026. This file replaces the previous `WMT2027_PLAN.md`, whose thesis
Phase 0 refuted (§0.3: the in-domain results of the lr-1.0 follow-up to E0.3 and of E0.5, not
the E0.3 gate verdict itself). The previous version, with its correction note (2026-09-04 to
2026-09-12), is in git history.

**What this plan holds.** What runs, when, at what cost and on what it depends. The outcome
rules of E0.6, E0.7 and E1.0, the reading of each outcome, and the numbered OPEN DECISIONS are
stated only in the pre-registration drafts [PR]; this plan points to them by section and number
("[PR] E0.6 OD 5") and does not restate them.

**Sources.** Every number in this plan comes from one of the files below and carries its key.
"Derived" marks arithmetic done for this plan on numbers from those files. "To check" marks
anything nobody has verified.

| key | file |
| --- | --- |
| [R] | `phase0/README.md`: E0.3, E0.4 and E0.5 results; the corpus and scoring sections |
| [P] | `PROTOCOL.md`: E0.4, E0.5 and the closing sections |
| [PA] | `phase0/PRIOR_ART_E04.md` |
| [Gn:key] | `phase0/next/groundwork.json`, `notes`: full-text reading notes on one paper (key as in the file) |
| [Gl], [Ge], [Gv] | `phase0/next/groundwork.json`, design memos `llm`, `ende`, `vocab` |
| [F] | `phase0/next/e06_feasibility.md` |
| [B5] | `phase0/e05_build.py` (how the E0.5 sets were built) |
| [L] | `~/mt_local/phase0_final/e05_src/ft_topk.source` (sha256 `a4ad4586…`, pinned in `e05_build.py`): one source label per row of the 926,999-row band; PROTOCOL says the labels reproduce the control manifest's source counts exactly [P] |
| [OLD] | the previous `WMT2027_PLAN.md` |
| [PR] | `phase0/prereg_drafts.md`: the E0.6, E0.7 and E1.0 pre-registration drafts, with their outcome rules, readings and OPEN DECISIONS (OD) |
| [W] | a web lookup during planning (EAMT 2026 call and proceedings), not saved in any file above: to check |

**Where the sources live.** `groundwork.json` and `e06_feasibility.md` are in `phase0/next/`,
with every script (`.py`, 24 files), JSON report and log from the two analysis directories; in this
plan `vocab/` and `ende/` mean `phase0/next/vocab/` and `phase0/next/ende/`. The bulk derived files
are not in git (the repo keeps data out of git, and GitHub rejects files over 100 MB): the 24 arm
index `.npy` files (R0.npy, A_narrow_42_58.npy, the A_V*_nn.npy arms and the rest; 33 MB),
`ende/s2_idx_*.json` (31 MB), `vocab/rows.pkl` (334 MB), `ende/rows.npz` (127 MB), `vocab/pre_en.pkl`
and `pre_fr.pkl` (about 100 MB each) and the other `.pkl`, `.npz` and per-row `.npy` caches are in
`~/mt_local/phase0_next/` on the author's Mac. `phase0/next/SHA256SUMS` lists the sha256 of all 68
files, committed or not, so a fail-closed rebuild (§3.3, §3.4) can check its output against them.

No index file or script for three E0.6 arms, rep_dedup, div_dd and div_narrow, exists in
`phase0/next/`; `e06_feasibility.md` records only their statistics, from a draw that was not
recorded. rep_dedup and div_dd can still be rebuilt from their definition: a scratch rebuild on
the Mac (2026-10-16) to the pre-registration draft's definition (ft_topk file order, first copy
kept, stratified length matching with `random.Random(42)`) gives 198,333 rows per side, a
repeated-8-gram share of 0.378, 25,256 source types for rep_dedup and 33,098 for div_dd; [F]'s
0.379 and 25,296 are the unrecorded feasibility draw. div_narrow's rule is not fixed yet ([PR]
E0.6 OD 2). The fail-closed `e06_build.py` must generate all three, and every figure (including
[F]'s 20,284 and 34,070 types for div_narrow) must be re-measured on the generated sets before it
goes into the pre-registration (§3.3).

---

## 0. Status, 17 October 2026

### 0.1 What the old plan claimed

[OLD] stated the thesis as: top-k quality-estimation (QE) filtering is "a covert
domain-selection operation", and the sign and size of its effect on translation quality can be
predicted before training from one divergence statistic between the filtered corpus and the
evaluation distribution. The program built on that thesis had an orthogonalising construction
(arms R, N, Q, D, C at 5M pairs each), a Moore-Lewis domain-shift statistic, a dose-response
validated out of sample, Base-vs-Big robustness runs, a second language pair, an LLM bridge
and a seed-noise floor. It budgeted about 600-800 H200-hours behind a one-GPU-day gate, E0.3.

### 0.2 What Phase 0 found

All results are on the CR-safe corpus. All fine-tuning starts from the same released,
pretrained v1.1 Transformer-base (step 105,000, averaged; newstest2014 35.31 BLEU, heldout_un
46.83) [R]. It is not called converged: README E0.2 finds every original cell under-trained
(Base capped 12.4 tokens per parameter against a reference of about 36-41), and the question
waits on re-measurement on the real cache [R].

| date | experiment | result | source |
| --- | --- | --- | --- |
| 2026-09-13 | corpus audit | The v2 en-fr corpus was about 45% misaligned from line 16,575,777 (carriage-return bug); en-de had 113,149 misaligned rows (2.71%); v1.1 is aligned. The CR-safe rebuild has 38,275,284 rows. | [R] |
| 2026-09-19 | E0.3 gate, lr_scale 0.15 (chosen by the frozen rule) | **NO-GO.** Top-1M by CometKiwi: −0.14 BLEU on newstest2014 against seed sd 0.16; gap to matched random 0.26, not significant (Welch t 2.59, df 2.6, crit 4.30). In-domain gain +0.02 against sd 0.04. | [R] |
| 2026-09-19 | lr_scale 1.0 probe (not pre-registered) | Matched random text alone loses 0.90, so half the rejected paper's effect is the learning rate. ft_topk loses 1.69, 0.79 below random (t 10.62). On heldout_un ft_topk loses 1.56 while random gains 0.52, although the top-1M set is 78.3% UN before deduplication (77.0% of the 926,999 rows trained on). | [R] [L] |
| 2026-09-20 | E0.4 (pre-registered) | The repetitive half of the top band is 2.73 BLEU worse on newstest2014 than the diverse half (floor 0.28, t 13.96); on heldout_un the gap is 0.11, inside the noise. Found later (2026-10-11): the halves differ in source mix, 90.9% against 74.2% UN. | [R] [P] |
| 2026-10-11 | prior-art check (36 papers opened) | The bare phenomenon is published at least four times: a narrow QE top slice is no better, or worse, than a same-size random draw (Waldendorf, Chimoto), a lower band (Tan), or the full data (Finkelstein, no random control); NRC remarked on top-rank redundancy in 2018 and 2019; no controlled test of the mechanism is in print. | [PA] |
| 2026-10-16 | E0.5 (pre-registered, 30 runs) | **Q1 YES**: inside UN the repetitive half is 2.63 BLEU worse (t 33.3, df 7.5, floor 0.14). **Q2 outcome B**: the diverse half is 0.58 BLEU worse than matched random UN text (t −7.0, df 7.9, bound 0.19). **E0.4 replicates**: 3.04 over all sources. On heldout_un both UN halves lose (−2.11, −2.65) while random UN gains 0.20. | [R] |

### 0.3 Why the thesis changes

1. **The domain account fails its own test.** If the top band hurt because it shifts the
   training mix toward UN text, in-domain UN quality should rise. It falls: −1.56 for the
   top band in the lr-1.0 probe, −2.11 and −2.65 for the two UN halves in E0.5, while random
   text from the same pool holds or gains (+0.52, +0.20) [R]. "QE filtering is covert domain
   selection" is refuted as the explanation, and the domain-divergence predictor built on it
   has nothing left to predict. Two caveats: these changes are against the averaged release
   while the runs evaluate final checkpoints, which biases them toward "degrades" by about
   0.2-0.4 BLEU [R]; and E0.5's heldout_un contrasts were registered as deciding nothing [P].
   The clean numbers are condition against condition: on heldout_un random text beats the top
   band by 2.08 BLEU in the lr-1.0 probe, and random UN text beats the UN halves by 2.31 and
   2.85 in E0.5 (derived; not tested by the decision rule).
2. **The repetitive half costs more with the source held fixed.** Inside UN text alone,
   matched on source, size and length histogram [B5], and with exact-pair pretraining exposure
   (36.7 / 36.7 / 37.9%) [P] and mean QE score (0.8990 against 0.8993 for the two halves) [Gv]
   measured nearly equal, not matched by design, the half whose sentences share 8-grams with
   other rows costs 2.63 BLEU more on news [R]. That half is a bundle: shared 8-grams, a
   narrower vocabulary (25,134 against 33,874 source word types) and 8.45% duplicate sources
   [P]. E0.5 did not separate these strands.
3. **The rest of the band is lexically narrow.** The non-repetitive half is still worse than
   random UN text and has 58% of its source word types at the same word count (33,874
   against 58,441) [R] [P].

The surviving thesis is about what the band selects (text that repeats across rows), tested
with the source held fixed; narrow vocabulary is a candidate cause still to test. Section 1
states it. The target is EAMT 2027 (deadline expected in mid-March 2027), with WMT 2027
(summer 2027) as the extended follow-up (section 2).

### 0.4 What survives from the old plan

| old item | fate |
| --- | --- |
| E0.1 exact provenance | Done. The source labels make the within-UN control possible [R]. |
| E0.2 token accounting | Done and corrected twice; the token-counter patch is in use. Its compute table must still be re-measured on the real cache before any figure is quoted [R]. |
| E0.3 gate | Ran; NO-GO. No later probe can revive it [R] [P]. |
| E1.1 Base-vs-Big instrument, E2.2 capacity robustness | Dropped. PROTOCOL.md §1-7 (run-to-overshoot protocol) stays as a record; it is still a draft and was never tagged [P]. Two of its reporting rules are kept as binding rules for every paper (§6.1, item 8): §6.6, a section titled "Deviations from pre-registration" appears in the submitted paper whether or not it is empty, and §6.7, every pre-registered cell is reported regardless of outcome [P]. |
| E1.2 orthogonalising construction, E2.4 domain-shift statistic and dose-response | Dropped: the domain account they serve is refuted. |
| E2.1 14-condition grid, E2.3 convergence evidence | Dropped. |
| E2.5 filter swap | Survives as W1 (a second QE selector). |
| E2.6 second language pair | Survives as E0.7, en-de. [OLD] preferred en-zh or en-ru; en-de is the only second pair already scored [R]. |
| E3.1 LLM bridge (~100 GPU-h) | Survives as E1.0, narrower: about 53-55 GPU-h [Gl]. |
| E3.2 current test sets; E2.5 anti-circularity | Survives: WMT24++ and MetricX-24 for the LLM arm; the selector's metric family is never the only evaluation metric [Gl]. |
| E4.1 seed-noise floor | Survives as method, not as a contribution: every contrast uses the noise-floor rule [P]. |
| Pre-registration (E1.3), the red-team pass (month 12), citation discipline, no arXiv | Kept; section 6. |

---

## 1. Thesis

### 1.1 Statement

> When the released, pretrained v1.1 en-fr Transformer-base (step 105,000, averaged) is
> fine-tuned on the extreme top band of a CometKiwi-22 ranking, the half of the selected text
> that repeats across rows does more out-of-domain damage than the other half, even with the
> source held fixed. "Repeats" is a bundle (shared 8-grams, a narrower vocabulary, 8.45%
> duplicate sources) whose strands E0.5 did not separate. E0.6 tests whether the repetitive
> half costs more than a diverse set matched to it in vocabulary coverage, and whether its
> duplicate-free part costs more than a diverse set; and it divides the gap between the
> diverse half and repetition-free random UN text into the cost of a vocabulary-capped random
> set (N) and the rest of QE selection (S), each read by its own verdict, never as a share.
> It does not isolate a single strand and cannot show that narrow vocabulary explains that
> gap ([PR] E0.6, N and S).

Terms, as used in every experiment:

- **Top band:** the top 1M rows by CometKiwi-22 of the 16,646,992-row reused en-fr pool
  (score band 0.8959-0.9145), 926,999 rows after deduplication. By provenance it is 78.3% UN
  before deduplication [R] [P] and 77.0% after (713,881 of 926,999 rows; derived from [L]), so
  deduplication removed mostly UN rows. About 6% of the pool (derived: 1M / 16,646,992).
- **rep(row):** the share of a row's source 8-gram tokens that occur elsewhere in the band.
  As run, an 8-gram counts when its total count over the band is above one; 44 of 926,999 rows
  change side of the median relative to the registered wording [P]. In the random arms the
  reference set is their own parent, ft_random (926,999 rows), not the band: R0's and
  A_narrow's rep = 0 is computed against ft_random [Gv], and so is ft_un_random's 25% of rows
  with rep > 0 (the 25% is from [Gv]; `e05_build.py` computes the random arms' rep within
  ft_random, lines 295 and 302 [B5]).
- **Diverse half:** rep = 0, which is the median. **Repetitive half:** rep > 0 [P].
- **Out-of-domain test:** newstest2014. **In-domain test:** heldout_un, 2,000 rows [R] [Gl].

### 1.2 Parts and their status

| part | statement | status |
| --- | --- | --- |
| C1 | Inside one source, matched on size and length histogram, with exact-pair pretraining exposure and mean QE score measured nearly equal (not matched by design), the repetitive half of the band costs more out-of-domain BLEU than the diverse half. | **Established** for en-fr, CometKiwi-22, one base model, lr_scale 1.0: 2.63 BLEU inside UN (E0.5) [R]. The all-source gaps (2.73 in E0.4, 3.04 in E0.5) are consistent replications of E0.4, not support for the within-source statement: they mix source and exposure [R] [P]. **Open:** whether it holds without duplicate sources (rep_dedup against div_dd), and separately against a diverse set matched on the coverage measure of [PR] E0.6 OD 1 (ft_un_rep against div_narrow), both in E0.6, and both at once only if [PR] E0.6 OD 5(b) adds the ninth set; in en-de (E0.7); and in an LLM (E1.0). |
| C2 | The diverse half is still worse than random text from the same source at the same size and length histogram. | **Established:** 0.58 BLEU (E0.5 Q2) [R]. **Cause open.** Narrow vocabulary is the obvious suspect, a hypothesis for E0.6, not a result [R]. The rival is everything else QE selection changes: more literal and more formulaic French, and a shift from procedural to narrative text [Gv]. E0.6 asks whether a vocabulary-capped random set (N) or the rest of QE selection (S) carries a detectable part of the gap between the diverse half and repetition-free random UN text (P = R0 − ft_un_div, not E0.5's Q2); neither can show that narrow vocabulary explains it ([PR] E0.6, N and S). The E0.5 random arm is not QE-matched (0.8629 against 0.8990) [Gv]. |
| C3 | Fine-tuning on either half also lowers in-domain quality, which random same-source text does not. | **Reported, not decisive:** −2.11 and −2.65 against +0.20 on heldout_un [R]. These are against the averaged release, so biased toward "degrades" by about 0.2-0.4 BLEU [R], and heldout_un contrasts were registered as deciding nothing [P]; condition against condition, random UN text beats the halves by 2.31 and 2.85 (derived). One reading fits every number but is untested: the whole band is lexically narrow, and its repetitive part adds a large extra cost out of domain [R]. |

Scope of every part: fine-tuning the released v1.1 Transformer-base; the extreme top band;
lr_scale 1.0 (risk 1); one language pair until E0.7 reports, one base model until E1.0 reports,
one selector (CometKiwi-22) until W1 reports; no claim about QE metrics in general even then.
E0.7 and E1.0 both select with CometKiwi-22 alone.

### 1.3 Checked against prior art

| part of the thesis | nearest prior art | what it already contains | what is left for this project |
| --- | --- | --- | --- |
| A narrow QE top slice is no better, or worse, than a same-size random draw, a lower band, or the full data | Waldendorf et al. 2025; Chimoto et al. 2024; Tan et al. 2024; Finkelstein et al. 2024 | The phenomenon. Waldendorf: CometKiwi-22 top 1% against same-size random, from scratch, 3 seeds; En-Ar 45.2 against 49.7 chrF++, En-De a tie on chrF++ and lower COMET. Chimoto: COMET-QE top 10% far below random on en-de, gap mostly closed at 30% kept. Tan: COMETKiwi > 89 below > 87 at 53k pairs, single runs. Finkelstein: top ~0.3% by BLEURT-QE, no random control. None tests a mechanism. [Gn:waldendorf2025, chimoto2024, tan2024, finkelstein2024] | Credited, not claimed. |
| The top of a ranking is redundant and lexically thin | Lo et al. 2018; Bernier-Colborne & Lo 2019 | A qualitative remark for YiSi-based scorers, and two remedies: drop pairs that add no new source bigram (bicov), or discount them by 20%. No measurement; the 2019 discount was never ablated. [Gn:lo2018, berniercolborne2019] | A measurement for a learned reference-free QE metric, and a controlled test of the harm. |
| Fine-tuning loss tracks the vocabulary of the fine-tuning set | Saunders & DeNeefe 2024 | Forgetting correlates with target vocabulary coverage across the 5 de-en and 3 en-ja domains (Kendall tau 0.9 with the COMET drop; pooled, n = 8, or per pair: to check), with no seeds reported; a coverage-completing mix-in removes most of it, with no same-size random control. [Gn:saunders2024] | Whether repetition hurts beyond coverage (E0.6), and the link to QE selection. Coverage is their idea and is credited. |
| QE filtering helps | Peter et al. 2023 | Keeping the top 50% and training from scratch to convergence: both QE metrics at or above random in every COMET22 cell of their Table 2; apparently one run per condition (to check). [Gn:peter2023] | Not contradicted. Our claims are scoped to the extreme top band and to fine-tuning; W2 maps where the sign turns. |
| A QE top slice beats random in LLM LoRA fine-tuning | Fujita de Mello & Takada 2025 | Japanese-English, four 7B LLMs with LoRA, 10k of a 440k single-domain pool: COMET-Kiwi top-10k beat same-size random in BLEU in 8 of 8 model-direction cells and in COMET in 6 of 8 (Tables III-IV), on a pool already deduplicated and filtered for repetition. Single seed, not length-matched; the tables have apparent copy errors. [Gn:llmselect] | The closest published result against C2 for LLM LoRA fine-tuning, and the reviewer objection the note itself names; E1.0 tests that regime. |
| Redundancy controls on top of a ranking | Kirchhoff & Bilmes 2014 (principle); Lo et al. 2018; Azpeitia et al. 2018; Lu et al. 2018; Bernier-Colborne & Lo 2019; Xu et al. 2020; Koehn et al. 2019 (practice) | The principle and the practice. The evidence that it helps NMT is weak to negative. Separately, deduplication before ranking: Rossenbach et al. 2018's near-duplicate filter gave +0.0 to +0.4 BLEU once pairs are model-ranked. [PA] | An intervention result of our own (W4). |
| "It is just duplicates" | Koehn et al. 2018; Waldendorf et al. 2025 | The WMT18 pool was deduplicated before scoring. Waldendorf near-deduplicated their data (whether this covers every setup is to check) and the CometKiwi top-1% deficit on En-Ar remained. [Gn:koehn2018, waldendorf2025] | E0.6 tests repetition with duplicates removed. |

### 1.4 What is explicitly not claimed

1. Not the bare phenomenon, which is published: a narrow QE top slice no better, or worse,
   than a same-size random draw (Waldendorf 2025, Chimoto 2024), a lower band (Tan 2024), or
   the full data (Finkelstein 2024, no random control). "First to observe" appears nowhere.
2. Not that top-ranked text is redundant or lexically thin. NRC said so in 2018 and 2019.
3. Not that QE filtering hurts. Peter et al. 2023 found the opposite at 50% kept, from scratch.
   Every claim is about the extreme top band and fine-tuning.
4. Not that QE metrics in general select repetitive text. One metric (CometKiwi-22) until W1
   reports, and no claim about QE metrics in general even then; one pool until E0.7 reports.
   E0.7 and E1.0 still select with CometKiwi-22 alone.
5. Not that vocabulary coverage predicts forgetting. That is Saunders & DeNeefe 2024.
6. Not that a redundancy or diversity control is a new idea (Kirchhoff & Bilmes 2014; NRC).
7. Not "predicts". A median split is a two-point contrast [PA]. The word waits for W3.
8. Not that repetitiveness explains a share (x%) of any gap. E0.5 registered the narrower
   rule: no claim is drawn from the size of the all-source gap against the UN one
   ("Repetitiveness explains x% of E0.4" is not a sentence those runs can support) [P]. The
   ban on a share of the gap between the top band and random data is this plan's own rule.
9. Not that the E0.5 random arm is QE-matched. It is not.
10. Nothing about training from scratch. Every NMT run fine-tunes the released, pretrained
    v1.1 Transformer-base (step 105,000, averaged).
11. Nothing about LLMs until E1.0 reports.
12. Not that domain or source mix plays no role. The claim is only that the repetitive half's
    cost is not reducible to source mix: the within-UN comparison holds the source fixed and
    the gap remains [R].
13. Not that deduplication or any redundancy control fixes QE selection. That waits for W4,
    against the NRC baselines [PA].

---

## 2. Two papers

### 2.1 EAMT 2027, research-technical track (expected deadline mid-March 2027)

**Venue facts.** Up to 10 pages. The 2027 call is not out. EAMT 2026's deadline was 18 March
2026, extended to 27 March; its research-technical track accepted 31 of 55 submissions (56%) [W].
This plan aims at a mid-March 2027 deadline and does not count on an extension. Anonymity
rules, appendix allowance and notification date: to check when the call appears.

**Question.** Why does fine-tuning on the extreme top band of a CometKiwi-22 ranking hurt
out-of-domain translation more than random data from the same pool, at lr_scale 1.0? At the
rule-selected 0.15 no top-vs-random difference was detected (§0.2).

**Working title** (free to change): the outline's option 1, "What Does CometKiwi Select? A
Controlled Study of Repetition in Top-Ranked Fine-Tuning Data", or its option 2, which scopes
the harm to the high learning rate. Any title names CometKiwi-22 and keeps the fine-tuning
scope; keep "vocabulary" and "not domain" out of it until E0.6 reports.

**Structure.** The EAMT outline draft (`eamt2027_outline.md`, 2026-10-16; commit it with this
plan) is the reference for section numbers and page budgets; this table follows it.

| § | content | experiments | pages |
| --- | --- | --- | --- |
| 1 | Introduction: the phenomenon is known and credited; the open question is why; the scope sentence | none | 1.0 |
| 2 | Related work | §1.3 | 1.2 |
| 3 | Setup: base model, pool, CometKiwi-22, the rep statistic, matching, seeds, decision rule, pre-registration, and the learning-rate regime (stated here, not left to Limitations) | none | 1.4 |
| 4 | Results: top band against random at two learning rates (the domain explanation fails its own test); what the band contains; the split over all sources and inside UN; the diverse half against random; 4.6 en-de inside Europarl; 4.7 an LLM, Qwen3-1.7B-Base | E0.3, lr-1.0 probe, E0.4, E0.5; E0.7; E1.0 | 2.3 |
| 5 | Analysis: the repetitive half against a coverage-matched diverse set and, without exact duplicates, against a diverse set; the vocabulary-capped random set (N) against the rest of QE selection (S); no single strand is isolated ([PR] E0.6); in domain against out of domain | E0.6 | 1.9 |
| 6 | Limitations: one selector, fine-tuning only, the learning rate, Peter et al. | none | 0.8 |
| 7 | Conclusion | none | 0.3 |

Title and abstract 0.3 pages; 0.8 pages of slack absorb the en-de and LLM tables.

**How the headline depends on E0.6.** Every E0.6 outcome already has its one drafted
reading in [PR] E0.6, "Outcomes, stated before the runs", to be registered with the tag: precondition P, N and S with their
Joint readings, C1-beyond-coverage, C1-beyond-duplicates, C1-beyond-both (only if OD 5(b)), V
(only if OD 18) and the Bridges, which gate the C1 readings on Q1′ and every N and S reading on
Q2′. The EAMT headline for an outcome is that reading, scoped as in §1.2; the outline
(`eamt2027_outline.md`, §3.5 and §9) maps each to its slot. Neither this plan nor the outline
words an outcome differently; a change is made in [PR] before the tag.

**Cut rule.** E0.6, the Europarl triple of E0.7 and the 15 core runs of E1.0 make up the
paper. If one of them has not reported by Friday 29 January 2027, its section (§5, §4.6 or
§4.7) moves to the WMT paper, where it joins W1-W4 as new runs (it never appeared in the EAMT
paper). Extras go first: the all-source pair of E0.7, then the second-model runs of E1.0.

### 2.2 WMT 2027 (summer 2027): the extension

**Deadline:** to check; the WMT 2027 call has not been seen.

**Question.** When does selecting the QE top band hurt, does it hold for another selector,
does the harm grow with repetition, and can a selection rule fix it better than the known
redundancy re-ranking?

**Contents.** W1 (second QE selector), W2 (threshold sweep), W3 (dose-response over
repetitiveness), W4 (remedy against the NRC redundancy baseline), plus any EAMT section the
cut rule moved here (§2.1), whose runs are new to both papers. The E0.x results the EAMT paper
reports appear only as a cited summary.

**What makes it substantially different from the EAMT paper:**

1. A different question. The EAMT paper explains a failure at one point of the ranking. The
   WMT paper maps where along the ranking the failure occurs (W2) and tests a fix (W4).
2. Every headline result comes from new runs. No EAMT table is reused.
3. Two claims the EAMT paper cannot make: a graded relationship between repetition and harm
   (W3), and an intervention result (W4) [PA].
4. Generality across selectors (W1), which the EAMT paper lists as a limitation.

**Contingencies.**

- EAMT accepts: the WMT paper cites it, summarises it in one paragraph and builds on it.
- EAMT rejects: its reviews feed the WMT paper, which may then carry the diagnosis and the
  remedy as one paper.
- EAMT has not decided by the WMT deadline: read both venues' rules on overlapping and
  concurrent submissions before submitting (to check).

---

## 3. Experiments

### 3.1 Overview

| id | paper | depends on | GPU | runs | GPU-hours and cost | answers |
| --- | --- | --- | --- | --- | --- | --- |
| E0.6 | EAMT | E0.5 sets; [PR] E0.6 OD 1-18, answered at the end of the reading week (§7); a post-decision dry run of `e06_build.py`; tag | 4x RTX 5090 | 40; 45 if [PR] E0.6 OD 5(b) adds the ninth set | about 14-16, about $8-13 (derived); with the ninth set about 16-18, about $10-14 (derived) | Does the repetitive half cost more than a diverse set matched on [PR] E0.6 OD 1's coverage measure (ft_un_rep against div_narrow), and does its duplicate-free part cost more than a diverse set (rep_dedup against div_dd)? Both fixed at once only with the ninth set. Does a vocabulary-capped random set (N), or the rest of QE selection (S), carry a detectable part of the gap between the diverse half and repetition-free random UN text (P)? No single strand is isolated, and N cannot show that narrow vocabulary explains the gap ([PR] E0.6, N and S). |
| E0.7 | EAMT | en-de decisions before the base is trained; the en-de base | 4x RTX 5090 | 15, plus 15 optional | fine-tunes about 6, plus 6 (derived); base training to measure | Does C1 replicate in a second pair? |
| E1.0 | EAMT | pilot gate; E0.6 for the optional 5- or 10-run arm | 4x B200 (the author's choice, 17 Oct; §6.5), one card per run | 30, plus 5 or 10 optional ([PR] E1.0 OD 11), plus 3 optional ([PR] E1.0 OD 1(d) and OD 14, recommended if the budget allows) | about 53, $32-42, under $90 with a 2x contingency ([PR] E1.0; the llm memo's rounded figure is 55, $35-45, under $100 [Gl]); plus about 7-14 for the OD 11 runs (derived: 5-10 LoRA runs × 1.4 [Gl]); plus about 6 for the OD 1(d) runs (derived: 3 full fine-tuning runs × 1.9 [Gl]) | Do C1 and C2 hold for an LLM under LoRA, and C1 under full fine-tuning and in a second model (a second family only if Gemma, [PR] E1.0 OD 1)? |
| W1 | WMT | choice of selector; scoring | 4x RTX 5090 | scoring, then 15-25 | scoring about 9.6 at CometKiwi-22's rate, more for a larger model (to measure); runs 6-10 (derived) | Is the effect specific to CometKiwi-22? |
| W2 | WMT | CPU descriptives | 4x RTX 5090 | about 40 | about 16 (derived) | At what depth does QE selection stop being worse than random? |
| W3 | WMT | E0.6 (which axis) | 4x RTX 5090 | about 25-30 | about 10-12 (derived) | Does the harm grow with repetition? |
| W4 | WMT | E0.6, W2, W3; NRC baselines implemented | 4x RTX 5090 | about 30, plus optional LLM runs | about 12 (derived); about 1.4 per LoRA run [Gl] | Does a selection rule beat the NRC re-ranking and deduplication? |

**Cost basis.** E0.5 ran 30 runs in one session on 4x RTX 5090, 21:24-00:18 UTC [R]: about
0.39 GPU-hours per run including evaluation (derived). Counted by waves of four runs instead
(E0.5: 8 waves in 2 h 54 min, 25 minutes of it the cache warm-up [R]), E0.6's 40 runs take
about 3.5 h, against about 3.9 h at 0.39 GPU-hours per run; this plan quotes 3.5-4 h and
14-16 GPU-hours (derived), and the pre-registration's cost line should use the same range.
Prices on 16 October 2026: $0.60 per GPU-hour for a single RTX 5090 and $0.80 in 4-GPU offers on
Vast.ai, $0.69 on Runpod [Gl].
Re-check prices at rental time. GPU money is not the constraint; the author's time to rent and
the pre-registration work are.

### 3.2 CPU work that needs no GPU (can start now)

1. **Coverage, Saunders & DeNeefe's way:** the share of the base model's subword vocabulary
   that appears in each E0.5 and E0.6 set, source and target separately [Gn:saunders2024].
   No SentencePiece model is on the Mac [Gv], but the HF release
   `euswbnix/transformer-wmt14-enfr-base` ships it, byte-identical to the original machine's
   `spm_enfr_v1_fixed.model` (`d1c10ea80fb95984…`) [R]. Fetch it from there and install the
   `sentencepiece` package into `~/mt_local/venv` (absent there; pyarrow 25.0.1 is present,
   checked 2026-10-16), so coverage can be computed on CPU now.
2. **Templated numerals:** rep recomputed with numbers, dates and UN document symbols masked,
   and the numeric-token share of each half [Gn:lo2018].
3. **Chimoto et al.'s rival explanation:** language-ID and length-ratio contamination in each
   half [Gn:chimoto2024].
4. **heldout_un hygiene:** its disjointness from the six E0.5 training sets is already checked
   (E0.5's second pre-flight, message of commit `0fb52a6`; a scratch recount on the
   run-of-record files on 17 October finds no shared source or target line). Repeat the check
   for the new E0.6 sets in `e06_build.py`'s manifest, and measure heldout_un's overlap with
   the pool [Gl] [Gn:koehn2018].
5. **Depth descriptives for W2:** share of rows with rep > 0, duplicate rate, UN share, and word
   types at matched word count, for the top 0.13%, 0.3%, 1%, 6%, 10%, 12%, 25%, 30% and 50% of the
   ranking (the retention levels of Tan, Finkelstein, Waldendorf, Chimoto and Peter, plus this
   band and W2's depths) [Gn:peter2023, tan2024, finkelstein2024, chimoto2024,
   waldendorf2025]. This is the one depth list; `reading_list.md` §3.2 item 10(d) and the
   outline's F3 use it. Descriptive only; no paper claim before W2 runs.
6. **Token-level forgetting:** ForgetGenUse, split into tokens inside and outside the
   fine-tuning vocabulary, on the E0.5 newstest2014 outputs [Gn:saunders2024]. Those outputs
   were not kept: `~/mt_local/phase0_final/e05_results/` (archive sha256 `b671180f…`) [R] holds
   only the BLEU cache (`.e03_collect_cache.json`, scores without hypotheses), and
   `e05_artifacts.tar.gz` holds data, logs, configs and the filtered held-out files (checked
   2026-10-16), because `e03_collect.py` parses only the BLEU score. So this item needs a
   re-decode of the 30 kept `final.pt` on newstest2014 on R1 first; after that it is CPU work.
   If [PR] E0.6 OD 11 picks FLORES+ devtest as the second out-of-domain set, the same R1
   re-decode also covers FLORES+ devtest for those 30 checkpoints (the outline's N1); WMT24++
   is paragraph-level and is not used for this arm (OD 11). E0.6's own evaluation must write
   its outputs ([PR] E0.6 OD 11).

### 3.3 E0.6: testing the strands of "repetitive" (EAMT)

**Model and recipe.** The project's own Transformer-base, the released v1.1 checkpoint
(step 105,000, averaged; "the Transformer-base arm" in this plan, with "from scratch" kept for
training on selected data only), fine-tuned exactly as in E0.5: en-fr, UN only, lr_scale
1.0, spike guard off, 10,000 steps, final checkpoint, five seeds (E0.5 used 42, 1, 2, 3, 4),
evaluation on newstest2014 and heldout_un with `--max-src-tokens 254` [F] [P]. Like E0.4 and
E0.5, it is an explanatory probe, not a gate.

**Eight sets, 40 runs** (nine sets and 45 runs if [PR] E0.6 OD 5(b) adds div_narrow_dd) [F] [Gv]:

| group | set | rows | built as | key facts |
| --- | --- | --- | --- | --- |
| anchors | ft_un_div, ft_un_rep, ft_un_random | 214,489 each | identical files to E0.5 | They bridge to E0.5: a new box and batch may shift BLEU levels [Gv]. |
| vocabulary cap (N, S) | R0 | 214,489 | random UN rows with rep = 0 (rep computed against ft_random, not the band [Gv]), ft_un_div's length histogram | 62,000 source word types; QE 0.861; literalness 0.693 |
| vocabulary cap (N, S) | A_narrow | 214,489 | R0's pool restricted to the 42,000 most frequent source and 58,000 most frequent French words; per-row rarity matched to ft_un_div | 33,562 types (ft_un_div 33,874); QE 0.866; literalness 0.698 (ft_un_div 0.899 and 0.783) |
| coverage | div_narrow | 214,489 (ft_un_rep's exact length histogram) | diverse UN candidates narrowed to the repetitive half's coverage on [PR] E0.6 OD 1's measure (within ±2%), by OD 2's rule | Measured low end (one heuristic: in each length bucket, the rows whose rarest word is most frequent; not a proven minimum): 20,284 source types; a random draw at that histogram: 34,070; ft_un_rep's 25,134 lies between |
| duplicates | rep_dedup | 198,333 | repetitive UN candidates with repeated sources removed (227,406 to 207,561), then length-matched | scratch rebuild (header): repeated-8-gram share 0.378, 25,256 source types; [F]'s unrecorded feasibility draw: 0.379, 25,296 (ft_un_rep 0.433); 0.00% duplicate sources |
| duplicates | div_dd | 198,333 | diverse candidates, length-matched to rep_dedup | the same 198,333-row histogram; 33,098 source types (scratch rebuild) |

div_narrow, rep_dedup and div_dd are not built sets. No index file or script for them survives
(header): div_narrow's figures come from a feasibility run [F] whose code and index files are
not in `phase0/next/`, and rep_dedup's and div_dd's from that run and from a scratch rebuild to
the pre-registration draft's definition (header). `e06_build.py` must generate them, and every
figure in their rows must be re-measured on the generated sets before the pre-registration
quotes it. R0 and A_narrow have index files (`R0.npy`, `A_narrow_42_58.npy`) [Gv].

**Contrasts, outcomes and limits: [PR] E0.6, not restated here.** The contrasts are N, S, the two
C1 contrasts (C1-beyond-coverage, ft_un_rep against div_narrow; C1-beyond-duplicates, rep_dedup
against div_dd), precondition P, the bridges Q1′ and Q2′, V if OD 18 registers it and
C1-beyond-both if OD 5(b) adds the ninth set. What each holds fixed and leaves free is in [PR]
E0.6's contrasts table and "What follows from the table"; every outcome's reading, including the
"N+ at the full gap" rule and the gating on P, Q1′ and Q2′, is in "Outcomes, stated before the
runs"; the held-out UN readout is under "Reported and deciding nothing".

**Resolution** ([PR] E0.6, "Why five seeds"). At E0.5's UN seed sds (0.11-0.14), E0.5's rule
could call gaps of 0.18-0.19 BLEU with five seeds [R]; at a seed sd of about 0.13, about 0.26
BLEU has 80% power [Gv]. PROTOCOL's simulation assumed E0.4's larger seed sds (0.20-0.28):
there five seeds call a true 0.5 BLEU gap 67-92% of the time and a 0.3 gap 30-51% [P].

**Follow-up arms, placed.** None of these runs before the EAMT submission. The NRC re-ranking
arms (bicov and the 20% discount) and exact-plus-near deduplication are W4 arms 3-5 (§3.9), the
only placement with a budget; in E0.6 a duplicate is an exact raw source string and there is
no NRC arm ([PR] E0.6 OD 4 and OD 13). A near-duplicate count on the repetitive candidates
with Waldendorf et al.'s tool may be reported descriptively before the tag (whether it needs
only CPU, and how long it takes, is to check); it adds no E0.6 set. The vocab memo's
Design C (literal against free split of the top band, 107,227 rows per half, 10 runs) is a WMT
follow-up, run only if S is detected and E0.6 gives S a reading: P holds and Q2′ replicates ([PR] E0.6, *Precondition P*, S+ and *Bridges*), and registered
separately [Gv]; about 4
GPU-hours (derived: 10 × 0.39).

**Decisions before the tag:** [PR] E0.6 OPEN DECISIONS 1-18, the full list; §7 gives the date
and the inputs to prepare for each. Fixed now, not open: the matching code is rewritten as a
fail-closed build script like `e05_build.py` (seed 42, pinned sources), and the deciding
measure is newstest2014 BLEU, as in E0.5. A_narrow's caps are still open ([PR] E0.6 OD 3).

**Compute and logistics.** 40 runs: about 3.5-4 hours on a 4x RTX 5090, about 14-16
GPU-hours, about $8-13 (derived from E0.5, §3.1). If [PR] E0.6 OD 5(b) adds the ninth set, 45
runs: about 2 GPU-hours more (5 × 0.39, derived), about 0.5 h more on 4 GPUs, about 16-18
GPU-hours and $10-14 in all (derived). The E0.5 box is destroyed, so upload the inputs from
`~/mt_local/phase0_final/e05_results/data/phase0_e05/` and `~/mt_local/phase0_final/e05_src/`
[Gl] [Gv].

### 3.4 E0.7: en-de replication inside Europarl (EAMT)

**Design** [Ge]:

- **Band:** top 1,000,000 rows by CometKiwi-22, E0.3's registered size. That is 994,881 rows
  after exact-pair deduplication: 23.5% of the 4,238,227-row en-de pool (derived; Ge's 23.6%
  is 1,000,000 / 4,238,227), QE 0.8599-0.9146.
- **Source held fixed:** Europarl, 71.0% of this band and the only source large enough. At this
  band Common Crawl gives 27,886 rows per half and News Commentary 2,055.
- **Three sets of 104,521 rows:** Europarl diverse (rep = 0), Europarl repetitive (rep > 0),
  and Europarl random drawn from the Europarl rows of a same-size uniform ft_random sample, as
  in en-fr. E0.4's stratified length rule with the exact histogram; tightest bucket slack
  2.78x, no bucket short.
- **Runs:** 5 seeds per set, 15 runs. 2,000 heldout_europarl rows are reserved and removed from
  the base's training data and from the pool before anything is drawn.
- **Secondary, optional:** the all-source pair at the same band (134,479 rows each) with its
  random control, 15 more runs; a Common Crawl single-source arm as a high-repetition
  comparison (rep_mean 0.518, 5.4% duplicate sources).
- **Evaluation:** the memo assumes newstest2014 as the out-of-domain set and heldout_europarl as
  the in-domain set [Ge]. Proposed secondary sets: the WMT18 multi-domain suite (newstest2018,
  IWSLT17, Acquis, EMEA, GlobalVoices, KDE) [Gn:koehn2018] and FLORES [Gn:chimoto2024]. Check
  the translation direction of each set.

**Decisions:** [PR] E0.7 OPEN DECISIONS 1-13 (band, steps, base, random arm, all-source triple,
sentence cap, CommonCrawl arm, held-out sets, base recipe and stopping rule, test sets, checks),
by Fri 6 Nov (§7). OD 3 (the base) and OD 9 (its recipe and stopping rule) must be answered
before the en-de base is trained, so before R2. The en-de base must be retrained in any case,
because its corpus was misaligned [Ge] [R].

**What it can and cannot show:** [PR] E0.7's opening paragraphs (four ways weaker, four ways
stronger, stated before any run) and the Q1-de and Q2-de outcomes, drafted there to be
registered before the base is trained, which say in advance how a smaller or missing gap is
read. The length figures there are scoped: 26.6 against 20.4 tokens is
over all sources at the 1M band [Ge]; the Europarl figure the design uses is [PR]'s derived
29.2 against 21.5, to re-derive from the build.

**Compute.** Fine-tuning: about 6 GPU-hours for 15 runs (derived from E0.5; less at 4,896
steps). Base training: no estimate in the sources. The original Base en-de cells' best
checkpoints lie at 168,000-300,000 steps [R]; their wall-clock is to check. Measure this before renting. It is
the largest GPU item before EAMT without an estimate.

**Carry into the write-up.** The en-de CometKiwi scores come from a different software stack
than the en-fr scores; do not quote them as if from one run [R]. The rep split follows the
interpreter's Unicode tables, so build the run of record on the training box and pin every
sha256 [Ge] [P].

### 3.5 E1.0: the LLM arm (EAMT)

**Model** [Gl]: Qwen3-1.7B-Base (Apache-2.0, ungated, 1.7B parameters), a base model rather than
an instruct model. Second model: Gemma-3-4b-pt (gated; a second family), Qwen3-4B-Base (ungated;
same family), or another use of the 6 runs ([PR] E1.0 OD 1); the author chooses.

**Thirty runs** [Gl]:

| runs | what | purpose |
| --- | --- | --- |
| 1-3 | LoRA on random UN text, one seed each at 1e-4, 2e-4, 4e-4, chosen on FLORES+ dev only, then frozen | learning-rate pick that cannot favour a condition |
| 4-18 | LoRA on ft_un_rep, ft_un_div, ft_un_random (214,489 pairs each), 5 seeds | Q1 (C1) and Q2 (C2) |
| 19-24 | full fine-tuning, repetitive against diverse, 3 seeds, lr 2e-5, 8-bit AdamW | LoRA is reported to forget less, which could hide the damage |
| 25-30 | second-model LoRA, repetitive against diverse, 3 seeds | a second model (a second family only if Gemma) |
| optional +5 or +10 | E0.6's files unchanged, LoRA, 5 seeds each: A_narrow only (5 runs) or R0 and A_narrow (10 runs) | an LLM counterpart of E0.6's N and S; options and rule in [PR] E1.0 OD 11, decided by an amendment after E0.6 reports and before E1.0's first run |
| optional +3 | full fine-tuning on random UN text, 3 seeds | Q2 under full fine-tuning; [PR] E1.0 OD 1(d) and OD 14, recommended there if the budget allows |

**Recipe** [Gl]: ALMA prompt, loss on target tokens only, no chat template; one epoch over the
full sets (no subsampling, which would thin out the treatment); LoRA on all linear layers,
r = 64, alpha = 32; effective batch 128 sequences; learning-rate schedule as [PR] E1.0 OD 4
decides (still open); weight decay 0.01; maximum length 512; bf16; greedy decoding.

**Evaluation** [Gl]: primary WMT24++ en-fr_FR (998 paragraph-level rows; post-edited
references; rows flagged as bad sources dropped, which removes the canary row) scored with
MetricX-24-hybrid-XL. Secondary: FLORES+ devtest, heldout_un, and newstest2014 for continuity
only (it is almost certainly in pretraining); chrF and BLEU; COMET-22 reported but flagged,
because it is the selector's model family. Reference point: the base model with 8-shot
prompting. WMT24, WMT25 and WMT26 have no en-fr [Gl]; WMT15 had an en-fr newstest
[Gn:chimoto2024]; WMT16-23 not checked (to check).

**Gate before training** [Gl]: pre-register the arm in PROTOCOL.md. A pilot must reproduce the
published 8-shot FLORES+ en-fr score for Qwen3-1.7B (39.7 spBLEU / 83.8 COMET, Shang et al.,
Table 5) and measure the off-target-language rate. If either check fails, the fallback is [PR]
E1.0 OD 3 (which model becomes primary is open there), re-registered before stage 1. To check
before the number becomes a gate: the Shang passages read so far do
not say whether that model is the base or the instruct version, and the memo matched numbers to
models by the order of the table columns, so the column-to-model mapping in Table 5 must be
confirmed [Gl].

**Design risks** [Gl]: the absolute gap may shrink (ALMA found large parallel sets wash out an
LLM's prior); Q2 may reverse (Pombal et al. found CometKiwi top-k beating random for a 2B LLM;
Fujita de Mello & Takada found COMET-Kiwi top-10k beating same-size random in BLEU in 8 of 8
cells for 7B LLMs with LoRA, on a pool deduplicated and filtered for repetition, single seed,
not length-matched, with apparent copy errors in the tables; the note names this as a
reviewer objection [Gn:llmselect]); contamination (UN text and newstest2014 are probably in
pretraining, so claim only contrasts between conditions and report each set's loss under the
base model); LoRA forgets less;
paragraph test segments against sentence training data; token imbalance (apply the 2% rule).
Power: five seeds give 79% power at a gap of 2 seed-sds and 93% at 2.5; E0.5's gap was about 20
seed-sds.

**Compute** [Gl]: about 53 GPU-hours (2 pilot, 4.2 learning-rate pick, 21 core, 11.4 full
fine-tuning, 14.4 Gemma), $32-42, under $90 with a 2x contingency, about 14 hours of wall-clock
on a 4x RTX 5090. The memo's summary rounds this to about 55 GPU-hours, $35-45, under $100.
These are RTX 5090 figures. E1.0 now runs on 4x B200 (§6.5), so its wall-clock and cost are
re-estimated from the offer at rental time and measured in the pilot (to measure).
The OD 11 runs add about 7-14 GPU-hours and 2-4 h of wall-clock (derived: 5-10 LoRA runs ×
1.4 GPU-hours [Gl], over four cards). The 3 runs of OD 1(d) / OD 14 add about 6 GPU-hours,
about $3-5, and about 2 h of wall-clock (derived: 3 full fine-tuning runs × 1.9 GPU-hours [Gl],
in parallel on three cards). Inputs to upload:
`~/mt_local/phase0_final/e05_results/data/phase0_e05/`.

### 3.6 W1: a second QE selector (WMT)

**Why.** One selector cannot support "QE top bands select repetitive text" [PA].

**Candidates and trade-off:**

- **Same family, larger:** CometKiwi-XL (`wmt23-cometkiwi-da-xl`) [OLD]. Tests model size, not
  model family.
- **Different family:** MetricX-24 used without references. Whether the hybrid checkpoint
  supports that mode: to check. Pombal et al. report a Jaccard overlap of 0.0307 between
  CometKiwi and MetricX top-k selections [Gl] in their setting (selections of 80,000 examples
  from a 10M OPUS pool); the overlap on our pool is to measure. If it is as low, the band would
  change substantially. That is a stronger test.
- **Not QE metrics:** Bicleaner-AI and LaBSE [OLD] score noise or similarity; using them would
  change the question.

**Recommendation:** a different-family QE metric, evaluated crosswise. MetricX-selected sets are
scored with COMET-22, chrF and BLEU, and CometKiwi-selected sets with MetricX, chrF and BLEU,
so no arm is judged by its own selector's family. Pombal et al. show that metrics favour
systems trained on data that the same or a related metric selected [Gl].

**Design:** score the reused pool (16,646,992 rows); rebuild the top 1M; apply the same rep rule
inside its dominant source; run the E0.5 triple (15 runs), plus the E0.6 contrast that mattered
(10 runs).

**Compute:** pilot the scorer on a sample first. CometKiwi-22 took about 22 GPU-hours for 38.3M
rows [R], which is about 9.6 GPU-hours for 16.6M rows at that rate (derived). An XL-size
model will be slower, by an amount to measure. Optional: the en-de pool, 4,238,227 rows, which
CometKiwi-22 scored in 1 h 35 min on one RTX 5090 [R].

### 3.7 W2: threshold sweep, the answer to Peter et al. (WMT)

**Why.** Peter et al. kept the top 50%, trained from scratch to convergence, and found QE at
or above random in every COMET22 cell [Gn:peter2023]. Waldendorf found CometKiwi beating random
from 10% kept on En-De [Gn:waldendorf2025]. Chimoto found the deficit mostly closed at 30% kept
[Gn:chimoto2024]. Tan saw damage only at the extreme top (> 89) [Gn:tan2024]. Our band is about
6%.

**Design:** at depths 6%, 12%, 25% and 50% of the ranking, draw a fixed-size sample from the
band and a random sample with the same size and length histogram; fine-tune with the E0.5
recipe; five seeds; about 40 runs. Fixed size keeps passes over the data equal. The sweep
answers "is the selected text worse than random at this depth", not "does keeping more data
help".

**What it cannot answer:** Peter et al.'s from-scratch regime. A from-scratch arm at two depths
would answer it. Its cost is not in the sources (to measure), and it is optional.

**Open:** all sources (as Peter et al.) or UN only (source-controlled), with source shares
reported either way; whether to add lr_scale 0.15 at each depth (risk 1).

### 3.8 W3: dose-response over repetitiveness (WMT)

**Why.** A median split is a two-point contrast. "Predicts" needs a graded relationship
[PA, risk 12].

**Design (feasibility to measure on CPU first):** split the UN top-band rows into rep = 0 plus
three or four bins of rep > 0 (227,406 repetitive UN candidates exist [F]); match length
histograms; draw one common set size; add random UN text at that size; five seeds. Fit BLEU
against mean rep with seed as a random effect, model and test registered before runs.

**The axis is tied to a named E0.6 outcome, and the choice stays the author's (§7, WMT).** The
default axis is repetitiveness. A dose-response on vocabulary (random UN rows narrowed to several
vocabulary caps) may be added, or substituted, only if E0.6's registered outcome is "N+ at the
full gap" ([PR] E0.6, N, which already requires S not detected) and its Joint reading applies;
what each other outcome licenses is that outcome's reading in [PR], not a new one here. The
vocabulary agent's scratch directory already holds candidate arms at several caps (files
`A_V30000_nn.npy` to `A_V50000_nn.npy`); their statistics are not in the memo and must be
measured. These index files (about 1.7 MB each) go into the repo with this plan (see the
header), or they are lost.

**Read first:** "The repetition rate of text as a predictor of the effectiveness of machine
translation adaptation". The prior-art search found it but nobody opened it [PA, Appendix B].
It bears on the EAMT paper too, not only on W3: open it before any EAMT sentence about
redundancy in fine-tuning data (§6.2, "Read before a design or claim depends on them", second
bullet), and in any case before any "predicts" wording.

### 3.9 W4: the remedy against the NRC redundancy baseline (WMT)

**Arms** at matched size, length and source, five seeds, all drawn from the QE top band unless
stated otherwise:

1. a raw sample of the top band;
2. the remedy, defined by the author after E0.6 reports (§7, WMT) and tied to a named E0.6
   outcome ([PR] E0.6, Outcomes): for example, drop rep > 0 rows as well as duplicate sources if
   C1-beyond-duplicates is YES and Q1′ replicates; re-rank to complete vocabulary coverage under
   "N+ at the full gap" only when its Joint reading applies (P holds and Q2′ replicates; [PR] E0.6,
   *Precondition P* and *Bridges*). A contrast that is not detected, or whose reading is withheld, selects no remedy;
3. NRC bicov: walk down the QE ranking and drop pairs that add no new source bigram
   [Gn:lo2018];
4. NRC discount: a 20% score penalty for pairs that add no new source bigram
   [Gn:berniercolborne2019];
5. exact plus near deduplication with BiCleaner `--aggressive_dedup`, Waldendorf et al.'s tool
   [Gn:waldendorf2025];
6. a random sample from the same source.

Optional: Saunders & DeNeefe's Minimal Mix-in, with a same-size random top-up as the control
their paper lacks [Gn:saunders2024]; arms 1, 2, 3 and 6 repeated in the LLM arm.

**Prior evidence the remedy must overcome.** NRC's bicov gained 2.44 BLEU for NMT at 10M words
in their own runs, 0.91 on average in the official six-domain evaluation, and less at 100M
words [Gn:lo2018]. Vicomtech's n-gram saturation lowered average NMT BLEU from 30.40 to 24.91 at
100M words. RWTH gained 0.0 to 0.4 under ranking; Volctrans 0.03 to 0.19 [PA]. The remedy is
reported whether or not it beats arm 3.

**Evaluation:** newstest2014, heldout_un and the secondary out-of-domain sets. The in-domain set
matters here: Waldendorf's narrow domain selections still won in domain [Gn:waldendorf2025].

---

## 4. Timeline, 17 October 2026 to submission

Rentals are R1 to R7. The author rents when told; each rental follows §6.5.

| dates | Claude (CPU) | author | rental |
| --- | --- | --- | --- |
| Sat 17 - Fri 23 Oct | Done 17 Oct: sources and scripts in `phase0/next/`, bulk files in `~/mt_local/phase0_next/`, sha256 of all of them in `phase0/next/SHA256SUMS` (header). By Wed 21 Oct, §3.2 item 1: fetch the SentencePiece model from the HF release, check `d1c10ea8…`, and compute coverage on the E0.5 sets and on R0 and A_narrow (the other E0.6 sets once `e06_build.py` generates them); [PR] E0.6 OD 1 and risk 2's mitigation need this statistic. Prepare the Mac for [PR] E0.6 OD 16's checks: use `~/mt_local/venv` (pyarrow 25.0.1 present), install `sentencepiece` there, and fetch the newstest2014 test split from the pinned `wmt/wmt14@b199e406`. Turn the E0.6 pre-registration draft into PROTOCOL.md text with its OPEN DECISIONS still open; write fail-closed `e06_build.py` and dry-run it on the Mac (it must reproduce `R0.npy` and `A_narrow_42_58.npy` row for row); CPU items 2-5 of §3.2. | Nothing to decide yet; the E0.6 decisions (§7) wait for the reading week. | none |
| Mon 26 Oct - Sun 1 Nov | Revise the existing drafts (the E0.6, E0.7 and E1.0 pre-registration drafts and the EAMT outline, all dated 2026-10-16) as the reading comes in. Prepare the inputs §7 lists for the E0.6 decisions: the piece-level and newstest2014 checks ([PR] E0.6 OD 16); Saunders & DeNeefe's coverage once the author has read them (OD 1); with `e06_build.py`, div_narrow's ±2% feasibility under each OD 1 option and each OD 2 rule, and whether div_narrow_dd's build passes (OD 5). | **Reading week**: the ten papers in §6.2, each with its first check; Erdmann & Gwinnup 2018 and Azpeitia et al. 2018 before the E0.6 decisions on Sunday 1 Nov (§6.2); then, as time allows, the other unopened pre-emption leads (§6.2). **Sun 1 Nov: answer [PR] E0.6 OD 1-18 (§7).** | none |
| Mon 2 - Sun 8 Nov | Write the E0.6 decisions into PROTOCOL.md; re-run `e06_build.py`'s dry run with the chosen options (div_narrow within ±2% under the chosen measure; div_narrow_dd if OD 5(b)) and fill [PR] E0.6's "Measured on the sets before any run" table from it; then the second pre-flight audit; tag `e06-prereg-v1` (target Tue 3 Nov); monitor R1, pull and sha-check results and decoded outputs; E0.6 report. Finish the E0.7 pre-registration; by Thu 5 Nov, estimate the en-de base cost from the original runs' logs (to check), so it is ready for the E0.7 decisions on Fri 6 Nov. | Rent R1 once the tag is pushed. If [PR] E0.6 OD 11 picks FLORES+ devtest, accept its terms on your own Hugging Face account and place the token (or the files) on the box before R1. Decide E0.7 (§7) by Fri 6 Nov. | **R1**, 4x RTX 5090, about 5-6 h with setup without E0.6-LR: E0.6 (40 runs, about 3.5-4 h, derived in §3.1; 45 runs and about 0.5 h more with OD 5(b)); the cache warm-up (25 minutes on E0.5's box, §6.5); re-decode E0.5's 30 kept checkpoints on newstest2014, whose outputs were not kept, and also on FLORES+ devtest if [PR] E0.6 OD 11 picks it (§3.2 item 6; time to measure). With E0.6-LR (OD 17), if registered separately, about 1.5 h more (15 × 0.39 / 4, derived): about 6-7 h. If the tag slips: a session of its own at the start of the R2 window, before the en-de work. |
| Mon 9 - Sun 22 Nov | Tag `e07-prereg-v1` before the base is trained; finish the E1.0 pre-registration. | Rent R2. Read Shang et al. (Table 5 setup) and Pang et al. 2024 (§6.2), then decide E1.0 (§7) by Fri 20 Nov. | **R2**, 4x RTX 5090: en-de base training (duration to measure), then 15 (or 30) fine-tunes. A free card can carry the zh-en run's continuation, which is outside this plan (§6.5). |
| Mon 23 Nov - Sun 6 Dec | Tag `e10-prereg-v1`; E0.7 report. | Rent R3. FLORES+ is gated: accept its terms on your own Hugging Face account and place the token on the box yourself, in every option; if Gemma, accept its licence too. | **R3**, 4x B200 (§6.5): E1.0 pilot and gate, then 30 runs (+5 or +10 optional, [PR] E1.0 OD 11; +3 optional, OD 1(d) / OD 14), about 14 h wall-clock as estimated for 4x RTX 5090, less on B200 (to measure in the pilot; about 2-4 h more with the OD 11 runs, about 2 h more with the OD 1(d) runs, on the same basis); extra metrics (chrF, MetricX-24, COMET-22) on E0.x outputs. |
| Mon 7 Dec - Sun 3 Jan | Analysis; EAMT draft v1 by Fri 18 Dec. | Second-tier reading (§6.2). | none |
| Mon 4 - Fri 29 Jan 2027 | Draft v2; red-team pass 1 (§6.3). **Experiment freeze Fri 29 Jan.** | Review. | **R4**, spare, only for a follow-up registered before 15 Jan. |
| Mon 1 - Fri 26 Feb | Draft v3; red-team pass 2. Check the EAMT 2027 call: deadline, anonymity, appendix. | Final read; decide on an OSF/Zenodo deposit and anonymisation. | none |
| Mon 1 Mar | **Submission-ready.** Submit when the portal opens, before the expected mid-March deadline. | Submit. | none |
| Mar - May 2027 | W2 and W3 CPU descriptives; W1-W4 pre-registrations, each tagged before its runs. | Choose the second selector; decide the W2 to W4 designs. | **R5** W1 scoring; **R6** W2 and W3 runs; **R7** W4. |
| Jun - Jul 2027 | WMT draft; red-team passes 1 and 2. | Handle EAMT notification per §2.2 (date to check). | none |
| WMT 2027 deadline (to check) | | Submit. | |

---

## 5. Risks, ranked

| # | risk | why it matters | mitigation |
| --- | --- | --- | --- |
| 1 | **The effect may exist only at an aggressive learning rate.** | Every repetitiveness run used lr_scale 1.0. At E0.3's rule-selected 0.15, the whole top band was not detectably worse than random (0.26, not significant) [R]. A reviewer can call the result an artefact of an aggressive fine-tuning rate or schedule [Gn:finkelstein2024]. | State the scope in the abstract. Propose **E0.6-LR** (not yet decided): the UN triple at lr_scale 0.15, 5 seeds, 15 runs, about 6 GPU-hours (derived), registered separately and run on R1; report whatever it shows. E1.0 picks its own learning rate on random text and dev data, which partly answers this for the LLM arm [Gl]. |
| 2 | **"Repetitive" may be a proxy for low vocabulary coverage.** | Saunders & DeNeefe tie forgetting to coverage. E0.5 matched size, length and source; exact-pair exposure and the halves' QE means were measured nearly equal, not matched by design; coverage was not matched; the repetitive half has the fewest types (25,134 against 33,874 and 58,441) [Gn:saunders2024]. | E0.6's coverage contrast (ft_un_rep against div_narrow); their coverage statistic computed on every set (§3.2); every outcome's reading registered before the runs ([PR] E0.6, Outcomes), so either outcome is a paper. |
| 3 | **"It is known."** | The phenomenon is published four times; NRC made the redundancy remark in 2018 and 2019 [PA]. Without the mechanism result there is no contribution. | §1.4 wording; credit in the introduction; the contribution is the controlled account of why. Correct the draft Related Work sentences per §6.2. |
| 4 | **The result may not generalise.** | One pool (77.0% UN after deduplication [L], 78.3% before [P]; §1.1), one pair and one metric [PA], one base model [R]. E0.7 is a weaker test of C1 ([PR] E0.7, opening paragraphs), and its Q2-de is diluted by 37% band rows in the random arm [Ge]. E1.0 may show a smaller gap, no gap, or a reversed Q2 (Pombal et al. [Gl]; Fujita de Mello & Takada [Gn:llmselect]). | The readings are drafted in [PR], to be registered before the runs: E0.7, Q1-de and Q2-de (the weaker en-de test); E1.0, Q1-LLM, Q2-LLM, F-Q1 and G-Q1. W1 adds a second selector. |
| 5 | **"Just deduplicate."** | WMT18 deduplicated its pool before scoring [Gn:koehn2018]; the repetitive UN half has 8.45% duplicate sources [P]. | E0.6's rep_dedup against div_dd; W4's deduplication arm. Waldendorf's En-Ar deficit was measured after near-deduplication (scope to check) [Gn:waldendorf2025]. |
| 6 | **Peter et al. found the opposite.** | QE at or above random in every COMET22 cell at 50% kept, from scratch [Gn:peter2023]. | Scope every claim to the extreme top band and fine-tuning; W2 locates where the sign turns. |
| 7 | **BLEU-only, and metric circularity.** | All E0.x verdicts are newstest2014 BLEU. Reviewers will ask for neural metrics; COMET-22 shares a family with the selector [Gn:peter2023, finkelstein2024] [Gl]. | chrF and MetricX-24 as secondary metrics and COMET-22 flagged, as [PR] E0.6 OD 11 and E0.7 OD 13 decide (recommended there); metric agreement in the LLM arm as [PR] E1.0 OD 7 fixes it (still open); a second out-of-domain set for the Transformer-base arm as [PR] E0.6 OD 11 fixes it (candidate FLORES+ devtest; WMT24++ is paragraph-level and not used for this arm). |
| 8 | **Schedule.** | The EAMT call is not out; the en-de base cost is unknown; rentals depend on the author's time; the reading week takes one week out of the run-up. | Freeze on 29 January; cut order in §2.1; estimate the en-de base cost by Thu 5 November, before the E0.7 decisions of Fri 6 November (§7); keep R4 as a spare. |
| 9 | **Process errors.** | This project has twice asserted unverified numbers or attributions [P]. The rep split depends on the interpreter's Unicode tables [P] [Ge]. Rented boxes are destroyed after use, and E0.3's stage-1 logs were lost that way [R]. | §6.2 citation rule; §6.3 red-team passes; build runs of record on the box and pin every sha256; the rental checklist in §6.5. |
| 10 | **Overlap and anonymity between EAMT and WMT.** | Both papers come from one project; PROTOCOL.md requires only that the E0.3 decisions file be in a tag and on a remote branch [P]. Whether the repository is public is to check; the anonymity mitigation depends on it. | Read both calls' rules (to check); anonymise the pre-registration link if required; keep the WMT headline results disjoint from EAMT's (§2.2). |
| 11 | **The WMT remedy may fail.** | Prior redundancy controls helped NMT little or hurt it [PA]. | Report it either way; W1 to W3 carry the WMT paper without it. |
| 12 | **"It is QE miscalibration."** | Waldendorf et al. attribute CometKiwi's pair-dependent behaviour to whether the pair was in its human DA training data; a reviewer can read our top band as CometKiwi-22 miscalibration on en-fr UN text rather than repetitiveness [Gn:waldendorf2025] [PA, risk 13]. | Check CometKiwi-22's DA training coverage for en-fr and en-de before submission [Gn:waldendorf2025]. Give the explanation a sentence next to Chimoto's (§3.2 item 3). E0.5's within-UN halves have equal QE means (observed, not built in), which only partly answers it. |

---

## 6. Standing rules kept from the old plan

### 6.1 Pre-registration discipline

[OLD] E1.3 asked for pre-registration before any run. E0.4 and E0.5 turned that into a house
style, which every experiment in this plan follows [P]:

1. **Before any run:** the hypothesis, the statistic, the sets and their build rule, the seeds,
   the decision rule, and every outcome with its reading, written in PROTOCOL.md and git-tagged
   (for example `e05-prereg-v1`). A second pre-flight audit before the tag; E0.5's amendments
   came from one (git log, commit `0fb52a6`).
2. **Decision rule:** a gap counts only if it exceeds the seed-noise floor (the larger seed sd of
   the two sets) and is Welch-significant at 0.05 with the exact critical value at the
   fractional df. A gap that fails is reported as "not detected", with the smallest gap the rule
   could have called beside it, never as "equal" [P].
3. **Fail-closed builds:** pinned input sha256s; exact matching or no build [P].
4. **Tokens:** sentence-matched sets; the 2% imbalance flag with a bound on its worth printed for
   each deciding pair [P].
5. **Not to be done after seeing results:** more sets, more seeds, other step counts, another
   statistic or threshold, or pooling with earlier experiments [P].
6. **Deviations:** recorded before the runs. If a registered decision is violated, the runs are
   thrown away, even when the violation turned out not to matter (the D9 spike-guard incident
   in E0.4) [P].
7. **Probes are not gates.** Nothing in this plan revives E0.3's NO-GO [P].
8. **Kept from PROTOCOL.md §6.6-6.7 as rules for every paper:** a section titled "Deviations
   from pre-registration" appears in the submitted paper whether or not it is empty, and every
   pre-registered cell is reported regardless of outcome [P]. The rest of PROTOCOL.md §1-7 is
   a record only (§0.4).

**Record.** Git tags in this repo are the timestamp of record. These sources show no OSF or
Zenodo deposit, which [OLD] planned (to check). Decide before the EAMT submission, with the
anonymity rules in mind.

### 6.2 Citation rule

Kept from PROTOCOL.md ("A non-reviewer risk") and PRIOR_ART_E04.md: this project has twice
asserted things about papers nobody had read [P] [PA]. Therefore:

1. A paper enters the `.bib` only after the author has opened it. The groundwork notes and the
   36-paper check are a floor, not a substitute [PA].
2. Every number attributed to a paper is located in that paper, by table or section. Numbers
   derived from its tables are labelled as ours.
3. Untested remarks in other papers are "observed" or "noted", never "showed" [PA].
4. No "first to observe" anywhere. "First to measure for CometKiwi" only if the reading still
   supports it [PA].
5. Every project number is re-derived from released artifacts before submission (§6.3).
6. New in this plan, not in PROTOCOL.md or PRIOR_ART_E04.md: at most one short verbatim quote
   per paper; paraphrase otherwise.

**Corrections to carry into every draft:**

- **[OLD] cites "Zhou & Savova (arXiv 2503.07329)". The arXiv page lists Bui, Savova and Wang
  (IJCNLP-AACL 2025), and the abstract does not mention LoRA.** Cite it as Bui, Savova and Wang
  (2025). [OLD]'s attributions to it (a ten-seed LoRA design, an audit of 85 ACL papers) were
  written before anyone read it, and only the abstract has been seen since [Gl]. Check the full
  text before attributing anything beyond the abstract.
- UN share of the top band: 78.3% of the 1,000,000 rows before deduplication [P]; 77.0% of the
  926,999 deduplicated rows the experiments use (713,881; derived from [L]), which is
  PRIOR_ART_E04.md's 77% [PA]. Name the set each time; the trained band is 77.0%.
- The E0.5 random arm is not QE-matched: 0.8629 against 0.8990 for the diverse half; only the
  two halves have equal QE means, 0.8990 against 0.8993, and that was observed, not built in
  [Gv]. Several draft sentences in [Gn]
  (`how_to_position`) say the random comparison is matched on QE score. Fix them.
- E0.5's UN sets are matched by construction on source, size and length histogram only [B5].
  Write "exact-pair pretraining exposure (36.7 / 36.7%) and mean QE (0.8990 / 0.8993) measured
  nearly equal, not matched by design"; README's "matched pretraining exposure" [R] is not
  carried over.
- The [Gn:chimoto2024] draft sentence saying repetitiveness "accounts for most of the
  out-of-domain loss" overstates. No share is claimed (§1.4, item 8).
- The [Gn:peter2023] and [Gn:llmselect] draft sentences (`how_to_position`) say the selected
  text, or the QE top band, is dominated by repetitive text. It is not: the median rep is 0.0,
  and only 250,008 of 926,999 band rows (27.0%) have rep > 0 [P] [Ge]. Write "about a quarter
  of the band has rep > 0"; never say the band is dominated by repetitive text.
- Nine of the ten [Gn] notes (all but Gn:saunders2024) call the base "converged", for example
  Gn:peter2023 `how_to_position` ("fine-tuning a converged en-fr model") and the setting
  comparisons in the others. Gn:tan2024 `relation_to_C1` calls it "a from-scratch
  Transformer-base", and Gn:llmselect `relation_to_C2` "our converged from-scratch
  Transformer-base". The base is the released, pretrained v1.1 Transformer-base (step 105,000,
  averaged), fine-tuned. Never "converged" (README E0.2 finds the original cells under-trained
  [R]) and never "from scratch". [F] line 29 ("PROPOSED E0.6 (from scratch, ...") and the llm
  memo [Gl] (six times, for example "the from-scratch vocabulary experiment") also call the
  Transformer-base arm "from scratch"; never carry that wording over.
- Draft sentences that make repetitiveness the cause: "this repetitiveness, rather than source
  mix, drives the out-of-domain loss" (Gn:peter2023 `how_to_position`); "to show that this
  repetitiveness, not source mix, drives the out-of-domain loss" (Gn:llmselect
  `how_to_position`);
  "attribute the out-of-domain loss to the repetitiveness ... rather than to its source mix"
  (Gn:tan2024); "we show it accounts for a 2.6 BLEU out-of-domain loss"
  (Gn:berniercolborne2019). The repetitive half is a bundle (shared 8-grams, narrower
  vocabulary, 8.45% duplicate sources) whose cost is not reducible to source mix (§1.1, §1.4
  item 12). Never write that it "drives" or "accounts for" the loss.
- Gn:finkelstein2024's draft says "the damage tracks how repetitive the selected text is". A
  median split is a two-point contrast: no "tracks" or other graded verb before W3 reports
  (§1.4, item 7).
- Chimoto et al. 2024 used COMET-QE (2020), not CometKiwi; the paper does not state which end
  of the ranking it kept; it has no en-fr QE arm [Gn:chimoto2024].
- Lo et al. 2018 scored with YiSi-1 on machine-translated sources, not a reference-free QE
  metric. Bernier-Colborne & Lo 2019 call YiSi-2 a QE metric, but it is an embedding
  similarity, and their coverage re-ranking was never ablated [Gn:lo2018, berniercolborne2019].
- Waldendorf et al. 2025, En-De top 1%: a tie with random on chrF++ (54.8 against 54.7), lower
  on COMET only. Do not cite it as below random on chrF++ for En-De [Gn:waldendorf2025].
- Finkelstein et al. 2024: no random control, no BLEU, BLEURT-QE [Gn:finkelstein2024].
- Tan et al. 2024: "100k Cleaned" is called a random sample in the caption and low-QE data in
  the text; hedge [Gn:tan2024].
- Peter et al. 2023 and Waldendorf et al. 2025 do not name their CometKiwi checkpoint. Do not
  write "the same metric as ours" [Gn:peter2023, waldendorf2025].
- Fujita de Mello & Takada 2025: Tables III and IV contain apparent copy errors; read the PDF
  before citing any cell [Gn:llmselect].
- [OLD]'s statistics about WMT papers (acceptance rates, seed use, paper genres) were never
  verified. Do not reuse them unchecked.

**Reading week, 26 October - 1 November 2026: the ten papers read in full in [Gn].**

| paper | check first |
| --- | --- |
| Waldendorf et al. 2025 (Findings of EMNLP) | the CometKiwi checkpoint; whether deduplication covers En-De |
| Chimoto et al. 2024 (Findings of ACL) | which end of the ranking was kept; which differences are bold (significant) |
| Tan et al. 2024 (WMT) | what "100k Cleaned" is; whether the > 87 set includes the > 89 pairs |
| Finkelstein et al. 2024 (WMT) | the size of the QE-filtered set; what the pool contains |
| Lo et al. 2018 (WMT) | pair counts behind 10M and 100M words; how bicov refills the budget |
| Bernier-Colborne & Lo 2019 (WMT) | that the coverage re-ranking was never ablated |
| Saunders & DeNeefe 2024 (EMNLP) | their coverage definition (subword, source and target); single runs; whether the Kendall tau pools the 5 de-en and 3 en-ja domains (n = 8) or is per pair |
| Peter et al. 2023 (WMT) | the CometKiwi checkpoint; one run per condition; Random left out of significance tests |
| Koehn et al. 2018 (WMT findings) | no random baseline; the pool was deduplicated before release |
| Fujita de Mello & Takada 2025 (arXiv 2512.11388) | the table copy errors; which test set was used |

**Second tier, December - January:** Kirchhoff & Bilmes 2014; Rossenbach
et al. 2018; Lu et al. 2018; Xu et al. 2020; Wang et al. 2018 (NICT, cite Anthology W18-6489);
Koehn et al. 2019; Pombal et al. (arXiv 2503.08327); ALMA (Xu et al., ICLR 2024). (Shang et al.
moved up: see the E1.0 item below.)

**Read before a design or claim depends on them.** This is the one list of extra reading and its
deadlines; `reading_list.md` §3.5 and the EAMT outline point here.

- **In the reading week, before the E0.6 decisions on Sun 1 Nov:** Erdmann & Gwinnup 2018 (AFRL,
  "Coverage and cynicism"; not opened by anyone) and Azpeitia et al. 2018 (Vicomtech, n-gram
  saturation; opened by the prior-art checkers [PA]), at least their coverage and saturation
  definitions. Gn:koehn2018 asks for both before the vocabulary arm is designed, so that an
  existing definition is reused rather than a new one invented; they bear on [PR] E0.6 OD 1 and
  OD 2, and once results exist [PR] E0.6 forbids another coverage measure.
- **Before any EAMT sentence about redundancy or a coverage control in the selected data** (in
  the reading week after the ten papers if time allows, at the latest before draft v1 on 18
  December), none opened by anyone: the WMT19 filtering system papers behind Koehn et al. 2019,
  Erdmann & Gwinnup 2019 (AFRL), Gonzalez-Rubio 2019 (Webinterpret) and Axelrod 2019 (DiDi,
  cynical selection), which PRIOR_ART_E04.md flags as where the real pre-emption risk for its C3
  (that QE selection needs a diversity or redundancy control; that a repetitiveness statistic
  predicts the harm) sits [PA, §5]; the repetition-rate predictor paper [PA, Appendix B], also
  before any "predicts" wording; Karpinska et al. 2022 (metric sensitivity to repetitions) [PA,
  §5].
- **Before the E1.0 decisions on Fri 20 Nov** (the LoRA design and the pilot gate are both fixed
  then): Shang et al. (arXiv 2602.11961), at least Table 5 and its evaluation setup (base or
  instruct model, FLORES+ split, spBLEU tokenizer, COMET model, and the column-to-model mapping),
  because its numbers become the pilot gate ([PR] E1.0 P1, OD 2); and Pang et al. 2024 (LLM
  terminology forgetting; not opened), since Gn:saunders2024 says to read it before fixing the
  LoRA design.
- **Others, not opened:** Biderman et al. 2024, Bui, Savova and Wang 2025, Kocyigit et al.
  (arXiv 2501.18771, contamination) and Tan, Chen, van Genabith and Dutta Chowdhury (EACL 2026
  short, FLORES contamination) beyond their abstracts [Gl]; the WMT20 filtering findings [PA];
  Littell et al. 2018 (the NRC unsupervised submission; Gn:lo2018). Pombal et al. (second tier,
  above) was read by the llm memo's author (sections 3.2, 4.1, Tables 1, 7-9, App. A [Gl]),
  not yet by the author.

### 6.3 Red-team pass

Kept from [OLD] (month 12: re-run every arithmetic check a reviewer would run) and from the
2026-09-04 audit practice (40 agents, 6 lenses, every finding independently refuted) [R]. Two
passes before each submission:

- **Pass 1, claims (late January for EAMT):** an adversarial audit of every claim against its
  evidence. Each "matched on X" sentence is checked against the build manifests. The §1.4 list
  is checked against the abstract, introduction and conclusion. Each Related Work sentence is
  checked against the opened paper.
- **Pass 2, numbers (February for EAMT):** every number re-derived by script from released
  artifacts: set sizes, sha256 pins, source shares, pretraining exposure, QE means, passes per
  arm (for example 9.0 against 7.9 in E0.5 [P]), applied-token imbalances and their bounds,
  Welch df and exact critical values, noise floors and the smallest callable gaps. Differences
  between registered wording and code are stated (the 44-row case [P]).

### 6.4 Publication path

Peer review only; no arXiv. V1 was desk-rejected on moderation, and the standing rule holds
[OLD]. Release with the paper: build scripts, manifests, sha256 lists, per-run logs with token
counts, and the report scripts.

### 6.5 Rentals and credentials

- The author rents; Claude says when and for what.
- Rented boxes are destroyed after use. Upload inputs from `~/mt_local`, verify sha256, run,
  then pull results, logs and checkpoints and check their sha256 before release. Copy logs
  first; E0.3's stage-1 logs were lost [R].
- Known setup issues from E0.5 [R]: the image's pyarrow 26.0.0 needs NumPy 2 while the env pins
  1.26.4 (pyarrow 19.0.1 was installed); the tokenisation pool is sized at `cpu_count() // 2`,
  192 processes on a 384-thread host, which made the cache warm-up take 25 minutes. Cap it.
- RTX 5090 (Blackwell) needs CUDA 12.8 or later builds; E0.5 ran torch 2.14.1+cu132 [Gl] [R].
- Hardware (the author's choice, 17 October 2026): every training and retraining job (the
  fine-tune matrices, the en-de base, the zh-en run) uses a standard 4-GPU rental, 4x RTX 5090;
  the LLM arm (E1.0) uses 4x B200. The author's own RTX 5090 is no longer used for this work. The
  MT trainer (`Machine_translation/train.py`) is single-GPU (no DDP): a single model, such as the
  en-de base, trains on one card while the other cards take other jobs, as E0.5's runs did.
  Multi-GPU training for one model would be a separate change, tested before use.
- Claude never handles credentials. FLORES+ (`openlanguagedata/flores_plus`) is gated and
  stages 0-1 of E1.0 need it (pilot gate, learning-rate pick), so in every option the author
  accepts its terms on their own Hugging Face account and places the token on the box. The
  model Qwen3-1.7B-Base is ungated; Gemma adds a second licence for the author to accept [Gl].

---

## 7. Decisions needed from the author, by date

The decisions themselves, with their options and recommendations, are [PR]'s OPEN DECISIONS;
this section gives only when each list is due and what must be ready before it.

**By Sunday 1 November, at the end of the reading week (E0.6)** [F] [Gv]: [PR] E0.6 OPEN
DECISIONS 1-18, the full list (OD 10 is a check, not a choice). They are taken after the reading,
so that the coverage measure (OD 1) is chosen once Saunders & DeNeefe, Erdmann & Gwinnup 2018 and
Azpeitia et al. 2018 have been read (§6.2); the post-decision dry run and the tag follow on Mon 2 -
Tue 3 Nov (§4). Inputs Claude prepares beforehand (§4, rows of 17 Oct and 26 Oct), so that each
answer is made with numbers in hand:
- OD 1 and OD 15: Saunders & DeNeefe's subword coverage on every set (§3.2 item 1);
- OD 1 and OD 2: div_narrow's ±2% feasibility under each coverage measure and narrowing rule;
- OD 4: the duplicate rates under the normalised definition, descriptively;
- OD 5: whether div_narrow_dd's build passes;
- OD 16: the piece-level and newstest2014 checks, both on the Mac (no rental);
- OD 11: if FLORES+ devtest is chosen, the author handles its gate before R1 (§6.5);
- OD 17 (E0.6-LR, risk 1): if yes, its own section and tag before R1.

**By Friday 6 November (E0.7)** [Ge]: [PR] E0.7 OPEN DECISIONS 1-13 (6 and 12 are checks). OD 3
(the base) and OD 9 (its recipe and stopping rule) come before the en-de base is trained, so
before R2. Input: the en-de base cost, estimated from the original runs' logs by Thu 5 Nov
(§3.4, Compute; §4).

**By Friday 20 November (E1.0)** [Gl]: [PR] E1.0 OPEN DECISIONS 1-14 (12 is a check; 15 is done).
OD 11, the vocabulary-matched set or sets, is answered later, by an amendment registered after
E0.6 reports and before E1.0's first run. FLORES+ needs the author's token on the box in every
option (§6.5). Shang et al. (Table 5 setup) and Pang et al. 2024 are read before the E1.0 decisions on Fri 20 Nov (§6.2).

**By Monday 1 March 2027 (EAMT):** an OSF or Zenodo deposit, or not; anonymisation of the
pre-registration link.

**By the end of March 2027 (WMT):** the second selector; the W2 design (fixed size or not; all
sources or UN only; lr_scale 0.15 or not; a from-scratch arm or not); the W3 axis (repetition
or vocabulary, tied to a named E0.6 outcome in [PR], §3.8); the W4 remedy definition (§3.9, arm 2);
whether to run Design C (only if E0.6's S is detected and E0.6 gives S a reading: P holds and Q2′ replicates, [PR] E0.6 *Precondition P*, S+ and *Bridges*; §3.3, follow-up arms). The NRC re-ranking
and near-duplicate arms are placed in W4, not before EAMT (§3.3).
