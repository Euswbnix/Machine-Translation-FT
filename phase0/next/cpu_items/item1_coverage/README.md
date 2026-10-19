# item1_coverage: subword vocabulary coverage, Saunders & DeNeefe's way

WMT2027_PLAN.md §3.2 item 1; feeds [PR] E0.6 OPEN DECISION 1 and plan risk 2. This is a descriptive CPU
analysis for the author's decisions. It is not a registered result and it is not linked to any BLEU.
Run 2026-10-19 on the Mac Studio (M5 Max, Python 3.14.3 arm64, Unicode 16.0.0, numpy 2.4.5,
sentencepiece 0.2.2, 4 encoding threads, `nice -n 5`). It took 223 s. All 37 checks in `results.json`
pass.

Files: `coverage.py` (the script that was run), `results.json` (every number below, plus more,
except the few marked "independent check"). Those come from `../item1_coverage_check/check_results.json`,
an independent re-derivation that does not import `coverage.py` and agrees with all 342 numbers it
compared. On the Studio, `~/cpu_items/item1_coverage/` holds the same two files plus `run.log`,
`sha256_check.log` and `used_inputs_SHA256SUMS`. That last file is
`~/mt_local/phase0_next/e06data_SHA256SUMS` with the lines for `v2_scored.tsv`, news-commentary and
`lex.pkl` removed, since this item does not read them. All 25 inputs it reads matched their sha256.

## Definition used

Sources:
- `groundwork.json`, note `saunders2024`: `setup` ("Coverage is the share of the generic subword
  vocabulary that appears at all in the domain set"), `author_must_check` item 8 ("at least once,
  computed separately for source and target"), `useful_for_design` item 1.
- `prereg_drafts.md` L560-574 (OD 1(c)).
- `e06_build_spec.md` A28 (L849-856): which ids count.
- `e06_build.py` `encode_pieces` and `piece_measures` (L2007-2032), which this script mirrors.
- `Machine_translation/src/data/dataset.py` L42-43 and L158-161 (rev `200f6c0`): how the trainer
  tokenises.

**Headline measure.** Each row is read with LF as the only line break (`e06_build.py` `read_lines`),
then `.strip()`ped and encoded with `spm_enfr_v1_fixed.model`'s defaults: no sampling, and no BOS/EOS.
`coverage` = the number of distinct piece ids in the set on that side, excluding the control ids
(`<pad>`, `<s>`, `</s>`) and `<unk>`, divided by 31,996. That is the model's 32,000 ids minus those 4.
Source and target are counted separately. The model is one joint en-fr BPE vocabulary (scores
= −rank, no byte or unused pieces), so both sides have the same denominator. Its TrainerSpec
(independent check) is BPE, vocab_size 32,000, trained on both `train.clean.en` and `train.clean.fr`,
with no byte fallback.

The trainer does not read LF-only: `dataset.py` opens the files in Python text mode (`open(path, "r")`,
universal newlines), which would also break a line at CR. That makes no difference here, because no
set file and no pretraining line contains a CR (`results.json` `cr_rows_in_files`, `lines_with_cr`).
Neither reader breaks at U+2028, VT or FS/GS/RS, which occur in a few `ft_random` rows (independent
check: 3, 2 and 2 English rows; 3, 2 and 0 French rows). `.strip()` changes no row in any set file
(independent check), so A28's "encode raw lines" and the trainer's strip give the same ids here.

**Where the paper is ambiguous, and what is reported for each:**

| question | headline choice | variant in `results.json` | does it change the gaps? |
| --- | --- | --- | --- |
| types or token-weighted | types ("appears at all") | `pretraining_token_share_uncovered_ppm`: share of the base's own pretraining piece tokens on that side whose piece is absent from the set | yes: relative gaps are much larger (table below) |
| special pieces | excluded (A28) | none needed: only 4 specials, and no set has more than 43 `<unk>` tokens | no |
| minimum count | k ≥ 1 | k ≥ 2, 5, 20 | at k = 20, larger for rep − div and div − random on both sides and for A_narrow − R0 on the source side; A_narrow − R0's target gap narrows (−8.9% → −8.4%) |
| denominator | all 31,996 countable ids | `coverage_of_side_vocab`: only ids the base saw on that side of `train.clean` (31,036 en, 31,695 fr) | relative gaps move by at most 0.035 points (largest: A_narrow − R0 source, −7.386% → −7.351%), so they agree at the one decimal printed. Set types outside that vocabulary: ≤ 8 for the 214k/242k sets, 23 source / 4 target for ft_random (ref.) |
| size | the whole set | `coverage_at_matched_pieces`: first 6.3M source / 7.7M target pieces in a `Random(0)` shuffle (`types_at`'s rule) | ≤ 0.16 points of relative gap for the three contrasts asked for; on the target side 0.24 for S and 0.25 for P (0.39 for ft_topk_div − ft_random_small, an E0.4 contrast that is only in `results.json`) |
| what the trainer keeps | all rows | `coverage_rows_kept_by_trainer` | the trainer drops 0 rows from every set except `ft_random` (268) |
| user-defined pieces | none exist | — | no. The independent check reads the model's piece types with a hand-written protobuf reader that agrees with sentencepiece id by id: 31,996 NORMAL, 3 CONTROL (`<pad>`, `<s>`, `</s>`), 1 UNKNOWN, 0 user-defined. `coverage.py` itself does not read piece types |

**Word types (prereg OD 1(a)/(b)), for comparison.** Lowercased `\w+` tokens.
- `source_word_types_D11`: over the whole set (prereg D11).
- `french_word_types_at_6500000_D12`: the first 6.5M French tokens, rows in a `Random(0)` shuffle
  (prereg D12, `vocab/common.py` `types_at`).
- `source_word_types_at_5600000`: the same rule on the source side, at a budget every set reaches.
  The smallest sets have 5,606,498 source words.

For all 214,489-row sets the source word count is identical (5,606,498), so D11 is already at matched
word count.

**R0 and A_narrow rows.** These are indices into `ft_random` in its LF line order (`e06_build.py`:
`Parent("ft_random")`, `build_r0`, `build_a_narrow`). The two pinned copies of `e06_build.py` (Studio
`64014a7b`, repo HEAD `577b13c5`) differ only in the reference-interpreter constant. Checked:
- 214,489 rows, strictly ascending, all labelled `un`, all with `rep_rand = 0`.
- RU0 = 450,709.
- Both sets have exactly ft_un_div's length histogram. The same indices read one line later do not.
- R0.npy is rebuilt row for row by `match_random` with seed 42.
- `ft_un_random`'s file is rebuilt line for line from ft_random by the same rule.
- A_narrow lies inside the 42k/58k capped pool, which has 402,010 rows.
- 136,916 rows are shared with R0.
- Word types reproduce every figure in the spec and E0.5's manifest: R0 62,000 / 72,574;
  A_narrow 33,562 / 43,703; ft_un_rep 25,134 / 32,684; ft_un_div 33,874 / 43,021;
  ft_un_random 58,441; ft_topk_div 38,783; ft_topk_rep 27,524; ft_random_small 94,897.

## Numbers

Coverage in % of 31,996 ids (piece types in brackets). "uncov." is the share of the base's pretraining
piece tokens, on that side, whose piece never occurs in the set.

| set | rows | src cov | tgt cov | src uncov. | tgt uncov. | src word types (D11) | src @5.6M | fr @6.5M (D12) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ft_un_rep | 214,489 | 54.10 (17,310) | 59.32 (18,979) | 1.72% | 2.33% | 25,134 | 25,131 | 32,684 |
| ft_un_div | 214,489 | 58.06 (18,576) | 63.47 (20,307) | 1.10% | 1.76% | 33,874 | 33,855 | 43,021 |
| ft_un_random | 214,489 | 65.02 (20,805) | 75.95 (24,300) | 0.45% | 0.94% | 58,441 | 58,420 | 69,320 |
| A_narrow | 214,489 | 60.66 (19,410) | 69.64 (22,283) | 0.67% | 1.21% | 33,562 | 33,555 | 43,703 |
| R0 | 214,489 | 65.50 (20,958) | 76.48 (24,471) | 0.42% | 0.92% | 62,000 | 61,958 | 72,574 |
| ft_topk_rep | 242,022 | 56.15 (17,966) | 61.84 (19,787) | 1.12% | 1.20% | 27,524 | 26,575 | 34,789 |
| ft_topk_div | 242,022 | 60.36 (19,314) | 66.24 (21,194) | 0.59% | 0.76% | 38,783 | 36,829 | 47,216 |
| ft_random_small | 242,022 | 73.38 (23,480) | 84.91 (27,167) | 0.04% | 0.03% | 94,897 | 88,876 | 105,692 |
| ft_topk (ref.) | 926,999 | 64.09 (20,507) | 70.83 (22,662) | 0.26% | 0.39% | 61,190 | 36,545 | 46,446 |
| ft_random (ref.) | 926,999 | 82.07 (26,258) | 91.95 (29,419) | 0.01% | 0.01% | 195,719 | 89,484 | 105,989 |

ft_topk and ft_random are 4.3 times larger than the other sets, so their whole-set coverage is not size-matched. At 6.3M / 7.7M pieces their coverage is 59.83 / 65.63 (ft_topk) and 72.29 / 83.46 (ft_random).

**Gaps for the E0.6 contrasts** (a − b, with the relative gap (a − b)/b in brackets):

| contrast | src cov | tgt cov | src uncov. | tgt uncov. | src word types | fr types @6.5M |
| --- | --- | --- | --- | --- | --- | --- |
| ft_un_rep − ft_un_div | −3.96 pp (−6.8%) | −4.15 pp (−6.5%) | +0.62 pp (+56%) | +0.58 pp (+33%) | −8,740 (−25.8%) | −10,337 (−24.0%) |
| ft_un_div − ft_un_random | −6.97 pp (−10.7%) | −12.48 pp (−16.4%) | +0.66 pp (+146%) | +0.81 pp (+86%) | −24,567 (−42.0%) | −26,299 (−37.9%) |
| A_narrow − R0 | −4.84 pp (−7.4%) | −6.84 pp (−8.9%) | +0.25 pp (+59%) | +0.29 pp (+32%) | −28,438 (−45.9%) | −28,871 (−39.8%) |
| *extra:* A_narrow − ft_un_div (S) | +2.61 pp (+4.5%) | +6.18 pp (+9.7%) | −0.43 pp (−39%) | −0.55 pp (−31%) | −312 (−0.9%) | +682 (+1.6%) |
| *extra:* ft_un_div − R0 (P) | −7.44 pp (−11.4%) | −13.01 pp (−17.0%) | +0.68 pp (+161%) | +0.84 pp (+91%) | −28,126 (−45.4%) | −29,553 (−40.7%) |

At k ≥ 20 the relative gaps widen for rep − div (−11.8% src / −10.3% tgt) and div − random (−22.3% /
−19.9%). A_narrow − R0 widens on the source side (−7.4% → −8.9%) but narrows on the target side
(−8.9% → −8.4%).

**How much the result depends on the draw.** The R0 rule and ft_un_random's rule were each redrawn
with seeds 1-5 at the same histogram. The draw-to-draw sd of coverage is:

| rule | source | target |
| --- | --- | --- |
| R0 | 0.13 pp (40 types) | 0.10 pp (33 types) |
| ft_un_random | 0.14 pp (44 types) | 0.18 pp (59 types) |

Every coverage gap above (the src cov and tgt cov columns) is at least 18 times the larger sd on its
own side. The smallest case is S on the source side: 2.61 pp, 18.9 times 0.138 pp. Against the largest
sd on either side (0.185 pp, ft_un_random target) it is 14 times. With only 5 draws the sd is itself
loose: a 95% interval for it runs from about 0.6 to 2.9 times the value shown (4 degrees of freedom).
Even at 2.9 times 0.185 pp, the smallest coverage gap is about 5 times the spread, so the coverage gaps
are well above draw-to-draw spread.

This does not carry over to word types. Their redraw sd is 150-245 source and 174-305 French types
(`results.json` `resampling`), so the S row's word-type gaps (−312, +682) are only 1-4 times it.

**Comparing the two measures (OD 1).**
1. For rep − div and div − random, subword coverage shrinks the gap. The relative gap in piece types
   is about a quarter of the relative gap in word types for rep − div (0.26 src, 0.27 tgt) and for
   div − random on the source side (0.25), and 0.43 for div − random on the target side.
2. Whether the two measures put rep − div and div − random in different proportions depends on the
   scale:
   - On absolute gaps, rep − div is 36% (src) and 39% (fr) of div − random in word types, and 57%
     (src) and 33% (tgt) in piece coverage.
   - On relative gaps, it is 61% (src) and 63% (fr) in word types, and 64% (src) and 40% (tgt) in
     piece coverage.
   - So on the source side the difference comes only from the absolute scale. Only the target side
     differs on both scales: there piece coverage gives rep − div a smaller share.
3. **Matching on word types did not match subword coverage** (S row). A_narrow is within −0.9% /
   +1.6% of ft_un_div on word types, but it covers 4.5% more source pieces and 9.7% more target
   pieces. That is outside ±2% on both sides.
4. On the subword measure, A_narrow's narrowing covers only part of the P gap:
   - In word types it covers 101% (src) and 98% (fr) of the gap between R0 and ft_un_div.
   - In piece types it covers 65% (src) and 53% (tgt).
5. If OD 1(c) were chosen, `div_narrow`'s target is ft_un_rep at 17,310 source and 18,979 target
   piece types. ±2% is about 16,964-17,656 and 18,600-19,358.

## Caveats

- **Sets not covered yet.** `div_narrow`, `rep_dedup` and `div_dd` do not exist yet, so they have no
  numbers here. Measure them the same way once `e06_build.py` writes them; `e06_build.py --measure`
  computes the same headline piece types. Point 3 above shows that a set matched under OD 1(b) cannot
  be assumed to match under (c).
- **Absolute levels are not comparable to the paper.** Saunders & DeNeefe report target coverage of
  0.23-0.72 on 16K/32K BPE vocabularies for other language pairs and other domains. Here the vocabulary
  is joint, so English-only and French-only pieces sit in both denominators. Only gaps between our own
  sets are meaningful.
- **The resampling sd is narrow by construction.** The redraws overlap heavily, because they come from
  the same finite pool. The sd is draw-to-draw spread under these rules, not population sampling error.
- **The pretraining side counts all lines.** The token-weighted variant and `coverage_of_side_vocab`
  count all 9,312,233 lines of `train.clean`. The trainer would drop 2,261 of those pairs (a side over
  256 pieces with BOS/EOS). Counting only the kept pairs changes the uncovered shares by ≤ 0.007 pp
  and their relative gaps by ≤ 0.8 points, and the seen vocabularies become 31,029 / 31,693 instead
  of 31,036 / 31,695 (independent check). Negligible.
- **Approximate figures.** The quotients in "Comparing the two measures", the sd multiples and the ±2%
  windows are arithmetic on the `results.json` values, rounded. The 0.6-2.9 interval for the sd is the
  chi-square interval with 4 degrees of freedom. Figures marked "independent check" come from
  `../item1_coverage_check/check_results.json`. Everything else is printed by the script.
- **Interpreter.** The Unicode tables (16.0.0) differ from the E0.5 build's (15.0.0). Every word-type
  figure from E0.5's manifest and the spec still reproduced exactly.
- **Not a mechanism.** Coverage counts a piece once even if it occurs only inside one repeated template.
  The k ≥ 2/5/20 variants address this only partly. These are descriptive statistics, and nothing here
  says which measure drives forgetting.

## Open points

- **Independent-check figures are not in `results.json`.** The piece types and TrainerSpec, the
  kept-pairs-only pretraining figures, and the U+2028 / VT / FS/GS/RS and `.strip()` row counts come
  from `../item1_coverage_check/check_results.json`. Having `coverage.py` report them itself would need
  a code change (a protobuf reader for piece types, a kept-pairs pass over `train.clean`) and a rerun.
