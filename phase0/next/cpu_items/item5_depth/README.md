# §3.2 item 5: depth descriptives for W2

WMT2027_PLAN §3.2 item 5 (lines 351-356). For the top 0.13%, 0.3%, 1%, 6%, 10%, 12%, 25%, 30% and
50% of the QE ranking: share of rows with rep > 0, duplicate rates, word types at a matched word
count, and UN share. This is a descriptive CPU analysis for the author's decisions. It is not a
registered result and makes no paper claim before W2 runs.

Run on the Mac Studio on 2026-10-19 (Python 3.14.3, unidata 16.0.0, numpy 2.4.5), at most 4
worker processes, under `nice -n 5`. Every number below comes from `item5_depth_results.json`,
written by `depth.py`, or from `verify_results.json`, written by `verify.py`. The exceptions are
marked "independent check": they come from `../item5_depth_check/check_results.json`, written by
that folder's `chk.py`, which imports nothing from `depth.py` or `verify.py`.

## Definition used

- **Pool and ranking: E0.3's rule exactly.** The pool is `pool_mask_reused.npy`: the 16,646,992
  rows that carry the paper's CometKiwi-22 score (`e03_decisions.json` "_pool", decision D1). E0.3's
  held-out reservation is removed from it: 4,652 rows share an exact pair, a normalised source or a
  normalised target with `heldout_un`, and 4,645 of them are inside the mask
  (`e03_build_controls.py` l.400, 426). That leaves **N = 16,642,347** rows. Scores are cast to
  float32 (l.218). The order is `argsort(-score, kind="stable")` (l.437), so ties go to the lower
  row index.
- **Depth.** Depth f is the top k = round(f × N) rows. The extra row "E0.3 band" is k = 1,000,000,
  E0.3's n_ft, which is 6.009% of N. The 6% depth is k = 998,541, which is 1,459 rows shorter than
  the band.
- **Subset.** Take the top k rows, then remove exact (src, tgt) duplicates, keeping the first copy
  in score order (l.440). Every statistic except the raw duplicate rate is computed on this
  deduplicated subset. The subsets are nested prefixes of one ranking, and the band's subset is
  `ft_topk` line for line.
- **rep (E0.4's statistic).** Computed within each depth's own subset, as in `e04_split.py`
  l.29-36 and `e05_build.py` `rep_within()` l.94. Tokens are the source side, lowercased `\w+`.
  The 8-grams are keyed by blake2b-64. A row's rep is the fraction of its 8-grams whose count over
  the subset is above 1. Rows under 8 tokens get rep = 0. The table gives the share with rep > 0,
  over all rows and over UN rows only.
- **Duplicate sources.** 1 − (distinct raw source strings / rows) on the deduplicated subset
  (`e05_build.py` `lexicon()` l.127; `prereg_drafts.md` l.203). **Exact-pair duplicates** is
  1 − (deduplicated rows / k) on the raw top k.
- **Word types at a matched word count.** This is `vocab/common.py` `types_at()` (l.31;
  `prereg_drafts.md` l.195). Shuffle the subset's rows (kept in score order) with
  `random.Random(seed)`, then count the distinct lowercased `\w+` words in the first B tokens.
  Source and French are counted separately. **B1 = 410,000** is the largest two-significant-digit
  count that every depth reaches on both sides; the 0.13% subset has 419,251 English tokens, so B1
  takes 97.8% of them (see "Seeds" under Caveats). B1 is reported for seed 0, with the range over
  seeds 0-4 in brackets. **B2 = 3,300,000** covers the depths from 1% down; 1% has 3,307,267
  English tokens. **B3 = 6,500,000** is E0.6's French budget. B2 and B3 use seed 0 only. The
  whole-set type count is not size-matched and is given for reference.
- **UN share.** This is the e01 exact provenance label `un` (`~/mt_local/rebuild/prov_v2_fixed/`;
  `phase0/README.md` l.181-215), on the deduplicated subset. The share on the raw top k is in
  brackets.

### How the labels were tied to `v2_scored.tsv` row by row

The labels index the rows of the fixed corpus. `v2_scored.tsv`'s source and target columns,
re-joined with newlines, have the fixed corpus's sha256 (`c8cc665c…` / `e4f5301a…`). All
38,275,284 rows have exactly 3 fields. So TSV row i is corpus row i. The labels file is the one
E0.3 used: its sha256 is `c9174045…`, the same as the bundle's, and its size and blake2b first and
last 16 MiB match `controls_manifest.json`. The TSV itself has sha256 `64874c21…` and E0.3's
blake2b fingerprint.

## Check against the known band (all pass)

- The reservation reproduces E0.3's manifest exactly: 4,652 rows (3,290 exact pair, 1,078 extra by
  source, 284 extra by target).
- The top 1,000,000 gives **926,999 rows after exact-pair dedup** (73,001 removed). The QE range is
  0.8959360-0.9144790, the same as the manifest. The source counts equal the manifest's:
  **un 713,881 = 77.01%**, europarl 155,483, commoncrawl 53,795, news-commentary 1,560, giga-fren
  39, unmatched 2,241.
- The band's pair hashes equal those of `ft_topk.{en,fr}` row for row, in file order, and its
  labels equal `ft_topk.source` row for row.
- **UN rows with rep > 0: 227,406. UN rows with rep = 0: 486,475.** These are exactly E0.6's
  "repetitive / diverse UN candidates" (`prereg_drafts.md` l.179-180).
- The QE mean (summed in float32) differs from the manifest by 2.4e-7. The independent check's mean
  computed in float64 differs from the manifest by 2.5e-7 as well, so the gap comes from E0.3's own
  float32 summation, not from the inputs.
- `verify.py` recomputes every statistic directly with the reference code, on Python strings, on
  the first n lines of `ft_topk`. It covers the 0.13%, 0.3%, 1%, 6% and band subsets: rep counts,
  mean rep, duplicate sources, UN share, token counts, whole-set types, and types at B1 (5 seeds),
  B2 and B3. All are equal to `depth.py`'s values (`verify_results.json`). `verify.py` does not
  cover the deeper depths (10-50%); the independent check does (see "Verification coverage" under
  Caveats).

## Numbers (deduplicated subset of each depth)

| depth | k (top rows) | rows after dedup | QE at cut | exact-pair dups (top k) | duplicate sources | rep > 0 | rep > 0, UN rows | UN share (raw top k) | en / fr types at 410k tokens (seed 0; seeds 0-4) | en / fr at 3.3M | en / fr at 6.5M | en / fr types, whole set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.13% | 21,635 | 20,621 | 0.9055 | 4.69% | 1.29% | 13.77% | 16.10% | 77.57% (78.30%) | 10,373 / 12,400 (10,367-10,373 / 12,400-12,469) | n/a | n/a | 10,447 / 13,167 |
| 0.3% | 49,927 | 47,413 | 0.9041 | 5.04% | 1.39% | 15.60% | 18.42% | 77.71% (78.46%) | 10,994 / 13,256 (10,898-11,086 / 13,176-13,275) | n/a | n/a | 14,705 / 18,872 |
| 1% | 166,423 | 156,689 | 0.9016 | 5.85% | 1.60% | 19.04% | 22.41% | 77.63% (78.56%) | 11,921 / 14,233 (11,921-12,042 / 14,233-14,434) | 24,402 / 30,180 | n/a | 24,420 / 31,889 |
| 6% | 998,541 | 925,633 | 0.8959 | 7.30% | 2.28% | 26.96% | 31.85% | 77.01% (78.27%) | 13,708 / 16,291 (13,507-13,743 / 16,106-16,291) | 30,045 / 36,189 | 38,920 / 46,646 | 61,144 / 76,819 |
| E0.3 band | 1,000,000 | 926,999 | 0.8959 | 7.30% | 2.28% | 26.97% | 31.85% | 77.01% (78.27%) | 13,733 / 16,228 (13,582-13,770 / 16,167-16,338) | 29,812 / 36,183 | 38,735 / 46,446 | 61,190 / 76,870 |
| 10% | 1,664,235 | 1,531,764 | 0.8935 | 7.96% | 2.53% | 29.90% | 35.25% | 76.80% (78.22%) | 14,084 / 16,821 (14,084-14,403 / 16,779-16,960) | 32,014 / 38,527 | 41,787 / 49,740 | 82,908 / 102,204 |
| 12% | 1,997,082 | 1,832,220 | 0.8924 | 8.26% | 2.65% | 31.07% | 36.60% | 76.72% (78.21%) | 14,572 / 17,124 (14,493-14,625 / 17,032-17,230) | 32,905 / 39,143 | 42,955 / 50,874 | 93,203 / 114,125 |
| 25% | 4,160,587 | 3,753,364 | 0.8867 | 9.79% | 3.27% | 36.39% | 42.69% | 76.54% (78.41%) | 15,621 / 18,329 (15,606-15,797 / 18,160-18,329) | 37,132 / 43,560 | 49,448 / 57,511 | 155,453 / 183,467 |
| 30% | 4,992,704 | 4,478,997 | 0.8846 | 10.29% | 3.45% | 37.77% | 44.31% | 76.54% (78.54%) | 15,997 / 18,511 (15,784-16,060 / 18,363-18,645) | 38,486 / 44,661 | 51,374 / 59,113 | 178,138 / 208,514 |
| 50% | 8,321,174 | 7,361,851 | 0.8753 | 11.53% | 3.97% | 41.41% | 48.66% | 76.32% (78.64%) | 17,403 / 19,890 (17,036-17,403 / 19,529-19,890) | 42,804 / 49,351 | 58,194 / 66,277 | 272,376 / 312,018 |

The QE score at the top is 0.9145. The JSON also has, per depth: mean rep, mean source length,
token counts, full source composition, and how many rows tie with the cut's float32 score (at most
396 in the pool at any cut).

### Supplement: fixed-size samples (`depth.py sized`)

rep and duplicate rates computed within each depth's own subset grow with the subset's size: a
bigger set gives an 8-gram or a source more chances to recur. So the main table's rise of rep > 0
with depth is mostly a size effect. For comparisons across depths, the supplement draws uniform
random samples without replacement (numpy `default_rng(seed)`) of fixed size n from each depth's
subset and computes the statistics within the sample. The sizes are n = 20,621 (the 0.13% subset's
size; seeds 0-2, seed 0 shown, the range in brackets), 214,489 (the E0.5/E0.6 set size) and 926,999
(the band). The samples are not length-matched. The 0.13% row at n = 20,621 and the band row at
n = 926,999 are the whole subset, so they equal the main table.

| depth | n | rep > 0 | rep > 0, UN rows | mean rep | duplicate sources | UN share | mean source length |
|---|---|---|---|---|---|---|---|
| 0.13% | 20,621 | 13.77% | 16.10% | 0.0652 | 1.29% | 77.57% | 20.33 |
| 0.3% | 20,621 | 10.39% (9.83-10.45) | 12.36% (11.69-12.39) | 0.0395 | 0.67% (0.50-0.68) | 77.78% | 20.59 |
| 1% | 20,621 | 6.85% (6.50-6.98) | 8.40% (7.83-8.45) | 0.0200 | 0.27% (0.22-0.27) | 77.57% | 21.07 |
| 6% | 20,621 | 5.60% (5.60-5.75) | 6.93% (6.93-7.15) | 0.0120 | 0.11% (0.06-0.11) | 77.10% | 22.37 |
| E0.3 band | 20,621 | 5.97% (5.68-5.97) | 7.42% (7.11-7.42) | 0.0125 | 0.06% (0.05-0.06) | 77.35% | 22.51 |
| 10% | 20,621 | 5.90% (5.67-5.90) | 7.43% (7.01-7.43) | 0.0109 | 0.03% (0.03-0.07) | 76.52% | 23.05 |
| 12% | 20,621 | 6.34% (5.97-6.34) | 7.95% (7.55-7.95) | 0.0115 | 0.03% (0.03-0.07) | 76.34% | 23.24 |
| 25% | 20,621 | 7.01% (6.74-7.05) | 8.59% (8.34-8.78) | 0.0148 | 0.01% (0.01-0.03) | 76.97% | 24.24 |
| 30% | 20,621 | 7.79% (7.43-7.91) | 9.73% (9.28-9.73) | 0.0160 | 0.01% (0.01-0.04) | 76.86% | 24.70 |
| 50% | 20,621 | 8.23% (8.23-8.49) | 10.34% (10.34-10.71) | 0.0163 | 0.01% (0.01-0.01) | 76.35% | 25.80 |
| 6% | 214,489 | 15.71% | 18.98% | 0.0493 | 0.68% | 77.08% | 22.43 |
| E0.3 band | 214,489 | 15.53% | 18.75% | 0.0488 | 0.64% | 76.87% | 22.44 |
| 10% | 214,489 | 15.24% | 18.53% | 0.0449 | 0.47% | 76.85% | 23.05 |
| 12% | 214,489 | 15.03% | 18.24% | 0.0439 | 0.41% | 76.68% | 23.23 |
| 25% | 214,489 | 16.13% | 19.58% | 0.0472 | 0.28% | 76.69% | 24.24 |
| 30% | 214,489 | 16.46% | 20.05% | 0.0487 | 0.26% | 76.40% | 24.55 |
| 50% | 214,489 | 17.12% | 21.05% | 0.0499 | 0.20% | 76.46% | 25.67 |
| E0.3 band | 926,999 | 26.97% | 31.85% | 0.1124 | 2.28% | 77.01% | 22.46 |
| 10% | 926,999 | 25.32% | 30.07% | 0.0977 | 1.68% | 76.81% | 23.01 |
| 12% | 926,999 | 25.01% | 29.76% | 0.0944 | 1.51% | 76.69% | 23.22 |
| 25% | 926,999 | 24.84% | 29.73% | 0.0912 | 1.06% | 76.55% | 24.23 |
| 30% | 926,999 | 24.96% | 29.90% | 0.0918 | 0.97% | 76.56% | 24.56 |
| 50% | 926,999 | 25.20% | 30.56% | 0.0912 | 0.74% | 76.37% | 25.72 |

## What the numbers say (descriptive)

- **The UN share is flat.** It is 76.3-77.7% of deduplicated rows at every depth, and 78.2-78.6% of
  the raw top k. Depth does not change the source mix. W2's "all sources vs UN only" choice is
  about the same mix at every depth.
- **Duplicates concentrate at the very top.** Compared at a fixed size, duplicate sources are
  highest at the top, but the decline is only resolved in places. At n = 20,621 it is clear from
  0.13% to 6%: 1.29% at 0.13%, 0.50-0.68% at 0.3%, 0.22-0.27% at 1%, 0.06-0.11% at 6% (seeds 0-2).
  From 6% to 50% each sample holds only 2-23 duplicate source rows, and the seed ranges overlap
  (seeds 0-2: 6% 0.06-0.11%, 10% 0.03-0.07%, 50% 0.01%; seeds 0-4 in the independent check:
  0.04-0.11%, 0.03-0.07%, 0.01-0.03%). Over that range only seed 0's values (0.11% at 6%, 0.01% at
  25-50%) and the independent check's 5-seed means (0.073, 0.071, 0.051, 0.050, 0.031, 0.022 and
  0.017% for 6%, band, 10%, 12%, 25%, 30% and 50%) fall steadily; other single seeds do not (seed 2
  is higher at 10% and 12% than in the band). At n = 926,999 the decline is resolved: 2.28% in the
  band down to 0.74% at 50% (seed 0), and the independent check's seeds 0-1 differ by at most 0.02
  percentage points at each depth. Exact-pair duplicates in the raw top k rise with depth (4.7% to
  11.5%), but that figure is not size-matched.
- **rep > 0 at a fixed size is U-shaped.** At n = 20,621 the extreme top is the most repetitive:
  13.8% at 0.13% and 10.4% at 0.3% (seed 0), against 5.6-6.3% at 6-12%. It then rises again
  toward 50% as mean source length grows from 22.4 to 25.8 tokens: 8.2% for seed 0, which is the
  lowest of the seeds (8.23-8.49% over seeds 0-2; 8.23-9.09%, mean 8.49%, over seeds 0-4 in the
  independent check). The U-shape also holds on the independent check's 5-seed means: 13.77,
  10.12, 6.92, 5.69 (6%), 5.90 (band), 5.79, 6.15, 7.07, 7.65 and 8.49% (50%). Longer rows have
  more 8-grams, and the samples are not length-matched. At n = 926,999 the band is slightly above
  every deeper depth: 27.0% against 24.8-25.3% (seed 0).
- **Vocabulary narrows toward the top at matched token counts.** English types at 410k tokens rise
  from 10,373 (0.13%) to 13,708 (6%) to 17,403 (50%). At 6.5M tokens they rise from 38,920 (6%) to
  58,194 (50%). French follows the same order.
- **The QE scale is compressed.** The whole top 50% lies in 0.8753-0.9145. The 6% and 12% cuts are
  0.0035 apart.

## Caveats

- **Denominator.** The percentages are of the 16,642,347-row ranked pool: the reused-score pool
  minus the held-out reservation. They are not of the 38.3M full corpus. The 6% depth is not the
  band: k = 998,541 against 1,000,000. Both rows are given. Their shares differ by at most 0.02
  percentage points, and their type counts differ by less than the seed noise.
- **Ties.** At every cut, up to a few hundred rows share the cut's float32 score. As in E0.3,
  ascending row index decides which of them fall inside (counts in the JSON).
- **The rep statistic depends on the set and its size.** In the main table it is computed within
  each depth's own subset, as the plan asks, so it is not comparable across depths. Use the
  supplement for that, and even there length is not matched. The definition is E0.4's on the
  source side only, with within-row repeats counting, as in `e04_split.py`. rep was computed within
  each depth's all-source subset, not within its UN rows. For a UN-only W2 arm, rep would have to
  be recomputed within the UN rows (not done).
- **Labels.** These are e01 exact labels. A pair found in several sources goes to the first-listed
  source (europarl < commoncrawl < un < …). Unmatched rows are 0.12-0.28% of each subset.
  `phase0/README.md` l.181-215 gives the label bounds.
- **Hashes.** Word types in `depth.py` come from exact string vocabularies, with no hashing. The
  8-grams (as in E0.4), the pairs (as in E0.3) and the source strings are 64-bit hashes. The
  expected number of collisions is below 1e-3 over the 138,224,495 8-gram occurrences: negligible,
  but not zero. The independent check, with md5 for pairs and sources and a different 64-bit
  8-gram hash, gets the same numbers, so no collision is visible in any reported figure.
- **Unicode.** The Studio's Python 3.14.3 has unidata 16.0.0, and E0.5 was built with 3.12 / 15.0.0.
  `\w`, `lower()` and NFKC could differ on rare characters. The band check (reservation, the
  926,999 rows, 227,406 / 486,475) reproduces exactly, so any effect is invisible there. The
  independent check ran on the same Studio Python (3.14.3, unidata 16.0.0), so it does not test
  this caveat.
- **Seeds.** In `depth.py` the B2/B3 types and the n = 214,489 / 926,999 samples use a single
  seed, and the n = 20,621 samples use seeds 0-2. The independent check adds seeds 3-4 at
  n = 20,621 and seed 1 at the two larger sizes (quoted above where used). At B1 the spread over
  5 seeds is at most 367 types, and that sampling noise is not negligible against the gaps
  between neighbouring depths (for example 6% vs band, or 25% vs 30%). The 0.13% depth is the
  exception: B1 takes 97.8% of its 419,251 English tokens, so the five draws nearly coincide, and
  its English spread (6 types) is not comparable with the other depths' spreads.
- **Verification coverage.** `verify.py` covers the depths inside the band (0.13% to band) only.
  The independent check (`../item5_depth_check/chk.py`) covers all 10 depths, 10-50% included.
  It rebuilds the reservation from the `heldout_un` text, identifies pairs and sources by md5
  rather than blake2b-64, uses a different 8-gram hash, and computes rep directly within each
  depth rather than through the nested-prefix shortcut. Every main-table field at all 10 depths
  matches `item5_depth_results.json` (floats within 1e-12), and so do all 43 fixed-size samples
  in the supplement. It shares `depth.py`'s Python and Unicode version (see "Unicode").

## Files

Repo (`phase0/next/cpu_items/item5_depth/`):
- `depth.py`: the analysis, with stages textsha, pass1, rank, pass2, reduce and sized.
- `verify.py`: the direct recomputation.
- `run.sh`: the order the stages were run in.
- `item5_depth_results.json`: every number above except those marked "independent check", the
  definitions and all checks.
- `verify_results.json`: the verify output.
- `input_fingerprints_vs_e03.json`: the labels, mask and report against E0.3's manifest.

Studio (`~/cpu_items/item5_depth/`):
- The same scripts and JSON files.
- `inputs/`: the provenance labels and report, `pool_mask_reused.npy`, E0.3's
  `controls_manifest.json`, and the bundle's labels copy (sha256 identical).
- `logs/`: one log per stage.
- `work/`: intermediates, about 4.3 GB (per-row arrays, token and 8-gram files). It can be deleted;
  `depth.py` regenerates it in about 7 minutes.

## Open points

Not done here, because each needs a rerun or a code change:
- **Unicode.** Testing the Unicode caveat needs a run under Python 3.12 / unidata 15.0.0, the
  version E0.5 was built with. Both this item and its independent check ran on 3.14.3 / 16.0.0.
- **Seeds in this item's own results.** The 5-seed figures at n = 20,621 and the second seed at
  n = 214,489 / 926,999 exist only in `../item5_depth_check/check_results.json`. Having
  `item5_depth_results.json` carry them would need `depth.py sized` rerun with more seeds.
- **UN-only rep.** As noted under Caveats, rep within the UN rows alone (for a UN-only W2 arm) has
  not been computed.
