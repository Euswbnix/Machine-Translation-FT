# item5_depth: points still open after review (2026-10-19)

The headline numbers in README.md and every figure in results.json were re-derived by an independent check with its own code and agree. After one round of README edits, a second reviewer still found the wording and precision points below. Where they conflict with README.md, these notes are right; the README is to be corrected before any of it is quoted.

1. NEW claim, wrong at full precision (README lines 148-151, carried over from the checker's own wording): it says that from 6% to 50% at n = 20,621 'only seed 0's values (0.11% at 6%, 0.01% at 25-50%) and the independent check's 5-seed means ... fall steadily'. In item5_depth_results.json, seed 0 has 23, 12, 6, 6, 2, 3, 3 duplicate source rows for 6%, band, 10%, 12%, 25%, 30%, 50% (0.1115, 0.0582, 0.0291, 0.0291, 0.0097, 0.0145, 0.0145%). So seed 0 is flat from 10% to 12% and goes up from 25% to 30%. It only looks non-increasing at the 2-decimal rounding. Only the 5-seed means fall strictly (0.0727, 0.0708, 0.0514, 0.0504, 0.0310, 0.0223, 0.0175%). Fix: say only the 5-seed means fall steadily, or drop the seed-0 clause.

2. PRE-EXISTING wrong number (README line 184, Labels caveat; not in the editor's change list and missed by the checker): 'Unmatched rows are 0.12-0.28% of each subset'. From source_counts_dedup / n_dedup in item5_depth_results.json the values are 0.12, 0.10, 0.18, 0.24, 0.24, 0.26, 0.27, 0.29, 0.29, 0.28% for 0.13% through 50%, so the range is 0.10-0.29% (0.3% is 0.101%; 25% and 30% are 0.294% and 0.295%).

3. Minor overreach, NEW (README lines 189-190, Hashes caveat): 'so no collision is visible in any reported figure'. The two-hash cross-check covers only the figures both scripts computed (main table and the builder's 43 samples). The checker-only figures (seeds 3-4 at n = 20,621, seed 1 at n = 214,489 / 926,999) were computed with one hash family. Suggest 'in any figure computed by both'.

4. Minor, PRE-EXISTING (README lines 173-174, Denominator caveat): it says the 6% and band 'type counts differ by less than the seed noise'. That holds at B1 (en 25 / fr 63 apart, against 5-seed spreads of 236/185 and 188/171). Seed noise is not measured at B2/B3, which use one seed: there they differ by 233/6 and 185/200. Whole-set counts have no seed. Limit the claim to B1.

