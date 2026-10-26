# item6_forgetting: review notes

## 2026-10-25: the switch to Eq 1

The author read Eq 1 in Saunders & DeNeefe (2024) on 2026-10-25: ForgetGenUse_V = sum over t in V of
L(t), divided by the reference token occurrences of V, with O = min(c_B, c_R), A = min(c_F, c_R) and
L(t) = sum over sentences of max(O - A, 0); clipping per sentence; ID = the fine-tuning set's target
side. The script's headline now uses that denominator (it used B, the base model's correct uses,
before); the old ratio is kept as `forgetgenuse_per_base_correct`. The tests were re-derived by hand
(test_forgetgenuse.py docstring), 8 of 8 mutations are killed (mutation.log), and the real run was
repeated; the independent checker's own implementation (fgu_check.py, field alt_per_ref_occ) gives the
same integer counts for all 30 runs (real/crosscheck_real_words_eq1.json). README.md was rewritten;
points 1, 3, 5, 6 and 8 below are dealt with there, point 2's timing paragraph and point 4's macro
figures were dropped from it, and point 7's docstring sentence now names the per-split counts only.
Still open: the token unit and case Eq 1 counts in (README, Open points).

## 2026-10-19

The headline numbers in README.md and every figure in results.json were re-derived by an independent check with its own code and agree. After one round of README edits, a second reviewer still found the wording and precision points below. Where they conflict with README.md, these notes are right; the README is to be corrected before any of it is quoted.

1. NEW, inexact claim (fix first): Caveat 2, README line 130, says 'Both stand-ins set the reference to the base output.' The folder has three stand-in runs. Smoke A sets ref = base. Smoke B sets ref = the fine-tuned output (README line 84; results_smoke.json B_refft_*). The glue run sets ref = base. The conclusion still holds, because in the ref = ft run L = 0 everywhere, so every reading gives 0. But the sentence is wrong as written. Suggested wording: 'the stand-ins with nonzero loss (smoke A and the glue run) set ref = base; the ref = ft run has L = 0, so it cannot separate the readings either.'

2. NEW, softened claim: README lines 109-110 say 'An independent recount ... took a similar time.' The checker listed the 55.6 s as NOT re-verified. Its figures were 8.0 s for words and 33.9 s for spm in two parallel processes, using its own implementation (fgu_check.py) over the six sets, which it called 'consistent in size'. The README should give those figures, say it was a different implementation, and make clear that this does not verify the 55.6 s.

3. NEW, rationale leaves out a counterpoint: README lines 28-29 justify target-side ID by 'target coverage as the paper's strongest correlate (line 113)'. That is accurate, but line 113 says it is the strongest correlate for the COMET drop (tau = 0.9). groundwork.json line 105 (key_results item 5) says ForgetGenUse itself correlated significantly only with source coverage (tau = 0.7). Mention that, or justify the target side by the fact that reference tokens are French.

4. NEW, traceability: Caveat 2 (README lines 134-136) gives macro All 0.393/0.379 'in the full-size glue run'. It says results_glue_full3003.json keeps only the pooled numbers, but not where the macro values are. They appear nowhere in this folder; they are presumably in the Studio's ~/cpu_items/item6_forgetting/glue_test/out/ JSONs, and the checker confirmed them there. Elsewhere the README labels unlogged items 'not in this folder'; this should say the same.

5. Minor wording: README line 40 says per-type B/F/L are kept for 'the 25 most-lost and most-gained types'. The code (forgetgenuse.py lines 247-249) keeps up to 25 most-lost OOD types, up to 25 most-lost ID types and up to 25 most-gained types, and only types with L > 0 or G > 0. In the ref = base runs top_gained is empty.

6. Minor wording: README line 128 says 'Table 2 and Table 4 ranges (OOD 0.11-0.65, ID 0.07-0.17)'. Both ranges are from Table 4. Table 2 holds ForgetGenUse_All, 0.07-0.25 (groundwork.json line 105). Either add the All range or cite Table 4 only.

7. Outside the README (the editor had it read-only): the forgetgenuse.py docstring, lines 24-25, still says 'Every count behind the ratios is written, so another normalisation ... can be recomputed from the JSON without re-running.' This is the overclaim from checker problem 3, now fixed only in the README. It belongs with Open point 3, the code change.

8. Pre-existing, not from this edit, worth a question: README lines 85-87 say the glue test ran run_real.sh 'with the reference sha checks removed in a copy' and with the two hypotheses copied into all 31 slots. With the real redecode_report.json, run_real.sh lines 24-35 would refuse those copies. So the hypothesis sha check must also have been bypassed, or a substitute report used, and the README does not say which. Also, README line 4 and run_real.sh line 2 say the references are 'on no reachable machine'. The evidence only covers the Studio and this Mac's ~/mt_local and ~/mt_fetch.

