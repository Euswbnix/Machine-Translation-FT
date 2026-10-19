# item1_coverage: points still open after review (2026-10-19)

The headline numbers in README.md and every figure in results.json were re-derived by an independent check with its own code and agree. After one round of README edits, a second reviewer still found the wording and precision points below. Where they conflict with README.md, these notes are right; the README is to be corrected before any of it is quoted.

1. NEW WRONG CLAIM, README L52 (denominator row): 'so they agree at the one decimal printed' is false for the P row on the source side. The gaps table prints -11.4% (whole-vocab -11.366%). With the restricted denominator it is -11.346%, which rounds to -11.3%. The 'at most 0.035 points' bound itself is right: the max over all 7 contrasts is 0.0354, for A_narrow - R0 source. Only the one-decimal agreement claim fails. The checker's own wording had the same slip.

2. NEW MISATTRIBUTION, README L53 (size row): '(0.39 for ft_topk_div - ft_random_small ...)' sits right after 'on the target side', so it reads as a target-side figure. In fact 0.39 is the source-side change (-17.743% -> -17.348%), and the target-side change is 0.25 (-21.986% -> -21.741%). Related gap: the other contrast that is only in results.json, ft_topk_rep - ft_topk_div, moves 0.20 on the target side (-6.639% -> -6.443%). That is above the 0.16 bound and is not mentioned.

3. NEW INACCURATE CLAIM, README L55: '`coverage.py` itself does not read piece types' is not true. coverage.py L318-321 calls sp.is_control, is_unknown, is_unused and is_byte for every id. That is where results.json's unused_ids = 0 / byte_ids = 0 come from, and README L34 cites them ('no byte or unused pieces'). What coverage.py cannot see is the user-defined type, i.e. the full type breakdown, and the TrainerSpec. Open points L183/L186 ('the piece types', 'a protobuf reader for piece types') should be narrowed the same way.

4. SCOPE, README L168-169: the kept-pairs '<= 0.8 points' change in the uncovered-share relative gaps holds only for the five contrasts in the README's gaps table (max 0.75, div - random source). It does not hold for the two E0.4 contrasts in results.json: ft_topk_rep - ft_topk_div source changes by 1.01; ft_topk_div - ft_random_small changes by 2.6 (source) and 12.0 (target, on a +2607% gap). L53 now brings in an E0.4 contrast, so the scope should be stated. The '<= 0.007 pp' figure holds for all 10 sets.

5. MINOR, README L171-174: 'Everything else is printed by the script' overstates. The variants-table differences (0.035, 0.16, 0.24, 0.25, 0.39) are hand arithmetic on two results.json values each, and so is the '4.3 times' at L99. Neither is in the caveat's list of approximate figures, and neither is marked independent check.

6. MINOR WORDING, README L130: 'This does not carry over to word types' is literally true only because of the S row. Every other word-type gap is at least about 34 times the largest word-type sd (rep - div French: 10,337 / 305). The editor's report goes further than the README ('applies to coverage only'). The README text itself is acceptable, though it could say 'not to the S row's word-type gaps'.

7. PRE-EXISTING, not from this edit and not in the checker's list (FYI): L99 'ft_topk and ft_random are 4.3 times larger than the other sets' is true only against the 214,489-row sets; against the 242,022-row sets the factor is 3.8. L160-162 contrasts the paper's '16K/32K BPE vocabularies' with 'Here the vocabulary is joint'. But groundwork.json saunders2024 says de-en used 32K shared merges, so only en-ja was unshared.

