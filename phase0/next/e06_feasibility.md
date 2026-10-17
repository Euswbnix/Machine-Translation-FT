# Feasibility measured 2026-10-17 on the Mac, real data (E0.5 inputs, UN top band)

UN top band: 486,475 diverse candidates (rep = 0 = median), 227,406 repetitive candidates.
Run of record: ft_un_rep 214,489 rows, 25,134 source word types; ft_un_div 214,489 rows, 33,874.

rep_dedup — the repetitive UN candidates with repeated source sentences removed (first occurrence kept):
- 227,406 -> 207,561 candidates; stratified length matching against the diverse candidates gives
  198,333 rows per side.
- It keeps a repeated-8-gram share of 0.379 (ft_un_rep 0.433), 25,296 types, 0.00% duplicate sources.
- So "it is just duplicates" can be tested: rep_dedup vs a diverse draw of the same 198,333 rows.

div_narrow — the diverse UN candidates narrowed to the repetitive half's vocabulary:
- At ft_un_rep's exact length histogram, choosing in each length bucket the rows whose rarest word is
  most frequent reaches 20,284 source word types (no bucket short); a random draw at the same
  histogram has 34,070. ft_un_rep has 25,134, which lies inside that range, so a diverse set with
  the repetitive half's vocabulary size exists.
- So "repetitiveness is only low vocabulary coverage" (Saunders & DeNeefe 2024) can be tested:
  ft_un_rep vs div_narrow, matched on size, length and vocabulary, differing in repetition.
- Open: the coverage measure to match on. Source+target word types is the simplest; Saunders &
  DeNeefe use coverage of generic vocabulary. Must not use the test sets to define it.

From the groundwork workflow (vocabulary agent), Design A, all UN, rep = 0, 214,489 rows, ft_un_div's
length histogram: R0 (random UN with rep = 0; 62,000 types) and A_narrow (random UN with rep = 0,
restricted to the 42,000 most frequent source / 58,000 most frequent target words of the pool;
33,562 source types vs div 33,874; QE 0.866 vs R0 0.861 vs div 0.899; literalness 0.698 vs 0.693
vs 0.783). Contrasts N = R0 - A_narrow (narrowness alone), S = A_narrow - ft_un_div (the rest of
QE selection).

PROPOSED E0.6 (from scratch, en-fr, UN only, same recipe as E0.5: lr_scale 1.0, spike guard off,
10,000 steps, final checkpoint, five seeds), eight sets, 40 runs:
  anchors      ft_un_div, ft_un_rep, ft_un_random (bridge to E0.5; identical files)
  narrowness   R0, A_narrow                       -> N and S
  coverage     div_narrow                         -> ft_un_rep vs div_narrow (C1 beyond coverage)
  duplicates   rep_dedup, div_dd (198,333 each)   -> rep_dedup vs div_dd (C1 beyond duplicates)
