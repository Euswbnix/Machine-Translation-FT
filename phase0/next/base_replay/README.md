# base_replay — what the v1.1 base actually trained on (2026-10-18)

The trainer reseeds on every start and does not skip consumed batches, so a resume re-runs the
original batch order from its first batch (`WMT2027_PLAN.md` §0.2). The v1.1 en-fr base, the base of
every E0.x fine-tune, was resumed once: steps 0-100,000, then 100,000-122,294. This check rebuilds its
batch order exactly and asks whether the re-trained rows are spread evenly across the E0.5/E0.6 sets.

Run on the training box (AMD Ryzen 9 9950X3D, the CPU of the April runs), with the trainer code of
the April runs (`bde629a`, copied here as `build/april_dataset.py`, `build/april_trainer.py`).
`build/` is the builder; `verify_reimpl/` re-implemented everything from scratch without the builder's
code; `verify_method/` audited the method. All three agree on every number.

**Reconstruction, validated before use.**
- All 9,312,233 corpus lines re-tokenised the dataset's way; 2,261 dropped (over 256 tokens); the
  9,309,972 kept rows equal the trainer's own cache in every length and every token id.
- Batches per epoch 49,174 / 49,177 / 49,180, matching the logged epoch ends (49,174 and 98,351).
  (The 48,502 used earlier was lines // 192 + 1; the trainer printed 48,490, its `__len__`.)
- Order: the sampler also calls `random.shuffle(batches)` (Python's RNG), and swanlab drew 208 words of
  it before the first shuffle; `random.seed(42)` reproduces the 13 chart ids swanlab recorded in both
  sessions. Logged throughput against batch padding picks this order at rank 1 of 2,049 in both sessions.
- The batch order depends on numpy's AVX-512 argsort tie order on that CPU. A stable or AVX2 sort gives
  different epoch ends and keeps only about 34-42% of the replayed set. **The masks are valid only as
  built there**; their sha256 are in `BOX_SHA256SUMS` (files in `~/mt_local/phase0_next/base_replay/box/`
  and `~/replay_check/build/` on the box). A Mac run (`~/mt_local/phase0_next/base_replay/build/`) is wrong
  and marked SUPERSEDED.

**Result.**
- Re-trained in session 2: P1 batches 0-22,293 = 4,221,799 rows (45.35%). Times each row was seen over
  both sessions: 2 for 4,917,868 rows, 3 for 4,250,249, 4 for 141,855. No row unseen.
- The released average ("step 105,000") is the mean of steps 105,000 / 110,000 / 115,000 / 120,000 /
  122,294, so a replayed batch enters it with weight 1.0 down to 0.2 by position; the last two
  replayed batches (positions 22,292-22,293) never reached an optimizer step.
- The replay was **content-random with respect to every set**: per corpus line the replay rate is
  44.97-45.37% in each E0.5 set and in R0 / A_narrow; exposed rows whose pair sits on one corpus line
  were replayed at 45.0-45.3% everywhere; every between-set gap equals what the sets' duplicate pairs
  predict under a random replay (p >= 0.2 against 2,000 re-draws). The realized per-row gaps are not
  zero (e.g. ft_un_rep minus ft_un_div: +2.3 points of exposed rows replayed), and they are the
  duplicates' doing, not the replay's.
- Exposure shares reproduce the drafts exactly: 36.73 / 36.71 / 37.92% (ft_un_div / ft_un_rep /
  ft_un_random), 52.95 / 42.43 / 58.49% (ft_topk_div / ft_topk_rep / ft_random_small).
- **Found on the way, independent of the replay:** exposure *frequency* differs. ft_un_rep's exposed pairs
  sit on 1.158 corpus lines against 1.043 for ft_un_div, so per set row the base presented ft_un_rep's
  pairs 1.056 times against 0.951 (mean 2.88 against 2.59 per exposed pair). It is part of the
  duplicate strand of the "repetitive" bundle; whether it moves BLEU is not tested here.

`build/results.json` holds every per-set number; `build/05_sets.json` and `05b_dup_free.json` the
contrasts and re-draw nulls. Read the "ci95 design-based" fields as observed gap ± 1.96 sd of the re-draw
null, not as confidence intervals (the replay is one fixed event); report the gap, the null mean and p.
