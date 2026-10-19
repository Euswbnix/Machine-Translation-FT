# item6_forgetting: token-level forgetting (ForgetGenUse), ID vs OOD

WMT2027_PLAN.md §3.2 item 6 (line 357). **Status: implemented and tested; no real result yet.**
The newstest2014 references are on no reachable machine, so every number below checks the code. None
of them measures forgetting. This is a descriptive CPU analysis, not a registered result.

## Definition used

Source: groundwork.json, saunders2024 note, line 104 (field "setup", Evaluation/Metrics): "For each
reference token, ForgetGenUse counts how often the original model produced it correctly and the
adapted model no longer does. It can be split into tokens that appear in the adaptation set (ID) and
tokens that do not (OOD)." Line 113 ("useful_for_design") asks for ForgetGenUse_ID/OOD on the E0.5
newstest2014 outputs. The note gives this sentence but not the formula of Eq 1, so the counting
rule below is my implementation of that sentence:

For test segment s and reference token type v, let r, b, f be the counts of v in the reference, the
base hypothesis and the fine-tuned hypothesis.

- B = min(b, r) is the base model's correct uses, clipped the way BLEU clips unigram matches.
- F = min(f, r) is the fine-tuned model's correct uses.
- L = max(0, B − F) is the forgotten uses.
- G = max(0, F − B) is the gained uses. It is reported only.

**ForgetGenUse_X = Σ L / Σ B**, summed over the (s, v) pairs with v in X, where X is All, ID or OOD:

- **ID**: the types that occur at least once (`--id-min-count`, default 1) in the fine-tuning set's
  **target (French) side**, tokenised the same way. Using the target side only is my assumption: the
  note says just "appear in the adaptation set". I chose the target side because the note reports
  target coverage as the paper's strongest correlate (line 113).
- **OOD**: every other type.

Also reported:

- `_macro`: the mean over the types with Σ_s B > 0 of (Σ_s L / Σ_s B). This is an alternative
  aggregation.
- A copy split that is not in the paper: whether v also occurs in the source segment (names and
  numbers).
- Per-split totals (reference occurrences, B, F, L, G and type counts). A different denominator,
  such as reference occurrences, can be recomputed from these. Per-type B, F and L are kept only for
  the 25 most-lost and most-gained types (`--top`), so per-type netting, or an average over another
  set of types, needs a re-run (about a minute for everything).

Tokenisers:

- `words` (default): lowercased `\w+`, the project rule (e04_split.py:13,29; e05_build.py:49,79).
- `words_cased`: `\w+` with case kept.
- `spm`: pieces from the base model's `spm_enfr_v1_fixed` (sha256 d1c10ea8…, checked). The note's
  coverage measure (line 112) is defined on subwords.

## Files

| file | what it is |
| --- | --- |
| `forgetgenuse.py` | `--src --ref --base-hyp --ft-hyp (1..n) --ft-train <prefix>... --out` → JSON. Stdlib only; sentencepiece is needed only for `--tok spm`. It refuses to run if the line counts differ (lines split on `\n` only), if a training prefix's .en and .fr differ in length, or if `--expect-src-sha` / `--expect-ref-sha` (`newstest2014` = E0.5's 967dca93… / 82526d36…) do not match. |
| `test_forgetgenuse.py` | Hand-computed synthetic tests (example and arithmetic in its docstring), plus clipping, cased, min-count, invariant and refusal tests. `test.log`: 8 tests, OK. |
| `smoke_run.sh`, `smoke_check.py` | End-to-end run on the Studio, plus an independent recount that does not import the script. Outputs: `results_smoke.json`, `smoke.log`, `example_output_A_refbase_words.json`. |
| `run_real.sh` | The real run: 30 runs, 2 tokenisers, 12 calls, a hypothesis sha256 check against `redecode_report.json`, and a summary table. |
| `results_glue_full3003.json` | `run_real.sh` on full-size files with stand-in references; it tests the glue only. |

Studio copies: `~/cpu_items/item6_forgetting/` (scripts, `hyps/`, `smoke/`, `glue_test/out/`).

## Tests

- **Synthetic tests:** 8 of 8 pass on the Studio (Python 3.14.3). The hand example gives All 3/10 = 0.3,
  ID 0/6 = 0, OOD 3/4 = 0.75, and macro All 3/9.
- **Mutation check:** 8 of 8 deliberately broken copies fail the tests. This is my run; no log of
  it is kept in this folder. An independent check went through each mutation against the tests by
  reasoning and did not re-run them. The mutations:
  - unclipped F;
  - unclipped B;
  - ID and OOD swapped;
  - dividing by reference occurrences instead of B;
  - |B − F| in place of max(0, B − F);
  - the copy split inverted;
  - `>` in place of `>=` for the min count;
  - `splitlines()`.
- **Smoke run** (first 1,000 lines; fine-tuning set ft_un_rep). The inputs are the base
  hypothesis `baseline_v1.1_averaged` (sha256 21f41ffb…) and the fine-tuned hypothesis
  `ft_un_rep_lr1_s42_st2` (sha256 7455e7ba…). The reference is one of those two hypotheses, so it
  is fake, and the source is a stand-in. 34 of 34 independent checks pass:
  - With the reference set to the base hypothesis, B equals the reference occurrences, G = 0, and
    ForgetGenUse_All = 1 − (clipped matches / occurrences). Words: All 0.2258, ID 0.2018, OOD 0.5333
    (OOD base-correct 1,755 of 24,232). spm: All 0.2393, ID 0.2266, OOD 0.8715 (638 of 32,354).
  - With the reference set to the fine-tuned hypothesis, lost = 0 in every split.
- **Glue test:** `run_real.sh` ran with the reference sha checks removed in a copy. It used
  full-size 3,003-line files, the real training sets, and the two hypotheses copied into all 31
  slots. It also refused correctly when the references were missing, when the reference sha was
  wrong, and when a hypothesis was altered. These refusal runs are not logged in this folder. The
  independent check read the refusal code and did not re-run it.

## The real run

**What it needs:**

- newstest2014 `test.en` and `test.fr` exactly as E0.5 scored them: Machine_translation
  `data_enfr_v1/test.{en,fr}`, sha256 967dca935c7c… and 82526d36ae1c… (the cache in E0.5's results
  archive). Copy them to the Studio.
- The 31 `*.newstest2014.hyp` files plus `redecode_report.json` from
  `~/mt_local/phase0_final/e05_redecode/`.

**Command:**

```
cd ~/cpu_items/item6_forgetting && nohup ./run_real.sh <REF_DIR> <HYP_DIR> real > real.log 2>&1 &
```

**Time:** measured on full-size files with all 30 runs and both tokenisers, while the E0.6 job was
running, under `nice -n 5`: **55.6 s wall in one process**. This timing is not logged in this
folder. An independent recount, run as two parallel processes under a different load, took a
similar time.

- `words` takes about 2.0–2.3 s per set.
- `spm` takes about 6.2–8.6 s per set; encoding the 214–242k training rows dominates.

E0.6's 40 runs will need their outputs written (prereg OD 11). They cost the same per set.

## Caveats

1. **Eq 1 has not been read, so the counting rule is my reading.** The note gives only the one
   sentence quoted above, not a formula. Four parts of the rule are my choices, not the paper's:
   - pooling over all (s, v) pairs rather than averaging per type;
   - clipping per segment;
   - taking max(0, B − F) in each segment rather than netting each type over the whole test set;
   - B as the denominator, rather than reference occurrences.

   Three more choices are mine too, as noted above: ID comes from the French side only, tokens are
   lowercased, and the default unit is words rather than the model's subwords. Until Eq 1 is read
   in the PDF, do not compare these numbers with the paper's Table 2 and Table 4 ranges (OOD
   0.11–0.65, ID 0.07–0.17).
2. **The stand-ins cannot show how much the counting rule matters.** Both stand-ins set the
   reference to the base output. So B equals the reference count in every segment and G = 0
   (`results_smoke.json`: base_correct = ref_occ and gained = 0). B − F is then never negative, and
   three readings give the same number: Σ L / Σ B, Σ L / (reference occurrences), and per-type
   netting max(0, Σ_s B − Σ_s F) / Σ B. Only the per-type average differs. In the full-size glue
   run (ft_un_rep) the macro All is 0.393 against 0.234 pooled for words, and 0.379 against 0.245
   for spm. `results_glue_full3003.json` keeps only the pooled numbers. With a real reference the
   fine-tuned model also gains tokens, and the readings come apart. In this folder only the hand
   clipping test tells per-segment netting from per-type netting: it gives 0.5 per segment, and
   per-type netting would give 0. See Open points.
3. **The tokeniser matters.** With spm pieces nearly every reference token is ID. In the stand-in
   full run the OOD occurrences were 19 (ft_random_small) to 1,954 (ft_un_rep) of 98,166, so spm
   OOD rates will be noisy for the broad sets. With `words` they were 1,527 to 5,486 of 72,951.
   Words mode ignores punctuation; spm counts it. Lowercasing follows the project rule, while the
   paper's examples are cased; `words_cased` is available.
4. **Stand-in by-product** (not a result): these are the shares of the *base model's* newstest2014
   output tokens that occur in each set's French side, from the glue run.

   | set | words | spm |
   | --- | --- | --- |
   | ft_un_rep | 0.9248 | 0.9801 |
   | ft_un_div | 0.9425 | 0.9892 |
   | ft_un_random | 0.9611 | 0.9966 |
   | ft_topk_rep | 0.9344 | 0.9864 |
   | ft_topk_div | 0.9532 | 0.9939 |
   | ft_random_small | 0.9791 | 0.9998 |

   French-side vocabulary sizes are 32,771 / 43,420 / 70,046 / 36,235 / 49,892 / 113,161 word
   types, and 18,980 / 20,308 / 24,303 / 19,788 / 21,195 / 27,169 of 32,000 spm pieces. The real
   reference will give different shares.
5. **Statistics:** one test set and five seeds per set. The script reports the mean and sd over
   seeds and no significance test. The ID/OOD split is per type, so OOD mixes rare content words,
   names and numbers, which is why the copy split is there.
6. **Scope:** the fine-tuning target side defines ID. Base-pretraining coverage is not used.
   Changing `--id-min-count` changes ID.

## Open points

These need a recomputation, a code change or a source that is not on hand, so they are not done
here.

1. **Read Eq 1 in the paper's PDF.** No PDF was on hand for this check. Settle these questions:
   - the denominator (B or reference occurrences);
   - netting per segment or per type;
   - ID from the target side only, or from both sides;
   - case;
   - words or subwords.

   If the paper differs, change the default rule in `forgetgenuse.py`.
2. **Measure how much the rule moves the numbers.** This needs a stand-in where the fine-tuned
   model also gains tokens, for example a reference set to another fine-tuned output. An independent
   check reported such a run (reference = the ft_un_random s42 output, ft_un_rep vocabulary). For
   words, All was 0.151 under this rule, 0.101 with per-type netting, 0.122 per reference occurrence
   and 0.262 as a per-type average. Those numbers are not in this folder and were not reproduced
   here.
3. **Keep the full per-type counts** (or add an option for them) so that other per-type readings
   need no re-run. This is a code change.
4. **Re-run and log three things** in this folder: the mutation check, the `run_real.sh` refusal
   runs and the timing.

## Real run (2026-10-19)

newstest2014 `test.en` / `test.fr` copied from the training box (`data_enfr_v1/`), sha256
`967dca93…` / `82526d36…` = the src_sha / ref_sha in E0.5's BLEU cache; the 31 re-decoded
hypotheses sha-checked against `redecode_report.json`. `run_real.sh real_refs real_hyps real` on the
M5 Max, 5.2 s, exit 0. Outputs in `real/`. ForgetGenUse, `words` tokeniser, mean (sd) over the 5
seeds, pooled as defined above (ΣL / ΣB; ID = types in the set's French side):

| set | All | ID | OOD |
| --- | --- | --- | --- |
| ft_un_rep | 0.1503 (0.0012) | 0.1310 (0.0012) | 0.4077 (0.0014) |
| ft_un_div | 0.1168 (0.0019) | 0.1079 (0.0017) | 0.2717 (0.0073) |
| ft_un_random | 0.1067 (0.0013) | 0.1041 (0.0014) | 0.1743 (0.0012) |
| ft_topk_rep | 0.1249 (0.0011) | 0.1106 (0.0009) | 0.3474 (0.0048) |
| ft_topk_div | 0.0892 (0.0009) | 0.0844 (0.0010) | 0.1956 (0.0052) |
| ft_random_small | 0.0730 (0.0005) | 0.0735 (0.0005) | 0.0503 (0.0033) |

Cross-check: the independent checker's `fgu_check.py` (its own implementation), driven by
`real/item6_crosscheck.py`, gives the same means and sds to four decimals for all six sets
(`real/crosscheck_real_words.json`).

Reading, descriptive only: within UN, forgetting orders rep > div > random, as BLEU does; the gap is
mostly in tokens the fine-tuning set does not contain (OOD 0.41 / 0.27 / 0.17), while ID forgetting
differs by 2–3 points. The counting rule is this folder's reading of the note (Eq 1 unchecked, see
the caveats above), so compare sets with each other, not with the paper's values; `spm` OOD cells
rest on very few tokens and are not used.
