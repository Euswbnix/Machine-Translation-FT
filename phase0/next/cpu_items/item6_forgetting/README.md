# item6_forgetting: token-level forgetting (ForgetGenUse), ID vs OOD

WMT2027_PLAN.md §3.2 item 6. **Status: run on the real newstest2014 outputs (2026-10-19), re-run
2026-10-25 with the paper's Eq 1, cross-checked by an independent implementation.** This is a
descriptive CPU analysis, not a registered result.

**Correction, 2026-10-25.** The first run divided the lost uses by the base model's correct uses
(Σ L / Σ B). That was this folder's reading of a one-sentence note, made before anyone had read
Eq 1. The author read Eq 1 in the paper on 2026-10-25: the denominator is the number of reference
token occurrences. Every headline value below is now Eq 1. The old ratio is still written to the
JSON as `forgetgenuse_per_base_correct`, but it is not the paper's measure and should not be quoted.
The Eq 1 values are 0.64–0.65 of the old ones for All and ID and 0.57–0.61 for OOD. The ordering of
the sets and the ID/OOD split of the extra lost uses (counts, not rates) do not change.

## Definition used

Saunders & DeNeefe (2024, EMNLP), Sec 2.1, Eq 1, as the author checked it against the paper on
2026-10-25. For sentence i and token t, with c_R, c_B, c_F the counts of t in the reference, the base
hypothesis and the fine-tuned hypothesis:

- O_i(t) = min{c_B, c_R}, the base model's correct uses, clipped per sentence (this folder calls it B).
- A_i(t) = min{c_F, c_R}, the fine-tuned model's correct uses (F).
- L(t) = Σ_i max{O_i(t) − A_i(t), 0}, the lost uses. Clipping and the positive part are both taken
  per sentence before summing.
- **ForgetGenUse_V = Σ_{t∈V} L(t) / Σ_i Σ_{t∈V} c_{R_i}(t)**: total lost uses over the reference
  token occurrences of the token set V. The denominator is neither the base model's correct uses nor
  an average of per-token rates.
- V is All, ID or OOD. **ID** = the types that occur at least once (`--id-min-count`, default 1) in
  the fine-tuning set's **target (French) side**, tokenised the same way; the author confirmed that
  "in the fine-tuning set" means the target side. **OOD** = every other reference type.
- G = max(0, F − B) per sentence (gained uses) is reported only.

Also reported:

- `forgetgenuse_per_base_correct` = Σ L / Σ B, the first reading. Kept so the 2026-10-19 numbers can
  be traced; not the paper's measure.
- `forgetgenuse_macro`: the mean over the types with Σ_s B > 0 of (Σ_s L / Σ_s B). An alternative
  aggregation, not in the paper.
- A copy split that is not in the paper: whether t also occurs in the source sentence (names and
  numbers).
- Per-split totals (reference occurrences, B, F, L, G and type counts), so any pooled normalisation can
  be recomputed from the JSON. Per-type B, F and L are kept only for up to 25 most-lost OOD types, 25
  most-lost ID types and 25 most-gained types, and only for types with L > 0 or G > 0 (`--top`).
  Per-type netting, or an average over another set of types, needs a re-run (about a minute).

Tokenisers:

- `words` (default): lowercased `\w+`, the project rule (e04_split.py:13,29; e05_build.py:49,79).
- `words_cased`: `\w+` with case kept.
- `spm`: pieces from the base model's `spm_enfr_v1_fixed` (sha256 d1c10ea8…, checked). The paper's
  coverage measure is defined on subwords.

Which unit and case the paper's Eq 1 counts in has not been checked (Open points).

## Files

| file | what it is |
| --- | --- |
| `forgetgenuse.py` | `--src --ref --base-hyp --ft-hyp (1..n) --ft-train <prefix>... --out` → JSON. Stdlib only; sentencepiece is needed only for `--tok spm`. It refuses to run if the line counts differ (lines split on `\n` only), if a training prefix's .en and .fr differ in length, or if `--expect-src-sha` / `--expect-ref-sha` (`newstest2014` = E0.5's 967dca93… / 82526d36…) do not match. |
| `test_forgetgenuse.py` | Hand-computed synthetic tests (example and arithmetic in its docstring), plus clipping, cased, min-count, invariant and refusal tests. `test.log`: 8 tests, OK (Studio, Python 3.14.3, 2026-10-25). |
| `mutation_check.py` | Applies 8 single-line mutations to a copy of `forgetgenuse.py` and checks that the tests fail for each. `mutation.log`: 8 of 8 killed (Studio, 2026-10-25). |
| `smoke_run.sh`, `smoke_check.py` | End-to-end run on the Studio, plus an independent recount that does not import the script. Outputs: `results_smoke.json`, `smoke.log`, `example_output_A_refbase_words.json`. |
| `run_real.sh` | The real run: 30 runs, 2 tokenisers, 12 calls, a hypothesis sha256 check against `redecode_report.json`, and a summary table. |
| `results_glue_full3003.json` | `run_real.sh` on full-size files with stand-in references; it tests the glue only. |
| `real/` | The real run (Eq 1): per-set JSONs, `summary.json`, `real.log`, and the independent cross-check (`item6_crosscheck.py`, `crosscheck_real_words_eq1.json`). |

Studio copies: `~/cpu_items/item6_forgetting/` (scripts, `real/`, `real_refs/`, `real_hyps/`, `smoke/`,
`glue_test/out/`; the 2026-10-19 run is kept in `real_per_base_correct_20261019/`).

## Tests

- **Synthetic tests:** 8 of 8 pass on the Studio (Python 3.14.3). The hand example gives All 3/11,
  ID 0/7, OOD 3/4 (the Σ L / Σ B variant: 3/10, 0/6, 3/4) and macro All 3/9; the clipping test gives
  1/3 (variant 0.5), so it also separates per-sentence from per-type netting (per-type would give 0).
- **Mutation check** (`mutation_check.py`, `mutation.log`): each of these 8 mutations makes the tests
  fail:
  - unclipped F;
  - unclipped B;
  - ID and OOD swapped;
  - dividing by B instead of reference occurrences (the 2026-10-19 reading);
  - |B − F| in place of max(0, B − F);
  - the copy split inverted;
  - `>` in place of `>=` for the min count;
  - `splitlines()`.
- **Smoke run** (first 1,000 lines; fine-tuning set ft_un_rep). The inputs are the base
  hypothesis `baseline_v1.1_averaged` (sha256 21f41ffb…) and the fine-tuned hypothesis
  `ft_un_rep_lr1_s42_st2` (sha256 7455e7ba…). The reference is one of those two hypotheses, so it
  is fake, and the source is a stand-in. 34 of 34 independent checks pass (run 2026-10-19 under the
  first reading; with ref = base, B equals the reference occurrences, so Eq 1 gives the same numbers):
  - With the reference set to the base hypothesis, G = 0 and ForgetGenUse_All = 1 − (clipped
    matches / occurrences). Words: All 0.2258, ID 0.2018, OOD 0.5333 (OOD base-correct 1,755 of
    24,232). spm: All 0.2393, ID 0.2266, OOD 0.8715 (638 of 32,354).
  - With the reference set to the fine-tuned hypothesis, lost = 0 in every split.
- **Glue test:** `run_real.sh` ran on full-size 3,003-line files with the real training sets and the
  two hypotheses copied into all 31 slots; the reference sha checks were removed in a copy (whether
  the hypothesis sha check was also bypassed or a substitute report used is not recorded). It also
  refused correctly when the references were missing, when the reference sha was wrong, and when a
  hypothesis was altered. These refusal runs are not logged in this folder.

## The real run

newstest2014 `test.en` / `test.fr` copied from the training box (`data_enfr_v1/`), sha256
`967dca93…` / `82526d36…` = the src_sha / ref_sha in E0.5's BLEU cache; the 31 re-decoded
hypotheses sha-checked against `redecode_report.json`. 2026-10-25: `run_real.sh real_refs real_hyps`
on the M5 Max, exit 0, a few seconds. ForgetGenUse (Eq 1), `words` tokeniser, mean (sd) over the 5
seeds; ID = types in the set's French side:

| set | All | ID | OOD | OOD reference occurrences (of 73,616) |
| --- | --- | --- | --- | --- |
| ft_un_rep | 0.0966 (0.0007) | 0.0849 (0.0007) | 0.2356 (0.0008) | 5,722 |
| ft_un_div | 0.0751 (0.0013) | 0.0697 (0.0011) | 0.1631 (0.0044) | 4,265 |
| ft_un_random | 0.0686 (0.0008) | 0.0671 (0.0009) | 0.1058 (0.0007) | 2,916 |
| ft_topk_rep | 0.0803 (0.0007) | 0.0717 (0.0006) | 0.1981 (0.0027) | 5,025 |
| ft_topk_div | 0.0573 (0.0006) | 0.0545 (0.0007) | 0.1140 (0.0030) | 3,466 |
| ft_random_small | 0.0469 (0.0003) | 0.0473 (0.0003) | 0.0306 (0.0020) | 1,527 |

**Cross-check:** the independent checker's `fgu_check.py` (its own implementation), driven by
`real/item6_crosscheck.py`, gives for every one of the 30 runs the same integer counts (r, B, F, L, G
for All, ID, OOD, OOD_in_src, OOD_not_in_src); the largest difference in any mean or sd is 2.8e-17
(`real/crosscheck_real_words_eq1.json`).

**Reading, descriptive only:**

- Within UN, forgetting orders rep > div > random, as BLEU does; so does topk_rep > topk_div.
- The OOD rate differs most between sets (0.236 / 0.163 / 0.106), but by counts most of the extra
  loss of rep over div is on ID tokens. Mean lost uses: rep 7,114, div 5,526, random 5,052. Of the
  rep − div gap (+1,587), 935 (59%) are ID and 652 (41%) OOD. Of the div − random gap (+474), 387
  (82%) are OOD. For topk_rep − topk_div (+1,691), 64% are ID. All has the same denominator (73,616)
  for every set, so these shares are the same under Eq 1 and under the first reading.
- The OOD class itself differs in size by set (last column): the repetitive half's French side
  covers fewer of the test set's tokens, which is item 1's coverage gap seen from the test side. So
  the OOD rates compare different token populations; the All column compares like with like.
- `spm` results are in `real/*_spm.json`; with pieces nearly every reference token is ID (share
  0.980–0.997), so the spm OOD cells rest on few tokens and are not used.
- The paper's ranges (from the pre-reading note: Table 2, All 0.07–0.25; Table 4, OOD 0.11–0.65,
  ID 0.07–0.17) use the same formula now, but a different setting (8 whole-domain corpora, 20K steps,
  other test sets) and possibly another token unit, so a side-by-side comparison is only rough.

## Caveats

1. **The token unit and case are not checked against the paper.** Eq 1's formula, the per-sentence
   clipping, the denominator and the target-side ID set are now as in the paper. Whether the paper
   counts words or subwords, and cased or lowercased tokens, has not been read. The default here is
   lowercased words; `words_cased` and `spm` are available.
2. **The stand-ins could not separate the readings.** The stand-ins with nonzero loss (smoke A and
   the glue run) set ref = base, so B equals the reference occurrences and G = 0, and the ref = ft
   smoke run has L = 0 everywhere. Every reading gave the same number there. Only the real run shows
   the difference: Σ L / Σ B is about 1.56 times Eq 1 for All, because the base model gets about 64%
   of the reference occurrences right (47,332 of 73,616 with `words`).
3. **The tokeniser matters.** With spm pieces nearly every reference token is ID (above). Words mode
   ignores punctuation; spm counts it.
4. **Statistics:** one test set and five seeds per set. The script reports the mean and sd over seeds
   and no significance test. The ID/OOD split is per type, so OOD mixes rare content words, names and
   numbers, which is why the copy split is there.
5. **Scope:** the fine-tuning target side defines ID. Base-pretraining coverage is not used.
   Changing `--id-min-count` changes ID.

## Open points

1. **Check the token unit and case in the paper** (words or subwords; cased or not). If they differ,
   re-run with `--tok spm` or `words_cased` as the headline; both are already supported.
2. **Keep the full per-type counts** (or add an option for them) so that per-type readings need no
   re-run. This is a code change.
3. **Log the glue test's refusal runs and the timing** in this folder (the mutation check is now
   logged).
