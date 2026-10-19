# Item 3 (WMT2027_PLAN.md §3.2): Chimoto et al.'s rival explanation, contamination per half

A descriptive CPU analysis to inform the author's decisions. It is not a registered result.

**Status (2026-10-19, 14:00 EDT): RUN.** It ran on the Mac Studio between 13:54 and 13:57 EDT.
The Studio was reached over Tailscale as `alexs-mac-studio`, because the LAN address 10.0.0.126
cannot be reached from this MacBook. The independent recount (`check.py`) matches every count it
checks. The length-ratio numbers agree **exactly** with the checker's Apple-NaturalLanguage run:
59 sets, 2,065 counts and all 6 Tukey fences, 0 differences. The language-ID rates differ by tool,
as expected. They agree in direction for both requested contrasts. For C2 they also agree in size.
For C1, py3langid's difference is about 3 times Apple NL's, but that is the tool's 'pcm' false
positives (see the comparison section).

## Short answer

| | ft_un_rep | ft_un_div | ft_un_random | rep − div (sd) | div − random (sd) |
| --- | --- | --- | --- | --- | --- |
| **length-ratio outlier**, ws r > 1.5 or < 1/1.5 (both tools, identical) | 2.142% | 3.134% | 5.961% | **−0.99 pp** (0.05) | **−2.83 pp** (0.06) |
| **wrong language**, py3langid 0.4.0 | 0.368% | 0.152% | 1.090% | **+0.22 pp** (0.016) | **−0.94 pp** (0.024) |
| **wrong language**, Apple NL (checker) | 0.160% | 0.090% | 0.842% | **+0.07 pp** (0.011) | **−0.75 pp** (0.021) |
| wrong language, py3langid, rows flagged only by an English side labelled `pcm` left out | 0.126% | 0.047% | 0.779% | +0.08 pp (0.009) | −0.73 pp (0.020) |
| either headline flag, py3langid / Apple NL | 2.50 / 2.30% | 3.28 / 3.22% | 6.98 / 6.74% | −0.78 / −0.92 pp | −3.70 / −3.52 pp |

(214,489 rows per set; the sd is the binomial sd of the difference, a noise scale.)

- **The repetitive half is not dirtier by length ratio. It is cleaner.** It has 1 pp fewer ratio
  outliers than the diverse half, and the diverse half has 2.8 pp fewer than random UN. The order
  rep < div < random holds for every ratio variant with more than a handful of flags: all three
  units (whitespace, SentencePiece, characters) at 1.3 and 1.5, SentencePiece at 2.0, and all six
  Tukey fences. The one exception is characters at 2.0, with 2 vs 1 rows.
- **Wrong-language rows are rare everywhere: under 0.4% in either half under either tool.** The
  repetitive half has more of them than the diverse half (+0.07 to +0.22 pp). Most of these rows are
  English lists of names, dates and countries. The tools call them Nigerian Pidgin, Bikol, Latin,
  Indonesian or Romanian. They are not foreign-language text (see the examples and caveats).
- Neither contaminant can explain C1 on its own. The repetitive half has 1 pp *fewer* ratio
  outliers, and its wrong-language excess is 0.07 to 0.22 pp of rows, most of them false positives.
  The two quality-selected halves are both much cleaner than random UN on both measures, which is
  the C2 direction.

## Definition (what was measured)

**Where the rules come from.** (Unchanged from the builder.)
- The chimoto2024 note (groundwork.json `notes`, key `chimoto2024`) gives Chimoto et al.'s rival
  explanation as noise in the pool: "length outliers (some sentences of 500+ words) and
  foreign-language text" (App. A, Figs 7-8; their explanation, Sec 5.2). Its `useful_for_design`
  item 1 asks for language ID and source/target length-ratio filters, or at least contamination
  rates per half.
- The note names **no language identifier** and gives **no numeric length-ratio rule**. The koehn2018
  note lists "length, length ratio, language ID" pre-filters in general terms, also with no numbers.
- So both rules below are stated choices: a standard tool and a standard, fixed rule.

**Language ID.**
- Tool: py3langid 0.4.0, a fork of langid.py. The PyPI wheel is `py3langid-0.4.0-py3-none-any.whl`,
  sha256 `c25bd038147951ba3a89efe9ec12cc7d2116cb6f311ff8af84ade398a8bc0616`, and uv checked it with
  `--require-hashes`. The model is `py3langid/data/model.npz.xz`, sha256
  `f4f4a2c3465ca1f081541037f9cac23021d68151c2f709c55ae5eedfca522963`. It is a naive-Bayes model
  with **140 labels in 142 columns**: `sr` and `uz` each have two script columns, which the tool
  merges. It also includes `zxx` (no linguistic content) and many low-resource languages, for example
  `pcm`, `bcl` and `kik`. *(The builder's README said 97 languages. That was langid.py's old model,
  not 0.4.0's.)*
- Called through the **public API**: `LanguageIdentifier.from_model_file(MODEL_FILE,
  norm_probs=True).classify(line)` returns `(label, confidence)`. It is applied to the raw line,
  all labels, top-1. (The tool lower-cases a line that is entirely upper case, then applies NFC.)
- **Headline flag (`lid_any`):** a row is wrong-language if its English side's label is not `en`,
  or its French side's label is not `fr`. The two sides are also reported separately (`en_wrong`,
  `fr_wrong`).
- **The 0.9 confidence rule, stated unambiguously (`lid_any_conf`).** The flagged side's
  `classify()` confidence must be ≥ 0.9. With `norm_probs=True`, py3langid 0.4.0's confidence is
  **not** the plain naive-Bayes posterior. `_decide()` computes
  `p = softmax(s / sqrt(B))`, where `s` is the vector of raw NB log-scores, `B` is the byte length
  of the line after the tool's own normalisation (lower-case if all upper, NFC, UTF-8), and each
  duplicate script column (`sr`, `uz`) is added into the first. The rule uses `p[label]`, the
  number `classify()` returns. That is the builder's stated intent ("normalised top-1 probability",
  `classify(norm_probs=True)` in `check.py`). Sensitivity only: `lid_any_conf_T1` applies the same
  ≥ 0.9 rule to the plain posterior `softmax(s)[label]` (temperature 1, aliases summed), which is
  what langid.py's `norm_probs` used to return. **The choice matters.** The plain posterior is ≥ 0.9
  for almost every string, so `lid_any_conf_T1` comes out close to `lid_any`. The tempered one is
  far stricter.
- Other sensitivity variants, as the builder wrote them:
  - `swap_or_copy_2way`: the same public `classify()` on an identifier restricted by
    `set_languages(['en','fr'])`. A row is flagged if its English side comes out `fr` or its French
    side `en`, meaning a swapped or untranslated side.
  - `same_top1_both_sides` and `identical_sides` (the stripped strings are equal).
  - `lid_any_rawlabel` (new, a check): the label under `norm_probs=False` semantics (argmax of `s`).
    It differs from the headline label for 186 of 1,771,943 English strings and 89 of 1,789,619
    French strings. In the E0.5 sets, R0 and A_narrow it changes no `lid_any` count. Elsewhere it
    changes a count by at most 7 rows (`ft_random`).

**Length ratio.** (Unchanged from the builder.)
- r = n_fr / n_en, counted in whitespace tokens (`str.split()`). This is the unit of the pool's own
  cleaning rule.
- **Headline flag (`ratio_ws_1.5`):** a row is an outlier if r > 1.5 or r < 1/1.5. This is a
  symmetric fixed band: Moses `clean-corpus-n.perl`'s `-ratio` test at 1.5. The comparison is done
  in exact integer arithmetic. A zero length on either side counts as an outlier (none occur).
- Sensitivity variants: thresholds 1.3, 1.5 and 2.0; three units (whitespace tokens, pieces of the
  base model's `spm_enfr_v1_fixed.model` (`encode`, no BOS/EOS), characters of the stripped line);
  and Tukey fences on log r (Q1 − 1.5 IQR, Q3 + 1.5 IQR, `np.quantile` linear) computed on
  `ft_random` (all rows, and its UN rows).
- **L0, a check rather than a finding.** The pool's own cleaning rule (`phase0/rebuild_corpus.py`):
  3-200 whitespace tokens per side, an en/fr token ratio in [0.5, 2.0], and a Latin-script share
  ≥ 0.9 per side. **Result: 0 violations in every set**, and `ratio_ws_2.0` = 0 in every set, as
  expected. The longest line in the six E0.5 training sets has 128 whitespace tokens
  (`heldout_un`: 173), and no row anywhere has more than 200. Chimoto's "500+ words" outliers
  cannot occur here.

**Sets.** (Unchanged.) The six E0.5 sets and `heldout_un`. R0 and A_narrow, checked byte-identical
to their seed-42 feasibility twins. The whole of `ft_topk` and `ft_random`, and their UN rows. As
extras, all 46 feasibility draws: each sha256 is checked against `feasibility_report.json`, and the
three anchors are checked to reproduce E0.5's UN files line for line.

**Examples.** (Unchanged.) 20 rows flagged by `lid_any` or `ratio_ws_1.5`, drawn with seed 42: 7
from `ft_un_rep`, 7 from `ft_un_div`, 6 from `ft_un_random`. Another 10 per flag per set are in
`results.json`.

## What changed in the builder's scripts (and why)

1. **API.** py3langid 0.4.0 has no `from_pickled_model`, `instance2fv` or `nb_classprobs`. The
   scripts now use `LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)` and the public
   `classify()` / `set_languages()`. These were read from the installed
   `site-packages/py3langid/langid.py`. The headline labels, confidences and 2-way labels come
   straight from `classify()`. The private `_encode` / `_raw_score` calls (the same calls
   `classify()` makes) supply only the sensitivity quantities (`lid_any_conf_T1`,
   `lid_any_rawlabel`) and the consistency checks.
2. **Confidence rule.** It is defined as above. The builder's internal cross-check (|Δp| ≤ 1e-3
   against `classify()`) could never have passed, because its internal p was the temperature-1
   posterior.
3. **Install.** Nothing is installed into the E0.6 reference venv. There is a separate venv,
   `~/cpu_items/item3_contamination/venv`, made with `~/e06env/bin/uv` from
   `~/e06env/python/cpython-3.14.3-macos-aarch64-none` (`UV_CACHE_DIR=~/cpu_items/item3_contamination/uvcache`,
   `UV_NO_CONFIG=1`, `UV_PYTHON_DOWNLOADS=never`). It holds `numpy==2.4.5` (`--only-binary :all:
   --no-deps`; its installed files are identical to the reference venv's numpy per the RECORD
   hashes, apart from the two `bin/` shebang scripts) and the one hash-pinned py3langid wheel
   (`--only-binary :all: --no-deps --require-hashes`). `uv pip check` reports it compatible. See
   `logs/install.log`.
4. **SentencePiece.** The new venv has no sentencepiece, so the `spm` unit now comes from a first
   stage, `spm_lengths.py`. That stage is *run* with the reference interpreter, which already has
   sentencepiece 0.2.2. It writes a piece count for every line of the 18 text inputs, plus their
   sha256s. The main script refuses to run if any sha256 differs from its pins, or if one string has
   two different counts.
5. **Processes.** 5 workers plus the parent, at nice 5. Multiprocessing's spawn context also starts
   one idle resource-tracker process.
6. Everything else (the sets, the rules, the thresholds, the seeds and the contrasts) is as the
   builder wrote it.

## Checks (all passed)

- 24 pinned inputs, `feasibility_report.json` and 46 feasibility `.npy` files match their sha256s.
  So do the 18 sha256s recorded by `spm_lengths.py`.
- Every E0.5 pair occurs in its parent, the anchors reproduce E0.5's UN files line for line, and R0
  and A_narrow are byte-identical to their feasibility twins.
- **Public-API re-run:** 3,000 random unique strings per side were classified again in the parent
  with fresh identifiers. The label, the confidence (the exact float) and the 2-way label equal the
  reassembled worker arrays.
- The 2-way label equals the raw en-vs-fr score comparison for all 3,561,562 strings. The
  confidence equals a float64 recomputation of `softmax(s/sqrt(B))` to within 2.5e-6.
- **`check.py`, an independent recount** that shares no code with the analysis, calls the public
  API on every line of `ft_un_rep`, `ft_un_div`, `ft_un_random`, R0 and A_narrow. All 10 counts × 5
  sets and the 20 examples match (`check_results.json`, `"all_match": true`).
- `compare_tools.py` rebuilt the unique-string order and spot-checked 1,000 strings against a fresh
  `classify()`: all identical.
- 33 English and 30 French strings in the pools have no model features at all, for example
  "REGULATION No. 98". The tool then returns uniform scores, and the alias sum makes the label `sr`
  at p = 2/142. None of them is in the seven E0.5 set files.

## Numbers

### Length ratio (identical under both runs)

| set | rows | ws 1.3 | **ws 1.5** | ws 2.0 | fr long / short at ws 1.5 | spm 1.5 | char 1.5 | Tukey ws (ft_random UN) | L0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft_un_rep | 214,489 | 31,519 (14.69%) | **4,595 (2.14%)** | 0 | 4,523 / 72 | 12,149 (5.66%) | 1,363 (0.64%) | 496 (0.23%) | 0 |
| ft_un_div | 214,489 | 39,420 (18.38%) | **6,723 (3.13%)** | 0 | 6,666 / 57 | 14,590 (6.80%) | 2,450 (1.14%) | 597 (0.28%) | 0 |
| ft_un_random | 214,489 | 48,540 (22.63%) | **12,785 (5.96%)** | 0 | 11,339 / 1,446 | 20,683 (9.64%) | 7,661 (3.57%) | 3,754 (1.75%) | 0 |
| R0 | 214,489 | 51,701 (24.10%) | 14,087 (6.57%) | 0 | 12,649 / 1,438 | 22,781 (10.62%) | 8,657 (4.04%) | 3,884 (1.81%) | 0 |
| A_narrow | 214,489 | 50,818 (23.69%) | 13,530 (6.31%) | 0 | 12,232 / 1,298 | 23,349 (10.89%) | 7,952 (3.71%) | 3,642 (1.70%) | 0 |
| ft_topk_rep | 242,022 | 34,190 (14.13%) | 4,941 (2.04%) | 0 | 4,833 / 108 | 13,247 (5.47%) | 1,682 (0.69%) | 600 (0.25%) | 0 |
| ft_topk_div | 242,022 | 39,335 (16.25%) | 6,414 (2.65%) | 0 | 6,316 / 98 | 15,026 (6.21%) | 2,841 (1.17%) | 690 (0.29%) | 0 |
| ft_random_small | 242,022 | 53,118 (21.95%) | 15,527 (6.42%) | 0 | 12,086 / 3,441 | 23,737 (9.81%) | 11,470 (4.74%) | 6,976 (2.88%) | 0 |
| heldout_un | 2,000 | 490 (24.50%) | 145 (7.25%) | 0 | 128 / 17 | 221 (11.05%) | 84 (4.20%) | 47 (2.35%) | 0 |
| ft_topk | 926,999 | 169,840 (18.32%) | 33,393 (3.60%) | 0 | 32,808 / 585 | 70,535 (7.61%) | 16,655 (1.80%) | 4,667 (0.50%) | 0 |
| ft_topk[un] | 713,881 | 138,959 (19.47%) | 27,621 (3.87%) | 0 | 27,302 / 319 | 55,434 (7.77%) | 11,170 (1.56%) | 3,413 (0.48%) | 0 |
| ft_random | 926,999 | 213,347 (23.01%) | 67,438 (7.27%) | 0 | 54,129 / 13,309 | 98,673 (10.64%) | 53,067 (5.72%) | 30,556 (3.30%) | 0 |
| ft_random[un] | 617,634 | 143,216 (23.19%) | 41,605 (6.74%) | 0 | 36,816 / 4,789 | 62,505 (10.12%) | 26,390 (4.27%) | 13,880 (2.25%) | 0 |

Ratio contrasts in pp (binomial sd):

| contrast | ws 1.5 | ws 1.3 | spm 1.5 | char 1.5 | Tukey ws (UN) |
| --- | --- | --- | --- | --- | --- |
| ft_un_rep − ft_un_div (requested) | −0.99 (0.05) | −3.68 (0.11) | −1.14 (0.07) | −0.51 (0.03) | −0.05 (0.015) |
| ft_un_div − ft_un_random (requested) | −2.83 (0.06) | −4.25 (0.12) | −2.84 (0.08) | −2.43 (0.05) | −1.47 (0.03) |
| R0 − ft_un_div (E0.6 P) | +3.43 (0.07) | +5.73 (0.13) | +3.82 (0.09) | +2.89 (0.05) | +1.53 (0.03) |
| A_narrow − R0 (E0.6 N) | −0.26 (0.08) | −0.41 (0.13) | +0.27 (0.10) | −0.33 (0.06) | −0.11 (0.04) |
| ft_topk_rep − ft_topk_div | −0.61 (0.04) | −2.13 (0.10) | −0.74 (0.07) | −0.48 (0.03) | −0.04 (0.015) |
| ft_topk_div − ft_random_small | −3.77 (0.06) | −5.70 (0.11) | −3.60 (0.08) | −3.57 (0.05) | −2.60 (0.04) |
| ft_un_rep − div_narrow (OD 2 c2, OD 1 b) | −1.03 (0.05) | −3.86 (0.11) | −1.55 (0.08) | −0.51 (0.03) | −0.07 (0.016) |
| rep_dedup − div_dd (OD 4 a) | −1.03 (0.05) | −3.84 (0.12) | −1.33 (0.08) | −0.51 (0.03) | −0.04 (0.016) |

### Language ID, both tools (count, py3langid / Apple NL)

| set | lid_any py3langid | lid_any Apple NL | en_wrong | fr_wrong | conf ≥ 0.9 | 2-way swap/copy | identical sides |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ft_un_rep | 790 (0.368%) | 343 (0.160%) | 777 / 338 | 76 / 20 | 6 / 48 | 7 / 27 | 0 |
| ft_un_div | 326 (0.152%) | 192 (0.090%) | 320 / 183 | 22 / 30 | 12 / 59 | 12 / 29 | 0 |
| ft_un_random | 2,338 (1.090%) | 1,805 (0.842%) | 1,631 / 1,043 | 1,087 / 1,004 | 293 / 865 | 1,058 / 1,121 | 0 |
| R0 | 2,108 (0.983%) | 1,752 (0.817%) | 1,401 / 992 | 1,067 / 1,016 | 329 / 900 | 1,059 / 1,151 | 0 |
| A_narrow | 781 (0.364%) | 523 (0.244%) | 570 / 312 | 301 / 240 | 49 / 174 | 257 / 298 | 0 |
| ft_topk_rep | 848 (0.350%) | 378 (0.156%) | 820 / 357 | 92 / 36 | 21 / 73 | 17 / 34 | 0 |
| ft_topk_div | 367 (0.152%) | 239 (0.099%) | 323 / 192 | 66 / 67 | 58 / 110 | 29 / 47 | 0 |
| ft_random_small | 3,957 (1.635%) | 3,731 (1.542%) | 2,311 / 2,094 | 2,210 / 2,132 | 1,958 / 2,609 | 2,600 / 2,705 | 4 |
| heldout_un | 68 (3.40%) | 38 (1.90%) | 58 / 27 | 26 / 19 | 2 / 6 | 22 / 22 | 0 |
| ft_topk | 4,609 (0.497%) | 1,726 (0.186%) | 4,191 / 1,498 | 881 / 395 | 313 / 656 | 335 / 338 | 0 |
| ft_topk[un] | 3,719 (0.521%) | 1,121 (0.157%) | 3,540 / 1,091 | 574 / 143 | 41 / 258 | 158 / 170 | 0 |
| ft_random | 32,083 (3.461%) | 23,522 (2.537%) | 23,116 / 14,731 | 16,465 / 12,646 | 7,405 / 11,382 | 16,603 / 15,117 | 9 |
| ft_random[un] | 21,066 (3.411%) | 12,508 (2.025%) | 17,411 / 8,674 | 9,384 / 6,099 | 872 / 3,213 | 8,167 / 6,817 | 0 |

The temperature-1 confidence variant (`lid_any_conf_T1`, py3langid) gives 547 / 225 / 1,712 rows
for rep / div / random, against 6 / 12 / 293 for the headline confidence.

## Comparison with the checker (Apple NaturalLanguage)

Source: `~/cpu_items/item3_contamination_check/results.json` (sha256 `4437483c…1afa`; the same
file as the MacBook copy in `../item3_contamination_check/`). It was made with
`NLLanguageRecognizer` (macOS 27.2) under the same rules. Full detail is in `compare_tools.json`.

**Length ratio: exact agreement, as required.** 59 sets (the 13 named sets and 46 feasibility
draws) × 35 counts each (ws/spm/char × 1.3/1.5/2.0 with long/short splits, 6 Tukey flags, L0,
identical sides): 2,065 counts, 0 differences. The six Tukey fences are equal to the last digit.
The rows also agree (same sets, same sizes).

**Language ID: different tools, so different rates. Do the contrasts agree?**

| contrast | measure | py3langid | Apple NL | same direction? | rough size |
| --- | --- | --- | --- | --- | --- |
| **rep − div** | lid_any | +0.216 pp (sd 0.016), ×2.42 | +0.070 pp (sd 0.011), ×1.79 | **yes** | py3langid about 3× larger in pp; similar as a ratio |
| | en_wrong | +0.213 pp, ×2.43 | +0.072 pp, ×1.85 | yes | as above |
| | fr_wrong | +0.025 pp (sd 0.005) | −0.005 pp (sd 0.003) | **no** | both tiny (≤ 76 rows) |
| | conf ≥ 0.9 | −0.003 pp (sd 0.002) | −0.005 pp (sd 0.005) | yes | both within noise |
| | 2-way swap/copy | −0.002 pp (sd 0.002) | −0.001 pp (sd 0.003) | yes | both within noise |
| | either headline | −0.78 pp | −0.92 pp | yes | similar (the ratio part dominates) |
| **div − random** | lid_any | −0.938 pp (sd 0.024), ×0.14 | −0.752 pp (sd 0.021), ×0.11 | **yes** | **similar** (py3langid 1.25×) |
| | en_wrong | −0.611 pp | −0.401 pp | yes | similar |
| | fr_wrong | −0.497 pp | −0.454 pp | yes | very close |
| | conf ≥ 0.9 | −0.131 pp | −0.376 pp | yes | NL about 3× larger (different confidence scales) |
| | 2-way swap/copy | −0.488 pp | −0.509 pp | yes | very close |
| | either headline | −3.70 pp | −3.52 pp | yes | very close |

All 8 contrasts (the 2 requested and the 6 extra ones) agree in direction on `lid_any`. The 6
extra ones: R0 − div: +0.83 / +0.73 pp.
A_narrow − R0: −0.62 / −0.57 pp. All-source rep − div: +0.20 / +0.06 pp. All-source div − random:
−1.48 / −1.44 pp. rep − div_narrow: +0.26 / +0.12 pp. rep_dedup − div_dd: +0.17 / +0.07 pp.

**Why C1's size differs, and what that says.** py3langid labels English lines of names, dates and
country lists as `pcm` (Nigerian Pidgin, an English-based creole). On the English side that is
521 of its 777 flags in `ft_un_rep`, 227 of 320 in `ft_un_div` and 753 of 1,631 in `ft_un_random`.
Leave out the rows whose only flag is a `pcm` English side, and py3langid's C1 shrinks to
+0.080 pp (sd 0.009). That is Apple NL's +0.070 pp. C2 becomes −0.73 pp against Apple NL's
−0.75 pp. Apple NL has its own version of the same false positive: it calls the same kind of lines
`id`, `ro`, `hr` or `nb`.

**Row overlap** (rows flagged by py3langid / Apple NL / both, Jaccard). The fr side, which is
where real untranslated English shows up, overlaps well in the random-type sets. The en side,
which is mostly false positives, overlaps poorly.

| set | en_wrong | fr_wrong | lid_any |
| --- | --- | --- | --- |
| ft_un_rep | 777 / 338 / 157 (0.16) | 76 / 20 / 7 (0.08) | 790 / 343 / 157 (0.16) |
| ft_un_div | 320 / 183 / 64 (0.15) | 22 / 30 / 11 (0.27) | 326 / 192 / 70 (0.16) |
| ft_un_random | 1,631 / 1,043 / 581 (0.28) | 1,087 / 1,004 / 822 (0.65) | 2,338 / 1,805 / 1,262 (0.44) |
| R0 | 1,401 / 992 / 604 (0.34) | 1,067 / 1,016 / 832 (0.67) | 2,108 / 1,752 / 1,299 (0.51) |
| A_narrow | 570 / 312 / 147 (0.20) | 301 / 240 / 206 (0.61) | 781 / 523 / 347 (0.36) |

In random UN, the French side's wrong label is `en` for 601 of py3langid's 1,087 flags and 692 of
Apple NL's 1,004. Those are genuine untranslated English (titles, references). In both halves the
French side is almost never English.

## 20 examples (seed 42; 7 / 7 / 6 from the three UN sets; text cut at 130 characters)

1. `ft_un_rep:15258`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 21 / 35 tokens (r = 1.667)
   - EN: Romania ratified the major international human rights instruments, including those more closely related to child trafficking, chil…
   - FR: La Roumanie a ratifié les principaux instruments internationaux relatifs aux droits de l'homme, y compris ceux concernant plus spé…
2. `ft_un_rep:49668`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 19 / 29 tokens (r = 1.526)
   - EN: That is why we are firmly committed to uphold, implement and strengthen multilateral disarmament and non-proliferation treaties an…
   - FR: C'est pour cette raison que nous sommes fermement engagés à faire respecter, mettre en œuvre et consolider les traités et accords…
3. `ft_un_rep:60128`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 16 / 27 tokens (r = 1.688)
   - EN: Norway has repeatedly expressed support for the principle of irreversibility in arms control and nuclear disarmament.
   - FR: La Norvège a fait savoir à nombre de reprises qu'elle était en faveur du principe de l'irréversibilité en matière de maîtrise des…
4. `ft_un_rep:88646`, ratio_ws_1.5_fr_long; en `en` 0.99, fr `fr` 1.00; 13 / 20 tokens (r = 1.538)
   - EN: In recent years, some developing countries have performed well in trade in services.
   - FR: Ces dernières années, quelques pays en développement ont en effet obtenu de bons résultats au chapitre du commerce des services.
5. `ft_un_rep:95078`, en_wrong; en `bs` 0.55, fr `fr` 0.18; 8 / 6 tokens (r = 0.75)
   - EN: The capital of Bosnia and Herzegovina is Sarajevo.
   - FR: La capitale de Bosnie-Herzégovine est Sarajevo.
6. `ft_un_rep:103899`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 24 / 45 tokens (r = 1.875)
   - EN: The health effects of declining air and water quality impact particularly on the urban poor who lack access to clean water and ade…
   - FR: Du fait qu'ils n'ont accès ni à de l'eau salubre ni à des installations sanitaires adéquates, ce sont les pauvres vivant en milieu…
7. `ft_un_rep:210234`, ratio_ws_1.5_fr_long; en `en` 0.97, fr `fr` 1.00; 16 / 28 tokens (r = 1.75)
   - EN: To combat the island's drug problem, the police department continues to run preventive and outreach programmes.
   - FR: Afin de lutter contre le problème de la drogue sur l'île, le Département de la police continue à mettre en oeuvre des programmes d…
8. `ft_un_div:25965`, ratio_ws_1.5_fr_long; en `en` 0.99, fr `fr` 1.00; 19 / 31 tokens (r = 1.632)
   - EN: United Nations agencies, international donors, civil society, governments and other stakeholders must be mutually accountable for…
   - FR: Les institutions des Nations Unies, les donateurs internationaux, la société civile, les pouvoirs publics et les autres parties pr…
9. `ft_un_div:30688`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 29 / 44 tokens (r = 1.517)
   - EN: Forests and biodiversity conservation, including protected areas, is critically important for achieving the Millennium Development…
   - FR: Les forêts et la protection de la diversité biologique, y compris les zones protégées, sont d'une importance capitale pour la réal…
10. `ft_un_div:140976`, ratio_ws_1.5_fr_long; en `en` 0.97, fr `fr` 1.00; 16 / 26 tokens (r = 1.625)
    - EN: Community mobilization addressed barriers for girls' education and enhanced participation by community members in school managemen…
    - FR: La mobilisation communautaire visait à s'attaquer aux obstacles à l'éducation des filles et à améliorer la participation des membr…
11. `ft_un_div:151245`, ratio_ws_1.5_fr_long; en `en` 0.90, fr `fr` 1.00; 9 / 17 tokens (r = 1.889)
    - EN: KFOR troops and UNMIK police monitored the demonstrations closely.
    - FR: Les militaires de la KFOR et la police de la MINUK ont surveillé de près ces manifestations.
12. `ft_un_div:172458`, ratio_ws_1.5_fr_long; en `en` 0.99, fr `fr` 1.00; 17 / 27 tokens (r = 1.588)
    - EN: The Service also advises the Government and the public on matters concerning workers' health and workplace hygiene.
    - FR: Il conseille en outre le Gouvernement et la population sur des questions ayant trait à la santé des travailleurs et à l'hygiène su…
13. `ft_un_div:186703`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 21 / 32 tokens (r = 1.524)
    - EN: The situation on the ground continues to deteriorate, while Israel pursues its illegal destructive policies and practices against…
    - FR: La situation sur le terrain ne cesse de se détériorer, à mesure qu'Israël continue de se livrer à une politique et à des pratiques…
14. `ft_un_div:187612`, ratio_ws_1.5_fr_long; en `en` 0.97, fr `fr` 1.00; 14 / 22 tokens (r = 1.571)
    - EN: International human rights law provides the legal framework for gender-sensitive responses to the epidemic.
    - FR: Le droit international des droits de l'homme fournit le cadre juridique dans lequel peuvent s'inscrire des mesures sexospécifiques…
15. `ft_un_random:7087`, ratio_ws_1.5_fr_long; en `en` 0.99, fr `fr` 1.00; 32 / 52 tokens (r = 1.625)
    - EN: Accordingly, the UNDP integrated financial resources framework projections for 2008-2011 total $20.6 billion, or an average of $5.…
    - FR: Conformément à ce cadre intégré, il est prévu que le montant global des ressources du PNUD pour la période 2008-2011 s'élève à 20,…
16. `ft_un_random:7732`, ratio_ws_1.5_fr_long; en `en` 0.99, fr `fr` 1.00; 23 / 40 tokens (r = 1.739)
    - EN: The International Monetary Fund Development Committee had recently focused on potential new frameworks for debt sustainability and…
    - FR: Les participants espèrent que les nouveaux cadres pour assurer la viabilité de la dette et les politiques d'emprunt pour les pays…
17. `ft_un_random:21222`, ratio_ws_1.5_fr_long; en `en` 0.46, fr `fr` 1.00; 12 / 19 tokens (r = 1.583)
    - EN: Mr. Kaj Barlund, Director UNECE Environment, Housing and Land Management Division Director
    - FR: M. Kaj Barlund, Directeur de la Division de l'environnement, du logement et de l'aménagement du territoire de la CEE;
18. `ft_un_random:50443`, ratio_ws_1.5_fr_long; en `en` 0.90, fr `fr` 1.00; 28 / 47 tokens (r = 1.679)
    - EN: The revised gross and net biennial support budget appropriations for the biennium ended 31 December 2005 were $169.6 (2002-2003: $…
    - FR: Le montant révisé des crédits inscrits au budget d'appui biennal pour l'exercice clos le 31 décembre 2005 était de 169,6 millions…
19. `ft_un_random:53813`, fr_wrong; en `en` 0.73, fr `en` 0.83; 15 / 15 tokens (r = 1.0)
    - EN: No. 6 Roy Culpeper, Approaches to Globalization and Inequality within the International System, October 2005.
    - FR: No 6, Roy Culpeper, Approaches to Globalization and Inequality within the International System, octobre 2005
20. `ft_un_random:97947`, ratio_ws_1.5_fr_long; en `en` 1.00, fr `fr` 1.00; 18 / 31 tokens (r = 1.722)
    - EN: The value of time is obtained through direct methods of evaluating effects, through stated-preference surveys or revealed-preferen…
    - FR: La valeur du temps est en général établie à l'aide de méthodes directes d'évaluation des effets, notamment au travers d'enquêtes d…

Reading them: 17 of the 20 are ratio flags with a long French side. Nearly all are faithful
translations, where French simply needs more words. #16 is a free rewording and #17 a loose
expansion. #5 is a language-ID false positive. #19 is the one real case: a French-side reference
whose title stayed in English. (These 20 are not the checker's 20, because the "either flag" pools
differ by tool.)

## Caveats

1. **Wrong language mostly means "English with little ordinary text".** In the UN sets, most
   English-side flags under both tools are lists of names, dates, sessions or countries. Examples:
   "Goal 6: Combat HIV/AIDS, malaria and other diseases." → `pcm`; "Belarus, Bulgaria, Croatia, El
   Salvador, …" → `bcl`. Real foreign-language text (French- or Spanish-language titles on the English
   side, English on the French side) is a minority in the halves and a larger share in random UN.
2. **Part of the repetitive half's excess is the repetition itself.** 317 of the 777 py3langid
   English-side flags in `ft_un_rep` are rows whose line occurs more than once in the set (604
   distinct lines). For Apple NL it is 102 of 338. `ft_un_div` has no repeated lines.
3. **The ratio rule mostly measures how much longer French is, not misalignment.** 4,523 of the
   4,595 `ft_un_rep` flags are long-French rows. The sampled flags read as correct translations.
   The halves are length-matched (identical source-length buckets: 22,225 / 84,098 / 96,513 /
   11,653 rows in all three UN sets), so the rep < div < random ordering is not a source-length
   effect.
4. **Whitespace tokens count NBSP.** `str.split()` splits on U+00A0, the space French puts before
   `: ; ? !`, so French token counts include that punctuation. The checker's blank-only awk recount
   gives lower counts (ws 1.5: 4,244 / 6,240 / 11,989 for rep / div / random) but the same ordering.
   The headline rule stays `str.split()`, as the builder defined it.
5. **The confidence scales are not comparable across tools.** py3langid's tempered confidence is
   stricter than Apple NL's hypothesis probability: 6 vs 48 confident flags in `ft_un_rep`. Only the
   signs of the confident-flag contrasts should be compared, and those are within noise for C1.
6. **Apple NL's model is not pinned.** It is the macOS 27.2 system model. The py3langid wheel and
   its model are pinned by sha256.
7. Descriptive only, not registered. The sds are binomial noise scales over rows. They are not
   tests and do not account for the dependence that repeated lines create.

## Files

| file | what it is |
| --- | --- |
| `spm_lengths.py` | Stage 1, run by the reference interpreter. SentencePiece piece counts for every line of the 18 text inputs, plus their sha256s, written to `spm_lengths.npz` (on the Studio only, 26 MB). |
| `item3_contamination.py` | Stage 2, run by the py3langid venv. The analysis: fail-closed on sha256 pins, public `classify()` for the headline, 5 spawn workers. Writes `results.json` and `per_string.npz` (on the Studio only, 17 MB). |
| `check.py` | An independent recount with the public API on every line of 5 sets, compared with `results.json`. Writes `check_results.json`. |
| `compare_tools.py` | Compares with the checker's Apple-NL `results.json`: exact ratio agreement, LID rates and contrasts, row overlap, flag anatomy. Writes `compare_tools.json`. |
| `run.sh` | venv creation and install (skipped if the venv exists), then stage 1, stage 2 and the check, in order, on the Studio. |
| `results.json`, `check_results.json`, `compare_tools.json` | Fetched from `~/cpu_items/item3_contamination/` on the Studio (sha256 `fb1bd1a1…`, `7390b18c…`, `d2b4e15c…`). |
| `logs/` | `install.log`, `spm.log`, `run.log`, `check.log`, `compare.log`. |
| `req_lid.txt` | The one-line hash-pinned requirement for the py3langid wheel. |

To reproduce on the Studio: copy the four `.py` files and `run.sh` to
`~/cpu_items/item3_contamination/`, run `bash run.sh`, then
`nice -n 5 venv/bin/python -I -u compare_tools.py`.
