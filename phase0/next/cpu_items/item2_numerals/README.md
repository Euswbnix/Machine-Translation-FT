# Item 2 (WMT2027_PLAN.md §3.2): templated numerals

A descriptive CPU analysis to inform the author's decisions. It is not a registered result.
Run on the Mac Studio (arm64, Python 3.14.3, unidata 16.0.0, numpy 2.4.5, 4 workers, nice 5, 68 s):

    ~/e06env/venv-arm64/bin/python -I item2_numerals.py --data ~/e06data --out ~/cpu_items/item2_numerals --workers 4

Files: `masking.py` (the masking rules), `item2_numerals.py` (the analysis), `results.json`
(every number below, plus the regexes, sensitivity variants and examples), `run.log`.
The Studio holds the same files in `~/cpu_items/item2_numerals/`.

## Definition

**Statistic (unchanged).** Tokens are lowercased `\w+` (`e04_split.py` l.13, l.29). rep(row) is the
fraction of the row's 8-grams (blake2b-64 of `" ".join(8 tokens)`, l.16-17, l.30) whose count over
the whole parent set is above one (l.36). Rows under 8 tokens get 0. The parent is `ft_topk`
(926,999 rows) for `ft_topk_div/rep` and `ft_un_div/rep`. For `ft_un_random` and `ft_random_small`
it is `ft_random`, as in E0.5's manifest (`e05_build.py` `rep_within()` l.94, `rnd_rep` l.295).
The definitions match `prereg_drafts.md` "Definitions shared by every build" (l.172).

**Masking (`masking.py`).** No token is added or removed, and every token keeps its position.
Each token gets one class, checked in this priority order:
- **SYM**: the token overlaps a UN document symbol. A symbol is a series code that starts with
  a capital letter, followed by one or more `/` components, and the whole match must contain a
  digit. Only the first letter must be a capital: the rest of the code may be mixed case (up to
  12 letters in all), so mixed-case codes such as `Hawk/UH60` or `No.15/2003` also match (see
  Caveats).
  Regex: `(?<![\w/])[A-Z][A-Za-z]{0,11}(?:\.\d+)?(?:/[A-Za-z0-9][A-Za-z0-9.()\-]*)+`.
  Examples: `S/2005/86`, `A/C.3/58/L.12`, `E/CN.4/2004/L.10`, `CCPR/C/79/Add.25`.
  The digit requirement keeps out `HIV/AIDS`, `UN/ECE`, `and/or`.
- **DATE**: the token overlaps a date expression. Four shapes are covered:
  - [weekday] day [range] month [year]: `26 January 2004`, `26 and 27 January`, `1st of July`,
    `1 July 2006-30 June 2008`
  - [weekday] month day [range] [, year]: `June 30, 2002`, `January 26-28`
  - month [range] year: `July 2004`, `January-March 2004`
  - all-numeric dates: `26/01/2004`, `26.01.04`, `2004-01-26`

  In the first two shapes an optional weekday name in front is part of the date, so the
  weekday token is DATE too: `Monday, 15 March 2004`. A day or month range is joined by `-`,
  `–`, `—`, `to`, `and` or `or`. Month names must be capitalised, so the verb "may" never
  matches. Every shape needs a digit; "in March, June and September" is not masked. A comma
  between month and year with no day (`July, 2004`) is not a date: `2004` is still NUM, but
  `July` stays WORD.
- **NUM**: the token fully matches `\d+(?:st|nd|rd|th|s)?`. Examples: 1997, 22, 58th, 1990s, and
  the 51 and 2 of "51.2". Because `\w+` splits on punctuation, `57,902` becomes the two tokens
  57 and 902.
- **WORD**: every other token, including `co2`, `g77`, `km2` and spelled-out numerals.

The sources do not define "masked" (WMT2027_PLAN.md §3.2 item 2; the lo2018 note's design
item 4), so two readings are reported. Making M1 the headline is this analysis's choice. Lo et
al.'s number-blind dedup is closer to M2 and A17, and under those two readings rep rises
rather than falls (below).
- **M1, exclude (headline).** A masked token matches nothing. An 8-gram occurrence that contains a
  masked token never counts as repeated. An untouched 8-gram counts as repeated if it occurs
  untouched more than once in the parent. The denominator is still all of the row's 8-grams, so
  rep_M1 ≤ rep on every row. This answers "how much of rep runs through numerals", meaning
  repeated 8-grams that contain a number, date or symbol, which must then be the *same* number
  in every occurrence. It cannot see templates whose numbers differ. Their numeral-touching
  8-grams were never repeated, and their numeral-free repeated 8-grams survive M1.
- **M2, placeholder.** Masked tokens become `#num`, `#date` or `#sym`, and counts are recomputed
  over the parent. Rows that differ only in their numbers now share 8-grams. This is the reading
  that picks up templates whose numbers differ.

Sensitivity variants in `results.json`:
- **M1_digit**: mask exactly the tokens that contain a Unicode digit.
- **M1′**: M1's repeated 8-grams divided by the row's numeral-free 8-grams.
- **M2_A17**: every digit becomes `0` inside the token, which is `e06_build.py`'s `A17_RULE`
  applied to rep.

**Why these rules (lo2018 note).**
- Lo et al. (Sec. 2.1) dedup after replacing emails and URLs with a placeholder. They dedup again
  after removing all numbers and punctuation. M2 and the A17 key take the same number-blind view,
  but they do not remove anything. M2 puts in a placeholder, and A17 turns every digit into `0`.
  Punctuation is already gone, because `\w+` drops it. Lo et al.'s steps are whole-pair dedups,
  not an 8-gram statistic.
- Sec. 2.1 reports, qualitatively, that pairs such as lists of page numbers or dates were
  uninformative and that measures preferring them gave poor MT. Sec. 2.4 names many perfectly
  translated datelines among the top-ranked pairs. Both justify masking dates.
- The note's reviewer risk 5 and design item 4 ask whether "repetitive" UN text is just numeric
  or template boilerplate, such as dates or document symbols. M1 and M2 answer it from both
  sides. M1 removes the repetition that runs through identical numbers. M2 adds the repetition
  that differing numbers hide.
- Lo et al. do not mention document symbols; their pool was ParaCrawl. Masking symbols is not an
  addition made here. WMT2027_PLAN.md §3.2 item 2 and the note's design item 4 both ask for it,
  and reviewer risk 5 names symbols.

**Reproduction gate.** All 17 inputs match their pinned sha256. The unmasked statistic reproduces
E0.5's manifest exactly, as float equality, in 21 checks:
- the median, 0.0;
- the UN candidates: 486,475 with rep = 0 and 227,406 with rep > 0;
- rep_mean, rep_positive_fraction and duplicate_source_fraction of all six sets.

## Numbers

### (a) Rows that keep rep > 0 after masking, and (b) mean rep before and after

| set | rows | rep>0 before | rep>0 M1 | lose all rep (M1) | rep mean | M1 | M1_digit | M2 | M2_A17 |
|---|---|---|---|---|---|---|---|---|---|
| ft_un_rep | 214,489 | 214,489 | **204,134 (95.17%)** | 10,355 | 0.4332 | **0.3817 (−11.9%)** | 0.3824 | 0.4559 | 0.4503 |
| ft_topk_rep | 242,022 | 242,022 | **230,313 (95.16%)** | 11,709 | 0.4174 | **0.3681 (−11.8%)** | 0.3688 | 0.4393 | 0.4339 |
| ft_un_div | 214,489 | 0 | 0 | – | 0 | 0 | 0 | 0.0142 (16,024 rows >0) | 0.0097 |
| ft_topk_div | 242,022 | 0 | 0 | – | 0 | 0 | 0 | 0.0114 (14,878 rows >0) | 0.0077 |
| ft_un_random (rep over ft_random) | 214,489 | 54,628 | 51,152 | 3,476 | 0.0866 | 0.0727 (−16.1%) | 0.0734 | 0.1094 | 0.1020 |
| ft_random_small (rep over ft_random) | 242,022 | 47,694 | 44,897 | 2,797 | 0.0648 | 0.0549 (−15.3%) | 0.0554 | 0.0811 | 0.0760 |

- **The fall in mean rep is M1 only.** M1 lowers it by about 12% in the two repetitive halves.
  Under M2 and A17 it rises instead: `ft_un_rep` goes from 0.4332 to 0.4559 (M2) and 0.4503
  (A17), and `ft_topk_rep` from 0.4174 to 0.4393 and 0.4339. Under M2, 16,024 rows of `ft_un_div`
  (7.5%) and 14,878 of `ft_topk_div` (6.1%) gain rep > 0. Those rows repeat only once their
  numbers, dates or symbols are replaced: the same wording with different numbers.
- Among the rows that lose all rep under M1, many repeat as a whole line. That is 1,178 of
  10,355 in `ft_un_rep` and 1,297 of 11,709 in `ft_topk_rep`. Their source line occurs more than
  once in `ft_topk`, so they are verbatim reuse that happens to have a number in every window.
- In `ft_un_rep` and `ft_topk_rep`, 10.35% and 10.29% of repeated 8-gram occurrences touch a
  masked token. In the random sets the figure is 14.8% and 14.1%.
- **E0.4/E0.5 contrasts.** The rep gap between `ft_un_rep` and `ft_un_div` is 0.4332 unmasked,
  0.3817 under M1 (88% of it remains) and 0.4417 under M2. The gap between `ft_un_rep` and
  `ft_un_random` is 0.3466, 0.3090 under M1 and 0.3465 under M2. Masking does not close either
  gap.
- **Whole band, for E0.6's candidate pools.** Of the 227,406 repetitive UN candidates, 216,680
  keep rep > 0 under M1. Of the 486,475 diverse UN candidates (rep = 0), 29,309 would become
  rep > 0 under M2 (6.0%) and 21,150 under A17.
- **M1_digit ≈ M1.** Masking only the tokens that contain a digit gives nearly the same numbers.
  Month names and symbol letters add almost nothing.

### (c) Numeric-token share (source side, share of lowercased `\w+` tokens)

| set | NUM | DATE | SYM | masked total | tokens with a digit | rows with a masked token | fr tokens with a digit |
|---|---|---|---|---|---|---|---|
| ft_un_div | 2.71% | 0.73% | 0.016% | **3.46%** | 3.15% | 28.5% | 2.64% |
| ft_un_rep | 1.99% | 1.17% | 0.079% | **3.23%** | 2.76% | 28.4% | 2.34% |
| ft_topk_div | 2.40% | 0.63% | 0.013% | **3.04%** | 2.78% | 26.2% | 2.34% |
| ft_topk_rep | 1.95% | 1.10% | 0.071% | **3.12%** | 2.68% | 27.7% | 2.27% |
| ft_un_random | 2.23% | 1.01% | 0.777% | **4.02%** | 3.30% | 32.1% | 2.78% |
| ft_random_small | 1.94% | 0.78% | 0.522% | 3.24% | 2.76% | 28.2% | 2.37% |

- **UN halves:** the repetitive half is not more numeric than the diverse half. It has fewer
  masked tokens (3.23% vs 3.46%) and fewer digit tokens (2.76% vs 3.15%).
- **All-source halves:** these are about equally numeric. `ft_topk_rep` has slightly *more*
  masked tokens than `ft_topk_div` (3.12% vs 3.04%) and more rows with a masked token (27.7% vs
  26.2%). Only its share of digit tokens is lower (2.68% vs 2.78%).
- In both pairs, the repetitive half has fewer NUM tokens and more DATE tokens. The UN random
  control has the most masked tokens, mostly because of UN symbols: its SYM share is about 10x
  `ft_un_rep`'s.
- Symbol matches (`symbol_matches` in `results.json`):
  - `ft_un_rep`: 1,248
  - `ft_un_div`: 264
  - `ft_un_random`: 9,609
  - the whole top band: 1,740 in 926,999 rows

  CometKiwi's top band nearly excludes rows that cite documents.

**Duplicate-source fraction** (extra, as the lo2018 note's design item 4 asks):

| set | raw line | token sequence | A17 key | M2 key |
|---|---|---|---|---|
| ft_un_rep | 8.45% | 10.67% | 11.75% | 12.30% |
| ft_un_div | 0 | 0 | 0.014% | 0.028% |
| ft_un_random | 0.23% | 0.35% | 0.70% | 1.06% |

The raw-line column is E0.5's measure. The `ft_topk_*` and `ft_random_small` rows are in
`results.json`.

### 20 rows of ft_un_rep that lose all rep under M1

The rows are drawn with `random.Random(42).sample` from the 10,355 losing rows only. Those rows
are the 4.8% tail, so the sample does not show typical `ft_un_rep` rows. "line" is the
1-based line in `ft_un_rep.en`, and "in ft_topk" counts how often the exact source line occurs
there. The masked rendering and the top 4 repeated 8-grams of each row are in `results.json`.

| # | line | in ft_topk | rep | M2 | most-repeated 8-gram (count) | source |
|---|---|---|---|---|---|---|
| 1 | 2146 | 1x | 0.308 | 0.885 | `of which 51 2 per cent was male` (2) | According to the Central Intelligence Agency estimates for July 2004, the population of American Samoa was 57,902, of which 51.2 per cent was male and 48.8 per cent was female. |
| 2 | 2311 | 2x | 1.000 | 1.000 | `over 250 million people are directly affected by` (2) | Over 250 million people are directly affected by desertification. |
| 3 | 2813 | 1x | 0.154 | 0.154 | `during the first quarter of 2008 israeli authorities` (2) | During the first quarter of 2008, Israeli authorities demolished 124 structures for lack of permits.  Of those, 61 were residential buildings that caused the displacement of 435 Palestinians, 135 of whom were children. |
| 4 | 3065 | 1x | 0.333 | 0.500 | `30 per cent of children 5 to 14` (2) | Some 30 per cent of children 5 to 14 years old are working. |
| 5 | 13827 | 1x | 0.174 | 0.217 | `in 1999 alone there were more than 700` (4) | In 1999 alone, there were more than 700 large-scale disasters, resulting in the death of approximately 100,000 people and causing economic losses in excess of US$ 100 billion. |
| 6 | 15505 | 1x | 0.167 | 0.167 | `to all 12 united nations anti terrorism conventions` (2) | Slovakia is also a signatory to all 12 United Nations anti-terrorism conventions. |
| 7 | 17766 | 1x | 0.200 | 0.200 | `constitute a breach of article 3 of the` (12) | 3.1 The complainant claims that his extradition to the United States would constitute a breach of article 3 of the Convention. |
| 8 | 20024 | 2x | 1.000 | 1.000 | `the royal cayman islands police force comprises 295` (2) | The Royal Cayman Islands Police Force comprises 295 officers and 40 support staff. |
| 9 | 28438 | 1x | 0.250 | 1.000 | `the period from july 2006 to june 2007` (3) | The report covers the period from July 2006 to June 2007. |
| 10 | 53609 | 2x | 1.000 | 1.000 | `0 5 per cent of world exports and` (6) | On average, LDCs accounted for 0.5 per cent of world exports and 0.7 per cent of world imports. |
| 11 | 55028 | 1x | 1.000 | 1.000 | `the cost is estimated to be 1 552` (2) | The cost is estimated to be $1,552.8 million, reflecting a $503.8 million increase over the initial estimates. |
| 12 | 58639 | 1x | 0.333 | 1.000 | `the convention was ratified by brazil on 18` (2) | The Convention was ratified by Brazil on 18 November 1952. |
| 13 | 63270 | 2x | 1.000 | 1.000 | `this principle is also reflected in principle 10` (4) | This principle is also reflected in Principle 10 of the Rio Declaration and in Principle 23 of the World Charter for Nature. |
| 14 | 73948 | 1x | 0.417 | 0.417 | `according to 1998 estimates anguilla has a total` (2) | According to 1998 estimates, Anguilla has a total of 279 kilometres of motorable roads, of which 253 are paved. |
| 15 | 129260 | 2x | 1.000 | 1.000 | `pursuant to general assembly decision 48 432 of` (2) | Pursuant to General Assembly decision 48/432 of 20 December 1993, 3 May has been declared World Press Freedom Day. |
| 16 | 162562 | 1x | 0.241 | 0.276 | `of 0 7 per cent of gross national` (32) | In terms of official development assistance, along with most other donor countries, we are committed to reaching the target of 0.7 per cent of gross national product in ODA, consistent with the Millennium Development Goals. |
| 17 | 178422 | 1x | 0.067 | 0.133 | `4 2 the state party maintains that the` (4) | 4.2 The State party maintains that the complainant did not exhaust all domestic remedies before submitting his complaint to the Committee. |
| 18 | 184368 | 1x | 0.115 | 0.115 | `live on less than 2 a day and` (4) | For nearly half of the global population (3 billion people) who live on less than $2 a day, and the 1.5 billion who live in extreme poverty, poverty eradication is a dream. |
| 19 | 196972 | 1x | 0.200 | 1.000 | `exceptions authorized owing to medical conditions 1 july` (5) | Exceptions authorized owing to medical conditions, 1 July 2006-30 June 2008 |
| 20 | 201884 | 1x | 0.160 | 0.160 | `period from 1 july 2001 to 30 june` (12) | For the current period, $418,200 was authorized, and for the period from 1 July 2001 to 30 June 2002, estimated requirements total $412,600 (see annex III to the present report). |

Three kinds of row appear in this sample:
- **Reuse of a whole factual sentence** that contains a number. Rows 2, 8, 10, 13 and 15 occur
  twice in `ft_topk`. Row 11 occurs once but has every 8-gram repeated.
- **Fixed formulae with fixed numbers**: the 0.7% ODA target, "article 3 of the Convention",
  budget periods and "$2 a day".
- **Numbered paragraph openings** in treaty-body communications: "3.1 The complainant", "4.2 The
  State party".

Only the last two are "templated numerals" in the reviewer's sense. M1 also removes the first
kind, so the 4.8% that loses all rep is an upper bound on the share of rows whose repetition runs
entirely through the same numbers. It is not a bound on templated numerals in general.
Templates whose numbers differ are invisible to M1 and show up under M2 instead (above: 7.5% of
`ft_un_div` rows gain rep > 0).

## Caveats

- **M1 is deliberately harsh.** Any 8-gram that touches a numeral is dropped, even inside a
  sentence repeated verbatim. Take the 11.9% fall in mean rep and the 4.8% of rows that lose all
  rep as upper bounds on the repetition that runs through identical numerals. They do not bound
  what templated numerals account for overall: under M2, where differing numbers match, mean
  rep rises. Under either reading the E0.4/E0.5 gaps survive: `ft_un_rep` minus `ft_un_div` is
  0.3817 under M1 and 0.4417 under M2.
- **The classes depend on context.** "january" is DATE only inside a date, and "a" is SYM only
  inside `A/58/12`. So M2 ≥ rep is not guaranteed row by row:
  - It fails on 116 rows of `ft_topk` and 180 of `ft_random`.
  - That includes 98 rows of `ft_un_rep`, 19 of which drop to M2 = 0. This is why M2's rep>0 count
    there is 214,470, not 214,489.
  - The strict M1 count differs from the naive "drop touched grams" count on 33 rows of `ft_topk`.
  - The script checks the guaranteed inequalities (M1 ≤ rep, M1_digit ≤ rep, rep ≤ M2_A17) on
    every row and counts the M2 exceptions.
- **The rules are approximate.**
  - Roman numerals (annex III, chapter IV, resolution 1514 (XV)), spelled-out numbers
    ("fifty-eighth session") and list labels ("(a)") are not masked.
  - The date patterns are English only. The French side gets only the digit-token share.
  - Symbol false positives are rare. Of the 1,740 matches in `ft_topk`, the non-UN ones are
    `Hawk/UH60` (4), `Kwh/m2/day` (1) and a few law numbers such as `No.15/2003`. These are
    mixed-case codes, which match because the SYM regex requires only a capital *first*
    letter, not an all-caps code. The full non-A/E/S list is
    `ft_topk_symbol_matches.not_A_E_S` in `results.json`. An independent check that requires an
    all-caps code finds 8 fewer symbol tokens in `ft_un_rep` (4,406 against this run's 4,414).
  - A weekday name in front of a date is masked with the date (see Definition). The independent
    check counts 310 such dates in `ft_un_rep`, 100 in `ft_un_div` and 445 in `ft_topk_rep`.
    `Month, year` with a comma and no day is not a date here.
  - Prefixed codes such as `EB.AIR/...` are masked only from `AIR` onward.
- **Independent check.** The definition was re-implemented from its written statement, with
  its own regexes and no shared code (`~/cpu_items/item2_numerals_check/` on the Studio). It
  matches exactly every number that does not depend on the regexes: the reproduction gate, the
  8-gram counts, M1_digit, A17, and the duplicate rates on raw lines, token sequences and the
  A17 key. The regex-dependent numbers differ by a few rows or tokens. For example, 204,138
  rows of `ft_un_rep` keep rep > 0 under M1 against 204,134 here, and 1,732 symbol matches in
  `ft_topk` against 1,740. The largest relative differences are in small counts (113 against
  116 rows where M2 < rep). Treat the last digit of regex-dependent counts as approximate. The
  differences move one rounding: `ft_un_random`'s M1 fall is −16.06% here (shown as −16.1%)
  and −16.03% in the check. The E0.4/E0.5 gaps agree to four decimals.
- **Interpreter differences.** E0.5 was built with Python 3.12.15 and unidata 15.0.0; this run
  used 3.14.3 and 16.0.0. All 21 reproduction checks are exact, so tokenisation did not change on
  these inputs.
- **rep for the two random sets is computed over `ft_random`**, not `ft_topk`, as in E0.5.
- **Example sampling.** "lose all rep" means rep > 0 before and rep_M1 = 0. The 20 examples are a
  seed-42 sample, and the three kinds of row above are an eyeball reading of 20 rows, not a count.

## Open points

These need a code change or a recomputation, so they are not done here.
- **SYM regex.** Should a symbol require an all-caps series code? That would drop `Hawk/UH60`,
  `Kwh/m2/day` and `No.15/2003`. It needs a change to `masking.py` and a re-run. The expected
  effect is a few tokens: 8 in `ft_un_rep` by the independent check.
- **DATE regex.** Should a weekday name in front of a date stay in DATE? Should `Month, year`
  (`July, 2004`) become a date? The independent check masks 29 of the latter in `ft_un_rep`.
  Either change needs a change to `masking.py` and a re-run.
- **How much of rep templated numerals account for overall.** M1 bounds only the repetition
  that runs through identical numbers, and M2 only shows that number-blind matching adds
  repetition. Splitting M2's rise in `ft_un_rep` (0.4332 to 0.4559) into repeated 8-grams that
  touch a placeholder and those that do not would take a new computation.
