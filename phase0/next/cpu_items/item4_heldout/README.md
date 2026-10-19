# §3.2 item 4: heldout_un hygiene

WMT2027_PLAN §3.2 item 4 (lines 346-350): (a) check that heldout_un is disjoint from every E0.6
candidate set built by the feasibility run, and compare with the feasibility report's own
`heldout_un_shared_lines` field; (b) measure heldout_un's overlap with the scored pool
(`v2_scored.tsv`) and with the base's pretraining corpus (`train.clean`). This is descriptive only.

## Status: RUN, 2026-10-19 13:51-13:55 EDT, on the Mac Studio. Every number agrees with the independent check

- Run with `bash launch_from_macbook.sh start`, `status`, `fetch`. The Studio was reached over
  Tailscale (`alexs-mac-studio`; its own hostname is `Workstation.local`). The interpreter was the E0.6
  reference interpreter `~/e06env/venv-arm64/bin/python`: Python 3.14.3, numpy 2.4.5, Unicode
  16.0.0, arm64. Nothing was installed. `run.sh` exited with `EXIT=0` (`logs/reduce.log`).
- The fixture test ran first: 51 of 51 checks `ok`, 0 `FAIL` (`logs/test.log`).
- **(a)** None of the 46 feasibility index files shares anything with heldout_un, exact or
  normalised, counted from either side. The report's `heldout_un_shared_lines` is `{0, 0}` for all
  46, and the recount agrees for all 46.
- **(b) pool.** All 2,000 heldout_un rows have their exact pair, source and target in
  `v2_scored.tsv`, as expected by construction. 4,652 pool rows touch heldout_un, which is exactly
  E0.3's reservation (3,290 + 1,078 + 284).
- **(b) pretraining.** 0 of 9,312,233 `train.clean` rows share an exact or normalised source,
  target or pair with heldout_un.
- **Independent check.** Every number was compared with the checker's, which was measured with its
  own code (`../item4_heldout_check/`): 159 of 159 comparisons agree, and so do all 874 per-file
  values in (a). See "Comparison with the independent check" below.

What was changed before the run: only the host in `launch_from_macbook.sh`
(`euswbnic@10.0.0.126` → `euswbnic@alexs-mac-studio`). The LAN address does not work from the
MacBook now, and the Studio is reached only over Tailscale. `heldout_hygiene.py`, `check_awk.sh`,
`test_hygiene.py` and `run.sh` were run as written; no bug was found in them.

The earlier status (12:11-13:13 EDT): the Studio could not be reached over the LAN, so nothing was
run then, and the figures in this README were quotes. They are now measurements, except where a line
says it is quoted from E0.3.

## Results

All figures are from `results.json` (written by `reduce` from `parts/*.json` and `check/*_awk.txt`).

### Inputs (`parts/inputs.json`)

- The 28 `~/e06data` files read have the sha256 listed in `e06data_SHA256SUMS`: 28 of 28.
- E0.3's fingerprints (`controls_manifest.json` `inputs`: size and blake2b of the first and last
  16 MiB) match for `v2_scored.tsv`, `train.clean.en`, `train.clean.fr`, `provenance_labels.npy` and
  `pool_mask_reused.npy`: 5 of 5. The pool and pretraining corpus are the files E0.3 reserved against.
- The rental bundle's `provenance_labels.npy`, `provenance_report.json` and `pool_mask_reused.npy`
  have their recorded sha256: 3 of 3.

### (a) heldout_un against the E0.6 candidate sets (`parts/sets.json`)

- heldout_un (`e05_src`, identical to `phase0_e05/heldout_un.*`): 2,000 rows; 1,999 distinct
  sources, targets and pairs (also 1,999 normalised sources and targets); no empty line; no tab.
- Feasibility report `9c73852f…`. Its `inputs` sha256 are the parent and heldout_un files read
  here: 8 of 8.
- 46 `FEASIBILITY_*.npy` files: 30 index `ft_topk` and 16 `ft_random`, with 36 distinct index
  contents. All 46 have the sha256 the report gives. They are strictly ascending inside their parent
  and have the report's row count. The repo's `e06_feasibility/SHA256SUMS` matches all 47 entries
  (the 46 files and the report). The 5 options that failed (A_narrow, OD 6 = remove, seeds 42, 1, 2,
  3 and 4) have no file. Every row of every set is a UN row.
- For each of the 46 files, all of the following are 0:
  - set rows with an exact heldout source, target or pair, or a normalised source or target (the
    production check, spec 6.7, counts these rows);
  - heldout rows whose source, target or pair, exact or normalised, occurs in the set;
  - the distinct set lines found in heldout_un (= the report's field, which is `{0, 0}` for all 46,
    so 46 of 46 agree).
- Supplementary sets, all 0 on every count:
  - `vocab/R0.npy` (214,489 rows; the same index as `FEASIBILITY_ft_un_r0__od6-keep__seed42.npy`);
  - `vocab/A_narrow_42_58.npy` (214,489; the same as `FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy`);
  - the six E0.5 sets: `ft_topk_div`, `ft_topk_rep` and `ft_random_small` (242,022 rows each), and
    `ft_un_div`, `ft_un_rep` and `ft_un_random` (214,489 each);
  - the parents as wholes: `ft_topk` and `ft_random`, 926,999 rows each;
  - the parents' UN rows: 713,881 in `ft_topk` and 617,634 in `ft_random`.

### (b) heldout_un against the scored pool (`parts/pool.json`)

`v2_scored.tsv`: 38,275,284 rows. 0 rows have other than 3 fields.

Heldout side: 2,000 of 2,000 rows have their exact pair in the pool, and also their exact source,
exact target, normalised source and normalised target. That is 1,999 distinct pairs, sources and
targets. 0 heldout rows have no pool row.

Pool side (rows of `v2_scored.tsv`):

| pool rows that share with heldout_un | rows |
| --- | ---: |
| exact pair | 3,290 |
| exact source | 4,139 |
| exact target | 3,605 |
| exact source or target | 4,449 |
| exact source, not the pair | 849 |
| exact target, not the pair | 315 |
| normalised source | 4,368 |
| normalised target | 3,848 |
| normalised source, not the exact source | 229 |
| normalised target, not the exact target | 243 |

E0.3's reservation categories (`e03_build_controls.py` l.389-394), with e01 source labels and the
reused-score mask:

| category | rows | inside the reused mask | e01 sources |
| --- | ---: | ---: | --- |
| exact pair | 3,290 | 3,290 | un 3,290 |
| + normalised source, not the pair | 1,078 | 1,072 | un 1,072, giga-fren 6 |
| + normalised target, neither of the others | 284 | 283 | un 283, giga-fren 1 |
| **total reserved** | **4,652** | **4,645** | un 4,645, giga-fren 7 |
| (exact source, not the pair) | 849 | 843 | un 843, giga-fren 6 |
| (exact target, not the pair) | 315 | 315 | un 315 |

3,290 / 1,078 / 284 / 4,652 are exactly E0.3's manifest figures. 4,645 inside the mask is item 5's
figure (`../item5_depth/README.md`).

- **Pool copies per heldout pair.** 1,627 heldout rows have their pair once in the pool, and 373 more
  than once. The most is 44 copies. Distribution (copies: heldout rows): 1: 1,627; 2: 208; 3: 61;
  4: 28; 5: 18; 6: 10; 7: 7; 8: 3; 9: 2; 10: 2; 11: 4; 12: 4; 13: 2; 14: 2; 15: 2; 16: 1; 17: 1;
  18: 4; 20: 1; 23: 1; 24: 4; 25: 1; 32: 1; 34: 1; 37: 1; 40: 1; 41: 1; 42: 1; 44: 1.
- **The duplicated heldout pair** (0-based heldout rows 1054 and 1903) has **24** copies in the
  pool, pool rows 9,944,888-9,945,495 (the row list is the checker's, in its `results/reduce.json`). The
  draw took two of them. The most pool rows sharing one heldout source is 95, and one heldout target 48.
  (From `pool_per_heldout_row.json`, which stays on the Studio.)
- **QE scores** of the 3,290 exact-pair rows: min 0.200354, median 0.879948, max 0.911058; no NaN.
- **The evaluated copy** (`inputs/filtered_tests/heldout_un.*`, sha256 `c014363f…` / `4222f10b…`)
  has 1,999 rows: heldout_un without 0-based row 1764. That row's pair has 1 pool row, its source 1
  and its target 2. All 1,999 kept rows have their exact pair, source and target in the pool.

### (b) heldout_un against the pretraining corpus (`parts/pretrain.json`)

`train.clean.{en,fr}`: 9,312,233 lines on each side. The count is 0 for each of the following:
rows with an exact heldout source, exact target, exact pair, normalised source or normalised target;
heldout rows whose exact source, target or pair, or normalised source or target, occurs in it.

### awk recount (`check/*_awk.txt`, exact matching, no Python)

| file | rows | rows with a heldout source / target / pair | heldout rows (of 2,000) found: source / target / pair |
| --- | ---: | --- | --- |
| `v2_scored.tsv` | 38,275,284 | 4,139 / 3,605 / 3,290 | 2,000 / 2,000 / 2,000 |
| `train.clean` | 9,312,233 | 0 / 0 / 0 | 0 / 0 / 0 |
| `ft_topk` | 926,999 | 0 / 0 / 0 | 0 / 0 / 0 |
| `ft_random` | 926,999 | 0 / 0 / 0 | 0 / 0 / 0 |

The awk pass also found 0 pool rows with other than 3 fields, and equal line counts for the two sides
of each aligned pair of files. Every figure equals the Python figure above.

## Comparison with the independent check

The checker (`../item4_heldout_check/hcheck.py`, run on the Studio before this item, results in
`../item4_heldout_check/results/`) was written from the source scripts and not from
`heldout_hygiene.py`. It joins the pool and pretraining corpus through its own keyed 64-bit blake2b
keys, and through strings as a second path. `compare_with_check.py` compares every figure that both
produce and writes `compare_with_check.json`. To include the four per-row figures, it needs
`--per` with a copy of the Studio's `pool_per_heldout_row.json`. Result: **159 of 159 agree**, and in
(a) all 874 per-file values agree (46 files × 19 values each: rows, tag, parent, sha256, UN rows, the
report field, agreement with it, all-zero, the two shared-line counts, five set-side counts and five
heldout-side counts).

| section | compared | agree | figures (builder = checker) |
| --- | ---: | ---: | --- |
| inputs | 36 | 36 | sha256 of 28 e06data files and 3 bundle files; 5 E0.3 fingerprint verdicts |
| heldout_un | 8 | 8 | 2,000 rows; 1,999 distinct sources, targets, pairs, normalised sources, normalised targets; 0 tab; 0 empty |
| (a) sets | 51 | 51 | report sha256; 8 report-input sha256; 5 failed options; 46 file names; 46 files, 46 all zero, 46 agreeing with the report; 30 / 16 by parent; 36 distinct contents; 47 repo SUMS entries; 10 supplementary sets (rows, all zero, 5 counts each, the two twins); parents 926,999 / 926,999; the 874 per-file values |
| (b) pool | 44 | 44 | 38,275,284 rows; 0 odd rows; the 10 pool-side counts; E0.3 categories 3,290 / 1,078 / 284 / 4,652 (= checker = E0.3 manifest); 2,000 heldout rows found for each of the 5 keys; the copies distribution; max 44, min 1; source / mask / label splits of 6 categories; duplicated pair 24 copies; source max 95; target max 48; 373 rows with more than one copy |
| evaluated copy | 4 | 4 | 1,999 rows; both sha256; dropped row 1764; 1,999 kept rows found |
| (b) pretraining | 11 | 11 | 9,312,233 rows; all 10 counts 0 |
| awk lines | 5 | 5 | the four awk lines above and the 0 odd rows, against the checker's Python figures |

20 of the 159 are not one-to-one field comparisons, and `compare_with_check.json` says how each was
formed. 4 are against E0.3's manifest, read through the checker. 11 are derived from the checker's own
fields or use one under another name: for example, "exact source, not the pair" = checker exact source minus exact pair, and
"no pool row" = 2,000 minus rows found. 4 use the builder's Studio-side per-row file. The last is the
47 repo SUMS entries: the checker's 46 `.npy` matches plus its report sha256 check.

**The one discrepancy is in this README, not in a measurement.** The earlier text said the duplicated
heldout pair "occurs twice in the pool and both copies were drawn". The checker measured 24 pool
copies (`../item4_heldout_check/compare_results.json`, its only disagreement), and this run measures
24 as well. The sentence is corrected below: the pair occurs 24 times in the pool, and two of those
rows were drawn. The checker lists the 24 rows: all lie within 608 consecutive pool rows
(9,944,888-9,945,495), and all are labelled un.

The checker also re-derived figures that this item only quotes from E0.3 and does not measure. The
eligible rows are 16,646,992. 11,598,490 rows are labelled UN. The candidates are 6,767,225 after the
pretraining exclusion, with 4,831,265 excluded. The `RandomState(42)` draw equals heldout_un row by
row (2,000 of 2,000). The PROTOCOL D4 default draw has 793 of 2,000 pairs in pretraining. All of
these equal E0.3's records. They are the checker's measurements, not this run's.

## Files

| file | what it is |
| --- | --- |
| `heldout_hygiene.py` | `inputs` (sha256 checks), `sets` (a), `pool` and `pretrain` (b), `reduce` (writes `results.json`) |
| `check_awk.sh` | an independent check of the exact counts, written in awk with no Python: pool, pretraining corpus, and the two parents `ft_topk` / `ft_random` |
| `test_hygiene.py` | fixture test: plants every kind of overlap and checks each count against a brute-force recount and against the awk check |
| `run.sh` | the run order on the Studio |
| `launch_from_macbook.sh` | copies to the Studio, starts the run, fetches the results (host: `alexs-mac-studio`, over Tailscale) |
| `results.json` | every result: `inputs`, `sets`, `pool`, `pretrain`, `independent_check_awk` |
| `parts/*.json`, `check/*_awk.txt`, `logs/*.log` | the per-step outputs and logs, fetched from the Studio |
| `compare_with_check.py`, `compare_with_check.json` | the comparison with the independent check (stdlib only) |

Studio paths: `~/cpu_items/item4_heldout/`. That folder holds everything above, plus
`pool_hit_rows.npy`, `pool_hit_flags.npy`, `pool_per_heldout_row.json` and `test_fixture/`, which
stay on the Studio. Copied there into `inputs/`: `e06data_SHA256SUMS`, E0.3's
`controls_manifest.json`, `provenance_labels.npy`, `provenance_report.json` and
`pool_mask_reused.npy` (from `~/mt_local/rebuild/rental_bundle_v2/`), `feasibility_SHA256SUMS`
(= `phase0/next/e06_feasibility/SHA256SUMS`) and `filtered_tests/heldout_un.*` (the copy the runs
were scored on). Each has the same sha256 as the checker's copy in `~/cpu_items/item4_heldout_check/inputs/`.
Read only: `~/e06data/` and `~/e06runs/feasibility/`.

How it was run: the test runs first. Phase 1 runs `inputs`, `sets` and `check_awk.sh` (3 awk) at
the same time, 5 compute processes. Phase 2 runs `pool` and then `pretrain`, each with 6 worker
processes. The parent process and multiprocessing's resource tracker sat idle beside them at about
0% CPU. Everything ran under `nice -n 5`. Wall time: test done 13:51:50; awk pool pass done 13:54:49;
`pool` 27.7 s; `pretrain` 6.5 s; `EXIT=0` at 13:55:24.

## Definition used

- **heldout_un** is `~/e06data/e05_src/heldout_un.{en,fr}`: 2,000 rows, sha256 `f70e725e…` /
  `7daf654f…`. These are the values pinned in `e05_build.py` l.66-69 and `e06_build.py` l.121-124.
  `phase0_e05/heldout_un.*` is the same file (checked).
- **Exact matching (primary)** is the matching of E0.5's second pre-flight. The message of commit
  `0fb52a6` says "no row of the in-domain test set shares a source or a target with any of the six
  training sets". `e05_build.py` and `e06_build.py` implement it with `read_lines` (e05 l.72-75):
  UTF-8, LF as the only line break, one trailing LF removed, identical strings. A *pair* is the
  source and target of one row. For `v2_scored.tsv` the source and target are fields 2 and 3 of
  `line.rstrip("\n").split("\t")` (`e03_build_controls.py` `split_scored`).
- **Normalised matching (secondary)** is E0.3's reservation key, `e03_build_controls.py` `norm()`
  (l.115): NFKC, casefold, whitespace runs collapsed to one space, stripped. Source and target are
  compared separately. The pool step also counts E0.3's three reservation categories exactly as
  l.389-394 define them: exact pair; then normalised source and not the pair; then normalised
  target and neither of the others.
- **Counted from both sides.** "Heldout rows": how many of the 2,000 rows have their source,
  target or pair in the other file. "Rows": how many rows of the other file have a source, target or
  pair that is one of heldout_un's. "Distinct": the number of distinct strings. The feasibility
  report's field (`e06_build.py` `describe()` l.1369-1371) counts the set's **distinct** source and
  target lines found in heldout_un. The script computes the same quantity and compares the two per
  file. The production build's check (spec 6.7, `check_sets` l.1405-1413) counts set **rows**,
  and that count is reported too.
- **(a) Sets.** These are all 46 `~/e06runs/feasibility/FEASIBILITY_*.npy`. Each file is mapped to
  its tag, its parent (`ft_topk` or `ft_random`) and its row count through
  `feasibility_report.json`'s `anchors` and `results`. The five options that failed (A_narrow with
  OD 6 = remove, all five seeds) have no file. Each index file must have the sha256 the report gives
  for it and the one in the repo's `e06_feasibility/SHA256SUMS`. It must be strictly ascending inside
  its parent, and the report's `inputs` sha256 must equal the parent files that are read.
  Supplementary rows: `vocab/R0.npy` and `A_narrow_42_58.npy` (both index `ft_random`), the six E0.5
  files, and the two parents as wholes and their UN rows. Every E0.5 and E0.6 set is a subset of one
  of these parents.
- **(b) Pool and pretraining corpus.** These are `~/e06data/v2_scored.tsv` and
  `~/e06data/v1_fixed/train.clean.{en,fr}`, read line by line in step. Their E0.3 fingerprints are
  checked against `controls_manifest.json` `inputs`. For the pool rows found, the step also reports
  their e01 source labels, how many are inside the reused-score mask, and how many pool copies each
  heldout pair has.

## How heldout_un was built (E0.3, before any fine-tuning set was drawn)

This comes from `phase0/e03_build_controls.py`, run with the frozen `phase0/e03_decisions.json`
(PROTOCOL.md "E0.3 decisions", D1-D4). The figures are E0.3's own, from
`~/mt_local/phase0_results/controls_manifest.json`. Those marked † are measured by this run. Those
marked ‡ were re-derived by the independent checker and not by this run.

1. **Pool.** The 38,275,284† rows of `v2_scored.tsv`. The eligible rows are the 16,646,992‡ in
   `pool_mask_reused.npy`, the rows that carry the paper's CometKiwi-22 score (D1 "reused").
2. **Candidates** (l.373-374). A row is a candidate if it meets all of the following:
   - it is labelled `un` by e01 provenance (11,598,490‡ rows);
   - it is inside the mask (all of them are: `pool_rows_excluded_mask` is 0‡);
   - it is not already reserved;
   - it shares neither the exact pair, nor the normalised source, nor the normalised target with
     any row of the v1.1 pretraining corpus (9,312,233† rows; D4
     `exclude_pretrain_from_heldout: true`, `--pretrain-src/--pretrain-tgt`, l.288-291).

   That leaves **6,767,225‡ rows**. Another 4,831,265‡ were excluded because they touch pretraining.
3. **Draw** (l.383). `rng.choice(pool, 2000, replace=False)` with
   `rng = np.random.RandomState(42)` (l.200). This is the generator's first use; `ft_random` is
   drawn from the same generator later (l.455). The draw gives 2,000 rows (equal to heldout_un row
   by row‡) and **1,999 distinct pairs**†. One pair, at heldout rows 1054 and 1903, occurs **24
   times in the pool**†, and two of those 24 rows were drawn. (The earlier text of this README said
   "twice in the pool"; that was wrong.)
4. **Reservation** (l.389-400). Every row of the whole scored file that shares a drawn row's exact
   pair (3,290† rows), normalised source (+1,078†) or normalised target (+284†) is reserved: 4,652†
   rows in all. `ft_topk`, `ft_bottom` and `ft_random` are drawn only from `eligible & ~reserved`
   (l.426).
5. **Build-time checks.** These ran on the arrays (l.462-476) and again on the written files
   (l.556-570):
   - no fine-tuning set shares a pair, normalised source or normalised target with a held-out row
     († for `ft_topk` and `ft_random` and every set drawn from them);
   - no held-out pair, normalised source or normalised target occurs in the pretraining corpus (†).

   The manifest records `heldout_post_write_check` and `pretrain_overlap_fraction` 0.0. Without the
   D4 exclusion, the draw at the same seed had 793‡ of its 2,000 pairs in pretraining (PROTOCOL.md
   D4, a different draw from this one).
6. **Afterwards.** `e04_split.py` copies `heldout_un.{en,fr}` byte for byte into the E0.4/E0.5 data
   directory (l.88-89). The runs were scored on a 1,999-row copy: `--max-src-tokens 254` drops one
   over-long row (phase0/README.md l.580). That row is 0-based row 1764†. The checker counts 649
   source pieces for it; the longest kept row has 144.

## Predicted from E0.3's records, and measured

| prediction (written before the run) | measured |
| --- | --- |
| (a) 0 for every count and every file; report field `{0, 0}` for all 46 | 0 everywhere; 46 of 46 agree with the report |
| (b) pool: all 2,000 rows have their exact pair, source and target in it | 2,000 / 2,000 / 2,000 |
| (b) pool: pool rows touching heldout_un = E0.3's 3,290 / 1,078 / 284 = 4,652 | 3,290 / 1,078 / 284 = 4,652 |
| (b) pool: 4,645 of them inside the mask (item 5) | 4,645 |
| (b) pool: exact-source and exact-target row counts (never recorded before) | 4,139 and 3,605 |
| (b) pretraining: 0 exact or normalised | 0 |
| the evaluated copy drops 0-based row 1764 | row 1764, 1,999 rows kept |

The pool overlaps are expected by construction, since heldout_un was drawn from the pool. They are
not contamination of any training set. No E0.5 or E0.6 set, and neither parent, shares a line with
heldout_un, and neither does the base's pretraining corpus.

## Caveats

- Normalised counts depend on the interpreter's Unicode tables. The Studio's are 16.0.0, and E0.3
  did not record its own. Exact counts do not depend on them, and the normalised reservation counts
  here equal E0.3's.
- Only exact and E0.3-normalised matches are counted. There is no near-duplicate or fuzzy matching
  (Koehn et al. 2018 also filtered near-copies). A heldout sentence that differs from a training
  sentence by one word or a number is not counted.
- `pretrain` reads `train.clean` as files. The trainer additionally `strip()`s each line, but any
  match after stripping is also a normalised match, so it is covered.
- The exact-pair pool rows have QE scores down to 0.200354. heldout_un is drawn from UN rows in the
  reused mask without any QE condition, so this is a description, not a defect.
- 7 of the 4,652 reserved rows are labelled giga-fren, not un. They share a normalised source (6)
  or target (1) with a heldout row, and are reserved by E0.3's rule as intended.
- The check against E0.3's draw (that heldout_un is the `RandomState(42)` draw from the candidate
  pool) is the checker's, not this run's.
