# Build specification for `phase0/e06_build.py` (E0.6)

> **Update 2026-10-19.** The reference interpreter is now the arm64 build on the M5 Max
> (`~/e06env/venv-arm64`: Python 3.14.3, numpy 2.4.5, Unicode 16.0.0); the x86_64
> `/usr/local/bin/python3` named below was retired with the Intel Homebrew. On the M5 Max,
> `--reproduce-check` rebuilt `R0.npy` and `A_narrow_42_58.npy` row for row and the E0.5 anchors
> byte for byte, so the scratch files written under x86_64 are reproduced there.

**Status: SPECIFICATION ONLY, written 2026-10-18. No code is written here, and nothing here is
registered.** It says exactly what `e06_build.py` must do so that the E0.6 sets are what
`phase0/prereg_drafts.md` ("E0.6", lines 117-552, and its OPEN DECISIONS, lines 553-694) describes,
or so that the build stops. Where the draft, the scratch scripts and the existing code disagree or
leave something open, section 10 lists it, and every resolution there is marked **PROPOSAL**:
it is a suggestion for the author, not a decision.

Sources read: `phase0/prereg_drafts.md` (E0.6); `phase0/e05_build.py`; `phase0/e04_split.py`;
`phase0/probe_run.sh`; `phase0/e05_run.sh`; `tests/suites/e05.py`; `PROTOCOL.md` ("E0.4", "E0.5");
`WMT2027_PLAN.md` §3.2 item 7 and §3.3; the vocab memo's scratch scripts in `phase0/next/vocab/`
(`common.py`, `freq.py`, `qe_join.py`, `rowstats.py`, `arms.py`, `a1.py`-`a13.py`, `a8_lib.py`,
`dice.py`); `phase0/next/SHA256SUMS`; `phase0/next/groundwork.json` ("vocab"); also
`phase0/e03_collect.py`, `phase0/e03_run_matrix.py`, `phase0/controls_fingerprint.py` and
`phase0/e03_build_controls.py` for the interfaces the build must satisfy.

"Draft L*n*" means line *n* of `phase0/prereg_drafts.md` as it stands in the working tree on
2026-10-18. "Scratch probe" means a read-only check run on the Mac on 2026-10-18 for this spec, with
`/usr/local/bin/python3` (3.14.3, unidata 16.0.0, numpy 2.4.5), over the scratch pickles in
`~/mt_local/phase0_next/vocab/` and the pinned files. Nothing was written to disk. Scratch-probe
values are for planning the decisions only. Like every figure in the draft, they must be replaced by
`e06_build.py`'s own runs before the registered text quotes them.

---

## 1. Findings from the scratch probes that bear on the decisions (read first)

1. **R0.npy and A_narrow_42_58.npy reproduce, and everything they depend on can be derived.**
   `match_exact(hist(ft_un_div), len, RU0)` (seed 42) gives `R0.npy` row for row (tightest
   slack 1.2025). `nn_match(ft_un_div → capped RU0, seed 42)` with caps 42,000 / 58,000 counted
   over RU0 gives `A_narrow_42_58.npy` row for row (capped pool 402,010; RU0 450,709 rows, 91,142
   source and 105,953 French types). A single-threaded count of `train.clean.en`
   (sha256 `7230adc6…`, checked today) gives exactly `pre_en.pkl`'s counts *and its tie order*
   (704,146 types, 232,741,668 tokens), and from it both `maxrank_*.npy` files, element for element.
   Whole-band `rep` recomputed from `ft_topk.en` equals `rows.pkl`'s. The UN split rebuilt in
   ascending order reproduces E0.5's `ft_un_div`/`ft_un_rep` membership. In `rep`-sorted order
   (`e04_split.py`'s) it reproduces `ft_un_div` but not `ft_un_rep`, as the draft says.
   `ft_un_random` also reproduces (slack 1.6648). So nothing outside the pinned inputs plus
   `train.clean.en` is needed (section 8).
2. **`div_narrow` cannot be built by OPEN DECISION 2(a) or 2(b).** At `ft_un_rep`'s histogram, in
   every length bucket from 36 to 88 tokens (41 buckets, **30,197 rows**), E0.5's stratified
   matching already used *every* diverse UN candidate: the diverse side was the binding one. So
   any rule at that histogram must take all 30,197 rows, whatever it does elsewhere. A pool-wide cap
   that removes even one of them makes a bucket short, and the build stops. Keeping all of them
   needs Ke ≥ 43,502 of the pool's 43,555 source types and Kf ≥ 54,948 of its 54,950 French types,
   which is no cap at all. Uncapped nearest-neighbour matching gives 30,939 source and 40,195
   French types (+23.1% / +23.0%), outside ±2%. Probed: all nine points with Ke ∈ {24k, 28k, 32k} and
   Kf ∈ {32k, 40k, 48k} are short at L = 36. Caps that exempt only the 41 forced buckets are still
   short at L = 33-35 (bucket slack 1.143 / 1.069 / 1.016). At (40k, 52k) they are feasible but give
   +21.1% / +20.7%. The same holds for `div_narrow_dd`: 25,238 forced rows (L ≥ 37) at
   `rep_dedup`'s histogram, with minimum caps 43,502 / 54,932.
3. **OPEN DECISION 2(c) meets ±2% on source types alone, but not on source and French types
   together.** With the family defined in §6.5(c) (q from 0 to 1), source types run from 21,158
   to 33,858. At q = 0.80 they are 25,230 (+0.4%), but French types are then 35,883 (+9.8%). French
   types reach the target at q = 0.30 (+0.0%), where source types are −13.0%. So the draft's
   recommended pair, OD 1(b) with OD 2(a) falling back to (c), **stops the build** as written.
   Under OD 1(a) (source types only), (c) is feasible.
4. **A two-parameter variant reaches ±2% on both sides.** This is not one of the draft's options;
   see the PROPOSAL "c2" in §6.5. Each row is ordered by a blend λ of the log ranks of its rarest
   source word and its rarest French word, with q chosen as in (c). On a coarse grid, λ = 0.5,
   q = 0.70 gives 24,686 source types (−1.8%) and 32,724 French types (+0.1%).
5. **The feasibility file's range 20,284-34,070 is not reproduced.** The natural definitions give
   21,158 at the low end (the h_L rows with the smallest `maxrank` in each bucket) and 33,858 at the
   high end (identical to `match_random` at `ft_un_rep`'s histogram over the diverse candidates,
   seed 42). No script for the feasibility numbers exists, so they are not used.
6. **For R0 and A_narrow, "top-band row" by exact pair and by QE ≥ 0.895936 pick out the same
   rows.** `min(QE of ft_topk)` = 0.895936. R0 has 9.09% such rows and A_narrow 11.70%, with zero rows
   where the two tests disagree. RU0 holds 35,229 rows of `ft_topk`. OD 6's "remove" can therefore
   be defined from the pinned inputs alone, without `v2_scored.tsv`.
7. **No pool overlaps heldout_un.** No source or target line is shared with RU0, R0, A_narrow or
   either UN candidate list, so making overlap a hard stop costs nothing.
8. **`rep_dedup` / `div_dd` reproduce the scratch rebuild:** 207,561 repetitive candidates after
   dedup; the diverse side loses 252 rows (lengths 3:5, 4:16, 5:66, 6:65, 7:100); 198,333 per side;
   25,256 / 33,098 source types; mean `rep` 0.3784; 5,242,970 source words; **French tokens
   6,131,873 / 6,219,966**. So B = 6,000,000 is feasible for that pair, with a 2.2% margin.
9. **Forced overlap with `ft_un_div`.** Whatever narrowing rule is used, `div_narrow` shares at least
   30,197 rows with `ft_un_div` (bears on OD 18's V; uncapped nearest-neighbour matching shares
   127,222).
10. **Interface: set names must start with `ft_`.** `e03_collect.py` derives the condition from the
   config name with `(ft_[a-z0-9_]+)_lr` and exits on anything else. The draft's names
   (`R0`, `A_narrow`, `div_narrow`, …) would build and train, but evaluation would refuse them
   (§2.4).

---

## 2. Command line, modes and outputs

### 2.1 Modes

- **Production** (the default):
  `e06_build.py --dir D --src-dir S --e05-dir E --pretrain-en P --decisions J`
  - `S` holds `ft_topk.{en,fr,source}`, `ft_random.{en,fr,source}` and `heldout_un.{en,fr}`
    (`~/mt_local/phase0_final/e05_src/`).
  - `E` is E0.5's data directory (`…/e05_results/data/phase0_e05/`), read only, for the anchor
    pins and the byte check.
  - `P` is the v1.1 `train.clean.en`.
  - `J` is the decisions file (§7.1).

  The build constructs every set that the decisions call for, writes the training files, and writes
  `manifest.json` last with `"e06_complete": true`. If any set's defining decision is unset, it
  writes nothing and exits 2. If any check fails, it writes nothing and exits 1. In either case it
  names every blocked set and the reason.
- **`--feasibility --out F`**: builds every option of every set-defining decision, for
  measurement only (§7.3). It writes **no `.en`/`.fr` file and no `manifest.json`**, so neither
  `probe_run.sh` nor `e03_run_matrix.py` can train on its output. It writes index files whose names
  begin `FEASIBILITY_`, a `NOT_FOR_TRAINING` file and `feasibility_report.json`
  (`"for_training": false`). It refuses an `F` that equals `D` or that holds a `manifest.json`.
- **`--reproduce-scratch V`** (V = `~/mt_local/phase0_next/vocab`): builds R0 and A_narrow under
  OD 3 = keep, OD 6 = keep and OD 12 = one draw, then compares them with `V/R0.npy` and
  `V/A_narrow_42_58.npy` (§8). It writes nothing except a report on stdout. It refuses to run
  unless the interpreter is the one that wrote those files (§3.2).
- **`--verify --dir D`**: exits 0 only if `D/manifest.json` has `"e06_complete": true` and every
  listed file still has its recorded sha256. This mirrors `e05_build.py` `verify()`, lines
  146-167, and `e06_run.sh` calls it the way `e05_run.sh` lines 32-48 call `e05_build.py --verify`.
- **`--allow-unpinned-src`**: for the tests only, as in `e05_build.py` lines 180-182 and 204-209.
  The manifest records it, and outside the tests it is a protocol deviation.
- **Exit codes (PROPOSAL):** 0 built; 1 STOP (a check failed, nothing written); 2 REFUSED (a
  decision is unset, nothing written).

### 2.2 Publication (fail-closed)

The build follows `e05_build.py` lines 211-329 exactly. Every output goes first to
`D/.<name>.tmp`. Only after every set is built and every check has passed are all temporary files
renamed with `os.replace`; `manifest.json` is then written, also through a temporary name and a
rename. On any `Stop`, every temporary file is removed and the previous contents of `D` are left
untouched. The caches `*.cached_*.npz` in `D` are deleted by the wrapper before a build (as in
`e05_run.sh` line 45), not by the build itself.

### 2.3 Files

- `D/<set>.en`, `D/<set>.fr` for every set (anchors included). Rows are written in ascending
  parent-row order, `row + "\n"`, UTF-8, `newline="\n"` (`e05_build.py` lines 213-219).
- `D/index/<set>.npy`: int64 and ascending, the row indices into the parent (`ft_topk` or
  `ft_random`). The manifest records which parent each set comes from. These files are listed in
  `"written"`, so they fall inside the controls fingerprint.
- `D/heldout_un.en`, `D/heldout_un.fr`: byte copies of `S/heldout_un.*` once their pins have been
  checked.
- `D/manifest.json`, with `"written"` (every file above, then `manifest.json` last), `"sha256"`,
  `"heldout_sets": {"heldout_un": {"n": 2000}}` (required: `e03_collect.py` lines 103-126 default to
  `heldout_europarl` without it), `"e06"` (decisions, the decisions file's sha256, input pins,
  interpreter, every set's description, every check's result, the grid tables, the RNG register of
  §5) and finally `"e06_complete": true`.

### 2.4 Set names on disk

**PROPOSAL.** `e03_collect.py` line 93 accepts only names matching `ft_[a-z0-9_]+` (see
finding 10). The files are therefore published as:

| draft name | file and condition name |
| --- | --- |
| `ft_un_div`, `ft_un_rep`, `ft_un_random` | unchanged |
| `R0` | `ft_un_r0` |
| `A_narrow` | `ft_un_a_narrow` |
| `div_narrow` | `ft_un_div_narrow` |
| `rep_dedup` | `ft_un_rep_dedup` |
| `div_dd` | `ft_un_div_dd` |
| `div_narrow_dd` | `ft_un_div_narrow_dd` |

The manifest records the map from draft name to file name, and `e06_report.py` uses it. Under OD 12
"per seed", the draws are named `ft_un_r0_d<seed>` and `ft_un_a_narrow_d<seed>`.

---

## 3. Pinned inputs and the interpreter

### 3.1 Inputs (a mismatch is a STOP before anything is computed)

| file | sha256 | used for | source of the pin |
| --- | --- | --- | --- |
| `ft_topk.en` | `787339d08e0c47d9960cb263757edafd844ec80a45875ac083845a02a20ff2b2` | all top-band sets | `e05_build.py` L58; draft L277-278 |
| `ft_topk.fr` | `22263da18eaf62cb7deea7438068ce0cb00ce3849c1a83f7e0c55008c3d78797` | | L59 |
| `ft_topk.source` | `a4ad45860c92eae6b512c4a7178fe78fc92f00d4b9551d00a6637c80079244ca` | | L60 |
| `ft_random.en` | `937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0` | `ft_un_random`, R0, A_narrow | L61 |
| `ft_random.fr` | `382e0de574feaaa323939d73a24736b762c76a8b55a0ffd258d64b15e5303e29` | | L62 |
| `ft_random.source` | `fd675a1948ab4e30dce05aae4c85d2432aec99549c05fb794903a49e9b0b6923` | | L63 |
| `heldout_un.en` | `f70e725eae9972899b6129f6acf123ed5421021b2f5ba81bb165a6928e858bbc` | copied; disjointness check | L67; draft L279 |
| `heldout_un.fr` | `7daf654fa37cca653cbc9724c77ee721384a406e3ca3e6469a2a847b66b729ab` | | L68 |
| `E/manifest.json` | `979dafd76344444c56cad9afc52e0ec7ba7ebb74c73115cbeb237583419b8eb2` | the source of the six anchor pins | **PROPOSAL** (hashed 2026-10-18; the draft pins its contents, not the file) |
| `E/ft_un_div.en` / `.fr` | `51d9ec1157c58591a7b25b2a1bde52e9dc93342a34c55e55547f0e3857ba94da` / `1957872adf3f96b8e7fe551eeeae42bbd38f03811511e013f9017b65b954a7e1` | byte check | E0.5 manifest; draft L280-281 |
| `E/ft_un_rep.en` / `.fr` | `940cde0f051d6b473c55a6771af64900a20c9c52d66e1a4772e18a5bf9cafe68` / `9373be1fc7afb86a1b5ebf76d7cb567bde0ef349443ca9c3c6a87c175185edbd` | byte check | same |
| `E/ft_un_random.en` / `.fr` | `80fa6aea32f342cc43adec8e66063f8f08c20b6e3b66a60b0bfd7baf3e35c59c` / `1b54a882d989153697417b1f4b2300b81500acac258e29c542776d39a32094c0` | byte check (§6.1, PROPOSAL) | same |
| `train.clean.en` (v1.1 pretraining, English) | `7230adc6c823d4df51407e445b1181ada450363dfe43507885c9f529d49b6b6e` | `maxrank`; pretraining measures | draft L282 says "to record"; the value is in `~/mt_local/rebuild/v1_fixed/rebuild_stats.json` and RUNBOOK L35/L181, rehashed 2026-10-18 (**PROPOSAL**: pin it) |

Inputs used only by the descriptive measures (§9). The set-construction code path never opens any
of these (OD 16; §7.2):

| file | sha256 | used for |
| --- | --- | --- |
| `train.clean.fr` | `b0a6ba91eeba2b4bdf28e8a73816a2bdd7ccbddd5f889fcc7d3f3f0a52dad7e2` (rehashed today) | exact-pair exposure, the literalness lexicon; French pretraining ranks under PROPOSAL c2 (which then *is* set-defining, §6.5) |
| `~/mt_local/phase0_final/v2_scored.tsv` (14.1 GB) | to record | QE mean |
| `news-commentary-v9.fr-en.en` | to record | news coverage (OD 1(d), OD 15) |
| `spm_enfr_v1_fixed.model` (from the HF release) | `d1c10ea80fb95984f3f240b3579c87a16313b77aa9bc103ca0d6a0e06672a561` (full value from the repo) | piece-level checks (OD 16); set-defining only under OD 1(c) |
| newstest2014 (`wmt/wmt14@b199e406`) | to record | descriptive only; never a construction input (OD 16) |

### 3.2 Interpreter

- The build records `sys.version`, `unicodedata.unidata_version`, `numpy.__version__` and
  `platform.machine()` in the manifest, as `e05_build.py` lines 317-318 do.
- **Production (PROPOSAL, from OD 10):** it refuses to run unless `unicodedata.unidata_version` is
  `"15.0.0"`, the version of E0.5's 3.12.15 build recorded in E0.5's manifest. The byte check of
  §6.1 then gates the run as well.
- **`--reproduce-scratch`:** it refuses unless the interpreter is Python 3.14.3 with unidata
  `"16.0.0"`. The scratch files were written by `/usr/local/bin/python3` (numpy 2.4.5, scipy 1.18.0),
  not by `~/mt_local/venv` (numpy 2.5.3, no scipy; `common.py` cannot even be imported there). A
  numpy version other than 2.4.5 is recorded, and is not a refusal.
- **Dependencies:** the standard library and numpy only. scipy is not needed: in the scratch code it
  served only the AUC and Spearman analyses. `sentencepiece` is optional and needed only for OD 1(c)
  and the OD 16 measures.

---

## 4. Shared definitions (normative)

Each definition names the code it must reproduce exactly. "Ascending" means ascending row index in
the parent file.

- **D1, reading.** `read_lines`: `open(p, encoding="utf-8", newline="\n")`, then
  `ln.rstrip("\n")` for each line, never `splitlines()` (`e05_build.py` lines 72-75). `.en`, `.fr`
  and `.source` must have equal line counts, or STOP (lines 231-233).
- **D2, tokens.** `WORD = re.compile(r"\w+", re.UNICODE)`; `toks(s) = [w.lower() for w in
  WORD.findall(s)]` (lines 49, 78-79), on both sides. The source length is `len(toks(en))`.
  Draft L172-173.
- **D3, `rep_within(tok_lists)`.** `e05_build.py` lines 90-101, unchanged. Keys are
  `blake2b(" ".join(t[j:j+8]).encode("utf-8"), digest_size=8)`. The count is over the whole list,
  and a row's value is (number of its 8-gram occurrences whose key count is > 1) / (number of its
  8-grams), or 0.0 if the row has fewer than 8 tokens. Here `rep = rep_within(ft_topk source
  tokens)` and `rep_rand = rep_within(ft_random source tokens)` (draft L174-177; `rowstats.py`
  lines 14-18, 36).
- **D4, the median check.** `med = rep[sorted(range(n), key=rep.__getitem__)[n // 2]]` (line 240).
  **STOP unless `med == 0.0`**: the draft's "rep = 0" definitions (L176, L178) are e05's
  `rep <= med` only when the median is 0.0.
- **D5, candidates.** `div_cand` = the ascending i with `top_lab[i] == "un"` and `rep[i] == 0.0`;
  `rep_cand` = the ascending i with `top_lab[i] == "un"` and `rep[i] > 0.0` (`e05_build.py` lines
  253-254; draft L178-180). The build checks the counts 486,475 / 227,406 and stops if they differ.
  `RU0` = the ascending i with `rnd_lab[i] == "un"` and `rep_rand[i] == 0.0` (`a1.py` line 6;
  draft L207), and the build checks that it has 450,709 rows.
- **D6, pretraining ranks and `maxrank`** (draft L181-184; `freq.py` lines 9-14 and 24-28; `a5.py`
  lines 3-9). Read `P` with D1 and add `toks(line)` for each line, in file order, to one
  `collections.Counter`, single-threaded. Then `order = [w for w, _ in C.most_common()]` and
  `rank[w] = position in order`. Python's stable sort with `reverse=True` keeps first-occurrence
  order among equal counts. That is the draft's tie rule, and a single-threaded count gives the same
  order as the 14-process `freq.py` (verified, finding 1).
  `maxrank(row) = max(rank.get(w, 10**7) for w in toks(en))`, and 0 for a row with no tokens. STOP
  if `len(order) >= 10**7`, since unseen words would then collide with real ranks (it is 704,146).
  Ranks come from the English side only.
- **D7, frequency order over a pool** (for caps; `a12.py` lines 6-12). Over the pool's rows in
  ascending order: `pe.update(src_toks[i]); pf.update(fr_toks[i])`, then
  `oe = [w for w, _ in pe.most_common()]` and likewise `of`. The ties follow the same
  first-occurrence rule, inside the pool. `caps(Ke, Kf)` = the ascending pool rows `i` with
  `all(w in set(oe[:Ke]) for w in src_toks[i]) and all(w in set(of[:Kf]) for w in fr_toks[i])`. An
  empty row passes. An equivalent, faster form is allowed: precompute each row's largest pool rank
  on each side, and test `< Ke` and `< Kf`. The equivalence must be shown by reproducing
  `A_narrow_42_58.npy`.
- **D8, `match_random(hist, lens, eligible, seed=42)`.** `e05_build.py` lines 108-124, unchanged.
  Buckets are filled from `eligible` in ascending order. If any bucket is short, every short bucket
  is listed and the build STOPs. Then `rng = random.Random(seed)` and, for `L in sorted(hist)`,
  `kept += rng.sample(buckets[L], hist[L])`. It returns the sorted list and the tightest slack.
  `common.match_exact` with `frac=1.0` makes the same draws: `int(round(1.0*n)) == n`, the same
  bucket order and the same sampling order. Draft L185-187.
- **D9, `strat_match(A, B, lens, seed=42)`.** `e05_build.py` lines 255-268, unchanged. `A` and `B`
  are the ascending lists, and `rng = random.Random(seed)`. For
  `L in sorted(set(lens of A) | set(lens of B))`, take `k = min(|A_L|, |B_L|)`; if `k > 0`, draw
  `kA += rng.sample(A_L, k)` **and then** `kB += rng.sample(B_L, k)`. The A side (diverse) is always
  drawn first. An empty bucket draws nothing. Both lists are sorted at the end. Draft L188-192.
- **D10, `nn_match(t_idx, t_parent, p_idx, p_parent, seed)`.** `a8_lib.py` lines 3-21, unchanged.
  1. `rng = random.Random(seed)`.
  2. Targets: for i in `t_idx` (ascending),
     `tb[len_t(i)].append(float(np.log1p(np.int64(maxrank_t[i]))))`. A target is only a value.
  3. Pool: for i in `p_idx` (ascending, **all** pool rows, including those at lengths absent from
     the target histogram), `pb[len_p(i)].append((float(np.log1p(np.int64(maxrank_p[i]))),
     rng.random(), i))`. Exactly one draw is made per pool row, all before any matching.
  4. First check every target length: if `len(pb[L]) < len(tb[L])`, STOP and list every such L. This
     is a fail-closed message improvement on the `assert`, and leaves the output unchanged.
  5. For each target length L, in any order (the buckets are independent and the RNG has already
     been used): `pool = sorted(pb[L])`, so ties on the key are broken by the tie-break draw and then
     by the row index; `keys = [p[0] for p in pool]`; `used = [False] * len(pool)`;
     `med = np.median(keys)` (float64; the mean of the two middle values when the count is even).
  6. Process the target values in `sorted(vals, key=lambda v: -abs(v - med))`, a **stable** sort.
     Equal distances keep the order of `tb[L]`, which is ascending target-row order. For each `v`:
     `j = bisect.bisect_left(keys, v)`; `lo = j - 1`, moved down past used entries; `hi = j`, moved up
     past used entries. The candidates are `[lo, hi]` where they are in range, and
     `k = min(cands, key=lambda x: abs(keys[x] - v))`. On equal distance `min` returns the first, so
     the lower neighbour wins. Set `used[k] = True` and output `pool[k][2]`.
  7. Return the sorted output as int64.

  Draft L218-224. Every step must use the numpy/float operations named here, not `math.log1p`
  (§10, item A4).
- **D11, source word types.** `len({w for i in idx for w in src_toks[i]})` (`e05_build.py` lines
  127-132; draft L193).
- **D12, French types at budget B.** `types_at` (`common.py` lines 31-39): `idx` is the set's
  parent indices **in ascending order** (that is, file order), shuffled with
  `random.Random(0).shuffle(idx)`. The count stops just before the (B+1)-th token, so it counts
  exactly the first B tokens. The scratch function returns `None` both when the set has fewer than B
  tokens and when it has exactly B. The build treats a set with ≤ B French tokens as "budget not
  reached", which STOPs a production set (draft L194-196). Budgets are B = 6,500,000 for the
  214,489-row histogram and B_dd for the 198,333-row histogram (§7.1; draft L198-201).
- **D13, duplicate sources.** `1 - len(set(raw source lines of the set)) / rows` (`e05_build.py`
  line 132; draft L202-203).
- **D14, relative deviation and tolerance.** For a measure x with target x*,
  `dev = |x - x*| / x*`. "Within ±2%" means `50 * |x - x*| <= x*`, computed in integers when x is a
  count. This is inclusive (§10, item A13).
- **D15, top-band membership.** A row "is in the top band" if its exact `(en, fr)` raw pair occurs in
  `ft_topk`, which is derivable from the pinned inputs. For R0 and A_narrow this equals QE ≥ 0.895936
  (finding 6). The QE-threshold share is recorded beside it as a descriptive measure (§9).

---

## 5. Build order and RNG register

Order of the build:

1. Check the pins (§3.1) and the interpreter (§3.2).
2. Read and tokenise `ft_topk` and `ft_random`, both sides.
3. Compute `rep` and `rep_rand`, and run the D4 check.
4. Count the pretraining corpus and compute `maxrank` for both parents.
5. Rebuild the UN split and run the anchor check (§6.1).
6. Build R0.
7. Build A_narrow.
8. Build `rep_dedup` / `div_dd`.
9. Build `div_narrow`.
10. Build `div_narrow_dd` (OD 5(b)).
11. Check the cross-set invariants (§6.7).
12. Compute the descriptive measures (§9).
13. Publish (§2.2).

Every random generator is `random.Random(seed)` from the standard library, created fresh at the
point named and used by nothing else.

| id | step | seed | draws, in this order |
| --- | --- | --- | --- |
| G1 | UN split (anchor check) | 42 | D9 over `div_cand`, `rep_cand`: for each length ascending, diverse sample, then repetitive sample |
| G2 | `ft_un_random` (anchor check) | 42 | D8 over the ascending UN rows of `ft_random`, at the histogram of `ft_un_div` |
| G3 | R0 | 42 (OD 12 per seed: s ∈ {42, 1, 2, 3, 4}) | D8 over RU0, at the histogram of `ft_un_div` |
| G4 | A_narrow tie keys | 42 (OD 12 per seed: s) | D10 step 3: one `.random()` per **capped-pool** row (402,010), in ascending order |
| G5 | `rep_dedup` / `div_dd` | 42 | D9 over the deduplicated lists, diverse first |
| G6 | `div_narrow`, OD 2(a), each grid point | 42, fresh at each point | D10 step 3 over that point's capped candidates |
| G7 | `div_narrow`, OD 2(b), each grid point | 42, fresh at each point | D8 over that point's capped candidates, at `ft_un_rep`'s histogram |
| G8a | `div_narrow`, OD 2(c)/(c2), tie keys | 42, once | one `.random()` per diverse candidate (all 486,475, ascending) |
| G8b | `div_narrow`, OD 2(c)/(c2), each grid point | 42, fresh at each point | for each length ascending, `sample(E_L in ascending order, h_L)` (§6.5) |
| G9 | French types (D12) | 0, fresh for each measurement | one `shuffle` of the set's ascending index list |
| G10 | `div_narrow_dd` | as G6-G8 | as G6-G8, at `rep_dedup`'s histogram |

---

## 6. The sets

### 6.1 The anchors `ft_un_div`, `ft_un_rep`, `ft_un_random` (draft L156-158, L284-288)

- Inputs: `ft_topk.*` and `ft_random.*`, with `rep`, the labels and the lengths from §4.
- Steps (`e05_build.py` lines 253-273):
  1. `kd, kr = strat_match(div_cand, rep_cand)` (G1).
  2. `un_hist = Counter(len_t(i) for i in kd)`.
  3. `ku = match_random(un_hist, len_r, ascending UN rows of ft_random)` (G2).
  4. Write the files with D1/D2 conventions (`e05_build.py` lines 213-219).
- **Check.** All six files must equal E0.5's, sha256 for sha256 (§3.1). On a mismatch: STOP,
  naming the file. The draft requires only `ft_un_div` and `ft_un_rep` (L284-286).
  **PROPOSAL:** also `ft_un_random`. It costs nothing, and it is the only known output of D8 on real
  data, which R0 reuses.
- Size: 214,489 rows each. The 214,489-row sets share `un_hist`, the histogram of `ft_un_div`
  and `ft_un_rep`.

### 6.2 R0 (draft L207-211)

- Inputs: RU0 (D5, 450,709 rows) and `un_hist` from §6.1. Decisions: OD 6 and OD 12.
- Steps: `R0 = match_random(un_hist, len_r, RU0, seed=42)` (G3; `a1.py` lines 9-12).
  - Under OD 6 = remove: the pool is RU0 minus the rows that are in the top band (D15); the scratch
    count of those rows is 35,229.
  - Under OD 12 = per seed: one draw for each s, with `random.Random(s)`.
- Ties: none. Sampling is by position inside the ascending bucket list.
- Checks (STOP if one fails):
  - the tightest slack is ≥ 1 (D8; the scratch value is 1.2025);
  - no source row of R0 has `rep_rand > 0`, by construction;
  - there are no duplicate sources (D13 = 0);
  - the histogram equals `un_hist`.
- Expected under the defaults: 214,489 rows, 62,000 source types and 72,574 French types at 6.5M
  (scratch). Under the defaults it must equal `R0.npy` (§8).

### 6.3 A_narrow (draft L212-226)

- Inputs: RU0, the `ft_random` tokens on both sides, `maxrank` for both parents, and `kd`.
  Decisions: OD 3, OD 6, OD 7, OD 12.
- Steps (`a12.py` lines 6-12 and 22-27; `a8_lib.py`):
  1. Run D7 over RU0 (or over OD 6's reduced pool), giving `oe` and `of`.
  2. `pool = caps(42000, 58000)`, which must give 402,010 rows under the defaults.
  3. `A = nn_match(t_idx=kd (ft_topk), p_idx=pool (ft_random), seed=42)` (G4).
  4. Under OD 12 = per seed: only the seed of step 3 changes, and the caps stay as they are (as in
     `a12.py` line 24).
- Ties:
  - frequency ties in D7 go to the first occurrence inside the pool;
  - in D10, key ties in the pool go to the tie key, then to the index;
  - in the processing order, ties go to ascending target order;
  - a tie in neighbour distance goes to the lower neighbour.
- Checks: no short bucket (D10 step 4); no duplicate source; `rep_rand == 0` for every row; the
  histogram equals `un_hist`.
- Expected under the defaults: 33,562 source types, 43,703 French types at 6.5M, and 136,916 rows
  shared with R0 (scratch). It must equal `A_narrow_42_58.npy` (§8).

### 6.4 `rep_dedup` and `div_dd` (draft L244-264)

- Inputs: `rep_cand`, `div_cand` and the raw `ft_topk.en` lines. Decision: OD 4.
- Steps:
  1. Walk `rep_cand` in ascending order and keep a row if its duplicate key has not been seen
     earlier *in this list*. Under OD 4(a) the key is the raw source line; under (b), §7.2. This
     must give 207,561 rows under (a).
  2. Do the same on `div_cand`. This gives 486,223 rows under (a), with 252 removed, all of 3-7
     tokens. Record the count and the lengths removed.
  3. `div_dd, rep_dedup = strat_match(div_cand_dedup, rep_cand_dedup)` (G5). The diverse side is
     A, as in E0.5.
  4. `rep` is not recomputed (OD 4, recommended).
- Checks:
  - under (a), every row removed in step 2 has fewer than 8 tokens. Otherwise STOP: the draft's
    argument at L248-252 would then be false;
  - the two sides have equal histograms;
  - D13 = 0 on `rep_dedup`;
  - `rep > 0` on every `rep_dedup` row and `rep == 0` on every `div_dd` row;
  - `rows(rep_dedup) == rows(div_dd)`, recorded (scratch: 198,333);
  - French tokens > B_dd on both sides (scratch: 6,131,873 and 6,219,966).

### 6.5 `div_narrow` (draft L227-243; OD 1 and OD 2)

- Inputs: `div_cand` (486,475, all of them, not deduplicated), `kr` (`ft_un_rep`'s rows) and the
  target `h = Counter(len_t(i) for i in kr)` (the same as `un_hist`). The coverage target is
  computed from `kr`, not hard-coded:
  - source types (D11) = 25,134;
  - French types at 6.5M (D12) = 32,684;
  - OD 1(c) and (d) values per §7.2.

  Decisions: OD 1, OD 2 and the grid (§7.1).
- **Forced rows** (finding 2). `F = {L : |div_cand_L| == h_L}`. The build records `F`, the forced
  row count and the smallest caps that keep every forced row. Any rule must contain every candidate
  in a forced bucket.
- **OD 2(a): caps plus nearest-neighbour matching.** For each (Ke, Kf) on the registered grid:
  1. run D7 over `div_cand`;
  2. take `pool = caps(Ke, Kf)`;
  3. if any target bucket is short, the point is *infeasible*: it is recorded and skipped (not a
     STOP);
  4. otherwise compute `nn_match(kr → pool, seed=42)` (G6) and measure the coverage;
  5. choose the feasible point with the smallest `max(dev)` over the measures of OD 1. On ties, take
     the larger Ke, then the larger Kf (**PROPOSAL**: the least restrictive caps);
  6. if the best `max(dev)` exceeds 0.02, or no point is feasible: STOP.

  Scratch finding 2 says this stops.
- **OD 2(b): caps plus a random draw.** As (a), but step 4 is `match_random(h, len_t, pool)` (G7).
  This stops for the same reason.
- **OD 2(c): a one-parameter family.** The draft (L578-580) gives only "within each length bucket,
  draw at random from the fraction q of candidates with the most frequent rarest word". The
  **PROPOSAL** below makes that exact:
  1. Draw the tie keys once: `u[i] = rng.random()` for i in `div_cand`, ascending (G8a).
  2. For each L in h, order `div_cand_L` by `(maxrank[i], u[i], i)` ascending: rarest word most
     frequent first, using the pretraining rank of D6.
  3. With grid index `k ∈ {0, …, K}` (**PROPOSAL** K = 200, so q = k/K in steps of 0.005), take
     `m_L = h_L + (k * (n_L - h_L)) // K`, in integer arithmetic, and let `E_L` be the first
     `m_L` rows of the order, re-sorted ascending.
  4. `rng = random.Random(42)` fresh for each k (G8b); for `L in sorted(h)`,
     `out += rng.sample(E_L, h_L)`.
  5. Measure the coverage and choose the k with the smallest `max(dev)`. On ties take the larger k
     (closer to a random draw). STOP if the best value is above 0.02.

  At k = K the output equals `match_random(h, len_t, div_cand)` exactly; this was checked, and it
  gives 33,858 source types. At k = 0, each bucket's h_L most frequent-rarest-word rows are taken
  (21,158 source types). Forced buckets have `m_L = n_L = h_L` at every k. Each point costs about
  2 s.
- **PROPOSAL "c2" (not a draft option; needed if OD 1 is (b) and finding 3 holds).** As (c), but
  step 2 orders by
  `((1 - λ) * log1p(maxrank_en[i]) + λ * log1p(maxrank_fr[i]), u[i], i)`.
  - `maxrank_fr` is D6 applied to the French side of `train.clean.fr` (pinned `b0a6ba91…`, which
    then becomes a construction input).
  - The grid is λ ∈ {0, 0.05, …, 1} × k ∈ {0, …, 200}, with λ = 0 equal to (c).
  - On ties take the larger k, then the smaller λ.

  On the scratch probe, at λ = 0.5 and q = 0.70: −1.8% source and +0.1% French.
- Checks: the histogram equals h; `rep == 0` for every row (by construction); D13 = 0; French
  tokens > 6,500,000. The full grid table (every point: feasible or not, source types, French types,
  dev) goes into the manifest.

### 6.6 `div_narrow_dd` (OD 5(b) only; draft L265-268)

- The rule is exactly §6.5's registered rule with `kr` replaced by `rep_dedup` and h by
  `rep_dedup`'s histogram. The French coverage is measured at B_dd. Candidates are all 486,475
  diverse candidates, not the deduplicated 486,223 (§10, A19).
- The target values are computed from the `rep_dedup` built in §6.4: scratch values 25,256 source
  types, with French at B_dd to be measured. The feasibility file's 25,296 is not used (draft
  L600-602).
- Forced rows: 25,238 (L ≥ 37).
- The checks are the same as §6.5, with B_dd.

### 6.7 Cross-set invariants (any failure is a STOP)

- Every set's rows are UN-labelled in its parent.
- Each 214,489-row set has histogram `un_hist`, and the dd pair (with `div_narrow_dd`) shares one
  histogram.
- Every new set has zero source lines and zero target lines in common with `heldout_un`, compared
  as raw lines. **PROPOSAL:** a hard stop; the scratch probe found zero in every pool (finding 7).
- Each set's `.en` and `.fr` have equal line counts and keep each row's own translation (as
  `tests/suites/e05.py` lines 133-134).

---

## 7. OPEN DECISIONS: what the build depends on, and how it fails closed

### 7.1 The decisions file (PROPOSAL)

`--decisions J` is a JSON object. A key that is absent, or whose value is `"OPEN"`, counts as unset.
Its sha256 is recorded in the manifest, and the registered text names that sha256.

| key | values | default |
| --- | --- | --- |
| `od1_coverage` | `a`, `b`, `c`, `d` | none |
| `od2_rule` | `a`, `b`, `c`, `c2` | none |
| `od2_grid` | (a)/(b): `Ke` and `Kf` lists; (c): `K`; (c2): `K` and the λ list | none |
| `od2_tie` | the tie rule | none (§6.5 proposals) |
| `tolerance` | | 0.02 |
| `od3_caps` | `[42000, 58000]` or `"retune"` | none |
| `od4_duplicate` | `a`, `b` | none |
| `od4_recompute_rep` | `false` / `true` | none |
| `od5` | `a`, `b`, `c` | none |
| `od6_topband` | `keep`, `remove` | none |
| `od7_exposure` | `accept`, `match` | none |
| `od10_unidata` | | `"15.0.0"` |
| `od12_draws` | `one`, `per_seed` | none |
| `od15_narrowness` | `type_count`, `news_coverage` | none |
| `od18_register_v` | `true` / `false` | none |
| `fr_budget_214489` | | 6,500,000 |
| `fr_budget_198333` (B_dd) | | none: the draft proposes 6,000,000, "fixed before the dry run", L199-200 |
| `seeds` | | `[42, 1, 2, 3, 4]` |

### 7.2 Each decision, with the options as the draft states them

"Production" is what a production build does. "Feasibility" is what `--feasibility` builds or
measures; it is never for training.

| OD | draft lines | parameter and options (draft) | depends on it | production | feasibility |
| --- | --- | --- | --- | --- | --- |
| 1 | 559-573 | the coverage `div_narrow` is matched on: (a) source word types; (b) source and French word types (French at 6.5M); (c) Saunders & DeNeefe's share of the base model's subword vocabulary seen at least once, source and target separately; (d) coverage of a non-test text, e.g. news-commentary v9 tokens seen ≥ 5 times. Recommended: (b), ±2% on both sides | `div_narrow`, `div_narrow_dd` | unset → refuses both sets. (c) refuses unless the SPM model is present with the pin of §3.1 and the piece-counting rule (§10, A28) is registered. (d) refuses unless a tolerance for (d) is registered (§10, A29) | builds every option in turn, with each OD 2 option; (c) only if `sentencepiece` and the model are present, otherwise recorded as "n.m." |
| 2 | 574-583 | the narrowing rule: (a) caps plus nearest-neighbour matching on `log(1+maxrank)` to `ft_un_rep`'s rows; (b) caps plus a random draw at the histogram; (c) the one-parameter q-family. Recommended: (a), then (c) if (a) cannot reach ±2%; "the grid and the tie rule are fixed in the registered text" | `div_narrow`, `div_narrow_dd` | unset, or the grid or tie rule missing → refuses. Takes **exactly one** rule: "a then c" is resolved at registration, because the box must rebuild the registered sets (§10, A16) | runs (a), (b), (c) and PROPOSAL c2, each over its full grid, and records every point |
| 3 | 584-586 | A_narrow's caps: keep 42,000 / 58,000, or re-tune them to a stricter tolerance. Recommended: keep | A_narrow | `[42000, 58000]` → §6.3. `"retune"` → refuses: no grid, tolerance or objective is defined | reports the source and French types over a cap grid around 42k/58k (PROPOSAL: Ke 38k-46k × Kf 54k-62k in steps of 1k), measurement only |
| 4 | 587-596 | what counts as a duplicate: (a) the exact raw source string; (b) a normalised source (lowercased `\w+` sequence, digits masked); also whether `rep` is recomputed after dedup. Recommended: (a), no recompute, (b) measured descriptively | `rep_dedup`, `div_dd`, `div_narrow_dd` | unset → refuses those three. (b) refuses unless the masking rule is registered (§10, A17). `recompute = true` → refuses (undefined, A18) | builds (a); builds (b) under the PROPOSAL masking of A17 and reports its counts (and whether step 2 then removes rows of 8 tokens or more) |
| 5 | 597-607 | (a) eight sets; (b) add `div_narrow_dd`; (c) replace `div_narrow` with `div_narrow_dd`. Recommended: (b) if its build passes, else (a) | whether `div_narrow` and `div_narrow_dd` are built | unset → refuses both. (c) with `od18_register_v = true` → refuses: V needs `div_narrow` (A20) | always builds `div_narrow_dd` under every OD 1 × OD 2 option |
| 6 | 608-610 | top-band rows in the random draws: keep them (E0.5's rule) or remove them from every random arm. Recommended: keep | R0, A_narrow (`ft_un_random` is an anchor and cannot change) | unset → refuses R0 and A_narrow. `remove` uses D15 on the pool (§6.2-6.3); A_narrow's caps are then counted over the reduced pool, so OD 3's "keep, reproduce A_narrow_42_58" no longer describes the production set (A21) | builds both |
| 7 | 611-613 | the pretraining-exposure difference: accept and report it, or add exposure to A_narrow's matching. Recommended: accept | A_narrow | `accept` → no effect. `match` → refuses: no rule is defined | not built (no rule); the exposure is reported (§9) |
| 8 | 614-618 | steps for the 198,333-row pair: 10,000, or about 9,250-9,300 | the run, not the build | no effect; the manifest records the dd pair's row count, from which `e06_run.sh` would derive the step count | — |
| 9 | 619-622 | multiplicity: none, or Holm over the primary contrasts | `e06_report.py` | no effect | — |
| 10 | 623-625 | where the run of record is built: on the training box, Python 3.12.15, gated by the byte check. A check, not a choice | every set | refuses unless unidata `od10_unidata` (§3.2); the byte check STOPs (§6.1) | records the interpreter, and runs on any |
| 11 | 626-641 | secondary metrics and test sets (chrF, MetricX-24, COMET-22 flagged, FLORES+ devtest, ForgetGenUse) | evaluation only | no effect. The build never reads a test set | — |
| 12 | 642-649 | one draw per set, or a redraw per seed, for R0 and A_narrow. Recommended: one draw | R0, A_narrow | unset → refuses both. `per_seed` → five index files and five set pairs each (§2.4); the seed-42 draw must equal the `one` draw; the runner change is not in the build (A22) | builds both |
| 13 | 650-653 | an NRC re-ranking arm: not in E0.6 | — | the build has no code path for it | — |
| 14 | 654-656 | R0 as Q2's reference: report both | the report | no effect | — |
| 15 | 657-662 | what narrowness means for N: type count (as built), or news coverage, which needs one more arm (5 runs) | possibly a new arm | `type_count` → no effect. `news_coverage` → refuses: the arm has no defined rule | reports A_narrow's and R0's NC v9 coverage (§9) |
| 16 | 663-673 | the piece-level check (may change the design before the tag) and the newstest2014 check (descriptive only) | measures | construction never reads newstest2014 or the SPM model, except under OD 1(c), where the SPM model defines `div_narrow`. These measures run after every set is fixed and are written to the report, never fed back | computes piece types and frequency profiles of every set, if the model is present |
| 17 | 674-676 | E0.6-LR (the UN triple at lr_scale 0.15): yes or no | its own section and tag | no effect (it reuses the anchors) | — |
| 18 | 677-694 | register V = BLEU(`ft_un_div`) − BLEU(`div_narrow`) | the report | no effect on construction. The manifest always records `|ft_un_div ∩ div_narrow|` (at least the 30,197 forced rows) | records the overlap |

### 7.3 What "not for training" means in `--feasibility`

The feasibility output cannot be mistaken for a production build:

- no `.en`/`.fr` text and no `manifest.json`, so `probe_run.sh` lines 31-48 and
  `e03_run_matrix.py` lines 146-154 refuse it;
- `--verify` also refuses it;
- every index file is named `FEASIBILITY_<set>__<option-tuple>.npy`;
- the report says `"for_training": false` and `"decisions": "all options"`.

Its grid tables and set descriptions are what the author decides from, and the registered text's
numbers then come from a **production dry run**: the production mode with the registered decisions,
on the Mac (§10, A8).

---

## 8. How the build reproduces `R0.npy` and `A_narrow_42_58.npy`

- **Target files:**
  - `~/mt_local/phase0_next/vocab/R0.npy`, sha256 `22eeb53dd3cb621831939cb41a5b9280045dbc5f6b1ce48686234479ac354dc9`;
  - `A_narrow_42_58.npy`, sha256 `e9fac76975bb83e2a4b12ed913c333212cc4a70c440346278c12ad7c0bd7cfa8`.

  The values agree with `phase0/next/SHA256SUMS` and were rehashed today. The build checks them
  before comparing. Both hold int64 arrays of 214,489 strictly ascending row indices into
  `ft_random`.
- **Inputs used:** `ft_topk.{en,fr,source}`, `ft_random.{en,fr,source}` and `train.clean.en`, all
  pinned. `.fr` is needed only for A_narrow's French cap. The scratch pickles (`rows.pkl`,
  `maxrank_*.npy`, `pre_en.pkl`, …) are **not read**. **PROPOSAL:** after a failed comparison they
  may be read, hash-checked against SHA256SUMS, to say where the paths first part ("rep differs",
  "maxrank differs", "first differing bucket"). This is a diagnosis, never a substitute.
- **Steps mirrored**, under OD 3 keep, OD 6 keep and OD 12 one draw:

  | build step | scratch counterpart |
  | --- | --- |
  | the ft_topk / ft_random lengths and `rep_rand` (D2, D3) | `rowstats.py` lines 34-36 and 66, the fields `len`, `rep`, `un` |
  | the `ft_un_div` membership | `rowstats.py` lines 69-76, which read it from E0.5's published files. The build uses its own `kd` from §6.1, which the byte check has already proven equal to those files |
  | RU0 and the histogram | `a1.py` lines 6 and 9 |
  | R0 = D8 | `a1.py` lines 10-12 → `common.match_exact` (lines 41-51) |
  | ranks and `maxrank` (D6) | `freq.py` (counts), `a5.py` lines 3-9 (rank and maxrank) |
  | caps (D7) | `a12.py` lines 6-12 |
  | `nn_match` (D10), seed 42 | `a8_lib.py` lines 3-21, called at `a12.py` line 25 with `seed=42`. The file was saved at line 27 (`seed == 42`); the seed-1 and seed-2 calls in the same loop each make their own `Random` and do not touch the seed-42 result |
- **Acceptance:** `np.array_equal(build_R0, np.load(R0.npy))` and the same for A_narrow. Byte
  equality of a re-saved `.npy` is recorded as information only, because the `.npy` header can
  depend on the numpy version.
- **What is not derivable from the pinned primary inputs** (`e05_src`):
  1. `maxrank` needs `train.clean.en`, which is not in `e05_src`. The draft lists it with
     "sha256 to record" (L282); the value is `7230adc6…`, checked today, and **PROPOSAL** pins it.
     Nothing else in R0 or A_narrow comes from outside.
  2. The scratch run took `ft_un_div`'s membership from E0.5's box-built files (3.12.15), not from a
     recomputation. Recomputed on the Mac it is identical (finding 1), so this is a check, not a
     dependency.
  3. Environment, not data: the Unicode tables (16.0.0 for the scratch files) and numpy's
     `log1p`/`median` (numpy 2.4.5, arm64). They are why the comparison is run on the same
     interpreter (draft L288-290), and why A4 and A8 below exist.
  4. `qe.npz` (from `v2_scored.tsv`), `lex.pkl`, `nc_en.pkl`, `heldout_un` and `pre_en.pkl`'s line
     hashes reach `arms.report()` only. They do not affect which rows R0 and A_narrow contain.
  5. Not reproducible at all, because no script exists: the feasibility file's 20,284 / 34,070
     (div_narrow range) and 25,296 / 0.379 (rep_dedup). The draft already declines to use the last
     two (L260-261). The first two should be dropped too (finding 5; A27).
- Verified today with the scratch functions on the scratch pickles: both files reproduce, and so do
  every intermediate the pickles hold (finding 1). A full run of `--reproduce-scratch` costs about
  4 min (§11).

---

## 9. Measurements the build records (the draft's "Measured on the sets", L312-363)

All of these are descriptive and none feeds back into a set. For every set, at the dry run:

- **Rows, source words, source types** (D11), **French tokens** (exact), **French types at B**
  (D12; B = 6.5M or B_dd).
- **Repetition:**
  - the parent statistic, that is mean `rep` for `ft_topk` sets and mean `rep_rand` for `ft_random`
    sets (the table's "source repeated-8-gram share": 0.433 for `ft_un_rep`, 0.087 for
    `ft_un_random`);
  - the share of rows with a parent value > 0;
  - **PROPOSAL**, also: the same two values recomputed *within the set* (A10).
- **Duplicates:** raw (D13), and normalised under the A17 rule (OD 4(b), descriptive).
- **QE mean and the top-band share:** QE from `v2_scored.tsv`, joined on the exact
  `en + "\t" + fr` bytes, taking the max over conflicting duplicate scores (`qe_join.py` lines
  4-26). The top-band share is the D15 share, with QE ≥ 0.895936 beside it.
- **Literalness:** `arms.lexhit` (lines 8-17), the mean over rows with at least one
  lexicon-covered source token of hits / covered, matched on 5-character prefixes. It uses the
  lexicon rebuilt as `dice.py` does:
  - `random.Random(7).sample(range(9312233), 1500000)` pairs of `train.clean.{en,fr}`;
  - content vocabularies = pretraining ranks 100-19,999, excluding `isdigit()` words;
  - Dice ≥ 0.15, top 3.

  The build checks that the line count is 9,312,233.
- **Source in pretraining:** the `freq.py`/`rowstats.py` normalisation (NFKC, `casefold`,
  `\s+` → one space, strip) with a blake2b-64 key, as membership among the normalised sources of
  `train.clean.en`.
- **Pair in pretraining:** E0.5's exact-pair measure, as `e03_build_controls.py` `h64(src, tgt)`
  over raw `train.clean.{en,fr}` pairs (lines 109-112, 270-290). **PROPOSAL** wording: the draft
  calls these two measures different statistics (L329-330), and this is their exact definition.
- **Overlaps:**
  - with `ft_topk` (exact pair) and with each anchor (exact pair and source line);
  - R0 ∩ A_narrow (scratch 136,916);
  - `ft_un_div` ∩ `div_narrow` (OD 18) and the forced-row count;
  - the dd pair against `ft_un_rep` / `ft_un_div`.
- **Build diagnostics:**
  - the tightest slacks;
  - pool sizes (RU0, capped pool, candidates after dedup);
  - the full OD 2 grid table;
  - the dedup counts and removed lengths;
  - **nn_match near-ties** (A4): the number of neighbour choices with
    `|d_lo - d_hi| < 1e-9` and the number of target-order ties between distinct values within
    1e-9.
- **News coverage** (OD 1(d) and OD 15): `arms.cover` with k = 1, 5, 20 on news-commentary v9 (pin
  to record).
- **heldout_un:**
  - disjointness (§6.7);
  - in-domain coverage, `arms.cover` on `heldout_un.en` (k = 5), which the draft cites (L511-513).
- **OD 16, only if the SPM model is present:**
  - piece types per side and the share of the model's vocabulary;
  - the piece-frequency profile (counts of pieces at frequency 1, 2-5, 6-20, 21-100, >100);
  - the exact piece-token totals per side, for the token-imbalance lines of every contrast pair
    (draft L525-536).
- **Descriptive only, in a separate `--measure-test` step that cannot write into `D`:**
  newstest2014 coverage, with its pin recorded.

---

## 10. Ambiguities and contradictions, with proposals

A1. **Which pool gets A_narrow's tie keys.** Draft L220-221 says "each pool row". In the script the
keys are drawn over the **capped** pool (402,010), not R0's pool (450,709), and including rows at
lengths with no targets. **PROPOSAL:** state it so (D10 step 3).

A2. **Target processing order on equal distance.** This is unstated (L222). The script's stable
sort keeps ascending target-row order, and that matters when two different values lie at the same
distance from an even-count median. **PROPOSAL:** state it (D10 step 6).

A3. **The "upper neighbour" includes keys equal to the target value** (`bisect_left`). Equal pool
keys are ordered by tie key and then by index. Unstated (L223). **PROPOSAL:** state it (D10 steps
5-6).

A4. **Cross-platform float dependence in `nn_match`.**
- Keys are `np.log1p` of integers, and the choices compare float distances. numpy may compute
  `log1p` differently on the box (Linux x86, possibly SIMD) and on the Mac (arm64). A one-ulp
  difference can flip a near-tie (most plausibly the |v − median| order for the two middle values),
  and so change A_narrow or `div_narrow` on the box with no error raised.
- **PROPOSAL:** record the near-tie counts (§9). The box build must also reproduce the registered
  dry run's index-file sha256 for every set, or STOP (A8). The draft checks only the anchors on the
  box (L284-288).

A5. **Anchor check scope.** The draft checks `ft_un_div` and `ft_un_rep` (L284-286), but
`ft_un_random` is also an anchor, and R0 reuses its code path. **PROPOSAL:** check all six files
(§6.1).

A6. **"The three anchor files"** (L280) are six files (`.en` and `.fr`). In addition, the manifest
that carries their pins is not itself pinned. **PROPOSAL:** pin the E0.5 manifest file
(`979dafd7…`).

A7. **`train.clean.en`'s sha256 is "to record"** (L282), but it is already recorded
(`rebuild_stats.json`, RUNBOOK L35/L181) and matches today: `7230adc6…`. **PROPOSAL:** pin it.
Also, the scratch read it from `~/mt_local/rebuild/v1_fixed/`, the box has it as
`data_enfr_v1/train.clean.en` (RUNBOOK L181, "already equals the rebuild"), and the build takes it
by path and pin.

A8. **Which build produces the numbers the registration quotes.**
- The draft wants the registered table filled "from a dry run of the fail-closed build" (L313-315),
  says "the run of record is built where it trains" (L312), and gates the box only on the anchors.
- But the Mac has no Python 3.12 (only 3.14.3, unidata 16.0.0, and 3.9.6), and E0.4's note shows
  that `\w+` differs across interpreters. The registered numbers and the trained sets could differ,
  and nothing would notice.
- **PROPOSAL:**
  - install Python 3.12.15 on the Mac (no rental) and run the production dry run there with the
    box's numpy version;
  - register every set's file sha256 from that run;
  - make the box build STOP unless it reproduces them byte for byte;
  - keep the separate `--reproduce-scratch` check on 3.14.3.

A9. **Which side is drawn first in stratified matching** for `rep_dedup`/`div_dd`. L253 says
"between the two lists". `e05_build.py` lines 264-266 draws the diverse side first, and that fixes
the RNG stream. **PROPOSAL:** diverse first (D9).

A10. **The "source repeated-8-gram share" column mixes parents.** It is whole-band `rep` for the
top-band sets and `rep_rand` (the `ft_random` parent) for `ft_un_random` / R0 / A_narrow, so the
values are not one statistic. **PROPOSAL:** label the column per parent, and also record a
within-set value (§9).

A11. **French budget edge case.** `types_at` returns None for a set with exactly B tokens; the
draft says "fewer than B" stops (L195-196). **PROPOSAL:** "≤ B French tokens stops" (D12). This is
immaterial unless a set has exactly B.

A12. **OD 1(a) with French caps.** A_narrow's rule has two caps, but under (a) only source types
are matched. Kf then has no objective, and many grid points tie. **PROPOSAL:** under OD 1(a), use
no French cap (Kf = ∞) and search Ke only. French types are reported.

A13. **The ±2% rule is unspecified**: inclusive or exclusive, relative to what, and signed or not.
**PROPOSAL:** D14 (relative to the target, absolute value, inclusive, in integers).

A14. **The OD 2(a) grid range and its tie rule are unspecified.** Draft L234 gives only "steps of
1,000". Unstated also: what happens at a grid point whose pool is short, since L224's "stops the
build" applies to A_narrow's single point. **PROPOSAL:**
- Ke = 1,000·j for j ≥ 1 up to the pool's source type count, rounded up, and likewise Kf;
- a short point is recorded as infeasible and skipped;
- ties go to the larger Ke, then the larger Kf.

By finding 2 the only feasible points are those with no effective cap.

A15. **OD 2(c) is underspecified** (L578-580). Unstated:
- what "the most frequent rarest word" ranks on (the pretraining `maxrank` or the pool's own
  frequency);
- how the fraction q maps to rows per bucket;
- ties at the cut;
- the random generator;
- the q grid;
- the tie rule between q values.

In addition, the "two ends of this family" the draft says were measured (20,284 and 34,070) are not
what a natural definition gives (21,158 and 33,858; finding 5). **PROPOSAL:** §6.5(c) (pretraining
`maxrank`, `m_L = h_L + ⌊k(n_L − h_L)/K⌋`, tie keys from G8a, draws from G8b, K = 200, ties to the
larger k).

A16. **"(a), then (c) if (a) cannot reach ±2%"** (L582) is a conditional rule. If the box applies
it, the box could build a different set from the one registered; if registration resolves it, it
is a decision. **PROPOSAL:** resolve it at registration. Production takes exactly one rule.

A17. **"Digits masked"** (OD 4(b), L589) is undefined. Under (b), two diverse rows of 8 or more
tokens whose sources differ only in their digits are duplicates while both have `rep = 0`. So
step 2 *can* remove rows of 8 or more tokens, which makes the draft's argument at L248-252 false,
and the step must then really be applied. **PROPOSAL:**
- the key is `" ".join(re.sub(r"\d", "0", w) for w in toks(src))`, under `re.UNICODE`;
- first occurrence kept, within each list;
- record whether any removed diverse row has 8 or more tokens.

A18. **"Whether `rep` is recomputed after deduplication"** (L593-594) has no defined parent: the
deduplicated list, or `ft_topk` minus the removed rows. If recomputed, a row could also stop being
"repetitive". **PROPOSAL:** not recomputed for the set (as recommended). Report a recomputation over
`ft_topk` minus the removed rows, descriptively only.

A19. **The candidates for `div_narrow_dd`.** Unstated whether they are the 486,475 diverse
candidates or the deduplicated 486,223. Under OD 4(a) the 252 rows removed are all shorter than
8 tokens and cannot be chosen, but under OD 2(a)/(b) they still enter D7's frequency counts.
**PROPOSAL:** all 486,475, so that `div_narrow` and `div_narrow_dd` share the candidate pool and its
ranks.

A20. **OD 5(c) conflicts with OD 18.** V needs `div_narrow`, and C1-beyond-coverage also loses its
set. **PROPOSAL:** refuse the combination (§7.2).

A21. **OD 6 "remove" conflicts with OD 3 "keep" and with the reproduction requirement.**
- The caps were tuned on the full RU0, and A_narrow's caps are counted over the pool. Removing the
  35,229 top-band rows changes the counts, the capped pool and every draw, so
  `A_narrow_42_58.npy` (and `R0.npy`) cannot describe the production sets.
- "Every random arm" (L609) cannot include `ft_un_random`, an anchor that must not change.
- **PROPOSAL:** under "remove", the reproduction check still runs on the "keep" variant as a code
  check; the caps are re-decided for the reduced pool (OD 3); and `ft_un_random` stays as it is.

A22. **What OD 12 "redraw per seed" means for A_narrow.** The only randomness in A_narrow is the
tie key: per the vocab memo, seeds 42/1/2 differ by ±20 types. A "redraw" is therefore mostly the
same rows. Also, `probe_run.sh` trains every condition at every seed, so it cannot train "draw s
only at seed s". **PROPOSAL:**
- the redraw changes only G3/G4's seed (the `a12.py` precedent);
- the sets are published as `…_d<seed>`;
- `e06_run.sh` needs a per-seed condition list. That is outside the build, and flagged for the run
  script.

A23. **B_dd is a set-defining parameter without a decision number** (L198-201). **PROPOSAL:** give
it a key in the decisions file (§7.1) with no default. The scratch French tokens (6,131,873 /
6,219,966) make 6,000,000 feasible with a 2.2% margin; the draft's *derived* "about 6.08M" is an
underestimate.

A24. **Under OD 1(b), the draft's recommended path is infeasible** (findings 2-3):
- OD 2(a) and (b) cannot narrow at all, because 30,197 rows are forced and the caps that keep them
  are no caps;
- OD 2(c) hits source or French, not both.

**PROPOSAL** for the author's decision on 1 November, choose one of:
- (i) OD 1(a) with OD 2(c);
- (ii) OD 1(b) with PROPOSAL c2 (a second parameter, the French rarest-word rank);
- (iii) a different design: caps on the non-forced buckets only. This is still short at L = 33-35 in
  the scratch probe, so it would need per-bucket narrowing anyway.

Whatever is chosen, the readings of C1-beyond-coverage and V should state that every row of 36 or
more tokens in `div_narrow` is a row of `ft_un_div`, so the narrowing acts only on shorter rows.

A25. **"Measured feasibility … 20,284 … with no bucket short"** (L238-240) does not mention that
any rule is forced in the buckets of 36 tokens or more. The low end is reachable only because those
rows are kept. **PROPOSAL:** add the forced-row count to the draft's text and table.

A26. **Measured-table "n.m." cells and the mixed date of the feasibility file** (L312-315) are for
the registration, not the build. The build fills every cell of §9.

A27. **The scratch feasibility numbers without a script** (20,284; 34,070; 25,296; 0.379).
**PROPOSAL:** drop all four from the draft and plan once the feasibility run exists. Today the draft
drops only the last two.

A28. **OD 1(c) piece counting is unspecified**: which ids count (`<unk>`, `<s>`, `</s>`, `<pad>`),
whether lines are encoded raw, and whether the trainer's own preprocessing is mirrored.
**PROPOSAL:**
- encode raw lines with the pinned model's defaults (no sampling);
- count distinct ids, excluding control and unknown ids;
- check in `$MT/src` that this is how the trainer tokenises.

A ±2% relative rule does not depend on the denominator.

A29. **±2% does not separate the sets under OD 1(d).** `ft_un_div`'s news-commentary coverage at
k ≥ 5 (94.98, vocab memo) is already within +1.9% of `ft_un_rep`'s 93.22, so "matched within ±2%"
would hold with no narrowing. **PROPOSAL:** production refuses (d) unless a tolerance specific to
(d) is registered, for example on the uncovered share (6.78%).

A30. **Set names.** The draft's names fail `e03_collect.py`'s `(ft_[a-z0-9_]+)_lr` (finding 10).
**PROPOSAL:** the names in §2.4.

A31. **The interpreter for the scratch files.** "The interpreter that wrote them" (L289-290) is
`/usr/local/bin/python3` 3.14.3 with numpy 2.4.5, not `~/mt_local/venv`, which has numpy 2.5.3 and
no scipy. **PROPOSAL:** name it in the registered text, and make `--reproduce-scratch` check it
(§3.2).

A32. **R0's no-duplicate argument** (L209-211) holds for A_narrow and `div_narrow` too, since
neither has rows under 8 tokens. Add it to their rows of the table and to the checks (§6.3, §6.5).

A33. **Plan §3.2 item 4 asks for heldout_un disjointness "in `e06_build.py`'s manifest"** without
saying whether a non-zero count stops the build. **PROPOSAL:** stop (§6.7). It is zero today.

A34. **"Pair in pretraining" has no script** in the repository. **PROPOSAL:** define it as §9 does
(`h64` exact pairs), and recompute it for the anchors to check against PROTOCOL's 36.7 / 36.7 /
37.9%. A mismatch there means the definition is wrong, and the column is then labelled
"not E0.5's measure".

---

## 11. CPU cost (Mac, measured pieces from today's probes; single thread unless stated)

| step | time |
| --- | --- |
| sha256 of every pinned input (`e05_src` 0.63 GB, anchors 0.48 GB, `train.clean.en` 1.44 GB, `.fr` 1.66 GB) | about 10 s (measured: 3.4 s for `train.clean.en`) |
| read and tokenise `ft_topk` and `ft_random`, both sides | about 30 s (measured: 15 s for `ft_random` en+fr, 5.7 s for `ft_topk.en`) |
| `rep` and `rep_rand` | about 50 s (measured: 24 s for `ft_topk`) |
| pretraining counts and `maxrank` | about 90 s (measured: 86 s single-threaded) |
| UN split, anchors and byte check | about 10 s |
| R0 | about 2 s |
| A_narrow (caps 5 s, `nn_match` 7 s) | about 12 s (measured) |
| `rep_dedup` / `div_dd` | a few seconds |
| `div_narrow`, one rule. (a)/(b): the shortness test is per bucket and costs about nothing; a feasible point costs about 7-10 s. (c): 201 points × about 2 s | (a)/(b) about 1-2 min; (c) about 7 min; c2 (21 × 201 points) about 2.3 h, or about 10-15 min on 14 processes |
| `div_narrow_dd` (OD 5(b)) | the same again |
| descriptives: pretraining source and pair hash sets | about 3 min |
| descriptives: lexicon rebuild (1.5M pairs) and literalness | about 5 min (estimate) |
| descriptives: QE join over the 14.1 GB `v2_scored.tsv` | about 5-8 min (estimate) |
| descriptives: SPM piece counts | 1-3 min once `sentencepiece` is installed (estimate) |

- **`--reproduce-scratch` alone:** about 4 min.
- **Production dry run with one registered rule:**
  - about 25-35 min single-threaded under (c);
  - about 15 min for (a)/(b), if the box's checks pass;
  - add about 2.3 h single-threaded, or about 15 min on 14 processes, if c2 is registered.
- **`--feasibility` across every option** (OD 1 × OD 2 for both `div_narrow` and `div_narrow_dd`,
  OD 4(b), OD 6 remove, OD 12's four extra draws, OD 3's cap report):
  - about 5-6 h single-threaded, dominated by the c2 grids;
  - about 30-45 min on 14 processes;
  - about 1 h single-threaded without c2.

None of these figures includes writing and testing the script, nor installing Python 3.12.15 and
`sentencepiece` on the Mac (A8).

---

## 12. Acceptance tests (`tests/suites/e06.py`, mirroring `tests/suites/e05.py`)

Each is a synthetic fixture like `make_src` there, built with `--allow-unpinned-src`.

1. The anchors rebuild byte for byte from fixtures where `e04_split.py`'s order and ascending order
   differ. A fixture with one row changed is a STOP.
2. Every set keeps each row's own translation, uses only UN rows, and matches its histogram. A
   short bucket in R0, A_narrow, or a pool smaller than its targets, is a STOP with nothing
   written and no temporary file left.
3. `nn_match` on hand-made keys:
   - lower neighbour on an equal distance;
   - ascending target order on an equal |v − median|;
   - key ties broken by tie key, then index;
   - tie keys drawn for pool rows at lengths without targets (removing such a row changes the
     result).
4. Frequency ties at a cap follow first occurrence.
5. Dedup:
   - the first occurrence is kept;
   - the diverse side's removals are all under 8 tokens under (a);
   - a fixture where (b) removes a diverse row of 8 or more tokens is reported.
6. Fail-closed decisions:
   - any set-defining key absent or `"OPEN"` → exit 2 and nothing written;
   - OD 5(c) with OD 18 true → exit 2;
   - `"retune"`, `match`, `news_coverage` and `recompute_rep: true` → exit 2.
7. `--feasibility` writes no `.en`/`.fr` and no `manifest.json`, `probe_run.sh` refuses its
   directory, and it refuses an `--out` that holds a manifest.
8. The manifest lists every file once with `manifest.json` last, has `heldout_sets`,
   `"e06_complete": true`, every condition name matching `ft_[a-z0-9_]+`, and a stable fingerprint
   on a rebuild. `--verify` refuses a changed file or a missing completion flag.
9. A non-zero heldout_un overlap is a STOP.
10. Production refuses an interpreter whose unidata differs from `od10_unidata`.
11. The source of `e06_build.py` contains `WORD = re.compile(r"\w+", re.UNICODE)` (as
    `tests/suites/e05.py` lines 92-94).
