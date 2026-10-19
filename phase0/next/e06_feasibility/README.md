# E0.6 feasibility run (2026-10-19) — NOT FOR TRAINING

`phase0/e06_build.py --feasibility` built every option of the E0.6 OPEN DECISIONS for measurement
only, on the M5 Max (`~/e06env/venv-arm64`: Python 3.14.3, numpy 2.4.5, Unicode 16.0.0, arm64 — the
reference interpreter), from the sha256-pinned inputs, single process, 00:55-01:46. EXIT 0.
`feasibility_report.json.gz` holds every number; the 46 index files are in
`~/mt_local/phase0_next/e06_feasibility/` (sha256 in `SHA256SUMS`, not in git).

Checks that passed: E0.5's UN split and anchors reproduced byte for byte; `R0` and `A_narrow` at
seed 42 have the sha256 of the scratch `R0.npy` / `A_narrow_42_58.npy`; the rebuilt Dice lexicon is
identical to `lex.pkl` (12,340 entries).

**div_narrow against its target** (`ft_un_rep`: 25,134 source word types, 32,684 French word types
at 6.5M tokens; tolerance ±2% on both):

| OD 2 rule | OD 1 measure | source | French | both within ±2%? |
| --- | --- | --- | --- | --- |
| (a) | any | +23.1% | +23.0% | no |
| (b) | any | +34.7% | +31.8% | no |
| (c), the draft's recommendation | (b) | −5.7% | +5.8% | no |
| (c) | (a) | +0.1% | +9.5% | source only |
| (c) | (d) | −15.8% | −2.7% | no |
| **c2** (spec proposal, two parameters) | **(b)** | **−0.3%** | **+0.3%** | **yes, the only one** |
| c2 | (a) | +0.0% | −7.8% | source only |
| c2 | (d) | −15.7% | −4.9% | no |

The ninth set `div_narrow_dd` (198,333 rows) behaves the same way: only c2 with OD 1(b) is within
±2% on both sides (+0.3% / −0.1%). Every div_narrow option keeps mean QE at 0.8990 (= `ft_un_div`)
and shares about 127,000 of its 214,489 rows with `ft_un_div` (the 30,197 forced rows and the
narrow candidate pool). `A_narrow` with OD 6 = remove fails at caps 42,000 / 58,000 (4 length
buckets short); `od3_cap_report` in the report gives the cap grid for both pools.

Subword coverage (OD 1(c)) is not built by the script; see `../cpu_items/item1_coverage/` — matching
on word types does not match subword coverage (A_narrow is +4.5% / +9.7% off `ft_un_div` on pieces).

These numbers inform the author's E0.6 decisions; none is registered. A production dry run with the
chosen decisions is still required before the tag.
