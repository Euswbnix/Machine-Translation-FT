# phase0/next — sources for the post-E0.5 plan

What the EAMT 2027 / WMT 2027 drafts cite, kept here so the citations survive:

- `groundwork.json` — full-text reading notes on ten papers (`notes`) and the three design memos
  (`vocab`, `ende`, `llm`). Cited as [Gn:key], [Gv], [Ge], [Gl] in `WMT2027_PLAN.md`.
- `e06_feasibility.md` — what the E0.6 arms can reach (cited as [F]).
- `vocab/*.py` — the vocab memo's analysis scripts (the pipeline behind `rows.pkl`, `lex.pkl` and
  the arm index files).
- `ende/*.py`, `ende/*.json`, `ende/*.log` — the en-de memo's scripts, reports and logs.

Not in git (data stays out of the repo; some files exceed GitHub's 100 MB limit): the 24 arm index
`.npy` files, `ende/s2_idx_*.json`, and the `.pkl` / `.npz` / per-row `.npy` caches. They are in
`~/mt_local/phase0_next/` on the author's Mac. `SHA256SUMS` covers all 68 files, committed or not;
check the backup with `cd ~/mt_local/phase0_next && shasum -a 256 -c <path to this SHA256SUMS>`.

These are exploratory scratch analyses, not registered experiments. The scripts read their inputs
from `~/mt_local/phase0_final/` and `~/mt_local/rebuild/` and write next to themselves. Every figure that a pre-registration
quotes must be re-measured from the fail-closed builds (`WMT2027_PLAN.md` §3.3, §3.4).
