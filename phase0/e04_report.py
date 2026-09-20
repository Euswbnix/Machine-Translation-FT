#!/usr/bin/env python3
"""E0.4 — apply the PRE-STATED outcomes of the repetitiveness probe to its results.

Written before the six runs finish, for the same reason e03_decide.py was written before
E0.3: the three outcomes in PROTOCOL.md "E0.4" are stated in advance precisely so that
nobody gets to look at the numbers first and then decide which story they tell.

THE HYPOTHESIS (PROTOCOL.md, "E0.4")
  H. CometKiwi's top band selects text that is easy to translate -- short, formulaic,
     lexically thin -- rather than text that teaches, and it is that property, not the
     domain mix, that makes ft_topk worse than a matched random set.

THE OUTCOMES, in the protocol's own order
  1. H HOLDS        ft_topk_rep degrades the news set MORE than ft_topk_div, by more than
                    the seed-noise floor, and the difference is significant (Welch, 0.05).
                    Reported alongside: whether ft_topk_div lands closer to -- or better
                    than -- ft_random at the same lr_scale (from --reference).
  2. H REFUTED      the halves are indistinguishable: the gap does not clear the noise
                    floor, or the Welch test does not reach significance. The damage then
                    belongs to the whole top band, not to repetitiveness within it.
  3. H REFUTED THE  ft_topk_div is the worse half, beyond the noise floor and significant.
     OTHER WAY      QE rewards something else entirely; the protocol says report and stop.

FAIL-CLOSED. Both halves must be present on the news set with exactly --expect-seeds
values each, and the baseline must have a row there; otherwise this exits 2 (NO DECISION)
rather than guessing. A zero-variance pair (Welch undefined) is also exit 2.

Exit 0 means an outcome was decided and named -- 0 is not "H holds". E0.4 is an
explanatory probe, not a gate: nothing here can revive E0.3's NO-GO verdict.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e03_decide import mean_sd, welch  # noqa: E402  (reused, never retyped)

DIV, REP = "ft_topk_div", "ft_topk_rep"


def read_results(path: Path):
    """condition -> testset -> [bleu]. Same tolerant reader as e03_decide: comments and
    header skipped, unparseable and NaN rows reported and dropped."""
    r = defaultdict(lambda: defaultdict(list))
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4 or p[0] == "condition":
            continue
        try:
            bleu = float(p[3])
        except ValueError:
            print(f"  skipping unparseable row: {line.rstrip()}")
            continue
        if math.isnan(bleu):
            print(f"  skipping NaN row: {line.rstrip()}")
            continue
        r[p[0]][p[2]].append(bleu)
    return r


TOK_RE = re.compile(r"Tok\(applied\)\s+([0-9.]+)B")


def applied_tokens(logdir: Path, cond: str):
    """Last 'Tok(applied) <x>B' of each run log of this condition, in billions.

    The trainer prints it every log interval, so the last one is the run's total. A log
    with no such line is returned as None rather than silently counted as zero."""
    out = {}
    for f in sorted(logdir.glob(f"{cond}_lr*_s*_st2.log")):
        hits = TOK_RE.findall(f.read_text(encoding="utf-8", errors="replace"))
        out[f.name] = float(hits[-1]) if hits else None
    return out


def table(r, sets, base=None):
    print(f"{'condition':<14} {'testset':<16} {'n':>2} {'mean':>7} {'sd':>6} {'vs base':>9}")
    for cond in sorted(r):
        for ts in sets:
            if ts not in r[cond]:
                continue
            m, sd = mean_sd(r[cond][ts])
            b = base.get(ts) if base else None
            delta = f"{m - b:>+9.2f}" if b is not None else " " * 9
            print(f"{cond:<14} {ts:<16} {len(r[cond][ts]):>2} {m:>7.2f} "
                  f"{sd if not math.isnan(sd) else float('nan'):>6.2f} {delta}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path, help="e04_probe.tsv")
    ap.add_argument("--reference", type=Path,
                    help="lr1_probe.tsv: ft_random/ft_topk at the same lr_scale, for context")
    ap.add_argument("--news", default="newstest2014")
    ap.add_argument("--indomain", default="heldout_un",
                    help="secondary, reported and never decisive (PROTOCOL E0.4)")
    ap.add_argument("--expect-seeds", type=int, default=3)
    ap.add_argument("--logdir", type=Path,
                    help="training logs: report the applied-token imbalance between the "
                         "halves, which PROTOCOL E0.4 says to report (threshold 2%%)")
    a = ap.parse_args()

    if not a.results.exists():
        print(f"NO DECISION — no results file at {a.results}")
        return 2
    r = read_results(a.results)

    missing = [c for c in (DIV, REP) if c not in r or a.news not in r[c]]
    if missing:
        print(f"NO DECISION — no {a.news} rows for: {', '.join(missing)}")
        return 2
    wrong = [f"{c} ({len(r[c][a.news])} of {a.expect_seeds})" for c in (DIV, REP)
             if len(r[c][a.news]) != a.expect_seeds]
    if wrong:
        print(f"NO DECISION — wrong seed count on {a.news}: {', '.join(wrong)}")
        return 2

    base = {}
    if "baseline" in r:
        base = {ts: v[0] for ts, v in r["baseline"].items() if v}
    if a.news not in base:
        print(f"NO DECISION — no baseline row on {a.news}")
        return 2

    sets = [a.news] + [t.strip() for t in a.indomain.split(",") if t.strip()]
    table(r, sets, base)

    div, rep = r[DIV][a.news], r[REP][a.news]
    m_div, sd_div = mean_sd(div)
    m_rep, sd_rep = mean_sd(rep)
    noise = max(sd_div, sd_rep)
    gap = m_div - m_rep            # > 0: the repetitive half is the worse one
    w = welch(div, rep)
    print(f"\n{a.news}: {DIV} {m_div:.2f} vs {REP} {m_rep:.2f} "
          f"(baseline {base[a.news]:.2f})")
    print(f"  gap = {gap:+.2f} BLEU (positive = the repetitive half degrades more)")
    print(f"  seed-noise floor (max sd) = {noise:.2f}")
    if w is None:
        print("NO DECISION — Welch undefined (a half has zero seed variance or too few seeds)")
        return 2
    t, dfree, crit = w
    sig = abs(t) > crit
    print(f"  Welch t = {t:.2f}, df = {dfree:.1f}, crit(0.05) = {crit:.2f} -> "
          f"{'significant' if sig else 'NOT significant'}")

    if a.reference and a.reference.exists():
        ref = read_results(a.reference)
        for cond in ("baseline", "ft_topk", "ft_random", "ft_bottom"):
            if cond in ref and a.news in ref[cond]:
                m, sd = mean_sd(ref[cond][a.news])
                print(f"  [reference] {cond:<10} {a.news} mean {m:.2f} sd "
                      f"{sd if not math.isnan(sd) else float('nan'):.2f}")
    elif a.reference:
        print(f"  [reference] {a.reference} not found — context lines omitted")

    for ts in sets[1:]:
        if all(ts in r[c] for c in (DIV, REP)):
            md, _ = mean_sd(r[DIV][ts])
            mr, _ = mean_sd(r[REP][ts])
            b = base.get(ts)
            extra = f" (baseline {b:.2f})" if b is not None else ""
            print(f"  [secondary] {ts}: {DIV} {md:.2f} vs {REP} {mr:.2f}{extra} — "
                  f"reported, not decisive")

    if a.logdir and a.logdir.exists():
        tok = {c: applied_tokens(a.logdir, c) for c in (DIV, REP)}
        vals = {c: [v for v in d.values() if v is not None] for c, d in tok.items()}
        missing = [n for d in tok.values() for n, v in d.items() if v is None]
        if all(vals.values()):
            md = sum(vals[DIV]) / len(vals[DIV])
            mr = sum(vals[REP]) / len(vals[REP])
            mid = (md + mr) / 2
            imb = abs(md - mr) / mid if mid else float("nan")
            print(f"  [secondary] applied tokens: {DIV} {md:.4f}B ({len(vals[DIV])} runs) vs "
                  f"{REP} {mr:.4f}B ({len(vals[REP])} runs) — imbalance {imb * 100:.2f}% "
                  f"({'within' if imb <= 0.02 else 'ABOVE'} the 2% the protocol set)")
        else:
            print("  [secondary] applied tokens: not reported (no runs found for a half)")
        if missing:
            print(f"  [secondary] no Tok(applied) line in: {', '.join(missing)}")
    elif a.logdir:
        print(f"  [secondary] applied tokens: {a.logdir} not found")

    if not sig or abs(gap) <= noise:
        print(f"\nOUTCOME 2 — H REFUTED: the halves are indistinguishable "
              f"({'gap within the noise floor' if abs(gap) <= noise else 'not significant'}). "
              f"The damage belongs to the whole top band, not to repetitiveness within it.")
    elif gap > 0:
        print(f"\nOUTCOME 1 — H HOLDS: the repetitive half degrades {a.news} by "
              f"{gap:.2f} BLEU more than the diverse half, beyond the {noise:.2f} noise "
              f"floor and significant.")
    else:
        print(f"\nOUTCOME 3 — H REFUTED THE OTHER WAY: the DIVERSE half is worse by "
              f"{-gap:.2f} BLEU. QE is rewarding something other than easy text; "
              f"the protocol says report this and stop the line.")
    print("E0.4 is an explanatory probe. It does not change E0.3's NO-GO verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
