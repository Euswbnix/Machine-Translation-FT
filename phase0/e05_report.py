#!/usr/bin/env python3
"""E0.5 — apply the PRE-STATED outcomes of the matched-control probe.

Written before the runs, like e03_decide.py and e04_report.py and for the same reason: the
outcomes in PROTOCOL.md "E0.5" are fixed in advance so the numbers cannot choose the story.

WHAT E0.4 LEFT OPEN
  1. Its two halves differ in more than repetitiveness: measured before any run, the
     repetitive half is 91% UN text and the diverse half 74%. "Repetitive" and "more UN"
     cannot be told apart from E0.4 alone.
  2. It had no random arm of the same size, so it could not say whether the diverse half is
     as good as ordinary data.
E0.5 trains six sets on one machine: the two halves and a size- and length-matched random
control over ALL SOURCES, and the same three built inside UN text ONLY.

THE RULE, for every contrast: a gap counts only if it exceeds the seed-noise floor (the
larger seed sd of the two sets) AND is Welch-significant at 0.05, with the exact critical
value at the fractional df (phase0/probe_stats.py). Otherwise it is "not detected", and the
smallest gap the rule could have called with these seeds is printed beside it.

Q1  DOES THE REPETITIVENESS GAP SURVIVE INSIDE ONE SOURCE?   ft_un_rep vs ft_un_div
      YES       the repetitive half is worse. Repetitiveness is not reducible to source mix.
      NO        not detected. E0.4's contrast cannot be attributed to repetitiveness, and that
                reading is withdrawn.
      REVERSED  the diverse half is worse.

Q2  IS THE DIVERSE HALF AS GOOD AS MATCHED RANDOM DATA?      ft_un_div vs ft_un_random
      A  not detected   no net difference this design can see
      B  diverse worse  the top band costs something beyond repetition (this half has no
                        repeated 8-gram; the random set has some)
      C  diverse better what repetition alone would produce; not by itself evidence that
                        quality selection helps
    The same contrast over all sources (ft_topk_div vs ft_random_small) is printed too, but
    those two sets differ in source mix, so it is reported and decides nothing.

    The glosses on B and C assume Q1 = YES. If Q1 is NO or REVERSED they are not printed and
    the outcome is reported by its letter alone.

REPLICATION of E0.4 on these runs: ft_topk_rep vs ft_topk_div over all sources. Reported
whatever Q1 says. No claim is drawn from the SIZE of this gap against the UN one: the two
triples differ in rows (242,022 vs 214,489), hence in passes over the data (7.9 vs 9.0), and
in how much of each set the base model has already seen.

TOKENS. The arms are matched in sentence pairs seen, not in tokens: the sets share a length
histogram in words, but their texts break into different numbers of subword pieces, so at
equal steps they apply different token counts (known before the runs to exceed the 2% the
protocol flags). For each deciding pair the report prints the imbalance, which set got more,
and a bound on what it could be worth: the imbalance, as a share of the fine-tuning steps,
times the steeper of the two sets' own late-training dev-BLEU slopes. A bound above the
contrast's seed-noise floor QUALIFIES the outcome; one below it is immaterial.

FAIL-CLOSED. All six sets must have exactly --expect-seeds values on the news set and the
baseline must have a row there, else exit 2 (NO DECISION). An undefined Welch test in Q1, Q2
or the replication is also exit 2. Exit 0 means the outcomes were decided and named.
E0.5 is an explanatory probe; it cannot revive E0.3's NO-GO verdict.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e04_report import applied_tokens, read_results, table  # noqa: E402
from probe_stats import contrast, mean_sd  # noqa: E402

ALL = ("ft_topk_div", "ft_topk_rep", "ft_random_small")
UN = ("ft_un_div", "ft_un_rep", "ft_un_random")
THRESHOLD = 0.02            # applied-token imbalance the protocol flags


def show(label: str, c: dict | None) -> None:
    if c is None:
        print(f"  {label}: Welch undefined (zero seed variance on both sides, or too few seeds)")
        return
    verdict = "DETECTED" if c["clear"] else "not detected"
    print(f"  {label}: gap {c['gap']:+.2f} BLEU | noise floor {c['noise']:.2f} | "
          f"Welch t = {c['t']:.2f}, df = {c['df']:.2f}, crit = {c['crit']:.2f}, p = {c['p']:.3f} -> {verdict}")
    print(f"      smallest gap this rule could have called with these seeds: {c['bound']:.2f} BLEU")


EVAL_RE = re.compile(r"Step\s+([0-9,]+)\s*\|\s*Valid BLEU:\s*([0-9.]+)")


def applied(results: Path, logdir: Path | None, names: tuple[str, ...]):
    """-> ({set: mean applied target tokens}, how). Exact counts from the results meta that
    e03_collect wrote; the 4-decimal log lines only as a fallback."""
    meta = results.with_suffix(".meta.json")
    try:
        budget = json.loads(meta.read_text(encoding="utf-8")).get("budget", {})
    except (OSError, ValueError):
        budget = {}
    means = {}
    for nm in names:
        vals = [v.get("applied_target_tokens") for k, v in budget.items() if k.rsplit(" ", 1)[0] == nm]
        if vals and all(isinstance(x, (int, float)) and x > 0 for x in vals):
            means[nm] = sum(vals) / len(vals)
    if len(means) == len(names):
        return means, "exact counts"
    means = {}
    if logdir and logdir.exists():
        for nm in names:
            vals = [v for v in applied_tokens(logdir, nm).values() if v is not None]
            if vals:
                means[nm] = 1e9 * sum(vals) / len(vals)
    return means, "training logs, 4 decimals: rounding alone is about 0.14 points"


def late_slope(logdir: Path | None, cond: str):
    """Mean dev-BLEU change per step over the second half of fine-tuning, and the number of
    steps the dev trace spans, averaged over this set's runs. None when no run has a trace."""
    if not logdir or not logdir.exists():
        return None
    slopes, spans = [], []
    for f in sorted(logdir.glob(f"{cond}_lr*_s*_st2.log")):
        pts = [(int(s.replace(",", "")), float(b))
               for s, b in EVAL_RE.findall(f.read_text(encoding="utf-8", errors="replace"))]
        if len(pts) < 3:
            continue
        (s0, _), (s1, _), (sn, bn) = pts[0], pts[1], pts[-1]
        mid = min(pts[:-1], key=lambda q: abs(q[0] - (s0 + sn) / 2))
        if sn == mid[0]:
            continue
        slopes.append((bn - mid[1]) / (sn - mid[0]))
        spans.append(sn - s0 + (s1 - s0))          # one interval ran before the first eval
    if not slopes:
        return None
    return sum(slopes) / len(slopes), sum(spans) / len(spans)


def token_pair(label: str, a: str, b: str, c: dict | None, tok: dict, how: str, logdir: Path | None) -> str:
    """Print the applied-token imbalance of one deciding pair and bound its effect.
    Returns '' or a note to append to that question's verdict."""
    if a not in tok or b not in tok:
        print(f"  tokens applied, {label}: NOT REPORTED — no counts in the results meta and none in the logs")
        return " [tokens applied not reported]"
    ta, tb = tok[a], tok[b]
    imb = abs(ta - tb) / ((ta + tb) / 2)
    more = a if ta > tb else b
    over = imb > THRESHOLD
    print(f"  tokens applied, {label} ({how}): {a} {ta / 1e6:.3f}M vs {b} {tb / 1e6:.3f}M — "
          f"{imb * 100:.2f}% more for {more} "
          + ("(ABOVE the 2% the protocol set)" if over else "(within the 2% the protocol set)"))
    sl = [late_slope(logdir, x) for x in (a, b)]
    if any(s is None for s in sl) or c is None:
        print("      its effect is NOT BOUNDED: no dev-BLEU trace in the training logs")
        return " [token imbalance above 2% and not bounded]" if over else ""
    steep = max(abs(sl[0][0]), abs(sl[1][0]))
    span = max(sl[0][1], sl[1][1])
    bound = imb * span * steep
    qualifies = bound > c["noise"]
    print(f"      worth at most {bound:.3f} BLEU: {imb * 100:.2f}% of {span:,.0f} steps at the steeper late "
          f"dev slope ({steep * 1000:.4f} BLEU per 1,000 steps); noise floor {c['noise']:.2f} -> "
          + ("QUALIFIES the outcome" if qualifies else "immaterial"))
    return f" [QUALIFIED: the token imbalance could be worth {bound:.2f} BLEU, above the noise floor]" if qualifies else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path, help="e05_probe.tsv")
    ap.add_argument("--news", default="newstest2014")
    ap.add_argument("--indomain", default="heldout_un",
                    help="secondary, reported and never decisive (PROTOCOL E0.5)")
    ap.add_argument("--expect-seeds", type=int, default=5, help="PROTOCOL E0.5 registers five")
    ap.add_argument("--logdir", type=Path,
                    help="training logs: the dev-BLEU traces that bound the effect of the token "
                         "imbalance, and a fallback for the token counts themselves")
    a = ap.parse_args()

    if not a.results.exists():
        print(f"NO DECISION — no results file at {a.results}")
        return 2
    r = read_results(a.results)
    six = ALL + UN
    missing = [c for c in six if c not in r or a.news not in r[c]]
    if missing:
        print(f"NO DECISION — no {a.news} rows for: {', '.join(missing)}")
        return 2
    wrong = [f"{c} ({len(r[c][a.news])} of {a.expect_seeds})" for c in six
             if len(r[c][a.news]) != a.expect_seeds]
    if wrong:
        print(f"NO DECISION — wrong seed count on {a.news}: {', '.join(wrong)}")
        return 2
    base = {ts: v[0] for ts, v in r["baseline"].items() if v} if "baseline" in r else {}
    if a.news not in base:
        print(f"NO DECISION — no baseline row on {a.news}")
        return 2

    sets = [a.news] + [t.strip() for t in a.indomain.split(",") if t.strip()]
    table(r, sets, base)
    n = lambda c: r[c][a.news]

    q1 = contrast(n("ft_un_div"), n("ft_un_rep"))            # > 0: the repetitive half is worse
    q2 = contrast(n("ft_un_div"), n("ft_un_random"))         # > 0: the diverse half is better
    rep = contrast(n("ft_topk_div"), n("ft_topk_rep"))
    q2_all = contrast(n("ft_topk_div"), n("ft_random_small"))

    print(f"\nQ1 — inside UN text only ({a.news}):")
    show("ft_un_div - ft_un_rep", q1)
    print(f"\nQ2 — inside UN text only ({a.news}):")
    show("ft_un_div - ft_un_random", q2)
    print(f"\nREPLICATION of E0.4, all sources ({a.news}):")
    show("ft_topk_div - ft_topk_rep", rep)
    print(f"\nREPORTED, NOT DECISIVE ({a.news}):")
    show("ft_topk_div - ft_random_small (the sets differ in source mix)", q2_all)
    show("ft_topk_rep - ft_random_small", contrast(n("ft_topk_rep"), n("ft_random_small")))
    show("ft_un_rep - ft_un_random", contrast(n("ft_un_rep"), n("ft_un_random")))
    for ts in sets[1:]:
        for names in (UN, ALL):
            if all(ts in r[c] for c in names):
                vals = "  ".join(f"{c} {mean_sd(r[c][ts])[0]:.2f}" for c in names)
                b = f" (baseline {base[ts]:.2f})" if ts in base else ""
                print(f"  {ts}: {vals}{b}")
    tok, how = applied(a.results, a.logdir, six)
    print("\nTOKENS (the arms are matched in sentence pairs seen, not in tokens):")
    note1 = token_pair("Q1", "ft_un_div", "ft_un_rep", q1, tok, how, a.logdir)
    note2 = token_pair("Q2", "ft_un_div", "ft_un_random", q2, tok, how, a.logdir)
    note_r = token_pair("replication", "ft_topk_div", "ft_topk_rep", rep, tok, how, a.logdir)

    if q1 is None or q2 is None or rep is None:
        print("\nNO DECISION — Welch undefined in Q1, Q2 or the replication "
              "(a set has zero seed variance or too few seeds)")
        return 2

    print()
    if rep["clear"] and rep["gap"] > 0:
        print(f"E0.4 REPLICATES on this machine: over all sources the repetitive half is again "
              f"worse, by {rep['gap']:.2f} BLEU.{note_r}")
    elif rep["clear"]:
        print(f"E0.4 DOES NOT REPLICATE: over all sources the DIVERSE half is the worse one here, "
              f"by {-rep['gap']:.2f} BLEU.{note_r}")
    else:
        print(f"E0.4 DOES NOT REPLICATE: over all sources the halves are not distinguishable here "
              f"({rep['gap']:+.2f} BLEU; a gap under {rep['bound']:.2f} could not have been called).{note_r}")

    q1_yes = q1["clear"] and q1["gap"] > 0
    if q1_yes:
        print(f"Q1 — YES: inside UN text alone the repetitive half is worse by {q1['gap']:.2f} BLEU. "
              f"Repetitiveness is not reducible to source mix.{note1}")
    elif q1["clear"]:
        print(f"Q1 — REVERSED: inside UN text alone the DIVERSE half is worse by {-q1['gap']:.2f} BLEU.{note1}")
    else:
        print(f"Q1 — NO: inside UN text alone no gap was detected ({q1['gap']:+.2f} BLEU; a gap under "
              f"{q1['bound']:.2f} could not have been called). E0.4's contrast cannot be attributed "
              f"to repetitiveness; that reading is withdrawn.{note1}")

    if not q2["clear"]:
        print(f"Q2 — OUTCOME A: no gap detected between the diverse half and matched random UN text "
              f"({q2['gap']:+.2f} BLEU; a gap under {q2['bound']:.2f} could not have been called). "
              f"This is a failure to detect, not a demonstration that they are equal.{note2}")
    elif q2["gap"] < 0:
        print(f"Q2 — OUTCOME B: the diverse half is worse than matched random UN text by "
              f"{-q2['gap']:.2f} BLEU."
              + (" The top band costs something beyond repetition: this half has no repeated 8-gram, "
                 "the random set has some." if q1_yes else
                 " Reported by letter only: with Q1 not a YES, no reading in terms of repetition is offered.")
              + note2)
    else:
        print(f"Q2 — OUTCOME C: the diverse half is better than matched random UN text by "
              f"{q2['gap']:.2f} BLEU."
              + (" That is what repetition alone would produce (the random set contains some, this half "
                 "none); it does not by itself show that quality selection helps." if q1_yes else
                 " Reported by letter only: with Q1 not a YES, no reading in terms of repetition is offered.")
              + note2)
    print("E0.5 is an explanatory probe. It does not change E0.3's NO-GO verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
