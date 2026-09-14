#!/usr/bin/env python3
"""E0.3 stage 1 -> the lr_scale for stage 2, by an EXPLICIT, mechanical rule.

Training-path finding: the stage-1 rule was worded two ways ("flat" in README /
e03_decide; "does not decline monotonically from the first eval" in RUNBOOK,
rental_setup.sh and run_stage1.sh), no code applied either, and a person picked the
value by eye. This parses the per-eval validation BLEU (newstest2013; the gate uses
newstest2014) from the stage-1 logs and applies ONE named rule. The rule is a
pre-registration decision (phase0/e03_decisions.json "lr_selection_rule"); the default
here is the wording the runner prints today, NOT a recommendation.

Rules (a rung is one lr_scale; x_1..x_n its evals in step order):
  no-strict-monotone-decline-from-first-eval   FAIL iff x_{i+1} < x_i for every i. No tolerance.
  flat-endpoints     PASS iff |x_n - x_1| <= tol
  flat-slope         PASS iff |OLS slope over (step, x)| * (step_n - step_1) <= tol
  flat-vs-baseline   PASS iff |x_n - B| <= tol, B = pre-fine-tuning newstest2013 BLEU
                     (rental_setup.sh accept writes it; pass --baseline-bleu)
Selected = the LARGEST lr_scale that passes.

Log lines parsed: "Step <N> | Valid BLEU: <x>" (trainer.py). One log per rung:
<logdir>/ft_topk_lr<scale %g>_s42_st1.log (run_parallel.py names logs <config stem><suffix>).

Exit: 0 a rung selected; 1 no rung passes; 2 invalid inputs (missing log, fewer or more
evals than --expect-evals, a repeated step, a tolerance given to a rule that takes none).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RULES = {"no-strict-monotone-decline-from-first-eval": False, "flat-endpoints": True,
         "flat-slope": True, "flat-vs-baseline": True}
DEFAULT_RULE = "no-strict-monotone-decline-from-first-eval"
EVAL_RE = re.compile(r"Step\s+(\d+)\s*\|\s*Valid BLEU:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_log(path: Path) -> list[tuple[int, float]]:
    out = []
    with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2))))
    return out


def ols_slope(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0


def judge(rule: str, steps, bleu, tol, baseline):
    """-> (passes: bool, statistic: float|None)"""
    if rule == "no-strict-monotone-decline-from-first-eval":
        strictly_declining = all(b < a for a, b in zip(bleu, bleu[1:]))
        return (not strictly_declining), None
    if rule == "flat-endpoints":
        s = abs(bleu[-1] - bleu[0])
    elif rule == "flat-slope":
        s = abs(ols_slope(steps, bleu)) * (steps[-1] - steps[0])
    elif rule == "flat-vs-baseline":
        s = abs(bleu[-1] - baseline)
    else:
        raise ValueError(rule)
    return s <= tol, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--rule", choices=list(RULES), default=DEFAULT_RULE)
    ap.add_argument("--tol", type=float, default=None, help="BLEU tolerance for the flat-* rules")
    ap.add_argument("--baseline-bleu", type=float, default=None, help="pre-FT newstest2013 BLEU (flat-vs-baseline)")
    ap.add_argument("--lr-scales", default="1,0.5,0.15,0.05")
    ap.add_argument("--expect-evals", type=int, default=10,
                    help="evals per rung (10 for 105K->115K at eval_interval 1000)")
    ap.add_argument("--out", default="", help="JSON record of the selection")
    a = ap.parse_args(argv)

    def bad(msg):
        print(f"NO SELECTION — {msg}")
        return 2

    if RULES[a.rule] and (a.tol is None or a.tol < 0):
        return bad(f"rule {a.rule} needs --tol >= 0")
    if not RULES[a.rule] and a.tol is not None:
        return bad(f"rule {a.rule} takes no tolerance; do not pass --tol")
    if a.rule == "flat-vs-baseline" and a.baseline_bleu is None:
        return bad("flat-vs-baseline needs --baseline-bleu")
    scales = [float(x) for x in a.lr_scales.split(",") if x.strip()]
    rungs = []
    for s in scales:
        log = Path(a.logdir) / f"ft_topk_lr{s:g}_s42_st1.log"
        if not log.is_file():
            return bad(f"{log} missing")
        ev = parse_log(log)
        steps = [e[0] for e in ev]
        if len(set(steps)) != len(steps):
            return bad(f"{log}: a step is logged twice (restarted run?); ambiguous")
        if len(ev) != a.expect_evals:
            return bad(f"{log}: {len(ev)} evals, expected {a.expect_evals}")
        if steps != sorted(steps):
            return bad(f"{log}: evals are not in step order")
        bleu = [e[1] for e in ev]
        ok, stat = judge(a.rule, steps, bleu, a.tol, a.baseline_bleu)
        rungs.append({"lr_scale": s, "log": str(log), "steps": steps, "bleu": bleu,
                      "statistic": stat, "pass": ok})

    print(f"rule: {a.rule}" + (f"   tol: {a.tol} BLEU" if a.tol is not None else "")
          + (f"   baseline newstest2013 BLEU: {a.baseline_bleu}" if a.baseline_bleu is not None else ""))
    for r in rungs:
        st = "" if r["statistic"] is None else f"  stat={r['statistic']:.3f}"
        print(f"  lr_scale {r['lr_scale']:<5g} {'PASS' if r['pass'] else 'FAIL'}{st}  "
              + " ".join(f"{b:.2f}" for b in r["bleu"]))
    passing = [r["lr_scale"] for r in rungs if r["pass"]]
    selected = max(passing) if passing else None
    rec = {"rule": a.rule, "tol": a.tol, "baseline_bleu": a.baseline_bleu,
           "expect_evals": a.expect_evals, "rungs": rungs,
           "selected": None if selected is None else f"{selected:g}"}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        tmp.replace(a.out)
    if selected is None:
        print("NO RUNG PASSES the rule — stop and report; do not pick one by eye.")
        return 1
    print(f"SELECTED lr_scale {selected:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
