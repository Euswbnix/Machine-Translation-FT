#!/usr/bin/env python3
"""PROTOCOL 4.3 / 6.2 / 6.3 — the tagged decision script.

This is the artifact the pre-registration promises reviewers: it consumes ONLY
the released dev-CE traces and emits the convergence points and the Big-vs-Base
verdict with no human in the loop. Anyone can re-run it, and anyone can edit the
stopping rule at the top and see whether the conclusion moves.

The point of run-to-overshoot is that no stopping decision is ever EXECUTED
during training. Convergence is defined here, after the fact, over a curve that
was run far past it. A stopping rule you never executed cannot bias the run.

INPUT
-----
TSV, one row per evaluation:

    cell            seed  tokens      dev_ce
    enfr_base_cap   1     50000000    3.9120
    enfr_base_cap   1     100000000   3.6015
    enfr_big_cap    1     50000000    4.2210
    ...

`tokens` is cumulative APPLIED non-pad target tokens (PROTOCOL 1.2), which is
what the patched trainer logs as `train/applied_target_tokens`.

Optional --params gives each cell's parameter count, enabling slice (c).

USAGE
-----
    python phase1/converge.py --traces results/phase1_dev_ce.tsv \
        --params enfr_base_cap=60000000,enfr_big_cap=209000000 \
        --big-prefix enfr_big --base-prefix enfr_base
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

SMOOTH_WINDOW = 3          # PROTOCOL 4.3: 3-point median
NOT_CONVERGED_TAIL = 0.20  # PROTOCOL 4.5: argmin in the final 20% -> not converged
NULL_SD_MULTIPLE = 2.0     # PROTOCOL 6.3


def median3(xs: list[float]) -> list[float]:
    """3-point median smoothing; endpoints pass through unchanged."""
    if len(xs) < SMOOTH_WINDOW:
        return list(xs)
    out = list(xs)
    for i in range(1, len(xs) - 1):
        out[i] = sorted(xs[i - 1:i + 2])[1]
    return out


def pooled_within_sd(groups):
    """PROTOCOL 6.3: pooled WITHIN-CELL seed standard deviation.

    Concatenating per-seed values across cells and taking one sd measures the
    BETWEEN-cell spread, which is the effect being tested -- using it as the
    noise floor would declare every real effect null. Pool the within-cell
    variances instead:  s_p = sqrt( sum_c (n_c-1) s_c^2 / sum_c (n_c-1) ).

    Returns nan when no cell has >= 2 seeds, i.e. when there is NO variance
    estimate. That is not the same as a small effect and must not be reported
    as a null.
    """
    num, den = 0.0, 0
    for v in groups.values():
        if not v or len(v) < 2:
            continue
        m = sum(v) / len(v)
        num += sum((x - m) ** 2 for x in v)
        den += len(v) - 1
    return math.sqrt(num / den) if den else float("nan")


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def argmin_point(tokens, ce):
    """PROTOCOL 4.3: argmin of the smoothed trace, ties broken toward FEWER tokens."""
    sm = median3(ce)
    best_i, best_v = 0, sm[0]
    for i, v in enumerate(sm):
        if v < best_v:                    # strict < keeps the EARLIEST minimum
            best_i, best_v = i, v
    return best_i, tokens[best_i], best_v


def ce_at(tokens, ce, target):
    """Dev CE at the last evaluation at or before `target` tokens.

    Returns (value, in_range). in_range is False when the trace ENDS before the
    target, in which case the value is the trace's final point and must NOT be
    used: silently substituting a short seed's endpoint would compare cells at
    different budgets while claiming they are matched.
    """
    if not tokens:
        return None, False
    if target < tokens[0]:
        return None, False
    best = None
    for t, c in zip(tokens, ce):
        if t <= target:
            best = c
        else:
            break
    return best, tokens[-1] >= target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--params", default="", help="cell=N,cell=N — enables the tok/param slice")
    ap.add_argument("--base-prefix", default="")
    ap.add_argument("--big-prefix", default="")
    args = ap.parse_args()

    raw = defaultdict(list)   # (cell, seed) -> [(tokens, ce)]
    with open(args.traces, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 4 or p[0] in ("cell", "#"):
                continue
            try:
                raw[(p[0], p[1])].append((int(float(p[2])), float(p[3])))
            except ValueError:
                print(f"  skipping unparseable row: {line.rstrip()}")
    if not raw:
        print("NO DECISION — no usable rows")
        return 2

    params = {}
    for kv in args.params.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            params[k.strip()] = float(v)

    # ---- per (cell, seed): the post-hoc convergence point ----------------
    print(f"{'cell':22s} {'seed':>4} {'n_eval':>6} {'conv tokens':>13} "
          f"{'CE@conv':>8} {'CE@end':>8}  status")
    print("-" * 78)
    conv = defaultdict(list)     # cell -> [tokens at argmin]
    ce_conv = defaultdict(list)  # cell -> [CE at argmin]
    traces = {}
    any_unconverged = False
    for (cell, seed), pts in sorted(raw.items()):
        pts.sort()
        tokens = [t for t, _ in pts]
        ce = [c for _, c in pts]
        traces[(cell, seed)] = (tokens, ce)
        i, tk, v = argmin_point(tokens, ce)
        tail = i >= len(tokens) * (1 - NOT_CONVERGED_TAIL)
        any_unconverged |= tail
        conv[cell].append(tk)
        ce_conv[cell].append(v)
        print(f"{cell:22s} {seed:>4} {len(pts):>6} {tk:>13,} {v:>8.4f} {ce[-1]:>8.4f}  "
              f"{'NOT CONVERGED (argmin in final 20%)' if tail else 'ok'}")

    if any_unconverged:
        print("\n⚠️  PROTOCOL 4.5: at least one cell's argmin lies in the final 20% of "
              "its trace.\n    That cell is declared NOT CONVERGED; the ceiling must be "
              "doubled and the\n    extension reported as a documented deviation. The "
              "verdict below is PROVISIONAL.")

    # ---- the three slices (PROTOCOL 6.2) --------------------------------
    cells = sorted(conv)
    base = [c for c in cells if args.base_prefix and c.startswith(args.base_prefix)]
    big = [c for c in cells if args.big_prefix and c.startswith(args.big_prefix)]
    if not (base and big):
        print("\n(no --base-prefix/--big-prefix match; per-cell summary only)")
        for c in cells:
            m, sd = mean_sd(ce_conv[c])
            tm, _ = mean_sd([float(x) for x in conv[c]])
            print(f"  {c:22s} CE {m:.4f}+/-{sd:.4f}  at {tm:,.0f} tokens")
        return 0

    min_conv = min(min(conv[c]) for c in cells)   # slice (b) budget
    print(f"\nslice (b) equal-token budget = {min_conv:,} "
          "(minimum convergence-token count across all cells)")

    # Slice (c) needs a tokens-per-parameter ratio, and WHICH ratio is a free
    # parameter a reviewer will attack. Fix it conservatively and state it: the
    # largest ratio EVERY cell can actually reach inside the slice-(b) budget,
    # i.e. min_conv / max(params). Any other choice must be justified in the paper.
    # Slice (c) needs a tokens-per-parameter ratio, and WHICH ratio is a free
    # parameter a reviewer will attack. Two rules were considered:
    #   min_conv / max(params)  -- conservative on tokens, but it evaluates the
    #       SMALLER model far short of its own convergence, so the slice compares
    #       a converged Big against an unconverged Base. That is not a fairer
    #       comparison, only a differently unfair one.
    #   min over cells of (max trace tokens / params)  -- the largest ratio the
    #       released traces can actually support. This is what run-to-overshoot
    #       bought us, so use it.
    # Either way, equal tok/param with a 3.5x parameter ratio CANNOT put both
    # cells at their own convergence points unless their convergence budgets
    # happen to differ by the same 3.5x. That is a property of the alignment, not
    # a defect here, and it is why slice (c) is reported third. The distance from
    # each cell's own convergence is printed so it cannot be glossed over.
    ratio_c = None
    if params and all(c in params for c in cells):
        # MIN across seeds, not max: a ratio only one seed's trace can support is
        # not a budget every cell reaches, and max() would hide the short seed.
        max_tok = {c: min(max(tk) for (c2, _), (tk, _) in traces.items() if c2 == c)
                   for c in cells}
        ratio_c = min(max_tok[c] / params[c] for c in cells)
        print(f"slice (c) tokens/param ratio  = {ratio_c:,.1f} "
              "(largest ratio every cell's trace supports)")
        for c in cells:
            budget = ratio_c * params[c]
            conv_mean = sum(conv[c]) / len(conv[c])
            frac = budget / conv_mean
            note = ""
            if frac < 0.8:
                note = f"  <-- only {frac:.0%} of its own convergence point"
            elif frac > 1.5:
                note = f"  <-- {frac:.1f}x past its own convergence point"
            print(f"          {c:22s} budget {budget/1e9:6.2f}B  "
                  f"(convergence {conv_mean/1e9:.2f}B){note}")
        print("          NOTE: equal tok/param gives the LARGER model proportionally "
              "MORE tokens.\n          If Big wins only here, capacity and data "
              "quantity are not separable — say so.")

    def slice_values(cell, mode):
        """Per-seed CE values for one cell under one alignment, or None.

        None means "this cell cannot be evaluated at this alignment" -- either no
        params, or at least one seed's trace ends before the required budget.
        Returning None rather than a substituted endpoint is the point: a slice
        that silently compares cells at different budgets is worse than a missing
        slice, because it looks complete.
        """
        vals = []
        for (c, _), (tk, ce) in traces.items():
            if c != cell:
                continue
            if mode == "conv":
                vals.append(argmin_point(tk, ce)[2])
                continue
            if mode == "equal_tokens":
                target = min_conv
            else:
                if ratio_c is None or cell not in params:
                    return None
                target = ratio_c * params[cell]
            v, in_range = ce_at(tk, ce, target)
            if v is None or not in_range:
                return None
            vals.append(v)
        return vals or None

    print(f"\n{'slice':18s} {'Base CE':>16} {'Big CE':>16} {'Big-Base':>10} "
          f"{'2x within-sd':>13}  verdict")
    print("-" * 92)
    signs = {}
    for mode, label in (("conv", "(a) convergence"),
                        ("equal_tokens", "(b) equal tokens"),
                        ("equal_per_param", "(c) equal tok/param")):
        bg = {c: slice_values(c, mode) for c in base}
        gg = {c: slice_values(c, mode) for c in big}
        bv = [v for vals in bg.values() if vals for v in vals]
        gv = [v for vals in gg.values() if vals for v in vals]
        if not bv or not gv:
            miss = [c for c, v in list(bg.items()) + list(gg.items()) if not v]
            print(f"{label:18s} {'—':>16} {'—':>16} {'—':>10} {'—':>13}  "
                  f"NO DATA (cells short of budget or missing params: {', '.join(miss)})")
            continue
        bm, bsd = mean_sd(bv)
        gm, gsd = mean_sd(gv)
        diff = gm - bm                      # CE: lower is better -> negative = Big wins
        # PROTOCOL 6.3: the noise floor is the pooled WITHIN-CELL seed sd, pooled
        # over every cell on both sides -- not the sd of the concatenated values,
        # which would include the very between-cell spread under test.
        pooled = pooled_within_sd({**bg, **gg})
        thresh = NULL_SD_MULTIPLE * pooled
        if math.isnan(pooled):
            # No cell has >= 2 seeds. There is no variance estimate at all, so
            # neither "null" nor "effect" is supportable. Do NOT default to null:
            # that would let a single-seed run produce a headline.
            signs[label] = None
            verdict = "NO VARIANCE ESTIMATE (need >= 2 seeds/cell)"
        else:
            null = abs(diff) < thresh
            signs[label] = 0 if null else (1 if diff > 0 else -1)
            verdict = "NULL (PROTOCOL 6.3)" if null else (
                "Big WORSE" if diff > 0 else "Big BETTER")
        print(f"{label:18s} {bm:>9.4f}+/-{bsd:<6.4f} {gm:>9.4f}+/-{gsd:<6.4f} "
              f"{diff:>+10.4f} {thresh:>13.4f}  {verdict}")

    # ---- sign invariance (PROTOCOL 6.2, the headline sentence) ----------
    print()
    NAME = {0: "NULL", 1: "Big worse", -1: "Big better"}
    usable = {k: v for k, v in signs.items() if v is not None}
    unusable = [k for k, v in signs.items() if v is None]
    if unusable:
        print("SIGN INVARIANCE: excluded for lack of a variance estimate — "
              + ", ".join(unusable))
    if len(usable) < 2:
        print("SIGN INVARIANCE: not assessable — fewer than two alignments yielded "
              "a usable verdict.")
    elif len(set(usable.values())) == 1:
        word = NAME[next(iter(usable.values()))]
        print(f"SIGN INVARIANCE: the Big-minus-Base effect is INVARIANT across all "
              f"{len(usable)} usable alignments ({word}).")
        print("  -> The alignment debate is moot for this result, and the paper should "
              "say so\n     in the abstract. This is the strongest available position.")
    else:
        print("SIGN INVARIANCE: the effect INVERTS across alignments — "
              + ", ".join(f"{k}: {NAME[v]}" for k, v in usable.items()))
        print("  -> WHERE it inverts is a more interesting finding than the null itself.\n"
              "     PROTOCOL 6.2 requires stating this explicitly; frame the paper around it.")

    print("\nPer-seed values (PROTOCOL 6.3 forbids reporting means alone):")
    for c in cells:
        print(f"  {c:22s} CE@conv " + ", ".join(f"{v:.4f}" for v in ce_conv[c])
              + "   tokens " + ", ".join(f"{t/1e9:.2f}B" for t in conv[c]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:                                   # noqa: BLE001
        print(f"NO DECISION — {type(exc).__name__}: {exc}")
        sys.exit(2)
