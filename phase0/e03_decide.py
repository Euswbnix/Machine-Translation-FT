#!/usr/bin/env python3
"""E0.3c — Apply the PRE-REGISTERED go/no-go rule to the E0.3 results.

This file is written BEFORE the runs. Its purpose is to stop us from doing to
ourselves what we did last time: staring at an outcome and constructing a story
that fits it. If the rule says NO-GO, the data-quality angle is dead and the
WMT 2027 program must rest on something else.

THE RULE (from phase0/README.md, fixed in advance)
--------------------------------------------------
Proceed to the full program only if, at the lr_scale selected from stage 1 by
phase0/e03_select_lr.py under the rule frozen in phase0/e03_decisions.json
("lr_selection_rule"), BOTH hold:

  (1) ft_topk degrades the news-domain eval MORE than the matched ft_random
      control, by more than the seed-noise floor; and
  (2) ft_topk IMPROVES the held-out in-domain (UN / Europarl) eval.

(1) alone is consistent with "fine-tuning on any 1M pairs hurts". (2) is what
makes it a DOMAIN story rather than a damage story -- the paper asserted a
domain shift but only ever showed a training-loss curve, which cannot tell
"learned the fine-tuning distribution" apart from "broke".

FAIL-CLOSED: every set named in --indomain must have a baseline row AND ft_topk
rows, the news set must have baseline, ft_topk and ft_random rows, and every
ft_topk/ft_random cell used by the rule must have exactly --expect-seeds values;
otherwise the script exits 2 (cannot decide). If ft_topk AND ft_random both have zero
seed variance on the news set, the noise floor and Welch test are undefined, which also
exits 2 (audit F8; before, this printed "too few seeds" and exited 1 = NO-GO). Exit 1 only ever means the rule was
applied to complete data and said no. It used to judge criterion 2
on whichever sets happened to be present, so what the gate tested depended on
which rows made it into the TSV.

PENDING USER DECISION (audit F11, not resolved here): phase0/README.md item 2
and WMT2027_PLAN.md say criterion 2 is "improves the UN/legislative eval"
(UN only), while this code, with the default --indomain
heldout_un,heldout_europarl, requires a gain on EVERY listed set. Options:
(a) UN-only gate: pass/default --indomain heldout_un, Europarl reported only;
(b) both sets: amend the README and plan to say "improves heldout_un AND
heldout_europarl". The default here is left unchanged until the user freezes
one wording (before any stage-2 result exists). Test sets present in the TSV
but not in --indomain are printed as [aux] rows and never affect the verdict.

CHECKPOINT TYPES: e03_collect.py writes <results stem>.meta.json with ft_ckpt and
baseline_kind (a hand-built TSV may instead carry a '# ft_ckpt=... baseline_kind=...' line).
When the FT rows are single final checkpoints and the baseline is the averaged release
(the default, and what a header-less TSV is assumed to be), a WARNING is printed: the
paper puts averaging at +0.2-0.4 BLEU, which biases criterion 1's baseline check toward
'degrades' and criterion 2 against 'improves'. The pass/fail logic is unchanged.

A useful auxiliary, not part of the gate: if ft_bottom degrades LESS than
ft_topk, QE score is not monotonically driving the effect, and the "quality
filtering causes domain narrowing" framing is in trouble regardless.

INPUT
-----
TSV, one row per (condition, seed, testset):

    condition   seed   testset            bleu
    baseline    -      newstest2014       38.21
    ft_topk     42     newstest2014       36.67
    ft_topk     42     heldout_un         41.02
    ...

`baseline` is the pre-fine-tuning checkpoint; its seed column is ignored.

USAGE
-----
    python e03_decide.py --results results/phase0_bleu.tsv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# two-sided t critical values at alpha=.05, by FLOORED Welch df. t_.975 decreases in df,
# so t(floor(df)) >= t(df): conservative. Rounding (the old code) was anti-conservative
# for fractional df just below the next integer (df 2.6 used t(3)=3.182; exact ~3.48).
_T05 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
        28: 2.048, 29: 2.045, 30: 2.042}


def tcrit(df: float) -> float:
    if math.isnan(df) or df < 1:
        return float("inf")          # unusable df -> nothing can reach significance
    if df <= 30:
        return _T05[int(math.floor(df))]
    return 2.042 if df <= 120 else 1.96     # t(30) is conservative up to df 120


def welch(a: list[float], b: list[float]):
    """Return (t, df, crit) for a two-sample Welch test. None if under-powered."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se2 = va / na + vb / nb
    if se2 == 0:
        return None
    t = (ma - mb) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, tcrit(df)


def mean_sd(xs: list[float]):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--news", default="newstest2014")
    ap.add_argument("--indomain", default="heldout_un,heldout_europarl")
    ap.add_argument("--expect-seeds", type=int, default=3,
                    help="seeds per ft_topk/ft_random cell the rule uses; any other count -> exit 2")
    args = ap.parse_args()

    # condition -> testset -> [bleu per seed]
    r: dict = defaultdict(lambda: defaultdict(list))
    header: dict = {}
    budget_warnings: list = []
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                toks = line[1:].split()
                if toks and toks[0] == "budget_warning":
                    budget_warnings.append(" ".join(toks[1:]))
                elif toks and "=" in toks[0]:
                    header.update(dict(x.split("=", 1) for x in toks if "=" in x))
                continue
            p = line.split()
            if len(p) < 4 or p[0] in ("condition", "#"):
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

    missing = [c for c in ("baseline", "ft_topk", "ft_random") if c not in r]
    if missing:
        print(f"NO DECISION — missing conditions: {', '.join(missing)}")
        return 2
    if args.news not in r["baseline"]:
        print(f"NO DECISION — no baseline score on {args.news}")
        return 2

    base_news = r["baseline"][args.news][0]
    indomain = [t.strip() for t in args.indomain.split(",") if t.strip()]
    if not indomain:
        print("NO DECISION — --indomain is empty; criterion 2 has nothing to test")
        return 2
    lacking = [f"{ts} (no {' or '.join(c for c in ('baseline', 'ft_topk') if ts not in r[c])} rows)"
               for ts in indomain if ts not in r["baseline"] or ts not in r["ft_topk"]]
    if lacking:
        print("NO DECISION — in-domain gate set(s) missing from the results: " + "; ".join(lacking))
        print("  Every set named in --indomain must be evaluated for the baseline and ft_topk.")
        print("  Heldout sets never built or never evaluated: run e01_provenance.py and")
        print("  e03_build_controls.py, then e03_collect.py. The gate CANNOT return GO without them.")
        return 2
    short = []
    for cond, sets in (("ft_topk", [args.news] + indomain), ("ft_random", [args.news])):
        for ts in sets:
            n = len(r[cond].get(ts, []))
            if n != args.expect_seeds:
                short.append(f"{cond}/{ts}: {n} seed value(s), expected {args.expect_seeds}")
    if short:
        print("NO DECISION — incomplete cells: " + "; ".join(short))
        return 2
    zero = [f"{c}/{args.news}" for c in ("ft_topk", "ft_random") if len(set(r[c][args.news])) == 1]
    if len(zero) == 2:
        print(f"NO DECISION — zero seed variance in {' and '.join(zero)}: the seed-noise floor and the Welch")
        print("  test are undefined; the seeds did not vary the outcome. Check training determinism")
        print("  (train.py set_seed / --seed) and that e03_collect scored distinct checkpoints.")
        return 2
    aux = sorted({ts for c in r for ts in r[c]} - set([args.news] + indomain))

    mp = Path(args.results).with_suffix(".meta.json")
    if mp.is_file():
        meta = json.load(open(mp, encoding="utf-8"))
        header.update({k: str(meta[k]) for k in ("ft_ckpt", "baseline_kind") if k in meta})
        budget_warnings.extend(meta.get("budget_warnings") or [])
        print(f"collection: lr_scale {meta.get('lr_scale')}, stage-1 selected {meta.get('selected_lr')}, "
              f"decisions sha256 {meta.get('decisions_sha')}")
        if meta.get("lr_deviation"):
            print("  WARNING: this lr_scale is NOT the one stage 1 selected (a recorded deviation, "
                  "results/phase0/deviations.txt)")
    ft_kind = header.get("ft_ckpt", "final")
    base_kind = header.get("baseline_kind", "averaged")
    print(f"checkpoint types: fine-tuned = {ft_kind}, baseline = {base_kind}"
          + ("" if header else f"  (no {mp.name} and no header: legacy TSV assumed final vs averaged)"))
    if not (ft_kind == "avg-last5" and base_kind == "averaged") and ft_kind != base_kind:
        print("  WARNING: fine-tuned and baseline checkpoints are of different kinds. Averaging is")
        print("  worth ~+0.2-0.4 BLEU (paper 05_sft.tex), so criterion 1's baseline check is biased")
        print("  toward 'degrades' and criterion 2 against 'improves'. The rule itself is unchanged.")
    for w in budget_warnings:
        print(f"  WARNING (budget): {w}")

    print(f"baseline {args.news} BLEU = {base_news:.2f}\n")
    print(f"{'condition':<11} {'testset':<20} {'n':>2} {'mean':>7} {'sd':>6} {'Δ vs base':>10}")
    print("-" * 62)
    for cond in ("ft_topk", "ft_random", "ft_bottom"):
        if cond not in r:
            continue
        for ts in [args.news] + indomain + aux:
            if ts not in r[cond]:
                continue
            m, sd = mean_sd(r[cond][ts])
            b = r["baseline"].get(ts, [float("nan")])[0]
            label = f"[aux] {ts}" if ts in aux else ts
            print(f"{cond:<11} {label:<20} {len(r[cond][ts]):>2} {m:>7.2f} "
                  f"{sd:>6.2f} {m - b:>+10.2f}")

    # ---- criterion 1 --------------------------------------------------
    print("\n" + "=" * 62)
    topk = r["ft_topk"].get(args.news, [])
    rand = r["ft_random"].get(args.news, [])
    m_t, sd_t = mean_sd(topk)
    m_r, sd_r = mean_sd(rand)
    gap = m_r - m_t          # positive = topk is worse, as the paper predicts
    noise = max(sd_t, sd_r) if not math.isnan(sd_t) and not math.isnan(sd_r) else float("nan")

    print(f"(1) news-domain: ft_topk {m_t:.2f} vs ft_random {m_r:.2f}")
    print(f"    ft_topk is worse by {gap:+.2f} BLEU; seed-noise floor (max sd) = {noise:.2f}")
    w = welch(rand, topk)
    if w:
        t, df, crit = w
        print(f"    Welch t={t:.2f}, df={df:.1f}, |t| vs crit(.05)={crit:.2f} -> "
              f"{'significant' if abs(t) > crit else 'NOT significant'}")
        print("    ⚠️  n=3 per group: this test has low power. A null result here is "
              "weak evidence\n        of no effect, but the gate requires a POSITIVE result, "
              "so that asymmetry is safe.")
        # A gap vs the random control is NOT enough. If fine-tuning IMPROVED
        # news BLEU under both conditions, ft_topk being less good than
        # ft_random still satisfies every test above — and the paper's claim is
        # that top-k QE filtering DEGRADES news performance. Require it.
        degraded = (base_news - m_t) > noise
        print(f"    ft_topk vs BASELINE: {m_t - base_news:+.2f} BLEU -> "
              f"{'degrades beyond the noise floor' if degraded else 'does NOT degrade'}")
        c1 = degraded and gap > noise and abs(t) > crit and t > 0
    else:
        # unreachable after the completeness and zero-variance checks; never let an undefined
        # test turn into exit 1 (NO-GO)
        print("NO DECISION — Welch test undefined for the news-domain cells")
        return 2
    print(f"    criterion 1: {'PASS' if c1 else 'FAIL'}")

    # ---- criterion 2 --------------------------------------------------
    print()
    gains = []
    for ts in indomain:                      # completeness was checked above (exit 2)
        m, _ = mean_sd(r["ft_topk"][ts])
        d = m - r["baseline"][ts][0]
        gains.append(d)
        print(f"(2) in-domain {ts}: ft_topk Δ = {d:+.2f} BLEU")
    c2 = all(g > 0 for g in gains)
    if not c2:
        print("    ft_topk does NOT improve every in-domain set. The fine-tuning")
        print("    did not buy in-domain competence; it only cost news competence.")
    print(f"    criterion 2: {'PASS' if c2 else 'FAIL'}")

    # ---- auxiliary ----------------------------------------------------
    if "ft_bottom" in r and args.news in r["ft_bottom"]:
        m_b, _ = mean_sd(r["ft_bottom"][args.news])
        print(f"\n[aux] ft_bottom {args.news} = {m_b:.2f} vs ft_topk {m_t:.2f}")
        if m_b > m_t:
            print("      LOW-QE data hurts LESS than high-QE data. QE score is not")
            print("      monotonically driving the effect — the quality-filtering framing")
            print("      is in trouble even if the gate passes.")

    # ---- verdict ------------------------------------------------------
    print("\n" + "=" * 62)
    if c1 and c2:
        print("VERDICT: GO — both pre-registered criteria pass. The domain-shift")
        print("account survives its first real control. Proceed to Phase 1.")
        return 0
    print("VERDICT: NO-GO — the pre-registered rule is not met.")
    print("Per the pre-registration, the data-quality/domain angle does NOT carry")
    print("the WMT 2027 submission. Do not rescue it with a post-hoc reframing;")
    print("that is exactly the failure mode this file exists to prevent.")
    return 1


if __name__ == "__main__":
    # Exit 1 MUST mean "the rule was applied and it said no". A crash exiting 1
    # would be silently read as a NO-GO decision by anyone following RUNBOOK.md.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:                                    # noqa: BLE001
        print(f"NO DECISION — {type(exc).__name__}: {exc}")
        sys.exit(2)
