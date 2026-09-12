#!/usr/bin/env python3
"""E0.3c — Apply the PRE-REGISTERED go/no-go rule to the E0.3 results.

This file is written BEFORE the runs. Its purpose is to stop us from doing to
ourselves what we did last time: staring at an outcome and constructing a story
that fits it. If the rule says NO-GO, the data-quality angle is dead and the
WMT 2027 program must rest on something else.

THE RULE (from phase0/README.md, fixed in advance)
--------------------------------------------------
Proceed to the full program only if, at the chosen flat-LR setting, BOTH hold:

  (1) ft_topk degrades the news-domain eval MORE than the matched ft_random
      control, by more than the seed-noise floor; and
  (2) ft_topk IMPROVES the held-out in-domain (UN / Europarl) eval.

(1) alone is consistent with "fine-tuning on any 1M pairs hurts". (2) is what
makes it a DOMAIN story rather than a damage story -- the paper asserted a
domain shift but only ever showed a training-loss curve, which cannot tell
"learned the fine-tuning distribution" apart from "broke".

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
import math
import sys
from collections import defaultdict

# two-sided t critical values at alpha=.05, by rounded Welch df
_T05 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def tcrit(df: float) -> float:
    if math.isnan(df):
        return float("inf")          # unusable df -> nothing can reach significance
    d = max(1, min(10, int(round(df))))
    return _T05[d] if df <= 10 else 1.96


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
    args = ap.parse_args()

    # condition -> testset -> [bleu per seed]
    r: dict = defaultdict(lambda: defaultdict(list))
    with open(args.results, encoding="utf-8") as f:
        for line in f:
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

    print(f"baseline {args.news} BLEU = {base_news:.2f}\n")
    print(f"{'condition':<11} {'testset':<20} {'n':>2} {'mean':>7} {'sd':>6} {'Δ vs base':>10}")
    print("-" * 62)
    for cond in ("ft_topk", "ft_random", "ft_bottom"):
        if cond not in r:
            continue
        for ts in [args.news] + indomain:
            if ts not in r[cond]:
                continue
            m, sd = mean_sd(r[cond][ts])
            b = r["baseline"].get(ts, [float("nan")])[0]
            print(f"{cond:<11} {ts:<20} {len(r[cond][ts]):>2} {m:>7.2f} "
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
        print("    too few seeds for a test")
        c1 = False
    print(f"    criterion 1: {'PASS' if c1 else 'FAIL'}")

    # ---- criterion 2 --------------------------------------------------
    print()
    have = [ts for ts in indomain if ts in r["ft_topk"] and ts in r["baseline"]]
    if not have:
        print("(2) in-domain: NO DATA — heldout sets were never built or never "
              "evaluated.\n    Run e01_provenance.py, then e03_build_controls.py. "
              "Without these the domain\n    claim is untestable and the gate CANNOT return GO.")
        c2 = False
    else:
        gains = []
        for ts in have:
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
