#!/usr/bin/env python3
"""Compare two provenance runs, e.g. `--norm exact` against `--norm strip`.

A looser normalizer always matches at least as many rows, so "the match rate went
up" is not evidence it is right. It can recover lines that differ only in
whitespace, or it can merge lines that were genuinely different -- and a merge
shows up as a row that was ALREADY matched changing its source label. This reports
the two apart:

  newly matched   unmatched in A, matched in B        (recovery; broken down by source)
  lost            matched in A, unmatched in B        (should be 0 for a looser normalizer)
  relabelled      matched in both, different source   (merging distinct lines)

Verdict rule, stated so it can be argued with: ADOPT B only if nothing was lost and
relabelled rows are under 1% of newly matched rows; otherwise REVIEW.

USAGE
    python phase0/e01_compare_labels.py --a phase0/provenance_exact --b phase0/provenance_strip
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

UNMATCHED = 255


def load(prefix: str):
    labels = np.load(f"{prefix}_labels.npy")
    report = json.load(open(f"{prefix}_report.json", encoding="utf-8"))
    return labels, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="prefix of the stricter run")
    ap.add_argument("--b", required=True, help="prefix of the looser run")
    ap.add_argument("--max-relabel-frac", type=float, default=0.01)
    args = ap.parse_args()

    la, ra = load(args.a)
    lb, rb = load(args.b)
    if len(la) != len(lb):
        sys.exit(f"row counts differ ({len(la):,} vs {len(lb):,}); these are not runs over the same corpus")
    if ra["sources"] != rb["sources"]:
        sys.exit(f"source order differs: {ra['sources']} vs {rb['sources']}; label codes are not comparable")
    names = ra["sources"]
    n = len(la)

    ma, mb = la != UNMATCHED, lb != UNMATCHED
    newly = ~ma & mb
    lost = ma & ~mb
    relab = ma & mb & (la != lb)

    print(f"rows {n:,} | A ({ra.get('normalizer')}) matched {ma.sum():,} ({ma.mean():.2%}) | "
          f"B ({rb.get('normalizer')}) matched {mb.sum():,} ({mb.mean():.2%})")
    print(f"newly matched: {newly.sum():,}")
    for i, name in enumerate(names):
        c = int((lb[newly] == i).sum())
        if c:
            print(f"    {name:18s} {c:>12,}")
    print(f"lost:          {lost.sum():,}")
    print(f"relabelled:    {relab.sum():,}")
    if relab.any():
        pairs = {}
        for x, y in zip(la[relab], lb[relab]):
            pairs[(int(x), int(y))] = pairs.get((int(x), int(y)), 0) + 1
        for (x, y), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {names[x]:>16s} -> {names[y]:<16s} {c:>10,}")

    frac = relab.sum() / max(newly.sum(), 1)
    ok = lost.sum() == 0 and frac < args.max_relabel_frac
    print(f"\nrelabelled / newly matched = {frac:.3%} (threshold {args.max_relabel_frac:.0%}); lost = {lost.sum():,}")
    print("VERDICT:", "ADOPT B" if ok else "REVIEW — do not adopt B on its match rate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
