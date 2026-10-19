#!/usr/bin/env python3
"""Reconcile the awk recount (logs/awk_ratio.log) with results.json's ratio_ws counts: recount
with fields split on ASCII blanks only (what LC_ALL=C awk does), and count the rows whose
str.split() and blank-only token counts differ, by the character that makes the difference."""
import json, re, sys
from collections import Counter
from pathlib import Path
D = Path.home() / "e06data" / "phase0_e05"
BLANK = re.compile(r"[ \t]+")
res = json.loads((Path.home() / "cpu_items/item3_contamination_check/results.json").read_text())
out = {}
for s in ("ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div", "ft_random_small", "heldout_un"):
    en = (D / f"{s}.en").read_bytes()[:-1].decode().split("\n")
    fr = (D / f"{s}.fr").read_bytes()[:-1].decode().split("\n")
    c = Counter(); why = Counter()
    for e, f in zip(en, fr):
        ne, nf = len([t for t in BLANK.split(e.strip(" \t")) if t]), len([t for t in BLANK.split(f.strip(" \t")) if t])
        for num, key in ((13, "1.3"), (15, "1.5"), (20, "2.0")):
            if ne == 0 or nf == 0 or 10 * nf > num * ne or 10 * ne > num * nf:
                c[f"blank_{key}"] += 1
        for side, x in (("en", e), ("fr", f)):
            if len(x.split()) != len([t for t in BLANK.split(x.strip(" \t")) if t]):
                c[f"rows_split_differs_{side}"] += 1
                for ch in set(x):
                    if ch.isspace() and ch not in " \t":
                        why[f"{side}:U+{ord(ch):04X}"] += 1
    py = res["sets"][s]["counts"]
    out[s] = {"blank_only": dict(c), "str_split": {k: py[f"ratio_ws_{k}"] for k in ("1.3", "1.5", "2.0")},
              "whitespace_chars_behind_differences": dict(why.most_common(8))}
    print(s, json.dumps(out[s]), flush=True)
(Path.home() / "cpu_items/item3_contamination_check/split_reconcile.json").write_text(json.dumps(out, indent=1))
