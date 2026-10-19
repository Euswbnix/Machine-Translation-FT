#!/usr/bin/env python3
"""Describe the English-side wrong-language rows of ft_un_rep and ft_un_div: label, confidence,
whether the English line repeats inside the set, and 8 examples per set (seed 42)."""
import json, random
from collections import Counter
from pathlib import Path
W = Path.home() / "cpu_items/item3_contamination_check/work"
D = Path.home() / "e06data/phase0_e05"
lab = {}
for k in range(3):
    ins = (W / f"lid_in_{k}.txt").read_bytes()[:-1].decode().split("\n")
    outs = (W / f"lid_out_{k}.tsv").read_text().rstrip("\n").split("\n")
    assert len(ins) == len(outs)
    for s, o in zip(ins, outs):
        c = o.split("\t")
        lab[s] = (c[0], float(c[2]))
res = {}
rng = random.Random(42)
for s in ("ft_un_rep", "ft_un_div", "ft_un_random"):
    en = (D / f"{s}.en").read_bytes()[:-1].decode().split("\n")
    cnt = Counter(en)
    rows = [i for i, x in enumerate(en) if lab[x][0] != "en"]
    conf = sum(1 for i in rows if lab[en[i]][1] >= 0.9)
    dup = sum(1 for i in rows if cnt[en[i]] > 1)
    uniq = len(set(en[i] for i in rows))
    ex = [{"line": i + 1, "label": lab[en[i]][0], "p": round(lab[en[i]][1], 3), "copies_in_set": cnt[en[i]],
           "en": en[i][:200]} for i in sorted(rng.sample(rows, min(8, len(rows))))]
    res[s] = {"en_wrong_rows": len(rows), "distinct_lines": uniq, "rows_whose_line_repeats_in_set": dup,
              "conf_ge_0.9": conf, "examples": ex}
    print(s, json.dumps({k: v for k, v in res[s].items() if k != "examples"}))
    for e in ex:
        print("   ", e)
(Path.home() / "cpu_items/item3_contamination_check/peek_en_wrong.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
