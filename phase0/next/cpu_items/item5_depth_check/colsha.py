#!/usr/bin/env python3
"""sha256 of v2_scored.tsv's source and target columns re-joined as LF-terminated files, to tie
TSV row i to row i of the fixed v2 corpus (the rows prov_v2_fixed labels). Expected values are
the v2 fixed rebuild's (~/mt_local/rebuild/v2_fixed.log sha256_src / sha256_tgt)."""
import hashlib, json
from pathlib import Path
tsv = Path.home() / "e06data" / "v2_scored.tsv"
en, fr = hashlib.sha256(), hashlib.sha256()
n = odd = 0
with open(tsv, "rb", buffering=1 << 24) as f:
    for line in f:
        body = line[:-1] if line.endswith(b"\n") else line
        p = body.split(b"\t")
        odd += len(p) != 3
        en.update(p[1] + b"\n"); fr.update(p[2] + b"\n")
        n += 1
exp = {"en": "c8cc665c8fc9971bf689bb02d7409bc76bbb489cb7b7dab6b35d13778863e655",
       "fr": "e4f5301af4c49b8cd91ff81807db08116a7da94f952ba0ee3962e82c0618421a"}
out = {"rows": n, "rows_not_3_fields": odd, "en": en.hexdigest(), "fr": fr.hexdigest(),
       "en_equal": en.hexdigest() == exp["en"], "fr_equal": fr.hexdigest() == exp["fr"]}
print(json.dumps(out))
(Path.home() / "cpu_items" / "item5_depth_check" / "colsha.json").write_text(json.dumps(out, indent=1))
