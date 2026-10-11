#!/usr/bin/env python3
"""Per-row source labels (europarl / commoncrawl / un / news-commentary / giga-fren) for a
control set, by exact (source, target) match against the corrected pool.

e03_build_controls.py knows each set's source COUNTS but does not keep which row came from
where, and E0.5 has to hold the source fixed. This writes <set>.source next to the set: one
label per line, in the set's row order.

    e05_source_labels.py --pool-src train.clean.en --pool-tgt train.clean.fr \\
        --labels provenance_labels.npy --report provenance_report.json \\
        --set-dir <dir> --sets ft_topk,ft_random

FAIL-CLOSED: every row of every set must be found in the pool, or nothing is written. A pair
that occurs in the pool under more than one label takes the label with the lowest code and is
counted, so the choice is deterministic and visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

UNMATCHED = 255                      # e01_provenance.py's code for a row matched to no source


def key(a: str, b: str) -> bytes:
    return hashlib.blake2b((a + "\t" + b).encode("utf-8"), digest_size=10).digest()


def pairs(src: Path, tgt: Path):
    # LF is the only line break; strict: the two sides must have the same number of lines
    with open(src, encoding="utf-8", newline="\n") as f, open(tgt, encoding="utf-8", newline="\n") as g:
        for a, b in zip(f, g, strict=True):
            yield a.rstrip("\n"), b.rstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-src", required=True, type=Path)
    ap.add_argument("--pool-tgt", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path, help="provenance_labels.npy (one uint8 per pool row)")
    ap.add_argument("--report", required=True, type=Path, help="provenance_report.json (names the label codes)")
    ap.add_argument("--set-dir", required=True, type=Path)
    ap.add_argument("--sets", default="ft_topk,ft_random")
    ap.add_argument("--src-ext", default="en")
    ap.add_argument("--tgt-ext", default="fr")
    a = ap.parse_args()

    import numpy as np
    names = json.loads(a.report.read_text(encoding="utf-8"))["sources"]
    lab = np.load(a.labels, mmap_mode="r")
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    keys = {s: [key(x, y) for x, y in pairs(a.set_dir / f"{s}.{a.src_ext}", a.set_dir / f"{s}.{a.tgt_ext}")]
            for s in sets}
    want = set().union(*map(set, keys.values()))
    print(f"{sum(map(len, keys.values())):,} rows in {len(sets)} set(s), {len(want):,} distinct pairs", flush=True)

    found: dict[bytes, int] = {}
    clash: set[bytes] = set()
    n = 0
    for n, (x, y) in enumerate(pairs(a.pool_src, a.pool_tgt), 1):
        k = key(x, y)
        if k in want:
            code = int(lab[n - 1])
            old = found.get(k)
            if old is None:
                found[k] = code
            elif old != code:
                clash.add(k)
                found[k] = min(old, code)
    if n != len(lab):
        print(f"STOP: the pool has {n:,} rows but {a.labels} has {len(lab):,} labels")
        return 1
    missing = {s: sum(1 for k in ks if k not in found) for s, ks in keys.items()}
    if any(missing.values()):
        print(f"STOP: rows not found in the pool: {missing}. Nothing written.")
        return 1

    def name(code: int) -> str:
        return "unmatched" if code == UNMATCHED else names[code]

    out = {}
    for s, ks in keys.items():
        labels = [name(found[k]) for k in ks]
        (a.set_dir / f"{s}.source").write_text("".join(x + "\n" for x in labels), encoding="utf-8")
        counts = {nm: labels.count(nm) for nm in names + ["unmatched"]}
        out[s] = {"rows": len(ks), "counts": counts,
                  "pairs_with_more_than_one_label": sum(1 for k in ks if k in clash)}
        print(s, json.dumps(out[s]), flush=True)
    (a.set_dir / "source_labels.json").write_text(json.dumps(
        {"pool_rows": n, "label_names": names, "sets": out}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
