#!/usr/bin/env python3
"""Rebuild a WMT14 cleaned training corpus from the HF parquet shards, in two modes.

--mode legacy  reproduces the ORIGINAL pipeline byte for byte, bug included.
               download_wmt_*.py wrote each pair after strip() + replace('\\n', ' '),
               so carriage returns INSIDE a line survived into train.*;
               clean_data_*.py then read train.* in default text mode, where universal
               newlines turn each internal CR into a line break on that side, and zip()
               paired line i with line i+k from there on.
--mode fixed   identical, except an internal CR is replaced by a space, so every pair
               stays exactly one line on both sides.

Why two modes: if legacy output is byte-identical to the published corpus, this
emulation of the whole pipeline is exact, and fixed mode -- which differs only in CR
handling -- can be trusted. Filters, thresholds, duplicate counting and write order are
the original clean_data_enfr.py's, unchanged.

USAGE
    python phase0/rebuild_corpus.py --parquet-dir ~/mt_local/hf_wmt14/fr-en --src en --tgt fr \\
        --mode fixed --out-dir ~/mt_local/rebuild/v2_fixed
    # capped v1.1 = first 10M HF rows (download_wmt_enfr.py --max-train-samples 10000000)
    python phase0/rebuild_corpus.py ... --max-rows 10000000 --mode legacy --out-dir .../v11_legacy
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
import time
from collections import Counter, deque

_UNIVERSAL_NL = re.compile(r"\r\n|\r|\n")
_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ſ]")


def latin_ratio(s: str) -> float:
    """Same value as clean_data_enfr._latin_ratio: share of chars that are ASCII or
    U+00C0..U+017F. Counted with a C-level regex instead of a Python loop."""
    if not s:
        return 0.0
    return (len(s) - _NON_LATIN.subn("", s)[1]) / len(s)


def hf_rows(parquet_dir: str, src: str, tgt: str, max_rows: int = 0):
    """Raw (src, tgt) strings from the HF shards, in load_dataset order.
    max_rows applies to HF rows BEFORE empty pairs are skipped, as
    download_wmt_enfr.py's train_data.select(range(N)) does."""
    import pyarrow.parquet as pq                                  # lazy: tests need no pyarrow
    files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    if not files:
        sys.exit(f"no train-*.parquet under {parquet_dir}")
    n = 0
    for fn in files:
        for batch in pq.ParquetFile(fn).iter_batches(batch_size=100_000, columns=["translation"]):
            for t in batch.column(0).to_pylist():
                if max_rows and n >= max_rows:
                    return
                n += 1
                yield t[src], t[tgt]


def written_pairs(rows):
    """What download_wmt_*.py's _save_split writes, minus the newline."""
    for s, g in rows:
        s = s.strip().replace("\n", " ")
        g = g.strip().replace("\n", " ")
        if s and g:
            yield s, g


def as_lines(x: str, mode: str) -> list[str]:
    """The line(s) a reader sees for one written field x (x + '\\n' on disk)."""
    if mode == "fixed":
        return [x.replace("\r", " ") + "\n"]
    parts = _UNIVERSAL_NL.split(x + "\n")               # universal newlines, as open() does
    return [p + "\n" for p in parts[:-1]]


def _digest(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()


def rebuild(rows_factory, mode: str, out_src: str, out_tgt: str, dup_threshold=50,
            min_tokens=3, max_tokens=200, min_ratio=0.5, max_ratio=2.0, min_latin_ratio=0.9,
            log=print) -> dict:
    """rows_factory() must return a FRESH iterator of raw (src, tgt) strings; it is
    consumed twice, like the original's two passes over the files."""
    assert mode in ("legacy", "fixed")
    t0 = time.time()
    cnt: Counter = Counter()
    tgt_lines_total = 0
    for _, g in written_pairs(rows_factory()):
        for line in as_lines(g, mode):
            cnt[_digest(line)] += 1
            tgt_lines_total += 1
    bad = {k for k, c in cnt.items() if c > dup_threshold}
    del cnt
    log(f"  pass 1 ({mode}): {tgt_lines_total:,} target lines, blocking {len(bad):,} "
        f"duplicate lines ({time.time()-t0:.0f}s)")

    kept = dropped = 0
    reasons: Counter = Counter()
    hs, ht = hashlib.sha256(), hashlib.sha256()
    qs: deque = deque()
    qt: deque = deque()
    src_lines = 0
    with open(out_src, "w", encoding="utf-8", newline="\n") as os_, \
         open(out_tgt, "w", encoding="utf-8", newline="\n") as ot:
        for s_raw, g_raw in written_pairs(rows_factory()):
            ls, lt = as_lines(s_raw, mode), as_lines(g_raw, mode)
            src_lines += len(ls)
            qs.extend(ls)
            qt.extend(lt)
            while qs and qt:                                     # zip() semantics
                s, t = qs.popleft(), qt.popleft()
                s_stripped, t_stripped = s.strip(), t.strip()
                sl, tl = len(s_stripped.split()), len(t_stripped.split())
                if sl == 0 or tl == 0:
                    reasons["empty"] += 1; dropped += 1; continue
                if sl < min_tokens or tl < min_tokens:
                    reasons["too_short"] += 1; dropped += 1; continue
                if sl > max_tokens or tl > max_tokens:
                    reasons["too_long"] += 1; dropped += 1; continue
                r = sl / tl
                if r < min_ratio or r > max_ratio:
                    reasons["bad_ratio"] += 1; dropped += 1; continue
                if latin_ratio(s_stripped) < min_latin_ratio or latin_ratio(t_stripped) < min_latin_ratio:
                    reasons["non_latin"] += 1; dropped += 1; continue
                if _digest(t) in bad:
                    reasons["duplicate"] += 1; dropped += 1; continue
                os_.write(s); ot.write(t)
                hs.update(s.encode("utf-8")); ht.update(t.encode("utf-8"))
                kept += 1
    stats = {
        "mode": mode, "kept": kept, "dropped": dropped, "reasons": dict(reasons),
        "src_lines_seen": src_lines, "tgt_lines_seen": tgt_lines_total,
        "line_offset_src_minus_tgt": src_lines - tgt_lines_total,
        "unpaired_tail_lines": len(qs) + len(qt),
        "sha256_src": hs.hexdigest(), "sha256_tgt": ht.hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  pass 2 ({mode}): kept {kept:,}, dropped {dropped:,}, line offset "
        f"{stats['line_offset_src_minus_tgt']:+,} ({stats['seconds']}s)")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--src", default="en")
    ap.add_argument("--tgt", required=True, help="fr or de")
    ap.add_argument("--mode", choices=("legacy", "fixed"), required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-sha-src", default="")
    ap.add_argument("--expect-sha-tgt", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_src = os.path.join(a.out_dir, f"train.clean.{a.src}")
    out_tgt = os.path.join(a.out_dir, f"train.clean.{a.tgt}")
    st = rebuild(lambda: hf_rows(a.parquet_dir, a.src, a.tgt, a.max_rows), a.mode, out_src, out_tgt)
    import json
    json.dump(st, open(os.path.join(a.out_dir, "rebuild_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))
    rc = 0
    for side, want, got in (("src", a.expect_sha_src, st["sha256_src"]), ("tgt", a.expect_sha_tgt, st["sha256_tgt"])):
        if want:
            same = want == got
            print(f"{side}: {'IDENTICAL to expected' if same else 'DIFFERS from expected'} ({got[:16]} vs {want[:16]})")
            rc |= 0 if same else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
