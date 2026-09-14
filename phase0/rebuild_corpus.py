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

"Differs only in CR handling" has one hidden dependency: the >dup_threshold target-line
blocklist is counted over the lines each mode SEES, and CR fragments are extra lines in
legacy mode. A fragment can push a line across the threshold in one mode only, which
would change which rows are dropped EVERYWHERE in the corpus, not just near the CRs.
Pass 1 therefore computes the blocked set for BOTH modes and exits 3 if they differ
(override with --allow-blocked-diff; both counts and the difference go into
rebuild_stats.json either way). Measured on the real corpora: 4,661 / 4,661 for fr-en and
126 / 126 for de-en, difference 0.

Output safety: the corpus is written to train.clean.<lang>.tmp, then hashed again FROM
THE BYTES ON DISK. Only if the on-disk sha256 equals the in-memory one and, when
--expect-sha-* is given, equals the expected value, are the files moved to
train.clean.<lang>. Otherwise both sides are kept as train.clean.<lang>.rejected and the
exit code is 1. An existing train.clean.<lang> from an earlier run is never overwritten
by a rejected run.

Only "\\n" ends a line. U+2028, U+2029, NEL (U+0085), VT, FF and FS/GS/RS occur inside
~870k fixed-v2 lines and must stay there: never read these corpora with str.splitlines()
or universal newlines (tests/suites/data.py enforces this for phase0/, phase1/, scripts/).

EXIT CODES
    0  outputs verified and moved into place
    1  on-disk hash != in-memory hash, or != --expect-sha-*; outputs kept as *.rejected
    3  legacy and fixed blocked-duplicate sets differ (no outputs written)

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
import json
import os
import re
import sys
import time
from collections import Counter, deque

_UNIVERSAL_NL = re.compile(r"\r\n|\r|\n")
_NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ſ]")
MODES = ("legacy", "fixed")


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


def sha256_file(path: str) -> str:
    """sha256 of the bytes on disk (not of anything held in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


class BlockedSetMismatch(RuntimeError):
    def __init__(self, check: dict):
        super().__init__(
            f"legacy and fixed modes block DIFFERENT duplicate target lines "
            f"(legacy {check['legacy']:,}, fixed {check['fixed']:,}, symmetric difference "
            f"{check['symmetric_difference']:,}). CR fragments pushed a line across the "
            f"duplicate threshold in one mode only, so 'fixed differs from legacy only in CR "
            f"handling' does NOT hold for this corpus. Re-run with --allow-blocked-diff only "
            f"if you accept that.")
        self.check = check


def blocked_sets(rows_factory, dup_threshold: int):
    """Pass 1 for BOTH modes in one scan. A written target field without a CR is the same
    single line in both modes, so it goes into one shared counter; only CR-bearing fields
    (a few hundred) get per-mode counters. Returns ({mode: blocked digest set},
    {mode: target lines seen}, check dict)."""
    common: Counter = Counter()
    extra = {m: Counter() for m in MODES}
    lines = {m: 0 for m in MODES}
    text_of: dict = {}
    for _, g in written_pairs(rows_factory()):
        if "\r" in g:
            for m in MODES:
                ls = as_lines(g, m)
                lines[m] += len(ls)
                for line in ls:
                    k = _digest(line)
                    extra[m][k] += 1
                    text_of.setdefault(k, line)
        else:
            common[_digest(g + "\n")] += 1       # == as_lines(g, mode) for both modes
            for m in MODES:
                lines[m] += 1
    touched = set(extra["legacy"]) | set(extra["fixed"])
    base = {k for k, c in common.items() if c > dup_threshold and k not in touched}
    bad = {m: base | {k for k in touched if common.get(k, 0) + extra[m].get(k, 0) > dup_threshold}
           for m in MODES}
    only_l, only_f = bad["legacy"] - bad["fixed"], bad["fixed"] - bad["legacy"]
    check = {
        "legacy": len(bad["legacy"]), "fixed": len(bad["fixed"]),
        "only_legacy": len(only_l), "only_fixed": len(only_f),
        "symmetric_difference": len(only_l) + len(only_f),
        "tgt_lines_legacy": lines["legacy"], "tgt_lines_fixed": lines["fixed"],
        "dup_threshold": dup_threshold,
        "examples": [{"only_in": "legacy" if k in only_l else "fixed",
                      "line": text_of.get(k, "").rstrip("\n")[:120],
                      "count": common.get(k, 0) + extra["legacy" if k in only_l else "fixed"].get(k, 0)}
                     for k in sorted(only_l | only_f)[:20]],
    }
    return bad, lines, check


def rebuild(rows_factory, mode: str, out_src: str, out_tgt: str, dup_threshold=50,
            min_tokens=3, max_tokens=200, min_ratio=0.5, max_ratio=2.0, min_latin_ratio=0.9,
            log=print, allow_blocked_diff=False) -> dict:
    """rows_factory() must return a FRESH iterator of raw (src, tgt) strings; it is
    consumed twice, like the original's two passes over the files.
    Raises BlockedSetMismatch (before writing anything) if the legacy and fixed
    blocked-duplicate sets differ and allow_blocked_diff is False."""
    assert mode in MODES
    t0 = time.time()
    bad_by_mode, lines, check = blocked_sets(rows_factory, dup_threshold)
    bad = bad_by_mode[mode]
    tgt_lines_total = lines[mode]
    del bad_by_mode
    log(f"  pass 1 ({mode}): {tgt_lines_total:,} target lines, blocking {len(bad):,} "
        f"duplicate lines ({time.time()-t0:.0f}s)")
    log(f"  blocked sets: legacy {check['legacy']:,}, fixed {check['fixed']:,}, "
        f"symmetric difference {check['symmetric_difference']:,}")
    if check["symmetric_difference"] and not allow_blocked_diff:
        raise BlockedSetMismatch(check)

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
        for f in (os_, ot):
            f.flush()
            os.fsync(f.fileno())
    stats = {
        "mode": mode, "kept": kept, "dropped": dropped, "reasons": dict(reasons),
        "src_lines_seen": src_lines, "tgt_lines_seen": tgt_lines_total,
        "line_offset_src_minus_tgt": src_lines - tgt_lines_total,
        "unpaired_tail_lines": len(qs) + len(qt),
        "sha256_src": hs.hexdigest(), "sha256_tgt": ht.hexdigest(),
        "blocked_check": check,
        "seconds": round(time.time() - t0, 1),
    }
    log(f"  pass 2 ({mode}): kept {kept:,}, dropped {dropped:,}, line offset "
        f"{stats['line_offset_src_minus_tgt']:+,} ({stats['seconds']}s)")
    return stats


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--src", default="en")
    ap.add_argument("--tgt", required=True, help="fr or de")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-sha-src", default="")
    ap.add_argument("--expect-sha-tgt", default="")
    ap.add_argument("--allow-blocked-diff", action="store_true",
                    help="proceed even if legacy and fixed modes block different duplicate lines")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    final = {"src": os.path.join(a.out_dir, f"train.clean.{a.src}"),
             "tgt": os.path.join(a.out_dir, f"train.clean.{a.tgt}")}
    tmp = {k: v + ".tmp" for k, v in final.items()}
    rej = {k: v + ".rejected" for k, v in final.items()}
    stats_path = os.path.join(a.out_dir, "rebuild_stats.json")
    for p in tmp.values():
        if os.path.exists(p):
            os.remove(p)                                   # stale partial output of a crashed run

    try:
        st = rebuild(lambda: hf_rows(a.parquet_dir, a.src, a.tgt, a.max_rows), a.mode,
                     tmp["src"], tmp["tgt"], allow_blocked_diff=a.allow_blocked_diff)
    except BlockedSetMismatch as e:
        _write_json(stats_path, {"mode": a.mode, "outcome": "aborted_blocked_set_mismatch",
                                 "blocked_check": e.check})
        print(f"ERROR: {e}", file=sys.stderr)
        for ex in e.check["examples"]:
            print(f"  only blocked in {ex['only_in']}: count {ex['count']}: {ex['line']!r}", file=sys.stderr)
        print(f"no corpus written; details in {stats_path}", file=sys.stderr)
        return 3

    rc = 0
    problems = []
    for side in ("src", "tgt"):
        disk = sha256_file(tmp[side])
        mem = st[f"sha256_{side}"]
        want = getattr(a, f"expect_sha_{side}")
        st[f"sha256_{side}_disk"] = disk
        st[f"expect_sha256_{side}"] = want or None
        print(f"{side}: sha256 in-memory {mem}")
        print(f"{side}: sha256 on-disk   {disk}  ({tmp[side]})")
        if disk != mem:
            problems.append(f"{side}: on-disk sha256 differs from the in-memory sha256")
        if want:
            same = want == disk
            print(f"{side}: on-disk {'IDENTICAL to expected' if same else 'DIFFERS from expected'} "
                  f"({disk[:16]} vs {want[:16]})")
            if not same:
                problems.append(f"{side}: on-disk sha256 differs from --expect-sha-{side}")

    if problems:
        rc = 1
        for side in ("src", "tgt"):
            os.replace(tmp[side], rej[side])
        st["outcome"] = "rejected"
        st["problems"] = problems
        for p in problems:
            print(f"REJECTED: {p}", file=sys.stderr)
        print(f"outputs kept as {rej['src']} and {rej['tgt']}", file=sys.stderr)
        for side in ("src", "tgt"):
            if os.path.exists(final[side]):
                print(f"NOTE: {final[side]} exists from an EARLIER run and was left untouched; "
                      f"it is not this run's output", file=sys.stderr)
    else:
        for side in ("src", "tgt"):
            if os.path.exists(rej[side]):
                os.remove(rej[side])                       # stale rejection from an earlier run
            os.replace(tmp[side], final[side])
        st["outcome"] = "accepted"
    _write_json(stats_path, st)
    print(json.dumps(st, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
