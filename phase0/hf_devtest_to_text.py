#!/usr/bin/env python3
"""Pinned WMT14 dev/test parquet -> valid.* / test.* text, byte-checked.

Replaces `rental_setup.sh accept`'s old call to download_wmt_enfr.py, which ran an
UNPINNED load_dataset("wmt14", "fr-en") and wrote the whole 40.8M-pair raw train split
only to copy valid/test out of it (audit F8, training-path finding "accept pinning").

The transform is download_wmt_enfr.py's _save_split, reused from rebuild_corpus.py
(written_pairs: strip, "\\n" -> " ", skip a pair if either side is empty), written with
newline="\\n". Output goes to <name>.tmp, is hashed ON DISK, and is renamed into place
only if the parquet sha256, the line count and both text sha256 values equal the
expected ones; otherwise both sides are left as <name>.rejected and nothing is replaced.

    python phase0/hf_devtest_to_text.py --parquet fr-en/test-00000-of-00001.parquet \\
        --out-src data_enfr_v1/test.en --out-tgt data_enfr_v1/test.fr \\
        --expect-parquet-sha 8b9f... --expect-sha-src 967d... --expect-sha-tgt 8252... --expect-lines 3003

Exit: 0 verified; 1 a hash or line count differs (nothing replaced); 2 bad input.
pyarrow is imported lazily so convert() is testable without it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_corpus import written_pairs  # noqa: E402  -- the one copy of the transform


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_rows(path: str, src: str, tgt: str):
    import pyarrow.parquet as pq                       # lazy
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=["translation"]):
        for t in batch.column(0).to_pylist():
            yield t[src], t[tgt]


def convert(rows, out_src: str, out_tgt: str, expect_sha_src="", expect_sha_tgt="", expect_lines=0,
            log=print) -> int:
    tmp_s, tmp_t = out_src + ".tmp", out_tgt + ".tmp"
    n = 0
    with open(tmp_s, "w", encoding="utf-8", newline="\n") as fs, \
         open(tmp_t, "w", encoding="utf-8", newline="\n") as ft:
        for s, t in written_pairs(rows):
            fs.write(s + "\n")
            ft.write(t + "\n")
            n += 1
        for f in (fs, ft):
            f.flush()
            os.fsync(f.fileno())
    got_s, got_t = sha256_file(tmp_s), sha256_file(tmp_t)
    problems = []
    if expect_lines and n != expect_lines:
        problems.append(f"{n} pairs written, expected {expect_lines}")
    if expect_sha_src and got_s != expect_sha_src:
        problems.append(f"{out_src}: sha256 {got_s} != expected {expect_sha_src}")
    if expect_sha_tgt and got_t != expect_sha_tgt:
        problems.append(f"{out_tgt}: sha256 {got_t} != expected {expect_sha_tgt}")
    if problems:
        os.replace(tmp_s, out_src + ".rejected")
        os.replace(tmp_t, out_tgt + ".rejected")
        for p in problems:
            log(f"REJECTED: {p}")
        return 1
    os.replace(tmp_s, out_src)
    os.replace(tmp_t, out_tgt)
    log(f"ok: {n} pairs -> {out_src} ({got_s[:12]}...), {out_tgt} ({got_t[:12]}...)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--src", default="en")
    ap.add_argument("--tgt", default="fr")
    ap.add_argument("--out-src", required=True)
    ap.add_argument("--out-tgt", required=True)
    ap.add_argument("--expect-parquet-sha", default="")
    ap.add_argument("--expect-sha-src", default="")
    ap.add_argument("--expect-sha-tgt", default="")
    ap.add_argument("--expect-lines", type=int, default=0)
    a = ap.parse_args(argv)
    if not os.path.isfile(a.parquet):
        print(f"{a.parquet} missing", file=sys.stderr)
        return 2
    if a.expect_parquet_sha:
        got = sha256_file(a.parquet)
        if got != a.expect_parquet_sha:
            print(f"REJECTED: {a.parquet} sha256 {got} != pinned {a.expect_parquet_sha}", file=sys.stderr)
            return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out_src)), exist_ok=True)
    return convert(parquet_rows(a.parquet, a.src, a.tgt), a.out_src, a.out_tgt,
                   a.expect_sha_src, a.expect_sha_tgt, a.expect_lines)


if __name__ == "__main__":
    sys.exit(main())
