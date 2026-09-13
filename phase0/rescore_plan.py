#!/usr/bin/env python3
"""Re-score only what changed after rebuilding a corpus.

A CometKiwi score is a function of the pair's text alone, not of its row, so every
pair of the rebuilt corpus that already appears in the old scored TSV keeps its old
score, and only genuinely new pairs need the GPU. For v2 that is roughly the ~13.5M
misaligned rows instead of all 30.1M (about 6 GPU-hours instead of 14).

  plan   old scored TSV + rebuilt corpus -> reuse_scores.npy (NaN where missing),
         missing_rows.npy, to_score.<src>/<tgt> (the pairs to send to score_with_comet.py)
  merge  plan dir + the scorer's TSV for to_score.* -> <out> scored TSV in rebuilt row order

Matching mirrors score_with_comet.py, which wrote src/tgt with tabs replaced by spaces:
the lookup key is blake2b-64 over (src, tgt) after the same replacement. At ~60M keys
the expected number of 64-bit collisions is ~1e-4, and a collision could only hand one
pair the score of another -- negligible, and stated rather than hidden.

USAGE
    python phase0/rescore_plan.py plan --old-scored v2_scored.tsv \\
        --new-src v2_fixed/train.clean.en --new-tgt v2_fixed/train.clean.fr --out-dir v2_rescore
    python score_with_comet.py --src v2_rescore/to_score.en --tgt v2_rescore/to_score.fr --out v2_rescore/new_scores.tsv
    python phase0/rescore_plan.py merge --plan-dir v2_rescore --new-scores v2_rescore/new_scores.tsv \\
        --new-src v2_fixed/train.clean.en --new-tgt v2_fixed/train.clean.fr --out v2_fixed_scored.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np


def key(src: str, tgt: str) -> int:
    s = src.rstrip("\n").replace("\t", " ")
    t = tgt.rstrip("\n").replace("\t", " ")
    return int.from_bytes(hashlib.blake2b(f"{s}\x00{t}".encode("utf-8"), digest_size=8).digest(), "little")


def cmd_plan(a) -> int:
    old_keys, old_scores = [], []
    with open(a.old_scored, encoding="utf-8", newline="\n") as f:
        for line in f:
            sc, s, t = line.rstrip("\n").split("\t", 2)
            old_keys.append(key(s, t))
            old_scores.append(float(sc))
    K = np.asarray(old_keys, dtype=np.uint64)
    S = np.asarray(old_scores, dtype=np.float32)
    del old_keys, old_scores
    order = np.argsort(K, kind="stable")
    K, S = K[order], S[order]
    print(f"old scored pairs: {len(K):,}")

    os.makedirs(a.out_dir, exist_ok=True)
    reuse, missing = [], []
    src_ext = a.new_src.rsplit(".", 1)[-1]
    tgt_ext = a.new_tgt.rsplit(".", 1)[-1]
    with open(a.new_src, encoding="utf-8", newline="\n") as fs, open(a.new_tgt, encoding="utf-8", newline="\n") as ft, \
         open(os.path.join(a.out_dir, f"to_score.{src_ext}"), "w", encoding="utf-8", newline="\n") as os_, \
         open(os.path.join(a.out_dir, f"to_score.{tgt_ext}"), "w", encoding="utf-8", newline="\n") as ot:
        for i, (s, t) in enumerate(zip(fs, ft)):
            k = np.uint64(key(s, t))
            j = int(np.searchsorted(K, k))
            if j < len(K) and K[j] == k:
                reuse.append(float(S[j]))
            else:
                reuse.append(math.nan)
                missing.append(i)
                os_.write(s if s.endswith("\n") else s + "\n")
                ot.write(t if t.endswith("\n") else t + "\n")
    R = np.asarray(reuse, dtype=np.float32)
    M = np.asarray(missing, dtype=np.int64)
    np.save(os.path.join(a.out_dir, "reuse_scores.npy"), R)
    np.save(os.path.join(a.out_dir, "missing_rows.npy"), M)
    summary = {"new_rows": int(len(R)), "reused": int(len(R) - len(M)), "to_score": int(len(M)),
               "first_missing_row": int(M[0]) if len(M) else None}
    json.dump(summary, open(os.path.join(a.out_dir, "plan.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_merge(a) -> int:
    R = np.load(os.path.join(a.plan_dir, "reuse_scores.npy"))
    M = np.load(os.path.join(a.plan_dir, "missing_rows.npy"))
    new = []
    with open(a.new_scores, encoding="utf-8", newline="\n") as f:
        for line in f:
            new.append(float(line.split("\t", 1)[0]))
    if len(new) != len(M):
        sys.exit(f"scorer returned {len(new):,} scores for {len(M):,} missing rows; refusing to merge")
    R = R.copy()
    R[M] = np.asarray(new, dtype=np.float32)
    if np.isnan(R).any():
        sys.exit(f"{int(np.isnan(R).sum()):,} rows still have no score after merge")
    n = 0
    with open(a.new_src, encoding="utf-8", newline="\n") as fs, open(a.new_tgt, encoding="utf-8", newline="\n") as ft, \
         open(a.out, "w", encoding="utf-8", newline="\n") as fo:
        for i, (s, t) in enumerate(zip(fs, ft)):
            fo.write(f"{R[i]:.6f}\t{s.rstrip(chr(10)).replace(chr(9), ' ')}\t{t.rstrip(chr(10)).replace(chr(9), ' ')}\n")
            n += 1
    if n != len(R):
        sys.exit(f"corpus has {n:,} rows but the plan covers {len(R):,}")
    print(f"wrote {a.out}: {n:,} rows ({n - len(M):,} reused, {len(M):,} newly scored)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--old-scored", required=True)
    p.add_argument("--new-src", required=True)
    p.add_argument("--new-tgt", required=True)
    p.add_argument("--out-dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--plan-dir", required=True)
    m.add_argument("--new-scores", required=True)
    m.add_argument("--new-src", required=True)
    m.add_argument("--new-tgt", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    return cmd_plan(a) if a.cmd == "plan" else cmd_merge(a)


if __name__ == "__main__":
    sys.exit(main())
