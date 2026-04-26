"""Filter a CometKiwi-scored TSV down to the top-K pairs by score.

Reads `score\\tsrc\\ttgt` lines, keeps top-K, writes parallel src/tgt
files (NOT preserving original order — sorted high-to-low quality).

For 30M lines the score column fits in RAM as a list of floats
(~240 MB), so we sort in-memory rather than two-pass.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input TSV from score_with_comet.py")
    ap.add_argument("--src-out", required=True)
    ap.add_argument("--tgt-out", required=True)
    ap.add_argument("--top-k", type=int, required=True,
                    help="Keep this many highest-scored pairs")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Optional: also drop pairs below this score")
    args = ap.parse_args()

    print(f"Reading {args.in_path} ...", file=sys.stderr)
    rows = []
    with open(args.in_path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            try:
                score = float(parts[0])
            except ValueError:
                continue
            if args.min_score is not None and score < args.min_score:
                continue
            rows.append((score, parts[1], parts[2]))

    print(f"  loaded {len(rows):,} valid rows", file=sys.stderr)
    rows.sort(key=lambda r: r[0], reverse=True)
    rows = rows[: args.top_k]

    print(f"  keeping top {len(rows):,} (score range "
          f"{rows[-1][0]:.4f} - {rows[0][0]:.4f})", file=sys.stderr)

    with open(args.src_out, "w", encoding="utf-8") as fs, \
         open(args.tgt_out, "w", encoding="utf-8") as ft:
        for _, s, t in rows:
            fs.write(s + "\n")
            ft.write(t + "\n")
    print(f"Wrote {args.src_out} and {args.tgt_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
