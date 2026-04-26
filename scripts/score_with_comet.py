"""Score parallel corpus with CometKiwi-22 (reference-free QE).

Output: TSV `<score>\t<src>\t<tgt>` per line, in the SAME order as input.

Memory note: streaming, never holds the full corpus. For 30M pairs on
a single GPU expect ~1-2h with batch_size=64.
"""

import argparse
import sys
from pathlib import Path

from comet import download_model, load_from_checkpoint
from tqdm import tqdm


def chunked(src_lines, tgt_lines, chunk_size):
    buf = []
    for s, t in zip(src_lines, tgt_lines):
        buf.append({"src": s.rstrip("\n"), "mt": t.rstrip("\n")})
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source side text file")
    ap.add_argument("--tgt", required=True, help="Target side text file")
    ap.add_argument("--out", required=True, help="Output TSV: score\\tsrc\\ttgt")
    ap.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da",
                    help="HF model id; CometKiwi-22 is the standard")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=50000,
                    help="Process this many pairs per predict() call")
    args = ap.parse_args()

    print(f"Downloading / loading {args.model} ...", file=sys.stderr)
    ckpt_path = download_model(args.model)
    model = load_from_checkpoint(ckpt_path)

    src_lines = open(args.src, encoding="utf-8")
    tgt_lines = open(args.tgt, encoding="utf-8")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with open(args.out, "w", encoding="utf-8") as f_out:
        for chunk in tqdm(chunked(src_lines, tgt_lines, args.chunk_size),
                          desc="scoring", unit="chunk"):
            preds = model.predict(
                chunk,
                batch_size=args.batch_size,
                gpus=args.gpus,
                progress_bar=False,
            )
            scores = preds["scores"]
            for d, s in zip(chunk, scores):
                # tabs in src/tgt would break TSV — replace defensively
                src_safe = d["src"].replace("\t", " ")
                tgt_safe = d["mt"].replace("\t", " ")
                f_out.write(f"{s:.6f}\t{src_safe}\t{tgt_safe}\n")
            n_total += len(chunk)
    print(f"Wrote {n_total:,} scored pairs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
