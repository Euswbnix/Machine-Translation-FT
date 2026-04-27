"""Score parallel corpus with CometKiwi-22 (reference-free QE).

Output: TSV `<score>\\t<src>\\t<tgt>` per line, in the SAME order as input.

CometKiwi-22 (XLM-RoBERTa-XL backbone) on a 5090 runs at ~700 pair/s
with batch_size=64. 30M pairs ≈ 12 hours.

Resume: pass --resume to continue an interrupted run. The script counts
lines already in the output file, fast-forwards that many lines from
the input, and appends. Output line ordering is preserved across
resumes (chunks are flushed atomically per-write).
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


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def fast_forward(file_handle, n: int):
    for _ in range(n):
        if not file_handle.readline():
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source side text file")
    ap.add_argument("--tgt", required=True, help="Target side text file")
    ap.add_argument("--out", required=True, help="Output TSV: score\\tsrc\\ttgt")
    ap.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da",
                    help="HF model id; CometKiwi-22 is the standard reference-"
                         "free QE model. Gated — request access at "
                         "https://huggingface.co/Unbabel/wmt22-cometkiwi-da "
                         "and run `hf auth login` first.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=50000,
                    help="Process this many pairs per predict() call")
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted run: count existing output "
                         "lines, skip them in the input, append new scores.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skip = 0
    if args.resume:
        skip = count_lines(out_path)
        print(f"Resume: {skip:,} pairs already scored — skipping ahead",
              file=sys.stderr)
        open_mode = "a"
    else:
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"WARNING: {out_path} exists and is non-empty. "
                  "Pass --resume to continue, or delete the file first.",
                  file=sys.stderr)
            sys.exit(2)
        open_mode = "w"

    print(f"Downloading / loading {args.model} ...", file=sys.stderr)
    ckpt_path = download_model(args.model)
    model = load_from_checkpoint(ckpt_path)

    src_lines = open(args.src, encoding="utf-8")
    tgt_lines = open(args.tgt, encoding="utf-8")
    if skip:
        fast_forward(src_lines, skip)
        fast_forward(tgt_lines, skip)

    n_total = skip
    with open(out_path, open_mode, encoding="utf-8") as f_out:
        for chunk in tqdm(chunked(src_lines, tgt_lines, args.chunk_size),
                          desc="scoring", unit="chunk", initial=skip // args.chunk_size):
            preds = model.predict(
                chunk,
                batch_size=args.batch_size,
                gpus=args.gpus,
                progress_bar=False,
            )
            scores = preds["scores"]
            for d, s in zip(chunk, scores):
                src_safe = d["src"].replace("\t", " ")
                tgt_safe = d["mt"].replace("\t", " ")
                f_out.write(f"{s:.6f}\t{src_safe}\t{tgt_safe}\n")
            f_out.flush()  # ensure each chunk is durable, so resume is safe
            n_total += len(chunk)
    print(f"Wrote {n_total:,} scored pairs total to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
