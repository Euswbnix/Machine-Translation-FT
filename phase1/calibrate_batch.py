#!/usr/bin/env python3
"""PROTOCOL 3.2 — calibrate the effective optimizer batch to match across arms.

WHY
---
The rejected paper's two arms did not train at the same effective batch:

    Base  batch_size 24576 x accumulate_steps 4 = 98,304
    Big   batch_size  8192 x accumulate_steps 4 = 32,768

Big trained at ONE THIRD of Base's effective batch, while Vaswani held it
constant. But those products are NOMINAL: `batch_size` is a `max_tokens` cap on a
PADDED budget, and it is almost never the binding constraint -- `max_sentences`
is (the two caps cross at max_tokens/max_sentences = 128 for both arms, far above
WMT subword lengths). Realized batches reach only ~34% of the cap.

So matching the nominal product does not match anything. The quantity that has to
be equal is the MEASURED mean non-pad target tokens per OPTIMIZER step:

    effective batch = accumulate_steps x E[non-pad tgt per micro-batch]

This script measures that on the real length distribution and solves for the
(max_sentences, accumulate_steps) pair that hits the target for each arm.

USAGE
-----
    python phase1/calibrate_batch.py --cache data_enfr_v1/train.cached_256.npz \
        --target 98304 --tolerance 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase0"))
from e02_token_accounting import build_batches            # noqa: E402

# (label, max_tokens) — max_tokens is kept at the arm's configured value; it is
# max_sentences that actually binds, so that is what we search over.
ARMS = [("Base", 24_576), ("Big", 8_192)]
SENTENCE_GRID = [32, 48, 64, 96, 128, 160, 192, 256, 320, 384]


def mean_nonpad_tgt(src_lens, tgt_lens, max_tokens, max_sentences, seeds=3):
    """E[non-pad target tokens per micro-batch], averaged over sampler shuffles."""
    vals, fills, capped = [], [], []
    for s in range(seeds):
        batches = build_batches(src_lens, tgt_lens, max_tokens, max_sentences, seed=s)
        # Count LABELS, not cached tokens. trainer.py:429 takes tgt[:, 1:] and
        # trainer.py:487 counts the non-pad entries of that, so a cached length L
        # contributes L-1. Counting L overcounts by one per sentence (~3% here),
        # which would certify a batch as inside a 2% tolerance while it is 3%
        # under target. The calibration unit MUST equal the trainer's unit.
        tgt = sum(int(tgt_lens[a].sum()) - len(a)
                  for a in (np.asarray(b) for b in batches))
        budget = sum(len(b) * int(max(src_lens[np.asarray(b)].max(),
                                      tgt_lens[np.asarray(b)].max())) for b in batches)
        vals.append(tgt / len(batches))
        fills.append((budget / len(batches)) / max_tokens)
        capped.append(sum(1 for b in batches if len(b) >= max_sentences) / len(batches))
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(fills)), float(np.mean(capped))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help=".cached_256.npz from TranslationDataset")
    ap.add_argument("--target", type=int, default=98_304,
                    help="target effective optimizer batch, in non-pad target tokens")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=2_000_000,
                    help="pairs to sample; 0 = all (slow on 30M)")
    ap.add_argument("--json-out", default="phase1/batch_calibration.json")
    args = ap.parse_args()

    d = np.load(args.cache)
    src_lens = np.diff(d["src_offsets"]).astype(np.int64)
    tgt_lens = np.diff(d["tgt_offsets"]).astype(np.int64)
    if args.limit and len(src_lens) > args.limit:
        idx = np.random.RandomState(0).choice(len(src_lens), args.limit, replace=False)
        src_lens, tgt_lens = src_lens[idx], tgt_lens[idx]
    print(f"{len(src_lens):,} pairs   src mean {src_lens.mean():.1f}   "
          f"tgt mean {tgt_lens.mean():.1f}\n")

    out = {"target": args.target, "tolerance": args.tolerance, "arms": {}}
    print(f"target effective batch: {args.target:,} non-pad target tokens/optimizer step\n")

    # ---- 1. measure every grid point for both arms ----------------------
    points = {}
    for label, max_tokens in ARMS:
        print(f"=== {label} (max_tokens {max_tokens:,}) ===")
        print(f"{'max_sent':>9} {'tgt/micro':>10} {'sd':>6} {'fill':>6} {'sent-cap':>9} "
              f"{'accum':>6} {'achieved':>10} {'err':>7}")
        rows = []
        for ms in SENTENCE_GRID:
            m, sd, fill, capped = mean_nonpad_tgt(src_lens, tgt_lens, max_tokens, ms, args.seeds)
            accum = max(1, int(round(args.target / m)))
            achieved = accum * m
            err = achieved / args.target - 1.0
            rows.append({"max_sentences": ms, "mean_tgt_per_micro": m, "sd": sd,
                         "nominal_fill": fill, "sentence_capped_frac": capped,
                         "accumulate_steps": accum, "achieved": achieved, "err": err})
            print(f"{ms:>9} {m:>10,.0f} {sd:>6.0f} {fill:>6.3f} {capped*100:>8.1f}% "
                  f"{accum:>6} {achieved:>10,.0f} {err*100:>+6.2f}%")
        points[label] = rows
        print()

    # ---- 2. joint selection ---------------------------------------------
    # Minimising |err| per arm independently is WRONG twice over:
    #   (a) it happily picks a tiny micro-batch with a huge accumulate_steps,
    #       which matches the target but wrecks throughput; and
    #   (b) two arms each within tolerance of the target can still be ~2x
    #       tolerance apart from EACH OTHER, which is the quantity that actually
    #       has to match.
    # So: require both arms AND the arm-to-arm spread inside tolerance, then
    # maximise the smaller micro-batch (throughput) among survivors.
    (la, _), (lb, _) = ARMS
    feasible = []
    for a in points[la]:
        for b in points[lb]:
            if abs(a["err"]) > args.tolerance or abs(b["err"]) > args.tolerance:
                continue
            spread = abs(a["achieved"] - b["achieved"]) / args.target
            if spread > args.tolerance:
                continue
            feasible.append((min(a["mean_tgt_per_micro"], b["mean_tgt_per_micro"]),
                             spread, a, b))
    if not feasible:
        print("NO FEASIBLE PAIR — no (max_sentences, accumulate_steps) combination "
              "puts both arms AND their mutual spread inside the tolerance.")
        print("  Widen SENTENCE_GRID, raise --tolerance deliberately, or change the "
              "--target. Do NOT proceed with mismatched arms and fix it in prose; "
              "that mismatch (98,304 vs 32,768) is the original defect.")
        json.dump({**out, "feasible": False}, open(args.json_out, "w"), indent=2)
        return 1

    feasible.sort(key=lambda x: (-x[0], x[1]))
    _, spread, best_a, best_b = feasible[0]
    out["arms"] = {la: best_a, lb: best_b}
    out["arm_spread"] = spread
    out["feasible"] = True
    out["n_feasible_pairs"] = len(feasible)

    print(f"{len(feasible)} feasible pairs; chosen by max throughput, ties to min spread.\n")
    for label, a in ((la, best_a), (lb, best_b)):
        print(f"  {label:5s} max_sentences {a['max_sentences']:>4}  "
              f"accumulate_steps {a['accumulate_steps']:>3}  -> "
              f"{a['achieved']:>9,.0f} ({a['err']*100:+.2f}%)   "
              f"fill {a['nominal_fill']:.3f}  sentence-capped {a['sentence_capped_frac']*100:.1f}%")
    print(f"\n  arm-to-arm spread: {spread*100:.2f}%  (tolerance {args.tolerance*100:.0f}%)")

    # Degeneracy warning: if NEITHER arm is near its token cap, the two arms have
    # the same memory footprint per micro-batch and the max_tokens difference --
    # the reason there are two settings at all -- has been calibrated away.
    if best_a["nominal_fill"] < 0.5 and best_b["nominal_fill"] < 0.5:
        print("\n  ⚠️  DEGENERATE: neither arm approaches its max_tokens cap "
              f"(fills {best_a['nominal_fill']:.2f} / {best_b['nominal_fill']:.2f}). "
              "The arms now differ only in model size, not in per-micro-batch "
              "footprint. That is defensible — but state it, do not let a reviewer "
              "discover that batch_size was inert.")

    print("\nConfig lines to write (PROTOCOL 3.2 requires publishing these):")
    for label, a in ((la, best_a), (lb, best_b)):
        print(f"  {label:5s}  max_sentences: {a['max_sentences']}   "
              f"accumulate_steps: {a['accumulate_steps']}   "
              f"# measured {a['mean_tgt_per_micro']:,.0f}+/-{a['sd']:,.0f} tgt/micro")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.json_out, "w"), indent=2)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
