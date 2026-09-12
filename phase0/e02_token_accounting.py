#!/usr/bin/env python3
"""E0.2 — Token-accounting audit (CPU only, seconds).

WHY THIS EXISTS
---------------
The rejected WMT 2026 paper reported "effective batch ~98K tokens/step (Base)
/ 32K (Big)" and derived per-cell token budgets from `steps x batch`. Two
reviewers computed training-token totals from those numbers and concluded Big
was severely undertrained. Their *relative* conclusion is almost certainly
right, but the *absolute* numbers cannot be, because C=6ND on those budgets
implies 1.6-3.2x an RTX 5090's bf16 dense peak (~209 TFLOPS) -- physically
impossible.

Root cause, from src/data/dataset.py TokenBatchSampler.__iter__:

    seq_len   = max(src_lens[idx], tgt_lens[idx])
    num_tokens = (len(batch) + 1) * new_max          # <-- PADDED, and uses
                                                     #     max(src,tgt)

So `batch_size: 24576` is a *padded* budget (sentences x longest sentence),
NOT the non-pad target-token count that the loss actually averages over
(src/training/loss.py:36, trainer.py:487: (tgt_labels != PAD_ID).sum()).

This script replicates the sampler exactly on the real length arrays and
measures the conversion factor, then recomputes every cell's true budget and
implied MFU. A paper whose premise is "the previous version mismatched token
budgets" cannot ship a second budget-accounting error.

USAGE
-----
    # fastest: point at the tokenizer cache the trainer already wrote
    python e02_token_accounting.py --cache data_enfr_v1/train.en.cached_256.npz

    # or from raw text (requires sentencepiece + the SPM model)
    python e02_token_accounting.py --src train.en --tgt train.fr \
        --spm data_enfr_v1/spm_enfr_v1_fixed.model
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# --- the six cells as reported in the paper -------------------------------
# (name, params, reported_steps, micro_batch_max_tokens, accum, wall_clock_h)
#
# CRITICAL: `reported_steps` are GLOBAL STEPS, and trainer.py increments
# global_step once per MICRO-BATCH (line 441, inside _train_step, called per
# batch at line 321); the optimizer only steps when
# `global_step % accumulate_steps == 0` (line 461), and `max_steps` is compared
# against global_step (line 310). So tokens = steps x micro_batch_budget, and
# multiplying by `accum` as the paper did double-counts accumulation 4x.
CELLS = [
    ("en-fr Base capped", 60.5e6, 122_000, 24_576, 4, 2.5),
    ("en-fr Big capped",  209.0e6, 260_000,  8_192, 4, 6.0),
    ("en-fr Base full",   60.5e6, 600_000, 24_576, 4, 14.0),
    ("en-fr Big full",    209.0e6, 416_000,  8_192, 4, 9.0),
    ("en-de Base",        60.5e6, 230_000, 24_576, 4, 7.0),
    ("en-de Big",         209.0e6, 466_000,  8_192, 4, 8.0),
]
RTX5090_BF16_PEAK = 209e12  # dense, no sparsity


def load_lengths(args) -> tuple[np.ndarray, np.ndarray]:
    if args.cache:
        d = np.load(args.cache)
        src_lens = np.diff(d["src_offsets"]).astype(np.int64)
        tgt_lens = np.diff(d["tgt_offsets"]).astype(np.int64)
        return src_lens, tgt_lens

    import sentencepiece as spm  # noqa: PLC0415
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    src_lens, tgt_lens = [], []
    with open(args.src, encoding="utf-8") as fs, open(args.tgt, encoding="utf-8") as ft:
        for i, (s, t) in enumerate(zip(fs, ft)):
            if args.limit and i >= args.limit:
                break
            # +2 mirrors the BOS/EOS the dataset adds; harmless if off by a
            # constant since we report ratios, but keep it explicit.
            src_lens.append(min(len(sp.encode(s.strip())) + 2, args.max_seq_len))
            tgt_lens.append(min(len(sp.encode(t.strip())) + 2, args.max_seq_len))
    return np.asarray(src_lens, np.int64), np.asarray(tgt_lens, np.int64)


def build_batches(src_lens, tgt_lens, max_tokens, max_sentences, seed=0):
    """Exact replication of TokenBatchSampler.__iter__ (shuffle=True path)."""
    n = len(src_lens)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    chunk = max_sentences * 100
    for i in range(0, n, chunk):
        c = indices[i : i + chunk]
        indices[i : i + chunk] = c[np.argsort(src_lens[c])]

    batches, batch, max_len = [], [], 0
    for idx in indices:
        seq_len = max(int(src_lens[idx]), int(tgt_lens[idx]))
        new_max = max(max_len, seq_len)
        if (len(batch) + 1) * new_max > max_tokens or len(batch) >= max_sentences:
            if batch:
                batches.append(batch)
            batch, max_len = [int(idx)], seq_len
        else:
            batch.append(int(idx))
            max_len = new_max
    if batch:
        batches.append(batch)
    return batches


def measure(src_lens, tgt_lens, max_tokens, max_sentences, seed=0):
    batches = build_batches(src_lens, tgt_lens, max_tokens, max_sentences, seed)
    budget = nonpad_tgt = nonpad_both = padded_tgt = 0
    for b in batches:
        b = np.asarray(b)
        sl, tl = src_lens[b], tgt_lens[b]
        budget      += len(b) * int(max(sl.max(), tl.max()))  # what the sampler counts
        # The loss counts SHIFTED labels: trainer.py:429 sets tgt_labels = tgt[:, 1:]
        # and trainer.py:487 counts (tgt_labels != PAD_ID). Cached lengths include
        # BOS and EOS (dataset.py:43 add_bos/add_eos), so a cached length L yields
        # L-1 label tokens. Counting L overcounts by one per SENTENCE -- about 3%
        # at WMT subword lengths, which is larger than the tolerances that depend
        # on this number.
        nonpad_tgt  += int(tl.sum()) - len(b)
        nonpad_both += int(sl.sum() + tl.sum()) - 2 * len(b)
        padded_tgt  += len(b) * (int(tl.max()) - 1)
    # WHICH CAP IS BINDING? max_tokens and max_sentences cross at
    # max_tokens/max_sentences (= 128 for BOTH arms here: 24576/192 = 8192/64).
    # WMT subword sentences run ~30 tokens, so the SENTENCE cap binds almost
    # always and the realized budget sits far below the nominal max_tokens.
    # Reporting max_tokens as "the batch size" is therefore wrong.
    sentence_capped = sum(1 for b in batches if len(b) >= max_sentences)
    return {
        "n_batches": len(batches),
        "sentence_capped_frac": sentence_capped / len(batches),
        "mean_sampler_budget_per_microbatch": budget / len(batches),
        "nominal_fill_frac": (budget / len(batches)) / max_tokens,
        "sampler_budget": budget,
        "nonpad_tgt": nonpad_tgt,
        "nonpad_src_plus_tgt": nonpad_both,
        "padded_tgt": padded_tgt,
        "nonpad_tgt_per_budget": nonpad_tgt / budget,
        "nonpad_both_per_budget": nonpad_both / budget,
        "tgt_padding_efficiency": nonpad_tgt / padded_tgt,
        "mean_nonpad_tgt_per_microbatch": nonpad_tgt / len(batches),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help=".cached_256.npz written by TranslationDataset")
    ap.add_argument("--src"); ap.add_argument("--tgt"); ap.add_argument("--spm")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--seeds", type=int, default=3, help="repeat to bound sampler noise")
    ap.add_argument("--json-out", default="phase0/e02_token_accounting.json")
    args = ap.parse_args()

    if not args.cache and not (args.src and args.tgt and args.spm):
        ap.error("give --cache, or all of --src/--tgt/--spm")

    src_lens, tgt_lens = load_lengths(args)
    print(f"pairs={len(src_lens):,}  "
          f"mean src={src_lens.mean():.1f}  mean tgt={tgt_lens.mean():.1f}\n")

    out = {"n_pairs": int(len(src_lens)), "configs": {}}

    for label, mt, ms in [("Base (24576/192)", 24_576, 192), ("Big (8192/64)", 8_192, 64)]:
        runs = [measure(src_lens, tgt_lens, mt, ms, seed=s) for s in range(args.seeds)]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        out["configs"][label] = agg
        print(f"--- {label} ---")
        print(f"  sampler 'tokens' budget consumed : {agg['sampler_budget']:>16,.0f}")
        print(f"  TRUE non-pad target tokens       : {agg['nonpad_tgt']:>16,.0f}")
        print(f"  non-pad src+tgt                  : {agg['nonpad_both']:>16,.0f}"
              if False else
              f"  non-pad src+tgt                  : {agg['nonpad_src_plus_tgt']:>16,.0f}")
        print(f"  mean non-pad tgt / micro-batch   : {agg['mean_nonpad_tgt_per_microbatch']:>16,.0f}"
              "   <== THE conversion factor")
        print(f"  mean sampler budget / micro-batch: {agg['mean_sampler_budget_per_microbatch']:>16,.0f}")
        print(f"  nominal fill fraction            : {agg['nominal_fill_frac']:.3f}"
              f"   (batches reach only this much of max_tokens)")
        print(f"  batches closed by max_sentences  : {agg['sentence_capped_frac']*100:.1f}%"
              "   <== if high, max_tokens is NOT the binding cap")
        print(f"  non-pad tgt / sampler budget     : {agg['nonpad_tgt_per_budget']:.3f}"
              "   (padding diagnostic only — NOT a budget conversion factor)")
        print(f"  target-side padding efficiency   : {agg['tgt_padding_efficiency']:.3f}\n")

    print("=" * 78)
    print("RECOMPUTED CELL BUDGETS  (C = 6ND, RTX 5090 bf16 dense peak 209 TFLOPS)")
    print("=" * 78)
    print(f"{'cell':20s} {'paper':>9s} {'TRUE tgt':>9s} {'MFU':>8s} {'tok/param':>9s} {'verdict':>10s}")
    rows = []
    for name, N, steps, mb, accum, hours in CELLS:
        key = "Base (24576/192)" if mb == 24_576 else "Big (8192/64)"
        cfg = out["configs"][key]
        # Correction 1 (exact, data-independent): global_step counts MICRO-batches
        # (trainer.py:441 inside _train_step), while the optimizer steps only every
        # accumulate_steps (trainer.py:461). steps x mb x accum double-counts by 4x.
        paper_claimed = steps * mb * accum
        # Correction 2: do NOT rescale the NOMINAL cap by a ratio whose denominator
        # is the REALIZED padded budget — mixing nominal numerator with realized
        # denominator inflates the result ~2-3x. Batches are capped by max_sentences,
        # not max_tokens, so they never reach mb. Use the directly measured mean
        # instead; steps x E[non-pad tgt per micro-batch] is the quantity we want.
        nominal = steps * mb                    # what "steps x batch_size" suggests
        true_tok = steps * cfg["mean_nonpad_tgt_per_microbatch"]
        mfu = 6 * N * true_tok / (hours * 3600) / RTX5090_BF16_PEAK
        verdict = "OK" if 0.15 <= mfu <= 0.65 else ("IMPOSSIBLE" if mfu > 1 else "check")
        tok_per_param = true_tok / N
        rows.append({"cell": name, "paper_claimed_tokens": paper_claimed,
                     "nominal_micro_batch_tokens": nominal,
                     "true_nonpad_tgt_tokens": true_tok, "implied_mfu": mfu,
                     "tokens_per_param": tok_per_param,
                     "nominal_over_true": nominal / true_tok if true_tok else None})
        print(f"{name:20s} {paper_claimed/1e9:8.1f}B {true_tok/1e9:8.1f}B "
              f"{mfu*100:7.1f}% {tok_per_param:8.1f} {verdict:>10s}")
    out["cells"] = rows

    # the number that actually matters for the paper's central confound
    print()
    for a, b, lbl in [(0, 1, "en-fr capped"), (2, 3, "en-fr full"), (4, 5, "en-de")]:
        r = rows[b]["true_nonpad_tgt_tokens"] / rows[a]["true_nonpad_tgt_tokens"]
        print(f"  {lbl:14s} Big/Base true-token ratio = {r:.2f}x   "
              f"(Vaswani 2017 gave Big 3.00x)")

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
