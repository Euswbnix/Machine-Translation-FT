#!/usr/bin/env python3
"""E0.1 — Exact per-pair provenance by hash-join (CPU only).

WHY THIS EXISTS
---------------
The rejected paper's Table `tab:qe_source` assigns each pair to a source
corpus by its LINE-NUMBER OFFSET in the HuggingFace stream ("Europarl+NC
occupy lines 0-2.2M; Common Crawl 2.2-5.4M; UN 5.4-18M; Giga-fren 18M+") and
reports every figure hedged with a tilde (~42% -> ~71% UN, ~40% -> ~5%
Giga-fren). Those boundaries were eyeballed from samples. The paper's central
mechanism -- that top-k QE filtering covertly performs DOMAIN selection -- rests
on them, so they cannot stay approximate.

This replaces the approximation with a ground-truth join: hash every pair in
each separately-downloaded constituent corpus, then look up every pair of the
cleaned 30M corpus.

Exact matching is valid because scripts/clean_data_enfr.py is a PURE FILTER: it
computes lengths on `s.strip()` but writes the ORIGINAL line (`os_.write(s)`),
so surviving lines are byte-identical to the stream they came from.

CAVEAT TO CHECK, NOT ASSUME
---------------------------
The 30M corpus was built from the HuggingFace `wmt14` stream, whereas the
constituent corpora here are downloaded from statmt.org. If HF applied its own
normalisation when building the dataset, exact matching will miss. The script
therefore reports the match rate and retries with progressive normalisation,
and tells you which regime you are in. Do not use the output until the match
rate is reported and acceptable.

CONSTITUENT CORPORA (WMT14 en-fr training data; verify URLs before use)
    europarl        https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
    commoncrawl     https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
    un              https://www.statmt.org/wmt13/training-parallel-un.tgz
    news-commentary https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
    giga-fren       https://www.statmt.org/wmt10/training-giga-fren.tar

USAGE
-----
    python e01_provenance.py \
        --corpus europarl:raw/europarl-v7.fr-en.en:raw/europarl-v7.fr-en.fr \
        --corpus commoncrawl:raw/commoncrawl.fr-en.en:raw/commoncrawl.fr-en.fr \
        --corpus un:raw/undoc.2000.fr-en.en:raw/undoc.2000.fr-en.fr \
        --corpus news-commentary:raw/news-commentary-v9.fr-en.en:raw/news-commentary-v9.fr-en.fr \
        --corpus giga-fren:raw/giga-fren.release2.fixed.en:raw/giga-fren.release2.fixed.fr \
        --clean-src data/v2_clean.en --clean-tgt data/v2_clean.fr \
        --qe-scores data/v2_scored.tsv \
        --out phase0/provenance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

NORMALIZERS = {
    "exact":      lambda s, t: (s, t),
    "strip":      lambda s, t: (s.strip(), t.strip()),
    "collapse_ws": lambda s, t: (" ".join(s.split()), " ".join(t.split())),
}


# Every corpus file is opened with newline="\n". Python's default text mode uses
# universal newlines, which treats a carriage return INSIDE a line as a line break.
# That is exactly the bug that misaligned the v2 en-fr and en-de training corpora
# (clean_data_enfr.py read train.en/train.fr that way). The first version of this
# script read the statmt constituents the same way, so it split news-commentary lines
# identically, reproduced the misalignment on the reference side, and reported the
# shifted pairs as exact matches -- placing the onset ~99k rows too late.
def h64(s: str, t: str) -> np.uint64:
    d = hashlib.blake2b(f"{s}\x00{t}".encode(), digest_size=8).digest()
    return np.uint64(int.from_bytes(d, "big"))


def hash_corpus(src_path: str, tgt_path: str, norm) -> np.ndarray:
    out = []
    with open(src_path, encoding="utf-8", errors="replace", newline="\n") as fs, \
         open(tgt_path, encoding="utf-8", errors="replace", newline="\n") as ft:
        for s, t in zip(fs, ft):
            a, b = norm(s, t)
            out.append(h64(a, b))
    return np.asarray(out, dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True,
                    metavar="LABEL:SRC:TGT", help="repeat once per constituent corpus")
    ap.add_argument("--clean-src", required=True)
    ap.add_argument("--clean-tgt", required=True)
    ap.add_argument("--qe-scores", help="score\\tsrc\\ttgt, same order as the clean corpus")
    ap.add_argument("--top-k", type=int, default=1_000_000,
                    help="reproduce the paper's top-K QE selection for the cross-tab")
    ap.add_argument("--norm", choices=list(NORMALIZERS), default="exact")
    # Escalating --norm maximises the match rate, which is also an ambiguity-maximising
    # search. Never adopt a looser normalizer on its match rate alone: rerun with a new
    # --out and compare the two runs with phase0/e01_compare_labels.py, which reports
    # rows newly matched versus rows whose label changed. (An earlier version declared a
    # --compare-norms flag here that was never implemented.)
    ap.add_argument("--out", default="phase0/provenance")
    args = ap.parse_args()

    norm = NORMALIZERS[args.norm]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. index the constituents -------------------------------------
    labels, all_h, all_lab = [], [], []
    for spec in args.corpus:
        label, src, tgt = spec.split(":", 2)
        idx = len(labels)
        labels.append(label)
        hs = hash_corpus(src, tgt, norm)
        print(f"  indexed {label:18s} {len(hs):>12,} pairs")
        all_h.append(hs)
        all_lab.append(np.full(len(hs), idx, dtype=np.uint8))

    H = np.concatenate(all_h)
    L = np.concatenate(all_lab)
    order = np.argsort(H, kind="stable")
    H, L = H[order], L[order]

    # Duplicate keys are duplicate LINES, full stop. Birthday bound at 64 bits:
    # for n = 6e7 keys, E[collisions] ~ n^2 / 2^65 = 3.6e15 / 3.69e19 ~ 1e-4.
    # So do NOT offer "maybe it is hash collisions" as an explanation — it is
    # numerically impossible at this scale, and offering it lets a real corpus
    # overlap be dismissed.
    uniq, inv = np.unique(H, return_inverse=True)
    dup = len(H) - len(uniq)
    exp_coll = len(H) ** 2 / 2.0 ** 65
    print(f"\nindex: {len(H):,} pairs, {len(labels)} sources, "
          f"{dup:,} duplicate keys ({dup/len(H)*100:.4f}%)")
    print(f"  expected 64-bit collisions at this scale: {exp_coll:.2e} "
          "— so duplicates are duplicate LINES, not hash accidents")

    # Split the statistic. Only CROSS-source duplicates threaten the labels:
    # the same pair appearing in two corpora makes its provenance genuinely
    # ambiguous, and np.searchsorted then silently returns whichever label
    # sorted first. Within-source duplicates are merely redundancy.
    if dup:
        # A key is cross-source iff its min and max label differ.
        Li = L.astype(np.int16)
        lo = np.full(len(uniq), 32767, dtype=np.int16)
        hi = np.full(len(uniq), -1, dtype=np.int16)
        np.minimum.at(lo, inv, Li)
        np.maximum.at(hi, inv, Li)
        n_cross = int((lo != hi).sum())
        print(f"  cross-source duplicate keys : {n_cross:,} "
              "(AMBIGUOUS provenance — the label assigned is arbitrary)")
        print(f"  within-source duplicate keys: {dup - n_cross:,} (harmless redundancy)")
        if n_cross / len(uniq) > 0.01:
            print("  WARNING: >1% of keys are claimed by MORE THAN ONE source. "
                  "The per-source shares below are not trustworthy; report the "
                  "ambiguous mass explicitly rather than silently assigning it.")

    # ---- 2. look up the cleaned corpus ---------------------------------
    UNMATCHED = np.uint8(255)
    prov, n = [], 0
    with open(args.clean_src, encoding="utf-8", errors="replace", newline="\n") as fs, \
         open(args.clean_tgt, encoding="utf-8", errors="replace", newline="\n") as ft:
        for s, t in zip(fs, ft):
            a, b = norm(s, t)
            k = h64(a, b)
            i = np.searchsorted(H, k)
            prov.append(L[i] if i < len(H) and H[i] == k else UNMATCHED)
            n += 1
            if n % 5_000_000 == 0:
                print(f"    …{n:,} pairs")
    prov = np.asarray(prov, dtype=np.uint8)

    matched = int((prov != UNMATCHED).sum())
    rate = matched / len(prov)
    print(f"\nmatched {matched:,} / {len(prov):,}  ({rate*100:.2f}%)  [norm={args.norm}]")
    if rate < 0.95:
        print("  ⚠️  LOW MATCH RATE — do NOT use these labels yet.")
        print("     Retry with --norm strip, then --norm collapse_ws.")
        print("     If all stay low, the HuggingFace stream text differs from the")
        print("     statmt.org distributions and a different join key is required.")

    np.save(f"{args.out}_labels.npy", prov)

    # ---- 3. the table that replaces tab:qe_source ----------------------
    full = Counter(prov.tolist())
    report = {"normalizer": args.norm, "match_rate": rate,
              "sources": labels, "full_corpus": {}, "top_k": {}}
    print(f"\n{'source':18s} {'full corpus':>14s} {'share':>8s}")
    for i, lab in enumerate(labels):
        c = full.get(i, 0)
        report["full_corpus"][lab] = c
        print(f"{lab:18s} {c:>14,} {c/len(prov)*100:7.2f}%")
    um = full.get(255, 0)
    print(f"{'(unmatched)':18s} {um:>14,} {um/len(prov)*100:7.2f}%")

    if args.qe_scores:
        scores = np.fromiter(
            (float(l.split("\t", 1)[0]) for l in open(args.qe_scores, encoding="utf-8", newline="\n")),
            dtype=np.float32)
        if len(scores) != len(prov):
            print(f"\n  ⚠️  QE rows ({len(scores):,}) != corpus rows ({len(prov):,}); "
                  "skipping cross-tab. They must be in the same order.")
        else:
            top = np.argpartition(-scores, args.top_k)[: args.top_k]
            tc = Counter(prov[top].tolist())
            print(f"\n{'source':18s} {'top-'+str(args.top_k//1000)+'K':>12s} "
                  f"{'share':>8s} {'enrichment':>11s}")
            for i, lab in enumerate(labels):
                c, f = tc.get(i, 0), full.get(i, 0)
                sh = c / args.top_k
                base = f / len(prov) if f else 0
                report["top_k"][lab] = {"count": c, "share": sh,
                                        "enrichment": (sh / base) if base else None}
                print(f"{lab:18s} {c:>12,} {sh*100:7.2f}% "
                      f"{(sh/base if base else float('nan')):10.2f}×")

    with open(f"{args.out}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}_labels.npy and {args.out}_report.json")
    print("These exact counts replace every tilde-hedged figure in tab:qe_source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
