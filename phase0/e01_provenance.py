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

AMBIGUOUS ROWS ARE NOT ARBITRARY
--------------------------------
A pair that occurs in more than one constituent has ambiguous provenance. The
label it gets is DETERMINISTIC: the index is a stable sort over the constituents
concatenated in --corpus order and the lookup takes the leftmost hit, so an
ambiguous row always goes to the FIRST-LISTED --corpus that contains it. Later
sources (giga-fren, last in rental_setup.sh's order) therefore never receive an
ambiguous row, and their counts are lower bounds. The report gives the number of
ambiguous corpus ROWS per source pair ("ambiguity" block) so shares can be quoted
as ranges.

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
from itertools import combinations
from pathlib import Path

import numpy as np

NORMALIZERS = {
    "exact":      lambda s, t: (s, t),
    "strip":      lambda s, t: (s.strip(), t.strip()),
    "collapse_ws": lambda s, t: (" ".join(s.split()), " ".join(t.split())),
}
LOOKUP_CHUNK = 1_000_000


# Every corpus file is opened with newline="\n". Python's default text mode uses
# universal newlines, which treats a carriage return INSIDE a line as a line break.
# That is exactly the bug that misaligned the v2 en-fr and en-de training corpora
# (clean_data_enfr.py read train.en/train.fr that way). The first version of this
# script read the statmt constituents the same way, so it split news-commentary lines
# identically, reproduced the misalignment on the reference side, and reported the
# shifted pairs as exact matches -- placing the onset ~99k rows too late.
# Likewise never use str.splitlines(): it also breaks on U+2028/U+2029, NEL, VT, FF and
# FS/GS/RS, which occur inside ~870k lines of the fixed v2 corpus.
def _h64_int(s: str, t: str) -> int:
    d = hashlib.blake2b(f"{s}\x00{t}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big")


def h64(s: str, t: str) -> np.uint64:
    return np.uint64(_h64_int(s, t))


def hash_corpus(src_path: str, tgt_path: str, norm) -> np.ndarray:
    out = []
    with open(src_path, encoding="utf-8", errors="replace", newline="\n") as fs, \
         open(tgt_path, encoding="utf-8", errors="replace", newline="\n") as ft:
        for s, t in zip(fs, ft):
            a, b = norm(s, t)
            out.append(_h64_int(a, b))
    return np.asarray(out, dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True,
                    metavar="LABEL:SRC:TGT",
                    help="repeat once per constituent corpus; ORDER MATTERS: a pair found in several "
                         "constituents is labelled with the first-listed one")
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
    if len(labels) > 64:
        sys.exit("at most 64 --corpus sources are supported (source sets are 64-bit masks)")
    order_txt = " < ".join(labels)

    H = np.concatenate(all_h)
    L = np.concatenate(all_lab)
    order = np.argsort(H, kind="stable")
    H, L = H[order], L[order]
    del order, all_h, all_lab

    # Duplicate keys are duplicate LINES, full stop. Birthday bound at 64 bits:
    # for n = 6e7 keys, E[collisions] ~ n^2 / 2^65 = 3.6e15 / 3.69e19 ~ 1e-4.
    # So do NOT offer "maybe it is hash collisions" as an explanation — it is
    # numerically impossible at this scale, and offering it lets a real corpus
    # overlap be dismissed.
    uniq, inv = np.unique(H, return_inverse=True)
    inv = inv.reshape(-1)
    dup = len(H) - len(uniq)
    exp_coll = len(H) ** 2 / 2.0 ** 65
    print(f"\nindex: {len(H):,} pairs, {len(labels)} sources, "
          f"{dup:,} duplicate keys ({dup/max(len(H), 1)*100:.4f}%)")
    print(f"  expected 64-bit collisions at this scale: {exp_coll:.2e} "
          "— so duplicates are duplicate LINES, not hash accidents")

    # Split the statistic. Only CROSS-source duplicates threaten the labels:
    # the same pair appearing in two corpora makes its provenance genuinely
    # ambiguous; the lookup below then returns the first-listed source's label
    # (deterministic, but biased against later-listed sources). Within-source
    # duplicates are merely redundancy.
    n_cross = 0
    mask = None                                   # per unique key: bit i set iff source i has it
    if dup:
        # A key is cross-source iff its min and max label differ.
        Li = L.astype(np.int16)
        lo = np.full(len(uniq), 32767, dtype=np.int16)
        hi = np.full(len(uniq), -1, dtype=np.int16)
        np.minimum.at(lo, inv, Li)
        np.maximum.at(hi, inv, Li)
        n_cross = int((lo != hi).sum())
        del Li, lo, hi
        print(f"  cross-source duplicate keys : {n_cross:,} "
              f"(AMBIGUOUS provenance — deterministically assigned to the first-listed "
              f"--corpus: {order_txt})")
        print(f"  within-source duplicate keys: {dup - n_cross:,} (harmless redundancy)")
        if n_cross / len(uniq) > 0.01:
            print("  WARNING: >1% of keys are claimed by MORE THAN ONE source. "
                  "The per-source shares below are not trustworthy; report the "
                  "ambiguous mass explicitly rather than silently assigning it.")
        if n_cross:
            mask = np.zeros(len(uniq), dtype=np.uint64)
            np.bitwise_or.at(mask, inv, np.left_shift(np.uint64(1), L.astype(np.uint64)))

    # ---- 2. look up the cleaned corpus ---------------------------------
    UNMATCHED = np.uint8(255)
    prov_chunks: list[np.ndarray] = []
    amb_by_mask: Counter = Counter()             # source-set mask -> ambiguous rows
    amb_by_label: Counter = Counter()            # assigned label  -> ambiguous rows
    n = 0

    def flush(buf: list[int]) -> None:
        if not buf:
            return
        K = np.asarray(buf, dtype=np.uint64)
        if len(H) == 0:
            prov_chunks.append(np.full(len(K), UNMATCHED, dtype=np.uint8))
            return
        i = np.searchsorted(H, K)                # leftmost hit == first-listed source
        ic = np.minimum(i, len(H) - 1)
        hit = (i < len(H)) & (H[ic] == K)
        lab = np.where(hit, L[ic], UNMATCHED).astype(np.uint8)
        prov_chunks.append(lab)
        if mask is not None:
            m = mask[inv[ic]]
            amb = hit & ((m & (m - np.uint64(1))) != 0)          # more than one bit set
            if amb.any():
                amb_by_mask.update(m[amb].tolist())
                amb_by_label.update(lab[amb].tolist())

    buf: list[int] = []
    with open(args.clean_src, encoding="utf-8", errors="replace", newline="\n") as fs, \
         open(args.clean_tgt, encoding="utf-8", errors="replace", newline="\n") as ft:
        for s, t in zip(fs, ft):
            a, b = norm(s, t)
            buf.append(_h64_int(a, b))
            n += 1
            if len(buf) == LOOKUP_CHUNK:
                flush(buf); buf = []
            if n % 5_000_000 == 0:
                print(f"    …{n:,} pairs")
    flush(buf)
    prov = np.concatenate(prov_chunks) if prov_chunks else np.zeros(0, dtype=np.uint8)

    matched = int((prov != UNMATCHED).sum())
    rate = matched / len(prov) if len(prov) else 0.0
    print(f"\nmatched {matched:,} / {len(prov):,}  ({rate*100:.2f}%)  [norm={args.norm}]")
    if rate < 0.95:
        print("  ⚠️  LOW MATCH RATE — do NOT use these labels yet.")
        print("     Retry with --norm strip, then --norm collapse_ws.")
        print("     If all stay low, the HuggingFace stream text differs from the")
        print("     statmt.org distributions and a different join key is required.")

    np.save(f"{args.out}_labels.npy", prov)

    # ---- ambiguity, in corpus ROWS -------------------------------------
    def set_name(m: int) -> str:
        return "+".join(labels[j] for j in range(len(labels)) if (m >> j) & 1)

    by_set, by_pair = {}, Counter()
    for m, c in sorted(amb_by_mask.items()):
        by_set[set_name(m)] = c
        members = [j for j in range(len(labels)) if (m >> j) & 1]
        for x, y in combinations(members, 2):
            by_pair[f"{labels[x]}+{labels[y]}"] += c
    amb_rows = sum(amb_by_mask.values())
    rule = ("a corpus row whose key occurs in more than one constituent is deterministically "
            "assigned to the FIRST-LISTED --corpus that contains it; order: " + order_txt)
    ambiguity = {
        "cross_source_keys": n_cross,
        "ambiguous_rows": amb_rows,
        "ambiguous_row_share": (amb_rows / len(prov)) if len(prov) else 0.0,
        "assignment_rule": rule,
        "corpus_order": labels,
        "ambiguous_rows_by_source_pair": {k: by_pair[k] for k in sorted(by_pair)},
        "ambiguous_rows_by_source_pair_note": "a row whose key is in k sources counts once in each of its k(k-1)/2 pairs",
        "ambiguous_rows_by_source_set": by_set,
        "ambiguous_rows_by_assigned_label": {labels[i]: amb_by_label.get(i, 0) for i in range(len(labels))},
    }
    print(f"\nambiguous rows (key in >1 source): {amb_rows:,} "
          f"({ambiguity['ambiguous_row_share']*100:.2f}% of the corpus)")
    print(f"  rule: {rule}")
    for k, c in ambiguity["ambiguous_rows_by_source_pair"].items():
        print(f"  {k:36s} {c:>12,} rows")

    # ---- 3. the table that replaces tab:qe_source ----------------------
    full = Counter(prov.tolist())
    report = {"normalizer": args.norm, "match_rate": rate,
              "sources": labels, "full_corpus": {}, "top_k": None, "top_k_note": None,
              "ambiguity": ambiguity}
    print(f"\n{'source':18s} {'full corpus':>14s} {'share':>8s}")
    for i, lab in enumerate(labels):
        c = full.get(i, 0)
        report["full_corpus"][lab] = c
        print(f"{lab:18s} {c:>14,} {c/max(len(prov), 1)*100:7.2f}%")
    um = full.get(255, 0)
    print(f"{'(unmatched)':18s} {um:>14,} {um/max(len(prov), 1)*100:7.2f}%")
    print(f"  (ambiguous rows are counted under the first-listed source: {order_txt})")

    if not args.qe_scores:
        report["top_k_note"] = ("NOT COMPUTED: --qe-scores was not given. null means the enrichment "
                                "table is absent, not that enrichment is zero.")
        print(f"\ntop-k cross-tab: {report['top_k_note']}")
    else:
        scores = np.fromiter(
            (float(l.split("\t", 1)[0]) for l in open(args.qe_scores, encoding="utf-8", newline="\n")),
            dtype=np.float32)
        if len(scores) != len(prov):
            report["top_k_note"] = (f"NOT COMPUTED: QE rows ({len(scores):,}) != corpus rows ({len(prov):,}); "
                                    "null means absent, not zero enrichment.")
            print(f"\n  ⚠️  QE rows ({len(scores):,}) != corpus rows ({len(prov):,}); "
                  "skipping cross-tab. They must be in the same order.")
        else:
            report["top_k"] = {}
            report["top_k_note"] = f"top-{args.top_k} rows by QE score from {args.qe_scores}"
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

    with open(f"{args.out}_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}_labels.npy and {args.out}_report.json")
    print("These exact counts replace every tilde-hedged figure in tab:qe_source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
