#!/usr/bin/env python3
"""E0.3a — Build the fine-tuning control sets and held-out domain test sets.

WHY THIS EXISTS
---------------
The rejected paper's QE-filtered fine-tuning result (-1.54 / -2.33 BLEU) has
NO CONTROL. Three explanations fit the evidence equally well:

  1. domain shift          (the paper's claim)
  2. restart-LR + forgetting  -- optimizer state was reset and Noam restarted at
     an effective peak of 2.73e-4 on an already-converged checkpoint, and BLEU
     falls monotonically from the FIRST eval, which is the textbook signature
  3. simply fine-tuning on 1M pairs at all, regardless of which 1M

Only (1) supports the paper. This script builds the sets that separate them.

A fourth candidate -- that filter_by_score.py writes top-K sorted high-to-low,
feeding a monotone quality curriculum -- was RULED OUT by reading the code:
TokenBatchSampler shuffles indices, re-sorts only within length chunks, then
shuffles batch order, and the trainer builds the loader with shuffle=True.
File order does not survive. No control needed for it.

WHAT IT BUILDS
--------------
  heldout_un / heldout_europarl : in-domain test sets. The domain story predicts
      QE-filtered FT should IMPROVE these while it degrades newstest. The paper
      only ever showed a training-loss curve, which cannot distinguish "learned
      the FT distribution" from "broke".
  ft_topk    : reproduction of the paper's selection
  ft_random  : same size, uniformly sampled -- isolates "fine-tuning at all"
  ft_bottom  : same size, lowest QE -- if this degrades LESS than top-k, quality
               is not what drives the effect

Held-out pairs are removed from the pool BEFORE any FT set is drawn, so no FT
set can contain them.

USAGE
-----
    python e03_build_controls.py \
        --qe-scores data/v2_scored.tsv \
        --provenance phase0/provenance_labels.npy \
        --sources europarl,commoncrawl,un,news-commentary,giga-fren \
        --out-dir data/phase0 --n-ft 1000000 --n-heldout 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def h64(src: str, tgt: str) -> int:
    """Stable 64-bit key for a sentence PAIR. Same function family as e01."""
    d = hashlib.blake2b(f"{src}\x00{tgt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little")


def write_pair(path_stem: str, rows, src_ext: str, tgt_ext: str) -> None:
    with open(f"{path_stem}.{src_ext}", "w", encoding="utf-8") as fs, \
         open(f"{path_stem}.{tgt_ext}", "w", encoding="utf-8") as ft:
        for s, t in rows:
            fs.write(s + "\n")
            ft.write(t + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qe-scores", required=True, help="score\\tsrc\\ttgt")
    ap.add_argument("--provenance", help="labels .npy from e01_provenance.py")
    ap.add_argument("--provenance-report",
                    help="e01's *_report.json; defaults to the --provenance path with "
                         "_labels.npy -> _report.json. The label ORDER comes from here, "
                         "not from --sources, because retyping it by hand silently "
                         "mislabels the held-out sets.")
    ap.add_argument("--sources", default="",
                    help="fallback only, used if no report is found; must match e01's "
                         "--corpus order exactly")
    ap.add_argument("--out-dir", default="data/phase0")
    ap.add_argument("--n-ft", type=int, default=1_000_000)
    ap.add_argument("--n-heldout", type=int, default=2000)
    ap.add_argument("--src-ext", default="en")
    ap.add_argument("--tgt-ext", default="fr")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print("reading scored corpus…")
    scores, src, tgt = [], [], []
    with open(args.qe_scores, encoding="utf-8", newline="\n") as f:  # never universal newlines on corpus text
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            scores.append(float(parts[0]))
            src.append(parts[1])
            tgt.append(parts[2])
    scores = np.asarray(scores, dtype=np.float32)
    n = len(scores)
    print(f"  {n:,} pairs, QE range [{scores.min():.4f}, {scores.max():.4f}]")

    if np.isnan(scores).any():
        sys.exit(f"{int(np.isnan(scores).sum()):,} NaN QE scores — argsort would "
                 "silently sort them to one end and poison ft_topk/ft_bottom. Fix upstream.")

    print("hashing pairs for text-level dedup…")
    row_h = np.fromiter((h64(a, b) for a, b in zip(src, tgt)), dtype=np.uint64, count=n)
    n_uniq = len(np.unique(row_h))
    if n_uniq < n:
        print(f"  corpus contains {n - n_uniq:,} duplicate pairs "
              f"({(n - n_uniq) / n * 100:.2f}%) — held-out reservation is by TEXT, not index")

    manifest: dict = {"n_pool_total": int(n), "n_ft": args.n_ft,
                      "n_heldout": args.n_heldout, "seed": args.seed}

    # ---- 1. reserve held-out domain test sets --------------------------
    reserved = np.zeros(n, dtype=bool)
    if args.provenance:
        prov = np.load(args.provenance)
        if len(prov) != n:
            print(f"  ⚠️  provenance rows ({len(prov):,}) != scored rows ({n:,}); "
                  "cannot build domain held-out sets. Fix the ordering first.")
        else:
            rep_path = args.provenance_report
            if not rep_path and args.provenance.endswith("_labels.npy"):
                rep_path = args.provenance[: -len("_labels.npy")] + "_report.json"
            names = []
            if rep_path and Path(rep_path).exists():
                rep = json.load(open(rep_path, encoding="utf-8"))
                names = list(rep.get("sources", []))
                rate = rep.get("match_rate")
                print(f"  source order from {rep_path}: {names}")
                print(f"  e01 match rate {rate*100:.2f}% (norm={rep.get('normalizer')})")
                if rate is not None and rate < 0.95:
                    sys.exit(f"e01 match rate {rate*100:.2f}% < 95%: the provenance "
                             "labels are not trustworthy, so neither would the "
                             "held-out domain sets be. Re-run e01 with a looser "
                             "--norm and report which one you needed.")
                cli_names = [s.strip() for s in args.sources.split(",") if s.strip()]
                if cli_names and cli_names != names:
                    sys.exit(f"--sources {cli_names} does not match e01's order "
                             f"{names}. Drop --sources; the report is authoritative.")
            else:
                names = [s.strip() for s in args.sources.split(",") if s.strip()]
                print(f"  ⚠️  no report at {rep_path}; falling back to --sources. "
                      "A wrong order here silently mislabels the held-out sets.")
            for want in ("un", "europarl"):
                idx = next((i for i, nm in enumerate(names) if want in nm.lower()), None)
                if idx is None:
                    print(f"  ⚠️  no source matching '{want}'; skipping its held-out set")
                    continue
                pool = np.flatnonzero(prov == idx)
                if len(pool) < args.n_heldout:
                    print(f"  ⚠️  only {len(pool):,} '{want}' pairs; skipping")
                    continue
                pick = rng.choice(pool, args.n_heldout, replace=False)
                # Index-level exclusion is NOT enough. clean_data_enfr.py is a
                # pure filter and does not deduplicate, so the same pair can
                # appear at many indices; reserving one index leaves its twins
                # free to land in an FT set, i.e. verbatim test pairs in train.
                # Reserve by TEXT: every row whose pair-hash matches a held-out
                # pair is removed from the pool.
                before = int(reserved.sum())
                reserved[np.isin(row_h, np.unique(row_h[pick]))] = True
                extra = int(reserved.sum()) - before - len(pick)
                if extra > 0:
                    print(f"    +{extra:,} duplicate rows of the same pairs also reserved")
                write_pair(str(out / f"heldout_{want}"),
                           [(src[i], tgt[i]) for i in pick],
                           args.src_ext, args.tgt_ext)
                manifest[f"heldout_{want}"] = int(len(pick))
                print(f"  reserved {len(pick):,} '{want}' pairs as a held-out test set")
    else:
        print("  (no --provenance: skipping domain held-out sets; "
              "run e01_provenance.py first — the domain claim cannot be tested without them)")

    avail = np.flatnonzero(~reserved)
    print(f"\npool after reserving held-out: {len(avail):,}")

    # ---- 2. the three FT sets: same UNIQUE size, each duplicate-free ----
    # The paper's FT set was built as top-n_ft rows by QE, THEN exact (src, tgt)
    # dedup: "Sorting and unique-ing the top-1M yields 931,366 unique (src, tgt)
    # pairs ... All SFT below uses this 931K-pair deduplicated set"
    # (paper_section_7.md:17). ft_topk reproduces that rule exactly. The controls
    # must match it in UNIQUE size and be duplicate-free themselves; otherwise
    # ft_random / ft_bottom carry repeated pairs the top-k arm does not, and the
    # arms differ in effective data, not only in QE.
    order = avail[np.argsort(-scores[avail], kind="stable")]
    top_sel = order[:args.n_ft]
    _, first = np.unique(row_h[top_sel], return_index=True)
    topk = top_sel[np.sort(first)]                      # score order, first copy kept
    k = len(topk)

    _, first_all = np.unique(row_h[avail], return_index=True)
    uniq = avail[np.sort(first_all)]                    # one row per distinct pair
    # With a unique pool < 2k, ft_topk and ft_bottom overlap and the controls stop
    # being controls. A warning is not enough — the run would look valid.
    if len(uniq) < 2 * k:
        sys.exit(f"unique pool {len(uniq):,} < 2 x unique set size ({2*k:,}): ft_topk "
                 "and ft_bottom would overlap and the comparison would be vacuous. "
                 "Lower --n-ft or use a larger corpus.")
    uorder = uniq[np.argsort(-scores[uniq], kind="stable")]
    sets = {
        "ft_topk":   topk,                                   # the paper's rule
        "ft_bottom": uorder[-k:],                            # lowest-QE unique pairs
        "ft_random": rng.choice(uniq, k, replace=False),     # uniform over unique pairs
    }
    manifest["n_ft_selected"] = int(args.n_ft)
    manifest["n_unique_per_set"] = int(k)
    manifest["topk_duplicates_removed"] = int(len(top_sel) - k)
    print(f"  top-{args.n_ft:,} by QE -> {k:,} unique pairs "
          f"({len(top_sel) - k:,} duplicates removed); all three sets use {k:,}")

    for name, idx in sets.items():
        write_pair(str(out / name), [(src[i], tgt[i]) for i in idx],
                   args.src_ext, args.tgt_ext)
        s = scores[idx]
        manifest[name] = {"n": int(len(idx)), "qe_mean": float(s.mean()),
                          "qe_min": float(s.min()), "qe_max": float(s.max())}
        print(f"  {name:10s} n={len(idx):,}  QE mean={s.mean():.4f} "
              f"[{s.min():.4f}, {s.max():.4f}]")

    # ---- 3. leakage assertions ----------------------------------------
    # (a) no held-out PAIR (by text, not index) in any FT set
    held_h = set(np.unique(row_h[reserved]).tolist())
    for name, idx in sets.items():
        leaked = held_h & set(np.unique(row_h[idx]).tolist())
        assert not leaked, f"{name} leaks {len(leaked)} held-out pairs by text"
    print("\n  ✓ no FT set contains any held-out pair (checked by text, not index)")

    # (b) ft_topk and ft_bottom must be disjoint; ft_random overlaps both by
    #     construction and that is fine, but the amount must be reported, since
    #     a large overlap weakens it as a control.
    top_s, bot_s, rnd_s = (set(v.tolist()) for v in
                           (sets["ft_topk"], sets["ft_bottom"], sets["ft_random"]))
    assert not (top_s & bot_s), "ft_topk and ft_bottom overlap"
    for name, idx in sets.items():
        assert len(np.unique(row_h[idx])) == len(idx), f"{name} contains duplicate pairs"
        assert len(idx) == k, f"{name} has {len(idx)} rows, expected {k}"
    ov = len(rnd_s & top_s)
    print(f"  ✓ ft_topk n ft_bottom = 0"
          f"   |   ft_random n ft_topk = {ov:,} ({ov/k*100:.2f}%, expected "
          f"{k/len(uniq)*100:.2f}% by chance)")
    manifest["ft_random_topk_overlap"] = ov

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {out}/ (3 FT sets + held-out test sets + manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
