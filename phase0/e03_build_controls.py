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
  heldout_<domain> (default un, europarl) : in-domain test sets. The domain
      story predicts QE-filtered FT should IMPROVE these while it degrades
      newstest. The paper only ever showed a training-loss curve, which cannot
      distinguish "learned the FT distribution" from "broke".
  ft_topk    : reproduction of the paper's selection
  ft_random  : same size, uniformly sampled -- isolates "fine-tuning at all"
  ft_bottom  : same size, lowest QE -- if this degrades LESS than top-k, quality
               is not what drives the effect

Held-out rows are removed from the pool BEFORE any FT set is drawn. Removal is
by exact pair text AND by normalised source AND by normalised target (NFKC,
casefold, whitespace collapsed), so no FT set contains a held-out source
sentence or a held-out reference under a different partner.

FAIL-CLOSED
-----------
Every condition that used to print a warning and skip the held-out sets is now
a hard exit (status 1): missing --provenance, provenance/scored row-count
mismatch, no e01 report and no --sources, a requested held-out domain that is
not a source name in the report, and a held-out pool smaller than --n-heldout.
Without held-out sets the gate cannot return GO, so building FT sets without
them only burns GPU time. --allow-no-heldout turns these back into skips; the
manifest then records why ("heldout_skipped").

A scored line with fewer than 3 tab-separated fields is also a hard exit: the
old reader skipped it, silently shifting every later row against the
provenance labels.

MEMORY
------
Two passes. Pass 1 streams the TSV into a float32 score and three uint64 hashes
per row (~28 B/row; ~1.1 GB for 38.3M rows). Selection and every leakage check
run on those arrays. Pass 2 streams the TSV again and keeps only the selected
rows' text, verifying each row's pair hash against pass 1 (so a file that
changed between passes is caught). Output files are identical to the one-pass
version whenever the source/target reservation removes nothing beyond the
exact-pair reservation.

OPTIONS THAT ENCODE PENDING USER DECISIONS (defaults = current pre-registration)
-------------------------------------------------------------------------------
  --pool-mask PATH      (F10) default: none = the full scored pool.
        Option A: mask = ~isnan(reuse_scores.npy), the 16,646,992 rows the paper
        selected from (ft_topk reproduces the paper, ft_random is a within-mix
        control, ~70% UN). Option B: no mask, the full 38.3M pool (ft_topk is a
        new selection, ft_random ~55% giga-fren, which the baseline never saw).
        With a mask, NaN scores are allowed outside it and are a hard error in it.
  --pretrain-src/--pretrain-tgt  (F9) default: none = held-out pools may contain
        pretraining pairs (as pre-registered; heldout_europarl is then ~100% and
        heldout_un ~40% inside Base v1.1's pretraining corpus). Given, held-out
        pools exclude any row whose exact pair, normalised source or normalised
        target occurs in the pretraining corpus. FT sets are NOT changed by it;
        their overlap fraction is only recorded.
  --heldout-domains     (F17) default "un,europarl". Adding e.g. giga-fren builds
        a reported-only set; whether it is part of the gate is e03_decide's
        --indomain, which this flag does not change. Names match the e01 report's
        source names exactly (not by substring).

USAGE
-----
    python e03_build_controls.py \
        --qe-scores data_enfr_v2/v2_scored.tsv \
        --provenance phase0/provenance_labels.npy \
        --out-dir data/phase0 --n-ft 1000000 --n-heldout 2000
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import unicodedata
from array import array
from pathlib import Path

import numpy as np

UNMATCHED = 255                      # e01_provenance.py's label for unmatched rows
FP_CHUNK = 16 * 1024 * 1024          # fingerprint window for large inputs
NORMALISATION = "NFKC, casefold, whitespace runs collapsed to one space, stripped"


def h64(src: str, tgt: str) -> int:
    """Stable 64-bit key for a sentence PAIR. Same function family as e01."""
    d = hashlib.blake2b(f"{src}\x00{tgt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def nh64(s: str) -> int:
    """64-bit key for ONE side after normalisation."""
    d = hashlib.blake2b(norm(s).encode("utf-8"), digest_size=8, person=b"e03norm").digest()
    return int.from_bytes(d, "little")


def die(msg: str):
    sys.exit(f"ERROR: {msg}")


def fingerprint(path) -> dict | None:
    if not path:
        return None
    p = Path(path)
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
        f.seek(max(0, size - FP_CHUNK))
        tail = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
    return {"path": str(p.resolve()), "size": size,
            "blake2b_first_16MiB": head, "blake2b_last_16MiB": tail}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_scored(line: str, lineno: int, path):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        die(f"{path} line {lineno}: {len(parts)} tab-separated field(s), need score\\tsrc\\ttgt. "
            "Skipping it would shift every later row against the provenance labels and pool mask.")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qe-scores", required=True, help="score\\tsrc\\ttgt")
    ap.add_argument("--provenance", help="labels .npy from e01_provenance.py (required unless --allow-no-heldout)")
    ap.add_argument("--provenance-report",
                    help="e01's *_report.json; defaults to the --provenance path with "
                         "_labels.npy -> _report.json. The label ORDER comes from here, "
                         "not from --sources, because retyping it by hand silently "
                         "mislabels the held-out sets.")
    ap.add_argument("--sources", default="",
                    help="fallback only, used if no report is found; must match e01's "
                         "--corpus order exactly")
    ap.add_argument("--heldout-domains", default="un,europarl",
                    help="comma list of e01 source names (exact match) to reserve as "
                         "heldout_<name> test sets, drawn in this order (default: un,europarl)")
    ap.add_argument("--allow-no-heldout", action="store_true",
                    help="skip (instead of failing on) held-out sets that cannot be built")
    ap.add_argument("--pool-mask",
                    help="bool .npy, one entry per scored row; FT and held-out sets are drawn "
                         "only from True rows (default: all rows)")
    ap.add_argument("--pretrain-src", help="pretraining corpus source side; held-out pools "
                    "exclude rows whose pair, normalised source or normalised target occurs in it")
    ap.add_argument("--pretrain-tgt", help="pretraining corpus target side")
    ap.add_argument("--out-dir", default="data/phase0")
    ap.add_argument("--n-ft", type=int, default=1_000_000)
    ap.add_argument("--n-heldout", type=int, default=2000)
    ap.add_argument("--src-ext", default="en")
    ap.add_argument("--tgt-ext", default="fr")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if bool(args.pretrain_src) != bool(args.pretrain_tgt):
        die("--pretrain-src and --pretrain-tgt must be given together")
    domains = [s.strip() for s in args.heldout_domains.split(",") if s.strip()]
    for dname in domains:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", dname):
            die(f"--heldout-domains entry {dname!r} is not a plain source name")
    if len(set(domains)) != len(domains):
        die(f"--heldout-domains lists a name twice: {domains}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- pass 1: scores + hashes only ---------------------------------
    print("pass 1: reading scores and hashing pairs (no text kept)…")
    sc_a, ph_a, sh_a, th_a = array("d"), array("Q"), array("Q"), array("Q")
    with open(args.qe_scores, encoding="utf-8", newline="\n") as f:  # never universal newlines on corpus text
        for ln, line in enumerate(f, 1):
            parts = split_scored(line, ln, args.qe_scores)
            try:
                sc_a.append(float(parts[0]))
            except ValueError:
                die(f"{args.qe_scores} line {ln}: unparseable score {parts[0][:40]!r}")
            s, t = parts[1], parts[2]
            ph_a.append(h64(s, t))
            sh_a.append(nh64(s))
            th_a.append(nh64(t))
            if ln % 5_000_000 == 0:
                print(f"  {ln:,} rows")
    scores = np.asarray(sc_a, dtype=np.float32)
    del sc_a
    row_h = np.frombuffer(ph_a, dtype=np.uint64)
    src_nh = np.frombuffer(sh_a, dtype=np.uint64)
    tgt_nh = np.frombuffer(th_a, dtype=np.uint64)
    n = len(scores)
    if n == 0:
        die(f"{args.qe_scores} has no rows")

    manifest: dict = {"n_pool_total": int(n), "n_ft": args.n_ft,
                      "n_heldout": args.n_heldout, "seed": args.seed}

    # ---- pool mask and NaN policy -------------------------------------
    nan = np.isnan(scores)
    if args.pool_mask:
        try:
            mask = np.load(args.pool_mask, allow_pickle=False)
        except Exception as exc:                                    # noqa: BLE001
            die(f"cannot load --pool-mask {args.pool_mask}: {exc}")
        if mask.dtype != np.bool_ or mask.ndim != 1:
            die(f"--pool-mask must be a 1-D bool array, got dtype={mask.dtype} shape={mask.shape}")
        if len(mask) != n:
            die(f"--pool-mask has {len(mask):,} entries but the scored file has {n:,} rows")
        bad_nan = int((nan & mask).sum())
        if bad_nan:
            die(f"{bad_nan:,} NaN QE scores inside --pool-mask — argsort would silently "
                "sort them to one end and poison ft_topk/ft_bottom. Fix upstream.")
        eligible = mask
        manifest["pool_mask"] = {"path": str(Path(args.pool_mask).resolve()),
                                 "sha256": sha256_file(args.pool_mask),
                                 "size": int(len(mask)), "n_true": int(mask.sum())}
        manifest["nan_scores_outside_mask"] = int(nan.sum())
        print(f"  pool mask {args.pool_mask}: {int(mask.sum()):,} of {n:,} rows eligible "
              f"({int(nan.sum()):,} NaN scores, all outside the mask)")
    else:
        if nan.any():
            die(f"{int(nan.sum()):,} NaN QE scores — argsort would silently sort them to one "
                "end and poison ft_topk/ft_bottom. Fix upstream, or pass --pool-mask that "
                "excludes them.")
        eligible = np.ones(n, dtype=bool)
        manifest["pool_mask"] = None
    manifest["n_pool_eligible"] = int(eligible.sum())
    es = scores[eligible]
    print(f"  {n:,} pairs, {len(es):,} eligible, QE range "
          f"[{es.min() if len(es) else float('nan'):.4f}, {es.max() if len(es) else float('nan'):.4f}]")
    del es

    n_uniq = len(np.unique(row_h))
    if n_uniq < n:
        print(f"  corpus contains {n - n_uniq:,} duplicate pairs "
              f"({(n - n_uniq) / n * 100:.2f}%) — held-out reservation is by TEXT, not index")

    # ---- pretraining corpus -------------------------------------------
    pre = None
    if args.pretrain_src:
        print(f"reading pretraining corpus {args.pretrain_src} / {args.pretrain_tgt}…")
        pp, ps, pt = array("Q"), array("Q"), array("Q")
        with open(args.pretrain_src, encoding="utf-8", newline="\n") as a, \
             open(args.pretrain_tgt, encoding="utf-8", newline="\n") as b:
            na = 0
            for s, t in itertools.zip_longest(a, b):
                if s is None or t is None:
                    die(f"pretraining corpus sides differ in length (first unpaired line {na + 1:,})")
                s = s[:-1] if s.endswith("\n") else s
                t = t[:-1] if t.endswith("\n") else t
                pp.append(h64(s, t)); ps.append(nh64(s)); pt.append(nh64(t))
                na += 1
        pre = {k: np.unique(np.frombuffer(v, dtype=np.uint64)) for k, v in
               (("pair", pp), ("source", ps), ("target", pt))}
        del pp, ps, pt
        in_pre = {"pair": np.isin(row_h, pre["pair"]),
                  "source": np.isin(src_nh, pre["source"]),
                  "target": np.isin(tgt_nh, pre["target"])}
        pre_any = in_pre["pair"] | in_pre["source"] | in_pre["target"]
        manifest["pretrain"] = {
            "rows": int(na), "distinct_pairs": int(len(pre["pair"])),
            "distinct_sources_norm": int(len(pre["source"])),
            "distinct_targets_norm": int(len(pre["target"])),
            "scored_rows_in_pretrain": {k: int(v.sum()) for k, v in in_pre.items()} | {"any": int(pre_any.sum())},
        }
        print(f"  {na:,} pretraining rows; {int(pre_any.sum()):,} scored rows share a pair, "
              "source or target with it")
    else:
        in_pre = None
        pre_any = np.zeros(n, dtype=bool)
        manifest["pretrain"] = None

    # ---- 1. provenance and held-out domain test sets ------------------
    manifest["heldout_domains"] = domains
    manifest["allow_no_heldout"] = bool(args.allow_no_heldout)
    skipped: list[str] = []

    def soft(msg: str) -> None:
        if not args.allow_no_heldout:
            die(msg + "\n  Without held-out sets the gate cannot return GO. Fix the inputs, or "
                "pass --allow-no-heldout to build FT sets only.")
        print(f"  ⚠️  {msg} (--allow-no-heldout: continuing)")
        skipped.append(msg)

    reserved = np.zeros(n, dtype=bool)
    held: dict[str, np.ndarray] = {}
    held_info: dict[str, dict] = {}
    prov = None
    names: list[str] = []
    rep_path = None
    rep = None
    if not args.provenance:
        soft("no --provenance: cannot build domain held-out sets (run e01_provenance.py first)")
    else:
        prov = np.load(args.provenance, allow_pickle=False)
        rep_path = args.provenance_report
        if not rep_path and args.provenance.endswith("_labels.npy"):
            rep_path = args.provenance[: -len("_labels.npy")] + "_report.json"
        if rep_path and Path(rep_path).exists():
            rep = json.load(open(rep_path, encoding="utf-8"))
            names = list(rep.get("sources", []))
            rate = rep.get("match_rate")
            print(f"  source order from {rep_path}: {names}")
            if rate is not None:
                print(f"  e01 match rate {rate*100:.2f}% (norm={rep.get('normalizer')})")
            if rate is not None and rate < 0.95:
                die(f"e01 match rate {rate*100:.2f}% < 95%: the provenance "
                    "labels are not trustworthy, so neither would the "
                    "held-out domain sets be. Re-run e01 with a looser "
                    "--norm and report which one you needed.")
            cli_names = [s.strip() for s in args.sources.split(",") if s.strip()]
            if cli_names and cli_names != names:
                die(f"--sources {cli_names} does not match e01's order "
                    f"{names}. Drop --sources; the report is authoritative.")
        else:
            rep_path = None
            names = [s.strip() for s in args.sources.split(",") if s.strip()]
            if names:
                print(f"  ⚠️  no e01 report found; falling back to --sources {names}. "
                      "A wrong order here silently mislabels the held-out sets.")
        if len(prov) != n:
            soft(f"provenance rows ({len(prov):,}) != scored rows ({n:,}): the labels do not "
                 "describe this file (partial or wrong scored TSV?)")
            prov = None
        elif not names:
            soft("no e01 report next to --provenance and no --sources: the label codes "
                 "cannot be named")
        if prov is not None and names:
            bad = (prov.astype(np.int64) >= len(names)) & (prov != UNMATCHED)
            if bad.any():
                die(f"{int(bad.sum()):,} provenance labels are outside 0..{len(names)-1} "
                    f"(and not {UNMATCHED}=unmatched): the report's source list {names} does "
                    "not belong to these labels")
            for want in domains:
                if want not in names:
                    soft(f"held-out domain '{want}' is not a source name in {names} "
                         "(names match exactly, not by substring)")
                    continue
                idx = names.index(want)
                in_dom = prov == idx
                cand = in_dom & eligible & ~reserved
                pool = np.flatnonzero(cand & ~pre_any)
                info = {"domain": want, "label_rows": int(in_dom.sum()),
                        "pool_rows": int(len(pool)),
                        "pool_rows_excluded_pretrain": int((cand & pre_any).sum()),
                        "pool_rows_excluded_mask": int((in_dom & ~eligible).sum()),
                        "pool_rows_excluded_prior_heldout": int((in_dom & eligible & reserved).sum())}
                if len(pool) < args.n_heldout:
                    soft(f"only {len(pool):,} eligible '{want}' pairs < --n-heldout {args.n_heldout:,}")
                    continue
                pick = rng.choice(pool, args.n_heldout, replace=False)
                # Index-level exclusion is NOT enough: the cleaner does not
                # deduplicate, and a held-out source can recur with another
                # target (or a reference with another source). Reserve every row
                # sharing the exact pair, the normalised source or the normalised
                # target of any held-out row.
                by_pair = np.isin(row_h, np.unique(row_h[pick]))
                by_src = np.isin(src_nh, np.unique(src_nh[pick]))
                by_tgt = np.isin(tgt_nh, np.unique(tgt_nh[pick]))
                new_pair = by_pair & ~reserved
                new_src = by_src & ~by_pair & ~reserved
                new_tgt = by_tgt & ~by_src & ~by_pair & ~reserved
                info.update({"n": int(len(pick)),
                             "distinct_pairs": int(len(np.unique(row_h[pick]))),
                             "reserved_rows_exact_pair": int(new_pair.sum()),
                             "reserved_rows_extra_by_source": int(new_src.sum()),
                             "reserved_rows_extra_by_target": int(new_tgt.sum())})
                reserved |= by_pair | by_src | by_tgt
                extra = info["reserved_rows_exact_pair"] - len(pick)
                if extra > 0:
                    print(f"    +{extra:,} duplicate rows of the same pairs also reserved")
                if info["reserved_rows_extra_by_source"] or info["reserved_rows_extra_by_target"]:
                    print(f"    +{info['reserved_rows_extra_by_source']:,} rows sharing a held-out "
                          f"source, +{info['reserved_rows_extra_by_target']:,} sharing a held-out "
                          "target (different partner) also reserved")
                held[f"heldout_{want}"] = pick
                held_info[f"heldout_{want}"] = info
                manifest[f"heldout_{want}"] = int(len(pick))
                print(f"  reserved {len(pick):,} '{want}' pairs as a held-out test set "
                      f"(pool {len(pool):,})")
    manifest["heldout_skipped"] = skipped
    if not args.allow_no_heldout:
        missing = [d for d in domains if f"heldout_{d}" not in held]
        if missing or not domains:
            die(f"held-out sets not built: {missing or '(no --heldout-domains)'}")
    manifest["reservation"] = {
        "normalisation": NORMALISATION,
        "rows_reserved_total": int(reserved.sum()),
        "exact_pair": int(sum(v["reserved_rows_exact_pair"] for v in held_info.values())),
        "extra_by_source": int(sum(v["reserved_rows_extra_by_source"] for v in held_info.values())),
        "extra_by_target": int(sum(v["reserved_rows_extra_by_target"] for v in held_info.values())),
    }

    avail = np.flatnonzero(eligible & ~reserved)
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
    if len(uniq) < 2 * k or k == 0:
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

    # ---- 3. leakage assertions (on the pass-1 arrays) ------------------
    # explicit exits, not `assert`: python -O must not disable them
    if held:
        allh = np.concatenate(list(held.values()))
        held_keys = {"pair": np.unique(row_h[allh]), "source": np.unique(src_nh[allh]),
                     "target": np.unique(tgt_nh[allh])}
        for name, idx in sets.items():
            for kind, arr in (("pair", row_h), ("source", src_nh), ("target", tgt_nh)):
                leaked = np.intersect1d(arr[idx], held_keys[kind])
                if len(leaked):
                    die(f"{name} leaks {len(leaked)} held-out {kind}(s) by text")
        print("\n  ✓ no FT set shares a pair, normalised source or normalised target with any held-out row")
        if in_pre is not None and pre_any[allh].any():
            die("a held-out row occurs in the pretraining corpus (array check)")
    for name, idx in list(sets.items()) + list(held.items()):
        if not eligible[idx].all():
            die(f"{name} contains rows outside --pool-mask")

    # (b) ft_topk and ft_bottom must be disjoint; ft_random overlaps both by
    #     construction and that is fine, but the amount must be reported, since
    #     a large overlap weakens it as a control.
    # Disjoint by ROW, as the one-pass version checked. The same pair text CAN sit
    # in both when its duplicates carry very different scores (top-k keeps the
    # high copy, the unique pool keeps the first copy); with real QE scores
    # duplicates score alike, so this is recorded rather than fatal.
    if len(np.intersect1d(sets["ft_topk"], sets["ft_bottom"])):
        die("ft_topk and ft_bottom overlap")
    tb_text = int(len(np.intersect1d(row_h[sets["ft_topk"]], row_h[sets["ft_bottom"]])))
    manifest["ft_topk_bottom_text_overlap"] = tb_text
    if tb_text:
        print(f"  ⚠️  {tb_text:,} pair texts occur in both ft_topk and ft_bottom (duplicates "
              "with very different QE scores)")
    for name, idx in sets.items():
        if len(np.unique(row_h[idx])) != len(idx):
            die(f"{name} contains duplicate pairs")
        if len(idx) != k:
            die(f"{name} has {len(idx)} rows, expected {k}")
    ov = len(np.intersect1d(sets["ft_random"], sets["ft_topk"]))
    print(f"  ✓ ft_topk n ft_bottom = 0"
          f"   |   ft_random n ft_topk = {ov:,} ({ov/k*100:.2f}%, expected "
          f"{k/len(uniq)*100:.2f}% by chance)")
    manifest["ft_random_topk_overlap"] = ov

    # ---- pass 2: text of the selected rows only -----------------------
    written = list(held.items()) + list(sets.items())
    need = np.unique(np.concatenate([v for _, v in written]))
    print(f"\npass 2: collecting text for {len(need):,} selected rows…")
    txt_s: list[str] = [""] * len(need)
    txt_t: list[str] = [""] * len(need)
    j, m = 0, len(need)
    nxt = int(need[0])
    with open(args.qe_scores, encoding="utf-8", newline="\n") as f:
        for i, line in enumerate(f):
            if i != nxt:
                continue
            parts = split_scored(line, i + 1, args.qe_scores)
            if h64(parts[1], parts[2]) != int(row_h[i]):
                die(f"{args.qe_scores} row {i + 1} changed between pass 1 and pass 2")
            txt_s[j], txt_t[j] = parts[1], parts[2]
            j += 1
            if j == m:
                break
            nxt = int(need[j])
    if j != m:
        die(f"{args.qe_scores} shrank between pass 1 and pass 2 ({j:,} of {m:,} selected rows found)")

    tmp_files = []
    for name, idx in written:
        pos = np.searchsorted(need, idx)
        stem = str(out / name)
        for ext, col in ((args.src_ext, txt_s), (args.tgt_ext, txt_t)):
            tmp = f"{stem}.{ext}.tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
                for p in pos.tolist():
                    fo.write(col[p] + "\n")
            tmp_files.append((tmp, f"{stem}.{ext}"))
    del txt_s, txt_t

    # ---- post-write checks, recomputed from the files ------------------
    wkeys = {}
    for name, idx in written:
        stem = str(out / name)
        sa, tb = [], []
        with open(f"{stem}.{args.src_ext}.tmp", encoding="utf-8", newline="\n") as a, \
             open(f"{stem}.{args.tgt_ext}.tmp", encoding="utf-8", newline="\n") as b:
            sa = [x[:-1] for x in a]
            tb = [x[:-1] for x in b]
        if len(sa) != len(idx) or len(tb) != len(idx):
            die(f"{name}: wrote {len(sa)}/{len(tb)} lines, expected {len(idx)}")
        ph = np.fromiter((h64(a, b) for a, b in zip(sa, tb)), np.uint64, len(sa))
        if not np.array_equal(ph, row_h[idx]):
            die(f"{name}: written text does not match the selected rows")
        wkeys[name] = (ph, np.fromiter((nh64(a) for a in sa), np.uint64, len(sa)),
                       np.fromiter((nh64(b) for b in tb), np.uint64, len(tb)))
    if held:
        hk = [np.unique(np.concatenate([wkeys[h][c] for h in held])) for c in range(3)]
        for name in sets:
            for c, kind in enumerate(("pair", "source", "target")):
                if len(np.intersect1d(wkeys[name][c], hk[c])):
                    die(f"written {name} shares a held-out {kind} (file re-check)")
    if pre is not None:
        for name in held:
            for c, kind in enumerate(("pair", "source", "target")):
                hit = np.isin(wkeys[name][c], pre[kind]).sum()
                if hit:
                    die(f"written {name}: {int(hit)} {kind}(s) occur in the pretraining corpus")
        manifest["pretrain"]["heldout_post_write_check"] = "no held-out pair, source or target occurs in pretraining"
        print("  ✓ no held-out pair, normalised source or normalised target occurs in the pretraining corpus")
    del wkeys

    # ---- composition, manifest ----------------------------------------
    label_names = (names if names else [f"code_{i}" for i in range(int(prov.max()) + 1 if prov is not None and len(prov) else 0)])

    def composition(idx):
        if prov is None:
            return None
        lab = prov[idx]
        cnt = {nm: int((lab == i).sum()) for i, nm in enumerate(label_names)}
        cnt["unmatched"] = int((lab == UNMATCHED).sum())
        return cnt

    def overlap(idx):
        if in_pre is None:
            return None
        return {k2: float(v[idx].mean()) for k2, v in in_pre.items()} | {"any": float(pre_any[idx].mean())}

    for name, idx in sets.items():
        s = scores[idx]
        manifest[name] = {"n": int(len(idx)), "qe_mean": float(s.mean()),
                          "qe_min": float(s.min()), "qe_max": float(s.max()),
                          "source_counts": composition(idx),
                          "pretrain_overlap_fraction": overlap(idx)}
        print(f"  {name:10s} n={len(idx):,}  QE mean={s.mean():.4f} "
              f"[{s.min():.4f}, {s.max():.4f}]")
    manifest["heldout_sets"] = {}
    for name, idx in held.items():
        manifest["heldout_sets"][name] = dict(held_info[name], source_counts=composition(idx),
                                              pretrain_overlap_fraction=overlap(idx))

    if prov is not None:
        cols = label_names + ["unmatched"]
        print("\ncomposition (% of rows by e01 source)")
        print(f"  {'set':<20} {'n':>9} " + " ".join(f"{c[:11]:>11}" for c in cols)
              + ("  in-pretrain" if in_pre is not None else ""))
        for name, idx in written:
            c = composition(idx)
            row = " ".join(f"{c[x] / len(idx) * 100:>10.1f}%" for x in cols)
            extra = f"  {pre_any[idx].mean() * 100:>10.1f}%" if in_pre is not None else ""
            print(f"  {name:<20} {len(idx):>9,} {row}{extra}")

    manifest["inputs"] = {
        "qe_scores": fingerprint(args.qe_scores),
        "provenance": fingerprint(args.provenance),
        "provenance_report": fingerprint(rep_path),
        "pool_mask": fingerprint(args.pool_mask),
        "pretrain_src": fingerprint(args.pretrain_src),
        "pretrain_tgt": fingerprint(args.pretrain_tgt),
        "report_match_rate": rep.get("match_rate") if rep else None,
    }
    final_files = [dst for _, dst in tmp_files] + [str(out / "manifest.json")]
    manifest["written"] = [Path(p).name for p in final_files]

    (out / "manifest.json").unlink(missing_ok=True)       # never leave a manifest describing old files
    for tmp, dst in tmp_files:
        os.replace(tmp, dst)
    with open(out / "manifest.json.tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    os.replace(out / "manifest.json.tmp", out / "manifest.json")

    stale = sorted(p.name for p in out.glob("heldout_*") if p.name not in manifest["written"])
    if stale:
        print(f"\n  ⚠️  files in {out} NOT from this run (not in manifest.json): {stale}")
    print(f"\nwrote {out}/: FT sets {list(sets)} ({k:,} pairs each), "
          + (f"held-out sets {list(held)} ({args.n_heldout:,} pairs each)" if held
             else "NO held-out sets (--allow-no-heldout) — the gate cannot return GO on these")
          + ", manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
