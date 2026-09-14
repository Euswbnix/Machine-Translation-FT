#!/usr/bin/env python3
"""Re-score only what changed after rebuilding a corpus.

A CometKiwi score is a function of the pair's text alone, not of its row, so every
pair of the rebuilt corpus that already appears in the old scored TSV keeps its old
score, and only genuinely new pairs need the GPU. For v2 (rebuilt, 38,275,284 rows) that is
21,628,292 pairs to score with 16,646,992 reused: ~10 GPU-hours on one RTX 5090 instead of
~17.5 for a full rescore (both derived from the paper's 13.8 h / 30.1M rate; PROTOCOL.md D7).

  plan     old scored TSV + rebuilt corpus -> reuse_scores.npy (NaN where missing),
           missing_rows.npy, to_score.<src>/<tgt> (the pairs to send to score_with_comet.py),
           plan.json (row counts + sha256 of the rebuilt corpus the plan is bound to)
  extract  plan dir + the same corpus rebuilt elsewhere -> to_score.* again. Lets a rented box
           rebuild the corpus itself and receive only the two .npy files (326 MB for v2:
           missing_rows.npy 173,026,464 B + reuse_scores.npy 153,101,264 B, measured with
           ls -l) instead of the 8 GB old scored TSV.
  merge    plan dir + the scorer's TSV for to_score.* -> <out> scored TSV in rebuilt row order.
           Every new_scores line must be exactly 3 fields whose text equals to_score.* line by
           line (tabs -> spaces) AND the corpus row at missing_rows[k]; the first bad row is
           named and nothing is written. Output goes to <out>.tmp and is os.replace'd only
           after all checks pass, so a failure never leaves a partial file or clobbers a good
           one. --allow-missing (no --new-scores): unscored rows get the literal score "nan".
  calibrate plan dir (reuse_scores.npy) + a FULL new scored TSV + e01 labels/report -> JSON
           comparing new vs reused scores overall and per source (mean diff, p99 |diff|,
           Spearman, top-k Jaccard at 50/30/15/5%); exit 0 pass / 1 fail / 2 insufficient /
           3 invalid inputs. Threshold defaults are proposals pending user confirmation.

extract and merge refuse a corpus whose sha256 differs from the one the plan was made on.

Matching mirrors score_with_comet.py, which wrote src/tgt with tabs replaced by spaces:
the lookup key is blake2b-64 over (src, tgt) after the same replacement. At ~60M keys
the expected number of 64-bit collisions is ~1e-4, and a collision could only hand one
pair the score of another -- negligible, and stated rather than hidden.

USAGE
    python phase0/rescore_plan.py plan --old-scored v2_scored.tsv \\
        --new-src v2_fixed/train.clean.en --new-tgt v2_fixed/train.clean.fr --out-dir v2_rescore
    python score_with_comet.py --src v2_rescore/to_score.en --tgt v2_rescore/to_score.fr --out v2_rescore/new_scores.tsv
        # several GPUs: python phase0/score_sharded.py --src ... --tgt ... --out ... --gpus N
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


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_plan(plan_dir: str, new_src: str, new_tgt: str) -> dict:
    pj = json.load(open(os.path.join(plan_dir, "plan.json")))
    for side, path in (("src", new_src), ("tgt", new_tgt)):
        got = sha256(path)
        if got != pj[f"new_{side}_sha256"]:
            sys.exit(f"{path}: sha256 {got[:16]}... but the plan was made on {pj[f'new_{side}_sha256'][:16]}...; "
                     f"this is not the same rebuilt corpus, refusing")
    return pj


def write_to_score(new_src: str, new_tgt: str, missing, out_dir: str) -> None:
    src_ext = new_src.rsplit(".", 1)[-1]
    tgt_ext = new_tgt.rsplit(".", 1)[-1]
    it = iter(missing.tolist() if hasattr(missing, "tolist") else missing)
    nxt = next(it, None)
    with open(new_src, encoding="utf-8", newline="\n") as fs, open(new_tgt, encoding="utf-8", newline="\n") as ft, \
         open(os.path.join(out_dir, f"to_score.{src_ext}"), "w", encoding="utf-8", newline="\n") as os_, \
         open(os.path.join(out_dir, f"to_score.{tgt_ext}"), "w", encoding="utf-8", newline="\n") as ot:
        for i, (s, t) in enumerate(zip(fs, ft)):
            if nxt is None:
                break
            if i == nxt:
                os_.write(s if s.endswith("\n") else s + "\n")
                ot.write(t if t.endswith("\n") else t + "\n")
                nxt = next(it, None)
    if nxt is not None:
        sys.exit(f"corpus ended before missing row {nxt:,}")


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
    with open(a.new_src, encoding="utf-8", newline="\n") as fs, open(a.new_tgt, encoding="utf-8", newline="\n") as ft:
        for i, (s, t) in enumerate(zip(fs, ft)):
            k = np.uint64(key(s, t))
            j = int(np.searchsorted(K, k))
            if j < len(K) and K[j] == k:
                reuse.append(float(S[j]))
            else:
                reuse.append(math.nan)
                missing.append(i)
    R = np.asarray(reuse, dtype=np.float32)
    M = np.asarray(missing, dtype=np.int64)
    np.save(os.path.join(a.out_dir, "reuse_scores.npy"), R)
    np.save(os.path.join(a.out_dir, "missing_rows.npy"), M)
    write_to_score(a.new_src, a.new_tgt, M, a.out_dir)
    summary = {"new_rows": int(len(R)), "reused": int(len(R) - len(M)), "to_score": int(len(M)),
               "first_missing_row": int(M[0]) if len(M) else None,
               "new_src_sha256": sha256(a.new_src), "new_tgt_sha256": sha256(a.new_tgt)}
    json.dump(summary, open(os.path.join(a.out_dir, "plan.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_extract(a) -> int:
    pj = load_plan(a.plan_dir, a.new_src, a.new_tgt)
    M = np.load(os.path.join(a.plan_dir, "missing_rows.npy"))
    out = a.out_dir or a.plan_dir
    os.makedirs(out, exist_ok=True)
    write_to_score(a.new_src, a.new_tgt, M, out)
    print(f"wrote to_score.* ({len(M):,} pairs of {pj['new_rows']:,}) to {out}")
    return 0


def norm(line: str) -> str:
    return line.rstrip("\n").replace("\t", " ")


def read_new_scores(new_scores: str, to_src: str, to_tgt: str, n_expected: int) -> np.ndarray:
    """Parse the scorer's TSV and check, line by line, that its text columns are exactly
    to_score.<src>/<tgt> (tabs -> spaces). Exits naming the first bad row."""
    for p in (new_scores, to_src, to_tgt):
        if not os.path.isfile(p):
            sys.exit(f"{p} missing; refusing to merge")
    new, keys = [], []
    with open(new_scores, encoding="utf-8", newline="\n") as f, \
         open(to_src, encoding="utf-8", newline="\n") as fs, open(to_tgt, encoding="utf-8", newline="\n") as ft:
        for k, line in enumerate(f):
            if not line.endswith("\n"):
                sys.exit(f"new_scores row {k:,}: partial last line (no newline); refusing to merge")
            parts = line[:-1].split("\t")
            if len(parts) != 3:
                sys.exit(f"new_scores row {k:,}: {len(parts)} tab-separated fields, expected 3; refusing to merge")
            s, t = fs.readline(), ft.readline()
            if not s or not t:
                sys.exit(f"new_scores has more rows than to_score ({k:,}+); refusing to merge")
            if parts[1] != norm(s) or parts[2] != norm(t):
                sys.exit(f"new_scores row {k:,} text differs from to_score row {k:,}; "
                         f"the scores are not for these pairs, refusing to merge")
            try:
                v = float(parts[0])
            except ValueError:
                sys.exit(f"new_scores row {k:,}: score {parts[0]!r} is not a number; refusing to merge")
            if not math.isfinite(v):
                sys.exit(f"new_scores row {k:,}: non-finite score {parts[0]!r}; refusing to merge")
            new.append(v)
            keys.append(key(parts[1], parts[2]))
        extra_to_score = bool(fs.readline() or ft.readline())
    if len(new) != n_expected:
        sys.exit(f"scorer returned {len(new):,} scores for {n_expected:,} missing rows; refusing to merge")
    if extra_to_score:
        sys.exit(f"to_score has more rows than the {len(new):,} in new_scores; refusing to merge")
    return np.asarray(new, dtype=np.float32), np.asarray(keys, dtype=np.uint64)


def cmd_merge(a) -> int:
    pj = load_plan(a.plan_dir, a.new_src, a.new_tgt)
    R = np.load(os.path.join(a.plan_dir, "reuse_scores.npy"))
    M = np.load(os.path.join(a.plan_dir, "missing_rows.npy"))
    if len(R) != pj["new_rows"] or len(M) != pj["to_score"]:
        sys.exit(f"plan files disagree with plan.json (reuse {len(R):,} vs {pj['new_rows']:,}, "
                 f"missing {len(M):,} vs {pj['to_score']:,}); refusing to merge")
    if len(M) and (np.any(np.diff(M) <= 0) or M[0] < 0 or M[-1] >= len(R)):
        sys.exit("missing_rows.npy is not strictly increasing within the corpus; refusing to merge")
    R = R.copy()
    if a.allow_missing:
        if a.new_scores:
            sys.exit("--allow-missing merges reused scores only; it takes no --new-scores")
        R[M] = np.nan
    else:
        if not a.new_scores:
            sys.exit("--new-scores is required (or pass --allow-missing to write nan for unscored rows)")
        tdir = a.to_score_dir or a.plan_dir
        src_ext = a.new_src.rsplit(".", 1)[-1]
        tgt_ext = a.new_tgt.rsplit(".", 1)[-1]
        R[M], new_keys = read_new_scores(a.new_scores, os.path.join(tdir, f"to_score.{src_ext}"),
                                         os.path.join(tdir, f"to_score.{tgt_ext}"), len(M))
        if not np.isfinite(R).all():
            sys.exit(f"{int((~np.isfinite(R)).sum()):,} rows still have no finite score after merge")
    tmp = a.out + ".tmp"
    n = 0
    k = 0
    Ml = M.tolist()
    nxt = Ml[0] if Ml else -1
    try:
        with open(a.new_src, encoding="utf-8", newline="\n") as fs, open(a.new_tgt, encoding="utf-8", newline="\n") as ft, \
             open(tmp, "w", encoding="utf-8", newline="\n") as fo:
            for i, (s, t) in enumerate(zip(fs, ft)):
                if i >= len(R):
                    raise ValueError(f"corpus has more rows than the plan's {len(R):,}")
                if i == nxt:
                    # to_score.* may be stale (another plan, same length): the scored text
                    # must be THIS corpus row, not just the to_score line.
                    if not a.allow_missing and int(new_keys[k]) != key(s, t):
                        raise ValueError(f"new_scores row {k:,} is not corpus row {i:,} "
                                         f"(to_score.* does not belong to this plan)")
                    k += 1
                    nxt = Ml[k] if k < len(Ml) else -1
                v = R[i]
                sc = "nan" if np.isnan(v) else f"{v:.6f}"
                fo.write(f"{sc}\t{norm(s)}\t{norm(t)}\n")
                n += 1
            fo.flush()
            os.fsync(fo.fileno())
        if n != len(R):
            raise ValueError(f"corpus has {n:,} rows but the plan covers {len(R):,}")
    except (ValueError, OSError) as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        sys.exit(f"{e}; refusing to merge (nothing written to {a.out})")
    os.replace(tmp, a.out)
    if a.allow_missing:
        print(f"wrote {a.out}: {n:,} rows ({n - len(M):,} reused, {len(M):,} written as nan)")
    else:
        print(f"wrote {a.out}: {n:,} rows ({n - len(M):,} reused, {len(M):,} newly scored)")
    return 0


# ------------------------------------------------------------------ calibrate
def avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (as scipy.stats.rankdata 'average')."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    boundaries = np.flatnonzero(np.diff(xs) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    avg = (starts + ends + 1) / 2.0  # mean of 1-based positions starts+1 .. ends
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = avg_ranks(a.astype(np.float64)), avg_ranks(b.astype(np.float64))
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def topk_jaccard(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    n = len(a)
    k = max(1, int(round(frac * n)))
    idx = np.arange(n)
    ta = np.lexsort((idx, -a.astype(np.float64)))[:k]
    tb = np.lexsort((idx, -b.astype(np.float64)))[:k]
    inter = len(np.intersect1d(ta, tb, assume_unique=True))
    return inter / (2 * k - inter)


CAL_FRACS = (0.50, 0.30, 0.15, 0.05)


def compare_group(new: np.ndarray, old: np.ndarray, a) -> dict:
    n = int(len(new))
    if n < a.min_rows:
        return {"n": n, "status": "insufficient"}
    d = new.astype(np.float64) - old.astype(np.float64)
    res = {"n": n, "mean_diff": float(d.mean()), "median_abs_diff": float(np.median(np.abs(d))),
           "p99_abs_diff": float(np.percentile(np.abs(d), 99)), "spearman": spearman(new, old),
           "topk_jaccard": {f"{int(f * 100)}%": topk_jaccard(new, old, f) for f in CAL_FRACS}}
    fails = []
    if not abs(res["mean_diff"]) <= a.max_mean_diff:
        fails.append(f"|mean diff| {abs(res['mean_diff']):.3g} > {a.max_mean_diff:g}")
    if not res["p99_abs_diff"] <= a.max_p99_diff:
        fails.append(f"p99 |diff| {res['p99_abs_diff']:.3g} > {a.max_p99_diff:g}")
    if not res["spearman"] >= a.min_spearman:
        fails.append(f"spearman {res['spearman']:.6f} < {a.min_spearman:g}")
    min_k = getattr(a, "jaccard_min_k", 0) or 0
    not_gated = []
    for f in CAL_FRACS:
        kk, j = f"{int(f * 100)}%", res["topk_jaccard"][f"{int(f * 100)}%"]
        if max(1, int(round(f * n))) < min_k:
            not_gated.append(kk)            # reported, not gated (--jaccard-min-k)
            continue
        if not j >= a.min_jaccard:
            fails.append(f"top-{kk} jaccard {j:.4f} < {a.min_jaccard:g}")
    if not_gated:
        res["jaccard_not_gated"] = not_gated
    res["status"] = "fail" if fails else "pass"
    res["failures"] = fails
    return res


def cmd_calibrate(a) -> int:
    def bad_input(msg):
        print(f"calibrate: {msg}", file=sys.stderr)
        return 3

    pj = json.load(open(os.path.join(a.plan_dir, "plan.json"), encoding="utf-8"))
    R = np.load(os.path.join(a.plan_dir, "reuse_scores.npy"))
    if len(R) != pj["new_rows"]:
        return bad_input(f"reuse_scores.npy has {len(R):,} rows, plan.json says {pj['new_rows']:,}")
    labels = np.load(a.provenance)
    if len(labels) != len(R):
        return bad_input(f"provenance labels have {len(labels):,} rows, plan covers {len(R):,}")
    rep = json.load(open(a.provenance_report, encoding="utf-8"))
    names = list(rep.get("sources", []))
    if not names:
        return bad_input(f"{a.provenance_report} has no 'sources' list")
    check_text = bool(a.new_src or a.new_tgt)
    if check_text:
        if not (a.new_src and a.new_tgt):
            return bad_input("--new-src and --new-tgt go together")
        load_plan(a.plan_dir, a.new_src, a.new_tgt)
    new = np.full(len(R), np.nan, dtype=np.float64)
    n = 0
    fs = open(a.new_src, encoding="utf-8", newline="\n") if check_text else None
    ft = open(a.new_tgt, encoding="utf-8", newline="\n") if check_text else None
    with open(a.new_scored, encoding="utf-8", newline="\n") as f:
        for line in f:
            if n >= len(R):
                return bad_input(f"{a.new_scored} has more rows than the plan's {len(R):,}")
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                return bad_input(f"{a.new_scored} row {n:,}: {len(parts)} fields, expected 3")
            if check_text and (parts[1] != norm(fs.readline()) or parts[2] != norm(ft.readline())):
                return bad_input(f"{a.new_scored} row {n:,}: text differs from the corpus row")
            try:
                new[n] = float(parts[0])
            except ValueError:
                return bad_input(f"{a.new_scored} row {n:,}: score {parts[0]!r} is not a number")
            n += 1
    if n != len(R):
        return bad_input(f"{a.new_scored} has {n:,} rows but the plan's new_rows is {len(R):,}")
    both = np.isfinite(new) & np.isfinite(R)
    report = {"plan_dir": a.plan_dir, "new_scored": a.new_scored, "rows": int(len(R)),
              "compared_rows": int(both.sum()),
              "thresholds": {"max_abs_mean_diff": a.max_mean_diff, "max_p99_abs_diff": a.max_p99_diff,
                             "min_spearman": a.min_spearman, "min_topk_jaccard": a.min_jaccard,
                             "topk_fracs": list(CAL_FRACS), "min_rows": a.min_rows,
                             "jaccard_min_k": a.jaccard_min_k,
                             "note": "defaults are proposals pending user confirmation"},
              "overall": compare_group(new[both], R[both], a), "per_source": {}}
    groups = [(i, nm) for i, nm in enumerate(names)] + [(255, "unmatched")]
    for i, nm in groups:
        sel = both & (labels == i)
        if not sel.any():
            report["per_source"][nm] = {"n": 0, "status": "absent"}
            continue
        report["per_source"][nm] = compare_group(new[sel], R[sel], a)
    statuses = [report["overall"]["status"]] + [g["status"] for g in report["per_source"].values()]
    verdict = "fail" if "fail" in statuses else ("insufficient" if "insufficient" in statuses else "pass")
    report["verdict"] = verdict
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as fo:
        json.dump(report, fo, indent=1)
        fo.write("\n")
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"verdict": verdict, "compared_rows": report["compared_rows"],
                      "overall": report["overall"],
                      "per_source": {k: v["status"] for k, v in report["per_source"].items()}}, indent=1))
    return {"pass": 0, "fail": 1, "insufficient": 2}[verdict]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--old-scored", required=True)
    p.add_argument("--new-src", required=True)
    p.add_argument("--new-tgt", required=True)
    p.add_argument("--out-dir", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--plan-dir", required=True)
    e.add_argument("--new-src", required=True)
    e.add_argument("--new-tgt", required=True)
    e.add_argument("--out-dir", default="")
    m = sub.add_parser("merge")
    m.add_argument("--plan-dir", required=True)
    m.add_argument("--new-scores", default="",
                   help="scorer TSV for to_score.*; its text columns are checked line by line")
    m.add_argument("--to-score-dir", default="",
                   help="where to_score.<src>/<tgt> live (default --plan-dir)")
    m.add_argument("--new-src", required=True)
    m.add_argument("--new-tgt", required=True)
    m.add_argument("--out", required=True, help="written to <out>.tmp, renamed only after every check passes")
    m.add_argument("--allow-missing", action="store_true",
                   help="merge reused scores only (no --new-scores): rows without a reused score get the "
                        "literal score 'nan' (for an E0.3 pool restricted to reused rows)")
    c = sub.add_parser(
        "calibrate", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compare a full new scoring run with the plan's reused (old-run) scores on rows that have "
                    "both, overall and per e01 source (+ 'unmatched' = label 255): mean diff, median and p99 "
                    "|diff|, Spearman (average ranks), top-k Jaccard at 50/30/15/5%% within the compared rows.\n"
                    "Exit: 0 pass, 1 any group fails a threshold, 2 insufficient data (a present group, or "
                    "overall, has fewer than --min-rows compared rows, and nothing failed), 3 invalid inputs.\n"
                    "THRESHOLD DEFAULTS ARE PROPOSALS PENDING USER CONFIRMATION (from the 2026-09-13 audit); "
                    "they are not pre-registered yet.")
    c.add_argument("--plan-dir", required=True, help="holds plan.json and reuse_scores.npy")
    c.add_argument("--new-scored", required=True, help="full scored TSV in fixed-corpus row order")
    c.add_argument("--provenance", required=True, help="e01 provenance_labels.npy for the same corpus")
    c.add_argument("--provenance-report", required=True, help="e01 provenance_report.json (source names)")
    c.add_argument("--out", required=True, help="JSON report")
    c.add_argument("--new-src", default="", help="optional: also check sha vs plan and every text column")
    c.add_argument("--new-tgt", default="")
    c.add_argument("--max-mean-diff", type=float, default=2e-4, help="|mean(new-old)| <= this (proposal)")
    c.add_argument("--max-p99-diff", type=float, default=2e-3, help="p99 |new-old| <= this (proposal)")
    c.add_argument("--min-spearman", type=float, default=0.999, help="Spearman >= this (proposal)")
    c.add_argument("--min-jaccard", type=float, default=0.99, help="top-k Jaccard >= this at every k (proposal)")
    c.add_argument("--min-rows", type=int, default=1000,
                   help="groups with fewer compared rows are 'insufficient' (proposal)")
    c.add_argument("--jaccard-min-k", type=int, default=0,
                   help="a top-k Jaccard whose k (= round(frac * group rows)) is below this is reported but "
                        "not gated. Default 0 gates every k (current behaviour). User decision, PROTOCOL.md D7")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        # Exit 1 must only ever mean "compared and failed": rental_setup.sh discards the scores'
        # shards on 1/2 but keeps them on anything else (audit F1). load_plan's sys.exit(msg)
        # and any read error (truncated .npy, bad JSON) are invalid input -> 3.
        try:
            return cmd_calibrate(a)
        except SystemExit as e:
            if isinstance(e.code, int):
                raise
            print(f"calibrate: {e.code}", file=sys.stderr)
            return 3
        except Exception as e:                                  # noqa: BLE001
            print(f"calibrate: invalid input: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
    return {"plan": cmd_plan, "extract": cmd_extract, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
