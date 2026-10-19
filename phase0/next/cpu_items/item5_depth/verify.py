#!/usr/bin/env python3
"""Independent re-computation of item 5's statistics for the depths that lie inside E0.3's band.

Every depth's dedup subset is a prefix of the score-ordered dedup ranking, and ft_topk.{en,fr}
holds the first 926,999 rows of that ranking in score order (depth.py's rank stage checks this
row for row). So the 0.13%, 0.3%, 1%, 6% and band subsets are the first n_dedup lines of
ft_topk. This script recomputes their statistics the direct way, with the code of
e04_split.py (rep), e05_build.py (duplicate sources, source word types) and vocab/common.py
(types_at), on Python strings, and compares them with item5_depth_results.json.

Usage: verify.py [--e05 DIR] [--results item5_depth_results.json]
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)


def key(gram):
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def read(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def types_at(t, idx, N, seed=0):
    """vocab/common.py types_at(), on token lists t"""
    rng = random.Random(seed); idx = list(idx); rng.shuffle(idx)
    seen = set(); n = 0
    for i in idx:
        for w in t[i]:
            if n >= N:
                return len(seen)
            seen.add(w); n += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e05", default=str(Path.home() / "e06data" / "e05_src"))
    ap.add_argument("--results", default=str(Path.home() / "cpu_items" / "item5_depth" / "item5_depth_results.json"))
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    B = R["definition"]["budgets"]
    src, tgt = read(Path(a.e05, "ft_topk.en")), read(Path(a.e05, "ft_topk.fr"))
    lab = read(Path(a.e05, "ft_topk.source"))
    te = [[w.lower() for w in WORD.findall(s)] for s in src]
    tf = [[w.lower() for w in WORD.findall(s)] for s in tgt]
    grams = [[key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in te]
    out = {}
    for d in R["depths"]:
        c = d["n_dedup"]
        if c > len(src):
            continue
        counts = collections.Counter()
        for g in grams[:c]:
            counts.update(g)
        rep = [sum(1 for k in g if counts[k] > 1) / len(g) if g else 0.0 for g in grams[:c]]
        del counts
        un = [x == "un" for x in lab[:c]]
        m = {"rep_positive_rows": sum(r > 0 for r in rep),
             "rep_mean": sum(rep) / c,
             "un_rows_rep_positive": sum(1 for r, u in zip(rep, un) if u and r > 0),
             "un_rows_rep_zero": sum(1 for r, u in zip(rep, un) if u and r == 0),
             "un_share_dedup": sum(un) / c,
             "duplicate_source_rate_dedup": 1 - len(set(src[:c])) / c,
             "tokens_en": sum(len(t) for t in te[:c]), "tokens_fr": sum(len(t) for t in tf[:c]),
             "types_whole_set_en": len({w for t in te[:c] for w in t}),
             "types_whole_set_fr": len({w for t in tf[:c] for w in t})}
        tm = {}
        for seed in range(5):
            for bname, b in B.items():
                if seed and bname != "B1":
                    continue
                tm.setdefault(f"{bname}_en", {})[f"seed{seed}"] = types_at(te, range(c), b, seed)
                tm.setdefault(f"{bname}_fr", {})[f"seed{seed}"] = types_at(tf, range(c), b, seed)
        m["types_matched"] = tm
        diffs = {}
        for k, v in m.items():
            w = d[k]
            same = abs(v - w) <= 1e-9 * max(1.0, abs(w)) if isinstance(v, float) else v == w
            if not same:
                diffs[k] = {"direct": v, "depth.py": w}
        out[d["name"]] = {"n_dedup": c, "all_equal": not diffs, "diffs": diffs}
        print(d["name"], c, "all equal" if not diffs else diffs, flush=True)
    Path(a.results).with_name("verify_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
