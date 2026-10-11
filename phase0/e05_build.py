#!/usr/bin/env python3
"""E0.5: the matched random controls, and the same three sets again inside ONE source.

Run after phase0/e04_split.py has written the two halves of the QE top band into --dir.
The rule is fixed in PROTOCOL.md "E0.5"; this only executes it. It adds four sets and
rewrites the directory's manifest.json to cover all six:

  ft_random_small   from ft_random, with exactly the halves' source-length histogram
  ft_un_div         UN rows of the top band at or below the median repetitiveness
  ft_un_rep         UN rows of the top band above it       } one shared length histogram
  ft_un_random      UN rows of ft_random with that histogram

    e05_build.py --dir data/phase0_e05 --src-dir <dir with ft_topk.* and ft_random.*>

The source directory must hold, for ft_topk and ft_random: .en, .fr and .source (one source
label per row, written by phase0/e05_source_labels.py).

WHY THE UN SETS. Measured before any run, the repetitive half of the top band is 91% UN
text and the diverse half 74%, so E0.4's contrast cannot separate "repetitive" from "more
UN". Inside one source that confound is gone by construction.

Length is counted in the same lowercased \\w+ tokens e04_split.py matched the halves on, and
repetitiveness is the same statistic, taken over the whole of ft_topk.

FAIL-CLOSED. Nothing is published unless every set can be built exactly: the halves must
share one length histogram, both random pools must fill every bucket, the two scripts must
agree on which rows are repetitive, and the source files must be the pinned ones (or
--allow-unpinned-src, which is recorded). Every output is written under a temporary name and
renamed only at the end, and manifest.json is written LAST: a manifest with
"e05_complete": true is the only thing that marks this directory as built.

Top-band rows are NOT removed from a random draw: a random draw from the pool contains them
at their base rate, and removing them would make the control "random minus the top band".
The overlap is counted instead.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)      # must stay identical to phase0/e04_split.py
SEED = 42
SOURCE = "un"                              # the one source large enough to hold all three sets
NEW_SETS = ("ft_random_small", "ft_un_div", "ft_un_rep", "ft_un_random")
ALL_SETS = ("ft_topk_div", "ft_topk_rep") + NEW_SETS

# sha256 of the E0.3 primary arm's control sets and of the source labels derived from them
# (PROTOCOL.md "E0.5"). A build from anything else is a different experiment.
PINNED_SRC = {
    "ft_topk.en": "787339d08e0c47d9960cb263757edafd844ec80a45875ac083845a02a20ff2b2",
    "ft_topk.fr": "22263da18eaf62cb7deea7438068ce0cb00ce3849c1a83f7e0c55008c3d78797",
    "ft_topk.source": "a4ad45860c92eae6b512c4a7178fe78fc92f00d4b9551d00a6637c80079244ca",
    "ft_random.en": "937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0",
    "ft_random.fr": "382e0de574feaaa323939d73a24736b762c76a8b55a0ffd258d64b15e5303e29",
    "ft_random.source": "fd675a1948ab4e30dce05aae4c85d2432aec99549c05fb794903a49e9b0b6923",
}
# the in-domain test set, as e04_split.py copies it into --dir from the same source directory
PINNED_HELDOUT = {
    "heldout_un.en": "f70e725eae9972899b6129f6acf123ed5421021b2f5ba81bb165a6928e858bbc",
    "heldout_un.fr": "7daf654fa37cca653cbc9724c77ee721384a406e3ca3e6469a2a847b66b729ab",
}


def read_lines(p: Path) -> list[str]:
    # LF is the only line break; never splitlines(), which also splits on U+2028/NEL
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def toks(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gram_key(gram: str) -> bytes:
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def rep_within(parent_toks: list[list[str]]) -> list[float]:
    """E0.4's statistic as phase0/e04_split.py computes it: per row, the fraction of its
    8-grams whose count over the whole set is above one. Rows under 8 tokens -> 0."""
    grams = [[gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in parent_toks]
    counts = collections.Counter()
    for g in grams:
        counts.update(g)
    return [sum(1 for k in g if counts[k] > 1) / len(g) if g else 0.0 for g in grams]


class Stop(Exception):
    pass


def match_random(hist: dict[int, int], pool_len: list[int], eligible: list[int], what: str) -> tuple[list[int], float]:
    """Exactly hist[L] rows of every length L, at random within the bucket (seed 42).
    eligible: candidate row indices, ascending. Raises Stop on any shortfall."""
    buckets: dict[int, list[int]] = {}
    for i in eligible:
        buckets.setdefault(pool_len[i], []).append(i)
    short = {L: (n, len(buckets.get(L, []))) for L, n in hist.items() if len(buckets.get(L, [])) < n}
    if short:
        lines = "".join(f"\n  length {L}: need {short[L][0]:,}, have {short[L][1]:,}" for L in sorted(short))
        raise Stop(f"{what} is short in {len(short)} length bucket(s) (total shortfall "
                   f"{sum(n - h for n, h in short.values()):,} rows):{lines}\n"
                   f"PROTOCOL.md 'E0.5' fixes an exact match; amend it there first.")
    rng = random.Random(SEED)
    kept: list[int] = []
    for L in sorted(hist):
        kept += rng.sample(buckets[L], hist[L])
    return sorted(kept), min(len(buckets[L]) / hist[L] for L in hist)


def lexicon(texts: list[str], tok_lists: list[list[str]]) -> dict:
    """How thin the set's vocabulary is, independently of the 8-gram statistic: distinct
    source words, and the share of rows whose source sentence already occurred in the set."""
    return {"source_words": sum(len(t) for t in tok_lists),
            "source_word_types": len({w for t in tok_lists for w in t}),
            "duplicate_source_fraction": 1 - len(set(texts)) / len(texts)}


def describe(idx: list[int], lens: list[int], rep: list[float], labels: list[str],
             texts: list[str], tok_lists: list[list[str]]) -> dict:
    k = len(idx)
    ls = sorted(lens[i] for i in idx)
    return {"rows": k, "len_mean": sum(ls) / k, "len_median": ls[k // 2],
            "sources": dict(sorted(collections.Counter(labels[i] for i in idx).items(), key=lambda x: -x[1])),
            "rep_mean": sum(rep[i] for i in idx) / k,
            "rep_positive_fraction": sum(1 for i in idx if rep[i] > 0) / k,
            **lexicon([texts[i] for i in idx], [tok_lists[i] for i in idx])}


def verify(d: Path) -> int:
    """0 if d is a finished E0.5 build and nothing in it changed since; 1 with a reason."""
    mp = d / "manifest.json"
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"NOT BUILT: {mp}: {e}")
        return 1
    if man.get("e05_complete") is not True:
        print(f"NOT BUILT: {mp} has no \"e05_complete\": true (an interrupted or pre-E0.5 build)")
        return 1
    want = man.get("sha256", {})
    lacking = [f"{n}.{e}" for n in ALL_SETS for e in ("en", "fr") if f"{n}.{e}" not in want]
    if lacking:
        print("NOT BUILT: the manifest records no sha256 for " + ", ".join(lacking))
        return 1
    bad = [w for w, h in sorted(want.items()) if not (d / w).is_file() or sha256_file(d / w) != h]
    if bad:
        print("CHANGED SINCE THE BUILD (sha256 differs from the manifest, or missing): " + ", ".join(bad))
        return 1
    print(f"built and intact: {len(want)} files in {d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path,
                    help="e04_split.py output: ft_topk_div.*, ft_topk_rep.*, manifest.json")
    ap.add_argument("--src-dir", type=Path,
                    help="ft_topk.{en,fr,source} and ft_random.{en,fr,source}")
    ap.add_argument("--verify", action="store_true",
                    help="build nothing: exit 0 only if --dir holds a COMPLETE build whose files "
                         "still have the sha256 the manifest recorded")
    ap.add_argument("--allow-unpinned-src", action="store_true",
                    help="build from source files other than the pinned E0.3 primary arm's "
                         "(tests; any real use is a protocol deviation and is recorded)")
    a = ap.parse_args()
    d, s = a.dir, a.src_dir
    if a.verify:
        return verify(d)
    if s is None:
        ap.error("--src-dir is required to build")

    src_files = [f"{n}.{e}" for n in ("ft_topk", "ft_random") for e in ("en", "fr", "source")]
    need = [d / f for f in ("ft_topk_div.en", "ft_topk_div.fr", "ft_topk_rep.en", "ft_topk_rep.fr",
                            "manifest.json")] + [s / f for f in src_files]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        print("MISSING: " + ", ".join(missing))
        return 1
    src_sha = {f: sha256_file(s / f) for f in src_files}
    for f in PINNED_HELDOUT:
        if not (d / f).is_file():
            print(f"MISSING: {d / f}")
            return 1
        src_sha[f] = sha256_file(d / f)
    pins = {**PINNED_SRC, **PINNED_HELDOUT}
    unpinned = sorted(f for f in src_sha if pins.get(f) != src_sha[f])
    if unpinned and not a.allow_unpinned_src:
        print("STOP: these source files are not the pinned E0.3 primary-arm sets "
              "(PROTOCOL.md 'E0.5'):" + "".join(f"\n  {f}  sha256 {src_sha[f]}" for f in unpinned))
        print("Nothing written.")
        return 1

    tmp = []                                   # (temporary path, final path)

    def write(name: str, ext: str, rows: list[str], idx: list[int]) -> None:
        final = d / f"{name}.{ext}"
        t = d / f".{name}.{ext}.tmp"
        with open(t, "w", encoding="utf-8", newline="\n") as f:
            for i in idx:
                f.write(rows[i] + "\n")
        tmp.append((t, final))

    try:
        # ---- the two all-source halves, as e04_split.py left them ----
        div_src, rep_src = read_lines(d / "ft_topk_div.en"), read_lines(d / "ft_topk_rep.en")
        hist = collections.Counter(len(toks(x)) for x in div_src)
        if hist != collections.Counter(len(toks(x)) for x in rep_src):
            raise Stop("ft_topk_div and ft_topk_rep do not share one source-length histogram; "
                       "this directory is not an e04_split.py output")

        top_src, top_tgt, top_lab = (read_lines(s / f"ft_topk.{e}") for e in ("en", "fr", "source"))
        rnd_src, rnd_tgt, rnd_lab = (read_lines(s / f"ft_random.{e}") for e in ("en", "fr", "source"))
        for name, x, y, z in (("ft_topk", top_src, top_tgt, top_lab), ("ft_random", rnd_src, rnd_tgt, rnd_lab)):
            if not len(x) == len(y) == len(z):
                raise Stop(f"{name}: .en has {len(x):,} lines, .fr {len(y):,}, .source {len(z):,}")
        top_toks, rnd_toks = [toks(x) for x in top_src], [toks(x) for x in rnd_src]
        top_len, rnd_len = [len(t) for t in top_toks], [len(t) for t in rnd_toks]

        # ---- repetitiveness over the whole top band, and agreement with e04_split.py ----
        rep = rep_within(top_toks)
        n = len(rep)
        med = rep[sorted(range(n), key=lambda i: rep[i])[n // 2]]
        low = {top_src[i] for i in range(n) if rep[i] <= med}
        high = {top_src[i] for i in range(n) if rep[i] > med}
        wrong = sum(1 for x in div_src if x not in low) + sum(1 for x in rep_src if x not in high)
        if wrong:
            raise Stop(f"{wrong:,} rows of the halves in {d} fall on the other side of the median "
                       f"by this script's statistic: {s} is not the set they were split from, or "
                       f"e04_split.py and e05_build.py no longer compute the same thing")

        # ---- ft_random_small: the all-source random control ----
        k_rs, slack_rs = match_random(hist, rnd_len, list(range(len(rnd_src))), "ft_random")

        # ---- the same three sets inside one source ----
        un_div = [i for i in range(n) if top_lab[i] == SOURCE and rep[i] <= med]
        un_rep = [i for i in range(n) if top_lab[i] == SOURCE and rep[i] > med]
        by = {"div": {}, "rep": {}}
        for nm, idx in (("div", un_div), ("rep", un_rep)):
            for i in idx:
                by[nm].setdefault(top_len[i], []).append(i)
        rng = random.Random(SEED)              # E0.4's stratified matching, unchanged
        kd, kr = [], []
        for L in sorted(set(by["div"]) | set(by["rep"])):
            x, y = by["div"].get(L, []), by["rep"].get(L, [])
            k = min(len(x), len(y))
            if k:
                kd += rng.sample(x, k)
                kr += rng.sample(y, k)
        kd.sort()
        kr.sort()
        if not kd:
            raise Stop(f"no '{SOURCE}' rows on both sides of the median: nothing to compare")
        un_hist = collections.Counter(top_len[i] for i in kd)
        k_ur, slack_ur = match_random(un_hist, rnd_len, [i for i in range(len(rnd_src)) if rnd_lab[i] == SOURCE],
                                      f"the '{SOURCE}' rows of ft_random")

        for name, rows_en, rows_fr, idx in (("ft_random_small", rnd_src, rnd_tgt, k_rs),
                                            ("ft_un_div", top_src, top_tgt, kd),
                                            ("ft_un_rep", top_src, top_tgt, kr),
                                            ("ft_un_random", rnd_src, rnd_tgt, k_ur)):
            write(name, "en", rows_en, idx)
            write(name, "fr", rows_fr, idx)

        # ---- descriptive record of all six sets ----
        pair_lab = {(x, y): l for x, y, l in zip(top_src, top_tgt, top_lab)}
        pair_rep = {(x, y): r for x, y, r in zip(top_src, top_tgt, rep)}
        div_set, rep_set = set(div_src), set(rep_src)
        sets = {}
        for name in ("ft_topk_div", "ft_topk_rep"):
            ps = list(zip(read_lines(d / f"{name}.en"), read_lines(d / f"{name}.fr")))
            ls = sorted(len(toks(x)) for x, _ in ps)
            sets[name] = {"rows": len(ps), "len_mean": sum(ls) / len(ls), "len_median": ls[len(ls) // 2],
                          "sources": dict(sorted(collections.Counter(pair_lab[p] for p in ps).items(), key=lambda x: -x[1])),
                          "rep_mean": sum(pair_rep[p] for p in ps) / len(ps),
                          "rep_positive_fraction": sum(1 for p in ps if pair_rep[p] > 0) / len(ps),
                          **lexicon([x for x, _ in ps], [toks(x) for x, _ in ps])}
        rnd_rep = rep_within(rnd_toks)         # the same statistic, within the random set's own parent
        sets["ft_random_small"] = {**describe(k_rs, rnd_len, rnd_rep, rnd_lab, rnd_src, rnd_toks), "tightest_bucket_slack": slack_rs,
                                   "source_lines_also_in_div": sum(1 for i in k_rs if rnd_src[i] in div_set),
                                   "source_lines_also_in_rep": sum(1 for i in k_rs if rnd_src[i] in rep_set)}
        sets["ft_un_div"] = describe(kd, top_len, rep, top_lab, top_src, top_toks)
        sets["ft_un_rep"] = describe(kr, top_len, rep, top_lab, top_src, top_toks)
        un_d, un_r = {top_src[i] for i in kd}, {top_src[i] for i in kr}
        sets["ft_un_random"] = {**describe(k_ur, rnd_len, rnd_rep, rnd_lab, rnd_src, rnd_toks), "tightest_bucket_slack": slack_ur,
                                "source_lines_also_in_un_div": sum(1 for i in k_ur if rnd_src[i] in un_d),
                                "source_lines_also_in_un_rep": sum(1 for i in k_ur if rnd_src[i] in un_r)}

        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        keep = [w for w in man.get("written", [])
                if w != "manifest.json" and w.rsplit(".", 1)[0] not in NEW_SETS]
        man["written"] = keep + [f"{nm}.{e}" for nm in NEW_SETS for e in ("en", "fr")] + ["manifest.json"]
        for t, final in tmp:                   # publish: every set at once, then the manifest
            os.replace(t, final)
        tmp.clear()
        man["e05"] = {"seed": SEED, "source_held_fixed": SOURCE, "median_rep": med,
                      "rep_statistic": "fraction of the row's 8-grams whose count over ft_topk is > 1",
                      "sets": sets,
                      "src_sha256": src_sha, "src_pinned": not unpinned,
                      "built_with": {"python": sys.version.split()[0],
                                     "unidata_version": unicodedata.unidata_version}}
        man["sha256"] = {w: sha256_file(d / w) for w in man["written"] if w != "manifest.json"}
        man["e05_complete"] = True
        mt = d / ".manifest.json.tmp"
        mt.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
        os.replace(mt, d / "manifest.json")
    except Stop as e:
        print(f"STOP: {e}\nNothing written.")
        return 1
    finally:
        for t, _ in tmp:
            t.unlink(missing_ok=True)

    for name in ALL_SETS:
        v = sets[name]
        src = "  ".join(f"{k} {100 * c / v['rows']:.1f}%" for k, c in list(v["sources"].items())[:3])
        print(f"{name:<16} {v['rows']:>8,} rows | len {v['len_mean']:.2f}/{v['len_median']} | "
              f"rep {v['rep_mean']:.3f} | {v['source_word_types']:>6,} word types | "
              f"{100 * v['duplicate_source_fraction']:.2f}% duplicate sources | {src}")
    if unpinned:
        print(f"NOTE: built from UNPINNED source files ({', '.join(unpinned)}); recorded in the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
