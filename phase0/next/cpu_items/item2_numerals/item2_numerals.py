#!/usr/bin/env python3
"""Item 2 of WMT2027_PLAN.md section 3.2: E0.4's rep statistic with numbers, dates and UN
document symbols masked, and the numeric-token share of the E0.5 sets. Descriptive only.

    item2_numerals.py --data ~/e06data --out ~/cpu_items/item2_numerals [--workers 4]

THE STATISTIC (unchanged). rep(row) = the fraction of the row's 8-grams whose count over the
whole parent set is above one; 8-grams over lowercased \\w+ tokens, keyed by a 64-bit BLAKE2b
digest of " ".join(8 tokens); rows under 8 tokens take 0 (phase0/e04_split.py lines 13-37,
phase0/e05_build.py rep_within()). The parent of the top-band sets is ft_topk (926,999 rows);
the parent of ft_un_random and ft_random_small is ft_random, as in E0.5's manifest.

MASKING (masking.py). Every token keeps its position; tokens are classed NUM / DATE / SYM / WORD.
Two readings of "masked" are computed, because they answer different questions:

  M1 "exclude" (headline): a masked token matches nothing (as if each were a fresh symbol). An
     8-gram occurrence that contains a masked token never counts as repeated; an untouched
     8-gram counts if it occurs untouched more than once in the parent. The denominator stays
     the row's number of 8-grams, so rep_M1 <= rep for every row.
  M2 "placeholder": every masked token is replaced by its class placeholder (#num, #date, #sym)
     and the counts are recomputed over the parent. Rows that differ only in their numbers now
     share 8-grams (Lo et al. 2018's number-aware view). Because DATE and SYM depend on context,
     rep_M2 >= rep is not guaranteed row by row; the rows where it fails are counted.

Sensitivity variants: M1-digit (mask exactly the tokens that contain a Unicode digit, nothing
else), M1' (M1's repeated 8-grams over the row's number of numeral-free 8-grams), and M2-A17
(every Unicode digit -> "0" inside the token: e06_build.py's A17 rule applied to rep).

FAIL-CLOSED. Every input is checked against its pinned sha256; the unmasked rep must reproduce
E0.5's manifest (median 0.0, the 486,475 / 227,406 UN candidates, every set's rep_mean and
rep_positive_fraction, compared as exact floats). Otherwise nothing is written.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import sys
import time
import unicodedata
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                          # noqa: E402

import masking as M                         # noqa: E402

PINNED = {   # ~/mt_local/phase0_next/e06data_SHA256SUMS (= e05_build.py PINNED_SRC, E0.5 manifest)
    "e05_src/ft_topk.en": "787339d08e0c47d9960cb263757edafd844ec80a45875ac083845a02a20ff2b2",
    "e05_src/ft_topk.source": "a4ad45860c92eae6b512c4a7178fe78fc92f00d4b9551d00a6637c80079244ca",
    "e05_src/ft_random.en": "937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0",
    "e05_src/ft_random.source": "fd675a1948ab4e30dce05aae4c85d2432aec99549c05fb794903a49e9b0b6923",
    "phase0_e05/manifest.json": "979dafd76344444c56cad9afc52e0ec7ba7ebb74c73115cbeb237583419b8eb2",
    "phase0_e05/ft_topk_div.en": "321474497238a74704fd29653878814e1742d419e740e0148fca4191742c47fa",
    "phase0_e05/ft_topk_div.fr": "2fdc192521c64f966bcbb1a321fd99ff55d0dce7a661fd60837b01e723d0357f",
    "phase0_e05/ft_topk_rep.en": "6db2a842ee9fbb21d9806738861ac0e94734310121d4ff5f5d3d684a442fa386",
    "phase0_e05/ft_topk_rep.fr": "0598ddfdeba978bbdbc354594889630dc175bb7ea0ab401bdce51a48c122ec75",
    "phase0_e05/ft_un_div.en": "51d9ec1157c58591a7b25b2a1bde52e9dc93342a34c55e55547f0e3857ba94da",
    "phase0_e05/ft_un_div.fr": "1957872adf3f96b8e7fe551eeeae42bbd38f03811511e013f9017b65b954a7e1",
    "phase0_e05/ft_un_rep.en": "940cde0f051d6b473c55a6771af64900a20c9c52d66e1a4772e18a5bf9cafe68",
    "phase0_e05/ft_un_rep.fr": "9373be1fc7afb86a1b5ebf76d7cb567bde0ef349443ca9c3c6a87c175185edbd",
    "phase0_e05/ft_un_random.en": "80fa6aea32f342cc43adec8e66063f8f08c20b6e3b66a60b0bfd7baf3e35c59c",
    "phase0_e05/ft_un_random.fr": "1b54a882d989153697417b1f4b2300b81500acac258e29c542776d39a32094c0",
    "phase0_e05/ft_random_small.en": "d75365502e1db21ad7975150c1a7a6b6f680e2432e9473dc5d3041a152881f84",
    "phase0_e05/ft_random_small.fr": "7072a04327a56ddd399ca48d885b6d2cdde737bca4db8b7fd3d01672ea8bb0f9",
}
SETS = {   # set -> parent of its rep
    "ft_topk_div": "ft_topk", "ft_topk_rep": "ft_topk", "ft_un_div": "ft_topk", "ft_un_rep": "ft_topk",
    "ft_un_random": "ft_random", "ft_random_small": "ft_random",
}
EXPECTED_COUNTS = {"ft_topk UN rows with rep = 0": 486_475, "ft_topk UN rows with rep > 0": 227_406}
SEED = 42
N_EXAMPLES = 20


class Stop(Exception):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(p: Path) -> list[str]:
    with open(p, encoding="utf-8", newline="\n") as f:      # LF only, as e05_build.py
        return [ln.rstrip("\n") for ln in f]


def gram_key(gram: str) -> bytes:                            # e04_split.py key()
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def keys_of(t: list[str]) -> list[bytes]:
    return [gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)]


def touch_flags(masked: list[bool]) -> bytes:
    """1 for every 8-gram window that contains a masked token."""
    n = len(masked)
    if n < 8:
        return b""
    c = [0]
    for m in masked:
        c.append(c[-1] + m)
    return bytes(1 if c[j + 8] - c[j] else 0 for j in range(n - 7))


def work(lines: list[str]) -> dict:
    """Per chunk: token counts per row, the three key streams and the two touch streams."""
    lens, n_num, n_date, n_sym, n_dig = [], [], [], [], []
    ko, km2, ka17 = [], [], []
    tf, td = bytearray(), bytearray()
    for s in lines:
        t, cls, has_digit = M.classify(s)
        k = keys_of(t)
        ko += k
        lens.append(len(t))
        if not has_digit:
            km2 += k
            ka17 += k
            z = bytes(max(len(t) - 7, 0))
            tf += z
            td += z
            n_num.append(0); n_date.append(0); n_sym.append(0); n_dig.append(0)
            continue
        dig = [bool(M.DIGIT.search(w)) for w in t]
        km2 += keys_of([w if c == M.W else M.PLACEHOLDER[c] for w, c in zip(t, cls)])
        ka17 += keys_of([M.DIGIT.sub("0", w) for w in t])
        tf += touch_flags([c != M.W for c in cls])
        td += touch_flags(dig)
        n_num.append(cls.count(M.NUM)); n_date.append(cls.count(M.DATE_C))
        n_sym.append(cls.count(M.SYM)); n_dig.append(sum(dig))
    return {"lens": np.array(lens, dtype=np.int64),
            "ko": b"".join(ko), "km2": b"".join(km2), "ka17": b"".join(ka17),
            "tf": bytes(tf), "td": bytes(td),
            "n_num": np.array(n_num, dtype=np.int64), "n_date": np.array(n_date, dtype=np.int64),
            "n_sym": np.array(n_sym, dtype=np.int64), "n_dig": np.array(n_dig, dtype=np.int64)}


def repeated(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-occurrence flag count > 1, sorted unique keys, their counts)."""
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return cnt[inv] > 1, uniq, cnt


def corpus_stats(lines: list[str], workers: int, label: str) -> dict:
    t0 = time.time()
    k = max(1, workers)
    step = (len(lines) + k - 1) // k
    chunks = [lines[i:i + step] for i in range(0, len(lines), step)]
    with get_context("spawn").Pool(k) as pool:
        parts = pool.map(work, chunks)
    lens = np.concatenate([p["lens"] for p in parts])
    n = len(lens)
    ng = np.maximum(lens - 7, 0)
    row = np.repeat(np.arange(n), ng)
    ko = np.frombuffer(b"".join(p["ko"] for p in parts), dtype="<u8")
    km2 = np.frombuffer(b"".join(p["km2"] for p in parts), dtype="<u8")
    ka17 = np.frombuffer(b"".join(p["ka17"] for p in parts), dtype="<u8")
    tf = np.frombuffer(b"".join(p["tf"] for p in parts), dtype=np.uint8).astype(bool)
    td = np.frombuffer(b"".join(p["td"] for p in parts), dtype=np.uint8).astype(bool)
    assert len(ko) == len(km2) == len(ka17) == len(tf) == len(td) == int(ng.sum()) == len(row)
    fo, uo, co = repeated(ko)
    # M1, strictly: a masked token matches nothing, so an 8-gram occurrence that touches one is
    # unique, and an untouched 8-gram is repeated only if it occurs untouched more than once.
    # (Classes depend on context -- "january" is DATE only inside a date, "a" is SYM only inside
    # A/58/12 -- so the same 8-gram can be touched at one occurrence and not at another.)
    ff, uf, cf = repeated(ko[~tf])
    fm1 = np.zeros(len(ko), dtype=bool)
    fm1[~tf] = ff
    fm2, _, _ = repeated(km2)
    fa17, _, _ = repeated(ka17)
    del km2, ka17, ff

    def per_row(flag: np.ndarray) -> np.ndarray:
        return np.bincount(row[flag], minlength=n)

    out = {
        "n": n, "ng": ng, "lens": lens,
        "c_orig": per_row(fo),                     # repeated 8-grams per row (integers)
        "c_m1": per_row(fm1), "c_m1_loose": per_row(fo & ~tf), "c_m1_digit": per_row(fo & ~td),
        "free": per_row(~tf),                      # numeral-free 8-grams per row (M1' denominator)
        "c_m2": per_row(fm2), "c_m2_a17": per_row(fa17),
        "touch": per_row(tf),
        "n_num": np.concatenate([p["n_num"] for p in parts]),
        "n_date": np.concatenate([p["n_date"] for p in parts]),
        "n_sym": np.concatenate([p["n_sym"] for p in parts]),
        "n_dig": np.concatenate([p["n_dig"] for p in parts]),
        "uniq": uo, "cnt": co, "uniq_free": uf, "cnt_free": cf,
        "grams": int(ng.sum()), "gram_types": int(len(uo)),
    }
    # what the definitions imply on every row: M1 <= rep (a subset of the repeated occurrences);
    # M1-digit is context-free (a token's digits), so the loose count is exact for it; A17 maps
    # each token by itself, so rep <= M2-A17. M2's classes depend on context, so rep <= M2 is
    # NOT guaranteed: the rows where it fails are counted and reported, not hidden.
    if not (np.all(out["c_m1"] <= out["c_m1_loose"]) and np.all(out["c_m1_loose"] <= out["c_orig"])
            and np.all(out["c_m1_digit"] <= out["c_orig"]) and np.all(out["c_orig"] <= out["c_m2_a17"])):
        raise Stop(f"{label}: M1 <= rep <= M2-A17 fails on some row; the implementation is wrong")
    out["rows_M2_below_rep"] = int(np.sum(out["c_m2"] < out["c_orig"]))
    out["rows_M1_strict_below_loose"] = int(np.sum(out["c_m1"] < out["c_m1_loose"]))
    print(f"{label}: {n:,} rows, {out['grams']:,} 8-grams, {out['gram_types']:,} types "
          f"({time.time() - t0:.1f} s)", flush=True)
    return out


def ratio(c: int, g: int) -> float:
    return c / g if g else 0.0                     # Python int/int, as e04_split.py


def _lookup(uniq: np.ndarray, cnt: np.ndarray, key) -> int:
    i = int(np.searchsorted(uniq, key))
    return int(cnt[i]) if i < len(uniq) and uniq[i] == key else 0


def gram_counts(st: dict, s: str) -> list[dict]:
    """Per 8-gram of one row: its text, its count over the parent, whether it touches a masked
    token here, and its count among the parent's untouched occurrences (M1's count)."""
    t, cls, _ = M.classify(s)
    out = []
    for j in range(len(t) - 7):
        g = " ".join(t[j:j + 8])
        key = np.frombuffer(gram_key(g), dtype="<u8")[0]
        out.append({"gram": g, "count": _lookup(st["uniq"], st["cnt"], key),
                    "touches_mask": any(c != M.W for c in cls[j:j + 8]),
                    "count_untouched": _lookup(st["uniq_free"], st["cnt_free"], key)})
    return out


def dup_rate(keys: list) -> float:
    return 1 - len(set(keys)) / len(keys)


def describe_set(name: str, src: list[str], fr: list[str], idx: list[int], st: dict,
                 line_count: collections.Counter) -> dict:
    """All numbers for one set; means are Python sums in the set's file order (as e05_build.py)."""
    k = len(src)
    g = [int(st["ng"][i]) for i in idx]
    rep = {v: [ratio(int(st[c][i]), gg) for i, gg in zip(idx, g)]
           for v, c in (("rep", "c_orig"), ("M1", "c_m1"), ("M1_digit", "c_m1_digit"),
                        ("M2", "c_m2"), ("M2_A17", "c_m2_a17"))}
    rep["M1_prime"] = [ratio(int(st["c_m1"][i]), int(st["free"][i])) for i in idx]
    variants = {v: {"rep_mean": sum(x) / k, "rows_rep_positive": sum(1 for r in x if r > 0),
                    "rep_positive_fraction": sum(1 for r in x if r > 0) / k} for v, x in rep.items()}
    pos = [j for j in range(k) if rep["rep"][j] > 0]
    lose = [j for j in pos if rep["M1"][j] == 0]
    gain = [j for j in range(k) if rep["rep"][j] == 0 and rep["M2"][j] > 0]
    c_orig = sum(int(st["c_orig"][i]) for i in idx)
    c_m1 = sum(int(st["c_m1"][i]) for i in idx)
    tok = sum(int(st["lens"][i]) for i in idx)
    cnt = {c: sum(int(st[c][i]) for i in idx) for c in ("n_num", "n_date", "n_sym", "n_dig")}
    masked_rows = sum(1 for i in idx if st["n_num"][i] + st["n_date"][i] + st["n_sym"][i] > 0)
    sym_matches = sum(len(M.symbol_spans(x)) for x in src if M.DIGIT.search(x))
    # French side: no date or symbol rule (English patterns); digits only
    fr_toks = [[w.lower() for w in M.WORD.findall(x)] for x in fr]
    fr_n = sum(len(t) for t in fr_toks)
    fr_dig = sum(1 for t in fr_toks for w in t if M.DIGIT.search(w))
    fr_num = sum(1 for t in fr_toks for w in t if M.NUMTOK.fullmatch(w))
    # duplicate sources, raw (E0.5's definition) and under normalised keys
    ks = [M.classify(x) for x in src]
    return {
        "rows": k, "source_tokens": tok, "rep_parent": SETS[name],
        "variants": variants,
        "rows_rep_positive_that_lose_all_rep_under_M1": len(lose),
        "of_which_source_line_occurs_more_than_once_in_parent": sum(1 for j in lose if line_count[src[j]] > 1),
        "rows_rep_positive_that_keep_rep_under_M1": len(pos) - len(lose),
        "rows_rep_zero_that_gain_rep_under_M2": len(gain),
        "rows_where_M2_is_below_rep": sum(1 for j in range(k) if rep["M2"][j] < rep["rep"][j]),
        "share_of_repeated_8gram_occurrences_touching_a_masked_token":
            (1 - c_m1 / c_orig) if c_orig else None,
        "token_share": {
            "num": cnt["n_num"] / tok, "date": cnt["n_date"] / tok, "sym": cnt["n_sym"] / tok,
            "masked_total": (cnt["n_num"] + cnt["n_date"] + cnt["n_sym"]) / tok,
            "tokens_containing_a_digit": cnt["n_dig"] / tok,
            "rows_with_a_masked_token": masked_rows / k,
            "symbol_matches": sym_matches,
            "counts": {**cnt, "source_tokens": tok},
            "fr_tokens_containing_a_digit": fr_dig / fr_n, "fr_num": fr_num / fr_n, "fr_tokens": fr_n,
        },
        "duplicate_source_fraction": {
            "raw_line": dup_rate(src),
            "token_sequence": dup_rate([" ".join(t) for t, _, _ in ks]),
            "A17_digits_to_0": dup_rate([" ".join(M.DIGIT.sub("0", w) for w in t) for t, _, _ in ks]),
            "full_mask_placeholders": dup_rate([" ".join(w if c == M.W else M.PLACEHOLDER[c]
                                                         for w, c in zip(t, cl)) for t, cl, _ in ks]),
        },
        "_lose": lose, "_gain": gain, "_rep": rep,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="~/e06data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.workers > 4:
        ap.error("at most 4 workers")
    t0 = time.time()
    try:
        bad = {}
        for f, h in PINNED.items():
            p = a.data / f
            got = sha256_file(p) if p.is_file() else "MISSING"
            if got != h:
                bad[f] = got
        if bad:
            raise Stop("inputs differ from their pins: " + json.dumps(bad, indent=1))
        print(f"sha256: {len(PINNED)} inputs match their pins", flush=True)
        man = json.loads((a.data / "phase0_e05/manifest.json").read_text(encoding="utf-8"))["e05"]

        par = {}
        for name in ("ft_topk", "ft_random"):
            lines = read_lines(a.data / f"e05_src/{name}.en")
            lab = read_lines(a.data / f"e05_src/{name}.source")
            if len(lines) != len(lab):
                raise Stop(f"{name}: {len(lines)} lines, {len(lab)} labels")
            st = corpus_stats(lines, a.workers, name)
            first = {}
            for i, s in enumerate(lines):
                first.setdefault(s, i)
            par[name] = {"lines": lines, "lab": lab, "st": st, "first": first,
                         "line_count": collections.Counter(lines)}

        # ---- reproduce E0.5 before anything else ----
        T = par["ft_topk"]
        st, n = T["st"], T["st"]["n"]
        rep_t = [ratio(int(st["c_orig"][i]), int(st["ng"][i])) for i in range(n)]
        med = rep_t[sorted(range(n), key=rep_t.__getitem__)[n // 2]]
        checks = {"median rep over ft_topk": (med, man["median_rep"])}
        un = [i for i in range(n) if T["lab"][i] == "un"]
        checks["ft_topk UN rows with rep = 0"] = (sum(1 for i in un if rep_t[i] == 0),
                                                  EXPECTED_COUNTS["ft_topk UN rows with rep = 0"])
        checks["ft_topk UN rows with rep > 0"] = (sum(1 for i in un if rep_t[i] > 0),
                                                  EXPECTED_COUNTS["ft_topk UN rows with rep > 0"])

        sets = {}
        for name, parent in SETS.items():
            src = read_lines(a.data / f"phase0_e05/{name}.en")
            fr = read_lines(a.data / f"phase0_e05/{name}.fr")
            P = par[parent]
            miss = sum(1 for s in src if s not in P["first"])
            if miss:
                raise Stop(f"{name}: {miss} source lines are not in {parent}")
            idx = [P["first"][s] for s in src]      # rep is a function of the source line alone
            d = describe_set(name, src, fr, idx, P["st"], P["line_count"])
            d["_src"], d["_idx"] = src, idx
            sets[name] = d
            e = man["sets"][name]
            checks[f"{name} rep_mean"] = (d["variants"]["rep"]["rep_mean"], e["rep_mean"])
            checks[f"{name} rep_positive_fraction"] = (d["variants"]["rep"]["rep_positive_fraction"],
                                                       e["rep_positive_fraction"])
            checks[f"{name} duplicate_source_fraction"] = (d["duplicate_source_fraction"]["raw_line"],
                                                           e["duplicate_source_fraction"])
        failed = {k: v for k, v in checks.items() if v[0] != v[1]}
        if failed:
            raise Stop("the unmasked statistic does not reproduce E0.5 exactly: "
                       + "; ".join(f"{k}: got {v[0]!r}, E0.5 {v[1]!r}" for k, v in failed.items()))
        print(f"reproduction: {len(checks)} checks equal E0.5's manifest exactly", flush=True)

        # ---- the whole band and its UN rows, for the E0.6 candidate pools ----
        def band(idx: list[int]) -> dict:
            k = len(idx)
            r = {v: [ratio(int(st[c][i]), int(st["ng"][i])) for i in idx]
                 for v, c in (("rep", "c_orig"), ("M1", "c_m1"), ("M1_digit", "c_m1_digit"),
                              ("M2", "c_m2"), ("M2_A17", "c_m2_a17"))}
            tok = sum(int(st["lens"][i]) for i in idx)
            return {"rows": k,
                    "variants": {v: {"rep_mean": sum(x) / k, "rows_rep_positive": sum(1 for y in x if y > 0)}
                                 for v, x in r.items()},
                    "rows_rep_positive_that_lose_all_rep_under_M1":
                        sum(1 for a_, b_ in zip(r["rep"], r["M1"]) if a_ > 0 and b_ == 0),
                    "rows_rep_zero_that_gain_rep_under_M2":
                        sum(1 for a_, b_ in zip(r["rep"], r["M2"]) if a_ == 0 and b_ > 0),
                    "rows_rep_zero_that_gain_rep_under_M2_A17":
                        sum(1 for a_, b_ in zip(r["rep"], r["M2_A17"]) if a_ == 0 and b_ > 0),
                    "token_share_masked_total": sum(int(st["n_num"][i] + st["n_date"][i] + st["n_sym"][i])
                                                    for i in idx) / tok,
                    "token_share_containing_a_digit": sum(int(st["n_dig"][i]) for i in idx) / tok}
        whole = {"ft_topk (all rows)": band(list(range(n))), "ft_topk UN rows": band(un)}

        # ---- examples: ft_un_rep rows with rep > 0 that have rep 0 under M1 ----
        rng = random.Random(SEED)
        d = sets["ft_un_rep"]
        pick = sorted(rng.sample(d["_lose"], min(N_EXAMPLES, len(d["_lose"]))))
        examples = []
        for j in pick:
            s = d["_src"][j]
            gc = [x for x in gram_counts(st, s) if x["count"] > 1]
            seen, rg = set(), []
            for x in sorted(gc, key=lambda x: -x["count"]):
                if x["gram"] not in seen:
                    seen.add(x["gram"])
                    rg.append(x)
            examples.append({"ft_un_rep_line": j + 1, "source": s,
                             "source_line_count_in_ft_topk": T["line_count"][s],
                             "rep": d["_rep"]["rep"][j], "rep_M1": d["_rep"]["M1"][j],
                             "rep_M2": d["_rep"]["M2"][j], "masked": M.render(s),
                             "repeated_8grams_touching_mask": sum(1 for x in gc if x["touches_mask"]),
                             "repeated_8grams_untouched": sum(1 for x in gc if not x["touches_mask"]),
                             "repeated_8grams_top4": rg[:4]})
        g = sets["ft_un_div"]
        pick_g = sorted(random.Random(SEED).sample(g["_gain"], min(10, len(g["_gain"]))))
        gain_ex = [{"ft_un_div_line": j + 1, "source": g["_src"][j], "rep_M2": g["_rep"]["M2"][j],
                    "masked": M.render(g["_src"][j])} for j in pick_g]

        # how often each masked class carries the lost rep, in ft_un_rep's losing rows
        sym_c, date_c = collections.Counter(), collections.Counter()
        for s in T["lines"]:
            if M.DIGIT.search(s):
                for x, y in M.symbol_spans(s):
                    sym_c[s[x:y].rstrip(".,;:)")] += 1
                for x, y in M.date_spans(s):
                    date_c[M.DIGIT.sub("0", s[x:y])] += 1

        res = {
            "item": "WMT2027_PLAN.md section 3.2 item 2: templated numerals",
            "descriptive_only": True,
            "definition": {
                "tokens": "lowercased \\w+ (re.UNICODE), phase0/e04_split.py line 13",
                "rep": "fraction of the row's 8-grams (blake2b-64 of ' '.join(8 tokens)) whose count over the "
                       "parent set is > 1; rows under 8 tokens take 0 (e04_split.py lines 29-37)",
                "parents": SETS,
                "token_classes": {"SYM": M.SYMBOL.pattern + "   (match must contain a digit)",
                                  "DATE": M.DATE.pattern,
                                  "NUM": "fullmatch " + M.NUMTOK.pattern + " on the lowercased token",
                                  "priority": "a token overlapping a SYM span is SYM, else overlapping a DATE "
                                              "span is DATE, else NUM if it matches, else WORD"},
                "M1": "exclude: a SYM/DATE/NUM token matches nothing, so an 8-gram occurrence containing one never "
                      "counts as repeated, and an untouched 8-gram counts only if it occurs untouched more than "
                      "once in the parent; denominator unchanged (headline)",
                "M1_digit": "M1 with exactly the tokens containing a Unicode digit masked",
                "M1_prime": "M1's repeated 8-grams over the row's number of 8-grams with no masked token (0 if none)",
                "M2": "placeholder: SYM/DATE/NUM tokens -> #sym/#date/#num, counts recomputed over the parent",
                "M2_A17": "every Unicode digit -> '0' inside each token (e06_build.py A17_RULE), counts recomputed",
            },
            "reproduction_checks": {k: {"got": v[0], "e05_manifest": v[1]} for k, v in checks.items()},
            "parents": {nm: {"rows": par[nm]["st"]["n"], "grams": par[nm]["st"]["grams"],
                             "gram_types_unmasked": par[nm]["st"]["gram_types"],
                             "rows_where_M2_is_below_rep": par[nm]["st"]["rows_M2_below_rep"],
                             "rows_where_strict_M1_is_below_loose_M1":
                                 par[nm]["st"]["rows_M1_strict_below_loose"]} for nm in par},
            "whole_band": whole,
            "sets": {nm: {k: v for k, v in x.items() if not k.startswith("_")} for nm, x in sets.items()},
            "examples_ft_un_rep_lose_all_rep_under_M1": examples,
            "examples_ft_un_div_gain_rep_under_M2": gain_ex,
            "ft_topk_symbol_matches": {"total": sum(sym_c.values()), "distinct": len(sym_c),
                                       "top30": sym_c.most_common(30),
                                       "series_codes": collections.Counter(
                                           x.split("/")[0] for x in sym_c.elements()).most_common(),
                                       "not_A_E_S": sorted(((x, c) for x, c in sym_c.items()
                                                            if x.split("/")[0] not in ("A", "E", "S")),
                                                           key=lambda y: (-y[1], y[0]))},
            "ft_topk_date_shapes": {"total": sum(date_c.values()), "top20": date_c.most_common(20)},
            "built_with": {"python": sys.version.split()[0], "unidata_version": unicodedata.unidata_version,
                           "numpy": np.__version__, "machine": platform.machine(), "node": platform.node(),
                           "workers": a.workers, "seconds": round(time.time() - t0, 1)},
            "inputs_sha256": PINNED,
        }
    except Stop as e:
        print(f"STOP: {e}\nNothing written.")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / ".results.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, a.out / "results.json")
    for nm, x in res["sets"].items():
        v = x["variants"]
        print(f"{nm:<16} rep {v['rep']['rep_mean']:.4f} ({v['rep']['rows_rep_positive']:,} >0) | "
              f"M1 {v['M1']['rep_mean']:.4f} ({v['M1']['rows_rep_positive']:,} >0) | "
              f"M2 {v['M2']['rep_mean']:.4f} ({v['M2']['rows_rep_positive']:,} >0) | "
              f"masked tokens {100 * x['token_share']['masked_total']:.2f}% "
              f"(num {100 * x['token_share']['num']:.2f} date {100 * x['token_share']['date']:.2f} "
              f"sym {100 * x['token_share']['sym']:.3f})", flush=True)
    print(f"wrote {a.out / 'results.json'} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
