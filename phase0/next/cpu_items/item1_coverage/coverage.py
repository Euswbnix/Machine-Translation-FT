#!/usr/bin/env python3
"""item1_coverage: WMT2027_PLAN.md section 3.2 item 1, vocabulary coverage "Saunders & DeNeefe's way".

Descriptive CPU analysis for the author's decisions (prereg_drafts.md E0.6 OPEN DECISION 1, risk 2),
not a registered result. It writes one JSON file and nothing else.

Headline measure (groundwork.json, note saunders2024: "the share of the generic model's SUBWORD
vocabulary that appears at least once, computed separately for source and target"), made exact
the way e06_build_spec.md A28 proposes and e06_build.py --measure implements it:
  - every row is read as the trainer reads it (Machine_translation/src/data/dataset.py): the line,
    .strip(), encoded with the base model's spm_enfr_v1_fixed.model defaults, no sampling, no
    BOS/EOS (those are control ids and never count);
  - piece types = distinct ids over all rows of the set on that side, control (<pad>, <s>, </s>)
    and unknown (<unk>) ids excluded;
  - coverage = piece types / countable ids (32,000 - 4 = 31,996: the model is one joint en-fr BPE
    vocabulary, so source and target share one denominator).
Variants, reported beside it because the paper leaves them open: minimum count k in {1, 2, 5, 20};
the denominator restricted to the pieces the base saw on that side of its pretraining corpus;
pretraining-token-weighted coverage; coverage at a matched number of pieces; coverage over the rows
the trainer keeps. Word types (lowercased \\w+) are counted the prereg's way (D11, D12) and at a
common word budget for every set.

usage: coverage.py --data ~/e06data --sums used_inputs_SHA256SUMS --out results.json [--threads 4]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import platform
import random
import re
import sys
import time
import unicodedata
from array import array
from pathlib import Path

import numpy as np
import sentencepiece as spm

WORD = re.compile(r"\w+", re.UNICODE)      # e04_split.py, e05_build.py, e06_build.py
SOURCE = "un"
SEED = 42
B_FR = 6_500_000            # prereg D12: French word types in the first 6.5M French tokens
B_SRC = 5_600_000           # common source-word budget, below the smallest set's 5,606,498 source words
KS = (1, 2, 5, 20)
MAX_SEQ_LEN = 256           # the trainer's max_tokens, BOS and EOS included
CAPS = (42_000, 58_000)
NOISE_SEEDS = (1, 2, 3, 4, 5)
E05 = ("ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small")
# numbers stated in phase0/prereg_drafts.md, e06_build_spec.md 6.2/6.3/6.5, e06_build.py and E0.5's manifest
EXPECTED = {
    "ft_random rows": 926_999, "ft_topk rows": 926_999,
    "RU0 (UN rows of ft_random with rep_rand = 0)": 450_709,
    "A_narrow capped pool at 42,000 / 58,000": 402_010,
    "rows shared by R0 and A_narrow": 136_916,
    "R0 source word types (D11)": 62_000, "R0 French word types at 6.5M (D12)": 72_574,
    "A_narrow source word types (D11)": 33_562, "A_narrow French word types at 6.5M (D12)": 43_703,
    "ft_un_rep source word types (D11)": 25_134, "ft_un_rep French word types at 6.5M (D12)": 32_684,
    "ft_un_div source word types (D11)": 33_874, "ft_un_div French word types at 6.5M (D12)": 43_021,
    "ft_un_random source word types (D11)": 58_441,
    "ft_topk_div source word types (D11)": 38_783, "ft_topk_rep source word types (D11)": 27_524,
    "ft_random_small source word types (D11)": 94_897,
}
CONTRASTS = (
    ("ft_un_rep vs ft_un_div (E0.5 Q1, E0.6 Q1')", "ft_un_rep", "ft_un_div", True),
    ("ft_un_div vs ft_un_random (E0.5 Q2, E0.6 Q2')", "ft_un_div", "ft_un_random", True),
    ("A_narrow vs R0 (E0.6 N)", "A_narrow", "R0", True),
    ("A_narrow vs ft_un_div (E0.6 S)", "A_narrow", "ft_un_div", False),
    ("ft_un_div vs R0 (E0.6 P)", "ft_un_div", "R0", False),
    ("ft_topk_rep vs ft_topk_div (E0.4 split)", "ft_topk_rep", "ft_topk_div", False),
    ("ft_topk_div vs ft_random_small (E0.4)", "ft_topk_div", "ft_random_small", False),
)

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def read_lines(p: Path) -> list[str]:
    # e06_build.py read_lines: LF is the only line break
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CSR:
    """Rows of token ids: lens, offsets, flat ids."""

    def __init__(self, lens, ids):
        self.lens = np.asarray(lens, dtype=np.int64)
        self.n = len(self.lens)
        self.off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(self.lens, out=self.off[1:])
        self.ids = ids
        assert int(self.off[-1]) == len(ids)

    def sel(self, idx):
        if idx is None:
            return self.ids
        m = np.zeros(self.n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return self.ids[np.repeat(m, self.lens)]

    def counts(self, idx, nv: int) -> np.ndarray:
        return np.bincount(self.sel(idx), minlength=nv)

    def tokens(self, idx) -> int:
        return int(self.lens.sum()) if idx is None else int(self.lens[np.asarray(idx, dtype=np.int64)].sum())

    def first_tokens(self, idx, budget: int):
        """e06_build.py Side.types_at / vocab/common.py types_at: the ids of the first `budget`
        tokens of the rows in a Random(0) shuffle of the ascending index list; None if the set has
        <= budget tokens."""
        order = list(range(self.n)) if idx is None else sorted(int(i) for i in idx)
        random.Random(0).shuffle(order)
        o = np.asarray(order, dtype=np.int64)
        ln = self.lens[o]
        cum = np.cumsum(ln)
        if not len(cum) or cum[-1] <= budget:
            return None
        r = int(np.searchsorted(cum, budget)) + 1
        o, ln = o[:r], ln[:r]
        start = np.repeat(self.off[o] - (np.cumsum(ln) - ln), ln)
        gidx = start + np.arange(int(ln.sum()), dtype=np.int64)
        return self.ids[gidx[:budget]]

    def types_at(self, idx, budget: int):
        x = self.first_tokens(idx, budget)
        return None if x is None else int(np.unique(x).size)


def word_csr(lines: list[str], vocab: dict) -> CSR:
    sd = vocab.setdefault
    lens = np.empty(len(lines), dtype=np.int64)
    buf = array("i")
    for k, s in enumerate(lines):
        t = WORD.findall(s)
        lens[k] = len(t)
        buf.extend(sd(w.lower(), len(vocab)) for w in t)
    return CSR(lens, np.frombuffer(buf, dtype=np.int32).copy())


def piece_csr(sp, lines: list[str], threads: int, chunk: int = 100_000) -> CSR:
    lp, ip = [], []
    for k in range(0, len(lines), chunk):
        enc = sp.encode([x.strip() for x in lines[k:k + chunk]], out_type=int, num_threads=threads)
        ln = np.fromiter((len(e) for e in enc), dtype=np.int64, count=len(enc))
        lp.append(ln)
        ip.append(np.fromiter(itertools.chain.from_iterable(enc), dtype=np.int32, count=int(ln.sum())))
    return CSR(np.concatenate(lp), np.concatenate(ip))


def stream_piece_counts(sp, path: Path, nv: int, threads: int, chunk: int = 200_000) -> dict:
    cnt = np.zeros(nv, dtype=np.int64)
    n = ncr = nempty = 0
    buf: list[str] = []

    def flush():
        nonlocal cnt
        enc = sp.encode(buf, out_type=int, num_threads=threads)
        tot = sum(len(e) for e in enc)
        cnt += np.bincount(np.fromiter(itertools.chain.from_iterable(enc), dtype=np.int32, count=tot), minlength=nv)
        buf.clear()

    with open(path, encoding="utf-8", newline="\n") as f:
        for ln in f:
            s = ln.rstrip("\n")
            n += 1
            if "\r" in s:
                ncr += 1
            s = s.strip()
            if not s:
                nempty += 1
            buf.append(s)
            if len(buf) >= chunk:
                flush()
    if buf:
        flush()
    return {"counts": cnt, "lines": n, "lines_with_cr": ncr, "empty_after_strip": nempty}


def rep_dup_counts(lines: list[str]):
    """rep_within (e06_build.py) in a vectorised form: per row, the number of its 8-grams (blake2b-64
    keys of the lowercased \\w+ tokens joined by one space) whose count over the whole list is above
    one, and the number of its 8-grams. rep = dup / grams; rep = 0 for rows under 8 tokens."""
    n = len(lines)
    keys = array("Q")
    ngr = np.zeros(n, dtype=np.int64)
    b2 = hashlib.blake2b
    for i, s in enumerate(lines):
        t = [w.lower() for w in WORD.findall(s)]
        g = len(t) - 7
        if g > 0:
            ngr[i] = g
            keys.extend(int.from_bytes(b2(" ".join(t[j:j + 8]).encode("utf-8"), digest_size=8).digest(), "little")
                        for j in range(g))
    k = np.frombuffer(keys, dtype=np.uint64)
    _, inv, c = np.unique(k, return_inverse=True, return_counts=True)
    dup = (c[inv] > 1).astype(np.int64)
    row_of = np.repeat(np.arange(n, dtype=np.int64), ngr)
    ndup = np.bincount(row_of, weights=dup, minlength=n).astype(np.int64)
    return ndup, ngr


def match_random(hist: dict, pool_len, eligible, seed: int):
    """e06_build.py match_random (= e05_build.py's): exactly hist[L] rows per length L, at random
    within the ascending bucket, lengths ascending, one Random(seed)."""
    buckets: dict[int, list[int]] = {}
    for i in eligible:
        buckets.setdefault(int(pool_len[i]), []).append(int(i))
    short = {L: (m, len(buckets.get(L, []))) for L, m in hist.items() if len(buckets.get(L, [])) < m}
    if short:
        raise SystemExit(f"STOP: match_random short in {len(short)} bucket(s)")
    rng = random.Random(seed)
    kept: list[int] = []
    for L in sorted(hist):
        kept += rng.sample(buckets[L], hist[L])
    return sorted(kept), min(len(buckets[L]) / hist[L] for L in hist)


def pool_maxranks(csr: CSR, rows) -> np.ndarray:
    """e06_build.py pool_order, one side: frequency ranks over the pool rows (ascending), ties to
    the first occurrence inside the pool (Counter.most_common order); each row's largest rank,
    -1 for an empty row."""
    rows = np.asarray(rows, dtype=np.int64)
    assert np.all(np.diff(rows) > 0)
    sel = csr.sel(rows)                       # ascending row order, tokens in order
    nv = int(sel.max()) + 1 if sel.size else 1
    cnt = np.bincount(sel, minlength=nv)
    uniq, first = np.unique(sel, return_index=True)
    order = np.lexsort((first, -cnt[uniq]))
    rank = np.full(nv, -1, dtype=np.int64)
    rank[uniq[order]] = np.arange(len(uniq), dtype=np.int64)
    ln = csr.lens[rows]
    r = rank[sel]
    mx = np.full(len(rows), -1, dtype=np.int64)
    nz = ln > 0
    starts = np.concatenate(([0], np.cumsum(ln)[:-1]))
    if nz.any():
        mx[nz] = np.maximum.reduceat(r, starts[nz])
    return mx


def py_types_at(tok_lists, budget: int):
    """vocab/common.py types_at, literally (rows of the set file in file order)."""
    rng = random.Random(0)
    idx = list(range(len(tok_lists)))
    rng.shuffle(idx)
    seen, n = set(), 0
    for i in idx:
        for w in tok_lists[i]:
            if n >= budget:
                return len(seen)
            seen.add(w)
            n += 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sums", type=Path, required=True, help="sha256sum list with paths 'e06data/...'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    D = a.data
    res: dict = {"what": "item1_coverage: Saunders & DeNeefe (2024) subword vocabulary coverage and word types "
                         "(WMT2027_PLAN.md 3.2 item 1); descriptive, not a registered result",
                 "interpreter": {"python": sys.version.split()[0], "unidata_version": unicodedata.unidata_version,
                                 "machine": platform.machine(), "numpy": np.__version__,
                                 "sentencepiece": getattr(spm, "__version__", "?"), "spm_threads": a.threads}}
    checks: list[dict] = []

    def check(name, ok, **kw):
        checks.append({"check": name, "ok": bool(ok), **kw})
        log(("OK   " if ok else "FAIL ") + name, kw if kw else "")
        return ok

    # ---------------- inputs, sha256 ----------------
    want = {}
    for ln in a.sums.read_text().splitlines():
        h, p = ln.split(None, 1)
        want[p.strip().removeprefix("e06data/")] = h
    used = ["e05_src/ft_topk.en", "e05_src/ft_topk.fr", "e05_src/ft_topk.source",
            "e05_src/ft_random.en", "e05_src/ft_random.fr", "e05_src/ft_random.source",
            "phase0_e05/manifest.json", "spm/spm_enfr_v1_fixed.model", "vocab/R0.npy", "vocab/A_narrow_42_58.npy",
            "v1_fixed/train.clean.en", "v1_fixed/train.clean.fr", "code/phase0/e06_build.py"]
    used += [f"phase0_e05/{s}.{e}" for s in E05 for e in ("en", "fr")]
    res["inputs"] = {}
    bad = []
    for f in used:
        h = sha256_file(D / f)
        res["inputs"][f] = {"sha256": h, "listed": want.get(f), "match": h == want.get(f)}
        if h != want.get(f):
            bad.append(f)
    if bad:
        print("STOP: sha256 mismatch: " + ", ".join(bad))
        return 1
    log(f"sha256: {len(used)} inputs match the list")

    # ---------------- the SentencePiece model ----------------
    sp = spm.SentencePieceProcessor()
    sp.Load(str(D / "spm/spm_enfr_v1_fixed.model"))
    nv = int(sp.get_piece_size())
    special = np.array([bool(sp.is_control(i)) or bool(sp.is_unknown(i)) for i in range(nv)], dtype=bool)
    unk = [i for i in range(nv) if sp.is_unknown(i)]
    unused = int(sum(bool(sp.is_unused(i)) for i in range(nv)))
    byte = int(sum(bool(sp.is_byte(i)) for i in range(nv)))
    countable = int((~special).sum())
    bpe_like = all(sp.get_score(i) == -(i - 4) for i in range(4, nv))
    res["spm"] = {"vocab_size": nv, "special_ids": {int(i): sp.id_to_piece(int(i)) for i in np.nonzero(special)[0]},
                  "countable_ids": countable, "unused_ids": unused, "byte_ids": byte,
                  "scores_are_minus_rank (BPE)": bpe_like,
                  "joint": "one model for both sides (dataset.py: self.tokenizer.sp encodes src and tgt)"}
    log(f"spm: {nv} ids, countable {countable}, special {res['spm']['special_ids']}, BPE-like {bpe_like}")

    # ---------------- the base's pretraining corpus, per side ----------------
    pre = {}
    for side, f in (("source", "v1_fixed/train.clean.en"), ("target", "v1_fixed/train.clean.fr")):
        r = stream_piece_counts(sp, D / f, nv, a.threads)
        c = r["counts"].copy()
        c[special] = 0
        pre[side] = c
        res.setdefault("pretraining", {})[side] = {
            "file": f, "lines": r["lines"], "lines_with_cr": r["lines_with_cr"], "empty_after_strip": r["empty_after_strip"],
            "piece_tokens": int(r["counts"].sum()), "unknown_tokens": int(r["counts"][unk].sum()),
            "piece_types_seen": int((c > 0).sum()), "piece_types_seen_share_of_countable": float((c > 0).sum() / countable)}
        log(f"pretraining {side}: {r['lines']:,} lines, {int(r['counts'].sum()):,} pieces, {(c > 0).sum():,} types seen")
    check("pretraining lines = 9,312,233 on both sides",
          res["pretraining"]["source"]["lines"] == res["pretraining"]["target"]["lines"] == 9_312_233)
    side_vocab = {s: pre[s] > 0 for s in pre}

    # ---------------- the sets ----------------
    wv = {"source": {}, "target": {}}           # one word vocabulary per side over every file
    store: dict[str, dict] = {}                 # file group -> CSRs
    sets: dict[str, tuple[str, object]] = {}    # set -> (file group, idx or None)

    def load_group(name: str, folder: str):
        en, fr = read_lines(D / folder / f"{name}.en"), read_lines(D / folder / f"{name}.fr")
        if len(en) != len(fr):
            raise SystemExit(f"STOP: {name}: .en {len(en):,} lines, .fr {len(fr):,}")
        g = {"rows": len(en),
             "cr": np.fromiter((("\r" in x) or ("\r" in y) for x, y in zip(en, fr)), dtype=bool, count=len(en)),
             "w": {"source": word_csr(en, wv["source"]), "target": word_csr(fr, wv["target"])},
             "p": {"source": piece_csr(sp, en, a.threads), "target": piece_csr(sp, fr, a.threads)}}
        store[name] = g
        log(f"loaded {folder}/{name}: {len(en):,} rows")
        return en, fr

    for s in E05:
        en, fr = load_group(s, "phase0_e05")
        sets[s] = (s, None)
        if s == "ft_un_rep":                     # literal re-checks on one set
            ft = [[w.lower() for w in WORD.findall(x)] for x in fr]
            lit = py_types_at(ft, B_FR)
            vec = store[s]["w"]["target"].types_at(None, B_FR)
            check("vectorised types_at = vocab/common.py types_at (ft_un_rep French, 6.5M)", lit == vec, literal=lit, vectorised=vec)
            one = sp.encode([x.strip() for x in en], out_type=int, num_threads=1)
            lit_ids = {t for e in one for t in e} - set(np.nonzero(special)[0].tolist())
            vec_ids = set(np.nonzero((store[s]["p"]["source"].counts(None, nv) > 0) & ~special)[0].tolist())
            check("piece types with 1 thread (Python set) = with N threads (bincount), ft_un_rep source",
                  lit_ids == vec_ids, one_thread=len(lit_ids), n_threads=len(vec_ids))
            del ft, one
        del en, fr
    for s in ("ft_topk", "ft_random"):
        en, fr = load_group(s, "e05_src")
        sets[s] = (s, None)
        check(f"{s} rows = 926,999", store[s]["rows"] == EXPECTED[f"{s} rows"], rows=store[s]["rows"])
        if s == "ft_random":
            R_en, R_fr = en, fr
        del en, fr
    R_lab = read_lines(D / "e05_src/ft_random.source")
    check("ft_random.source has one label per row", len(R_lab) == store["ft_random"]["rows"], lines=len(R_lab))

    # ---------------- R0 and A_narrow: row order mapping (e06_build.py: ft_random line order) --------
    Rw = store["ft_random"]["w"]
    lens_r = Rw["source"].lens
    un_hist = collections.Counter(int(x) for x in store["ft_un_div"]["w"]["source"].lens)
    check("ft_un_div and ft_un_rep share one source-length histogram",
          un_hist == collections.Counter(int(x) for x in store["ft_un_rep"]["w"]["source"].lens))
    log("rep_rand over ft_random ...")
    ndup, ngr = rep_dup_counts(R_en)
    un_rows = [i for i in range(len(R_lab)) if R_lab[i] == SOURCE]
    ru0 = [i for i in un_rows if ndup[i] == 0]
    check("RU0 (UN rows of ft_random with rep_rand = 0) = 450,709", len(ru0) == EXPECTED["RU0 (UN rows of ft_random with rep_rand = 0)"], got=len(ru0))
    idx = {}
    for nm, f in (("R0", "vocab/R0.npy"), ("A_narrow", "vocab/A_narrow_42_58.npy")):
        x = np.load(D / f, allow_pickle=False)
        ok = x.ndim == 1 and np.issubdtype(x.dtype, np.integer)
        check(f"{nm}: 1-d integer index ({x.dtype}, shape {x.shape})", ok)
        x = x.astype(np.int64)
        check(f"{nm}: 214,489 rows, strictly ascending, within [0, 926,999)",
              len(x) == 214_489 and bool(np.all(np.diff(x) > 0)) and x.min() >= 0 and x.max() < store["ft_random"]["rows"],
              rows=len(x), min=int(x.min()), max=int(x.max()))
        lab_ok = all(R_lab[i] == SOURCE for i in x)
        check(f"{nm}: every row is labelled '{SOURCE}' in ft_random.source", lab_ok)
        check(f"{nm}: rep_rand = 0 on every row", bool(np.all(ndup[x] == 0)), rows_with_rep=int((ndup[x] > 0).sum()))
        h = collections.Counter(int(v) for v in lens_r[x])
        check(f"{nm}: source-length histogram = ft_un_div's", h == un_hist)
        xs = x + 1
        xs = xs[xs < len(lens_r)]
        check(f"{nm}: power check: the same indices read one line later do NOT give ft_un_div's histogram",
              collections.Counter(int(v) for v in lens_r[xs]) != un_hist)
        idx[nm] = x
        sets[nm] = ("ft_random", x)
    r0b, slack = match_random(un_hist, lens_r, ru0, SEED)
    check("R0.npy = match_random(ft_un_div's histogram, RU0, seed 42) row for row (e06_build.py build_r0)",
          np.array_equal(np.asarray(r0b, dtype=np.int64), idx["R0"]), tightest_slack=slack)
    kub, slack_u = match_random(un_hist, lens_r, un_rows, SEED)
    ur_en = read_lines(D / "phase0_e05/ft_un_random.en")
    ur_fr = read_lines(D / "phase0_e05/ft_un_random.fr")
    check("ft_un_random file = ft_random rows match_random(histogram, UN rows, seed 42), both sides, line for line",
          [R_en[i] for i in kub] == ur_en and [R_fr[i] for i in kub] == ur_fr, tightest_slack=slack_u)
    del ur_en, ur_fr
    me = pool_maxranks(Rw["source"], ru0)
    mf = pool_maxranks(Rw["target"], ru0)
    capped = np.asarray(ru0, dtype=np.int64)[(me < CAPS[0]) & (mf < CAPS[1])]
    check("A_narrow capped pool at 42,000 / 58,000 = 402,010 rows", len(capped) == EXPECTED["A_narrow capped pool at 42,000 / 58,000"], got=int(len(capped)))
    check("A_narrow is inside the capped pool", bool(np.isin(idx["A_narrow"], capped).all()))
    shared = int(np.intersect1d(idx["R0"], idx["A_narrow"]).size)
    check("rows shared by R0 and A_narrow = 136,916", shared == EXPECTED["rows shared by R0 and A_narrow"], got=shared)
    del me, mf, capped

    # ---------------- resampling draws (scale of draw-to-draw spread) ----------------
    noise_sets = {}
    for sd in NOISE_SEEDS:
        noise_sets[f"R0_rule_seed{sd}"] = np.asarray(match_random(un_hist, lens_r, ru0, sd)[0], dtype=np.int64)
        noise_sets[f"ft_un_random_rule_seed{sd}"] = np.asarray(match_random(un_hist, lens_r, un_rows, sd)[0], dtype=np.int64)
    del R_en, R_fr

    # ---------------- budgets for coverage at matched piece count ----------------
    def ptoks(nm, side):
        g, ix = sets[nm]
        return store[g]["p"][side].tokens(ix)

    budget = {}
    for side in ("source", "target"):
        mn = min(ptoks(nm, side) for nm in sets)
        budget[side] = ((mn - 1) // 100_000) * 100_000
    res["budgets"] = {"source_words_common": B_SRC, "french_words": B_FR,
                      "pieces_matched": budget,
                      "pieces_matched_rule": "largest multiple of 100,000 below the smallest set's piece count on that side"}
    log(f"piece budgets {budget}")

    # ---------------- measures ----------------
    def piece_block(cnt: np.ndarray, side: str) -> dict:
        real = cnt.copy()
        real[special] = 0
        present = real > 0
        sv = side_vocab[side]
        w = pre[side]
        return {"piece_tokens": int(cnt.sum()), "unknown_tokens": int(cnt[unk].sum()),
                "piece_types": int(present.sum()),
                "coverage": float(present.sum() / countable),
                "coverage_min_count": {f"k{k}": float((real >= k).sum() / countable) for k in KS},
                "piece_types_min_count": {f"k{k}": int((real >= k).sum()) for k in KS},
                "coverage_of_side_vocab": float((present & sv).sum() / sv.sum()),
                "types_outside_side_vocab": int((present & ~sv).sum()),
                "pretraining_token_share_covered": float(w[present].sum() / w.sum()),
                "pretraining_token_share_uncovered_ppm": float(1e6 * w[~present].sum() / w.sum())}

    def measure(nm: str, g: str, ix) -> dict:
        G = store[g]
        P, W = G["p"], G["w"]
        rows = G["rows"] if ix is None else len(ix)
        d = {"parent_file": g, "rows": rows}
        for side in ("source", "target"):
            d[side] = piece_block(P[side].counts(ix, nv), side)
            ft = P[side].first_tokens(ix, budget[side])
            if ft is None:
                d[side]["coverage_at_matched_pieces"] = None
            else:
                c = np.bincount(ft, minlength=nv)
                c[special] = 0
                d[side]["coverage_at_matched_pieces"] = float((c > 0).sum() / countable)
                d[side]["piece_types_at_matched_pieces"] = int((c > 0).sum())
        # the rows the trainer keeps (dataset.py: both sides non-empty after strip, both <= 256 with BOS/EOS)
        ls = P["source"].lens if ix is None else P["source"].lens[ix]
        lt = P["target"].lens if ix is None else P["target"].lens[ix]
        keep = (ls > 0) & (lt > 0) & (ls + 2 <= MAX_SEQ_LEN) & (lt + 2 <= MAX_SEQ_LEN)
        d["rows_dropped_by_trainer"] = int((~keep).sum())
        if (~keep).any():
            base = np.arange(G["rows"], dtype=np.int64) if ix is None else np.asarray(ix, dtype=np.int64)
            kept = base[keep]
            for side in ("source", "target"):
                c = P[side].counts(kept, nv)
                c[special] = 0
                d[side]["coverage_rows_kept_by_trainer"] = float((c > 0).sum() / countable)
        else:
            for side in ("source", "target"):
                d[side]["coverage_rows_kept_by_trainer"] = d[side]["coverage"]
        d["words"] = {"source_words": W["source"].tokens(ix), "french_words": W["target"].tokens(ix),
                      "source_word_types_D11": int(np.unique(W["source"].sel(ix)).size),
                      "french_word_types_all_rows": int(np.unique(W["target"].sel(ix)).size),
                      f"source_word_types_at_{B_SRC}": W["source"].types_at(ix, B_SRC),
                      f"french_word_types_at_{B_FR}_D12": W["target"].types_at(ix, B_FR)}
        d["rows_with_cr"] = int(G["cr"].sum()) if ix is None else int(G["cr"][np.asarray(ix, dtype=np.int64)].sum())
        return d

    out = {}
    for nm, (g, ix) in sets.items():
        out[nm] = measure(nm, g, ix)
        s, t = out[nm]["source"], out[nm]["target"]
        log(f"{nm:<16} cov src {100 * s['coverage']:.2f}% ({s['piece_types']:,})  tgt {100 * t['coverage']:.2f}% "
            f"({t['piece_types']:,})  words src {out[nm]['words']['source_word_types_D11']:,}  "
            f"fr@6.5M {out[nm]['words'][f'french_word_types_at_{B_FR}_D12']}")
    res["cr_rows_in_files"] = {g: int(store[g]["cr"].sum()) for g in store}
    res["sets"] = out

    # expected values (provenance in EXPECTED's comment)
    for nm in ("R0", "A_narrow", "ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small"):
        k = f"{nm} source word types (D11)"
        if k in EXPECTED:
            got = out[nm]["words"]["source_word_types_D11"]
            check(f"{k} = {EXPECTED[k]:,}", got == EXPECTED[k], got=got)
        k = f"{nm} French word types at 6.5M (D12)"
        if k in EXPECTED:
            got = out[nm]["words"][f"french_word_types_at_{B_FR}_D12"]
            check(f"{k} = {EXPECTED[k]:,}", got == EXPECTED[k], got=got)

    # ---------------- resampling spread ----------------
    nz = {}
    for nm, ix in noise_sets.items():
        P, W = store["ft_random"]["p"], store["ft_random"]["w"]
        row = {}
        for side in ("source", "target"):
            c = P[side].counts(ix, nv)
            c[special] = 0
            row[f"{side}_coverage"] = float((c > 0).sum() / countable)
            row[f"{side}_piece_types"] = int((c > 0).sum())
        row["source_word_types_D11"] = int(np.unique(W["source"].sel(ix)).size)
        row[f"french_word_types_at_{B_FR}_D12"] = W["target"].types_at(ix, B_FR)
        nz[nm] = row
    summ = {}
    for rule in ("R0_rule", "ft_un_random_rule"):
        rows = [v for k, v in nz.items() if k.startswith(rule + "_seed")]
        summ[rule] = {}
        for m in rows[0]:
            xs = np.array([r[m] for r in rows], dtype=float)
            summ[rule][m] = {"mean": float(xs.mean()), "sd": float(xs.std(ddof=1)), "min": float(xs.min()), "max": float(xs.max())}
    res["resampling"] = {"what": "the R0 rule (RU0 pool) and ft_un_random's rule (UN rows of ft_random) redrawn with seeds "
                                 f"{list(NOISE_SEEDS)} at ft_un_div's histogram; draws overlap heavily (same finite pool), so "
                                 "the sd is the draw-to-draw spread of these rules, not sampling error from a population",
                         "draws": nz, "summary": summ}

    # ---------------- contrasts ----------------
    def flat(d: dict) -> dict:
        f = {}
        for side in ("source", "target"):
            b = d[side]
            f[f"{side}_coverage"] = b["coverage"]
            f[f"{side}_piece_types"] = b["piece_types"]
            for k in KS[1:]:
                f[f"{side}_coverage_k{k}"] = b["coverage_min_count"][f"k{k}"]
            f[f"{side}_coverage_of_side_vocab"] = b["coverage_of_side_vocab"]
            f[f"{side}_pretraining_token_share_uncovered_ppm"] = b["pretraining_token_share_uncovered_ppm"]
            f[f"{side}_coverage_at_matched_pieces"] = b["coverage_at_matched_pieces"]
        w = d["words"]
        f["source_word_types_D11"] = w["source_word_types_D11"]
        f[f"source_word_types_at_{B_SRC}"] = w[f"source_word_types_at_{B_SRC}"]
        f[f"french_word_types_at_{B_FR}_D12"] = w[f"french_word_types_at_{B_FR}_D12"]
        return f

    cs = []
    for label, x, y, main_ in CONTRASTS:
        fa, fb = flat(out[x]), flat(out[y])
        row = {"contrast": label, "a": x, "b": y, "asked_for": main_, "measures": {}}
        for m in fa:
            va, vb = fa[m], fb[m]
            if va is None or vb is None:
                row["measures"][m] = {"a": va, "b": vb}
                continue
            row["measures"][m] = {"a": va, "b": vb, "a_minus_b": va - vb, "relative_(a-b)/b": (va - vb) / vb if vb else None}
        cs.append(row)
    res["contrasts"] = cs
    res["checks"] = checks
    res["all_checks_ok"] = all(c["ok"] for c in checks)
    res["definitions"] = {
        "row": "the line as read with LF as the only break (e06_build.py read_lines), then .strip() (dataset.py)",
        "pieces": "sp.encode(line.strip(), out_type=int): spm_enfr_v1_fixed.model defaults, no sampling, no BOS/EOS",
        "coverage": "distinct piece ids over all rows of the set on that side, control and unknown ids excluded, / countable "
                    "ids (31,996 of one joint 32,000-id en-fr vocabulary): Saunders & DeNeefe's 'share of the generic subword "
                    "vocabulary that appears at least once', with e06_build_spec.md A28's PROPOSAL for which ids count",
        "coverage_min_count": "share of countable ids occurring at least k times in the set",
        "coverage_of_side_vocab": "the same, with numerator and denominator restricted to the ids the base saw at least once "
                                  "on that side of its pretraining corpus (train.clean.en / .fr, read the same way)",
        "pretraining_token_share_uncovered_ppm": "pretraining piece tokens on that side whose id never occurs in the set, per "
                                                 "million (token-weighted coverage = 1 - this / 1e6)",
        "coverage_at_matched_pieces": "coverage among the first N pieces of the rows in a Random(0) shuffle of the set's "
                                      "ascending row list (types_at's rule on pieces); N per side in budgets",
        "coverage_rows_kept_by_trainer": "coverage over the rows dataset.py keeps (both sides non-empty, <= 256 pieces with BOS/EOS)",
        "source_word_types_D11": "distinct lowercased \\w+ source tokens over all rows (prereg D11, e05_build.py lexicon)",
        f"source_word_types_at_{B_SRC}": f"distinct source tokens in the first {B_SRC:,} source tokens, types_at's rule",
        f"french_word_types_at_{B_FR}_D12": "prereg D12 (vocab/common.py types_at, Random(0) shuffle of the ascending rows)",
        "R0 / A_narrow rows": "row indices into ft_random in its LF line order (e06_build.py Parent('ft_random'), build_r0, "
                              "build_a_narrow); checked above"}
    res["runtime_seconds"] = round(time.time() - T0, 1)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {a.out}; all checks ok: {res['all_checks_ok']}")
    return 0 if res["all_checks_ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
