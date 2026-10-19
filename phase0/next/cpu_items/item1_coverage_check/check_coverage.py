#!/usr/bin/env python3
"""item1_coverage_check: an independent re-derivation of item1_coverage's numbers.

Written from the definitions, not from the builder's coverage.py (never imported, never copied):
  - groundwork.json saunders2024: coverage = share of the base model's subword vocabulary that
    appears at least once in the set, source and target separately;
  - prereg_drafts.md OD 1(c); e06_build_spec.md A28 (encode with the pinned model's defaults, no
    sampling; distinct ids, control and unknown ids excluded; mirror the trainer);
  - Machine_translation/src/data/dataset.py (rev 200f6c0): text-mode read, line.strip(), skip a pair
    with an empty side, sp.encode(add_bos=True, add_eos=True), drop a pair if either side has more
    than max_tokens (256) ids including BOS/EOS;
  - prereg "Definitions shared by every build": lowercased \\w+ tokens, rep over the parent's
    8-grams, the exact-histogram random draw (Random(42), lengths ascending), D11/D12 word types.

Independent choices on purpose: lines are split on b"\\n" from raw bytes; 8-gram repetition is
counted EXACTLY by composing token ids (2 -> 4 -> 8 spans, np.unique), not with blake2b keys; the
SentencePiece model's piece types are read by a hand-written protobuf parser and cross-checked
against the library.

Usage (Studio):  nice -n 5 python -I check_coverage.py --data ~/e06data --out <dir> --threads 3
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import statistics
import struct
import sys
import time
import unicodedata
from array import array
from pathlib import Path

import numpy as np
import sentencepiece as spm

WORD = re.compile(r"\w+")
T0 = time.time()
E05 = ("ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div", "ft_random_small")
ORDER = ("ft_un_rep", "ft_un_div", "ft_un_random", "A_narrow", "R0", "ft_topk_rep", "ft_topk_div",
         "ft_random_small", "ft_topk", "ft_random")
MAXTOK = 256
KS = (1, 2, 5, 20)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_lf(p: Path) -> tuple[list[str], dict]:
    raw = p.read_bytes()
    parts = raw.split(b"\n")
    trailing = parts[-1] == b""
    if trailing:
        parts.pop()
    rows = [x.decode("utf-8") for x in parts]          # strict
    odd = {"cr": "\r", "u2028": " ", "u2029": " ", "nel": "\x85", "vt": "\x0b", "ff": "\x0c",
           "fs_gs_rs": None}
    info = {"lines": len(rows), "ends_with_lf": trailing}
    for k, ch in odd.items():
        if ch is None:
            info["rows_with_" + k] = sum(1 for r in rows if ("\x1c" in r or "\x1d" in r or "\x1e" in r))
        else:
            info["rows_with_" + k] = sum(1 for r in rows if ch in r)
    info["empty_after_strip"] = sum(1 for r in rows if not r.strip())
    info["rows_changed_by_strip"] = sum(1 for r in rows if r != r.strip())
    return rows, info


# ---------------- hand-written protobuf reader for the .model file ----------------
def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b: bytes) -> list[tuple[int, int, object]]:
    out, i = [], 0
    while i < len(b):
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        out.append((fn, wt, v))
    return out


TYPE_NAMES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}


def parse_model(p: Path) -> dict:
    top = _fields(p.read_bytes())
    pieces = []
    trainer = {}
    for fn, wt, v in top:
        if fn == 1 and wt == 2:
            sub = _fields(v)
            piece, score, typ = None, 0.0, 1
            for f2, w2, v2 in sub:
                if f2 == 1:
                    piece = v2.decode("utf-8")
                elif f2 == 2 and w2 == 5:
                    score = struct.unpack("<f", v2)[0]
                elif f2 == 3 and w2 == 0:
                    typ = v2
            pieces.append((piece, score, typ))
        elif fn == 2 and wt == 2:                      # TrainerSpec: keep every scalar field raw
            for f2, w2, v2 in _fields(v):
                if w2 == 2:
                    try:
                        val = v2.decode("utf-8")
                    except UnicodeDecodeError:
                        val = v2.hex()
                elif w2 == 5:
                    val = struct.unpack("<f", v2)[0]
                else:
                    val = v2
                trainer.setdefault(str(f2), []).append(val)
    return {"pieces": pieces, "trainer_spec_fields": trainer}


# ---------------- piece encoding ----------------
class Pieces:
    def __init__(self, sp, rows: list[str], threads: int, chunk: int = 50000):
        lens = np.empty(len(rows), dtype=np.int64)
        parts = []
        for k in range(0, len(rows), chunk):
            enc = sp.encode([r.strip() for r in rows[k:k + chunk]], out_type=int, num_threads=threads)
            lens[k:k + len(enc)] = [len(e) for e in enc]
            parts.append(np.fromiter(itertools.chain.from_iterable(enc), dtype=np.int32))
        self.n = len(rows)
        self.lens = lens
        self.ids = np.concatenate(parts) if parts else np.zeros(0, np.int32)

    def mask_of(self, idx) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return m

    def counts(self, idx=None, V: int = 32000) -> np.ndarray:
        if idx is None:
            return np.bincount(self.ids, minlength=V).astype(np.int64)
        return np.bincount(self.ids[np.repeat(self.mask_of(idx), self.lens)], minlength=V).astype(np.int64)

    def first_budget_counts(self, idx, budget: int, V: int = 32000):
        """Rows in a Random(0) shuffle of the ascending index list; counts over the first `budget` ids."""
        order = sorted(int(i) for i in idx)
        random.Random(0).shuffle(order)
        o = np.asarray(order, dtype=np.int64)
        ln = self.lens[o]
        if int(ln.sum()) <= budget:
            return None
        off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(self.lens, out=off[1:])
        got, out = 0, []
        for i, L in zip(o.tolist(), ln.tolist()):
            take = min(L, budget - got)
            out.append(self.ids[off[i]:off[i] + take])
            got += take
            if got >= budget:
                break
        return np.bincount(np.concatenate(out), minlength=V).astype(np.int64)


# ---------------- word tokens ----------------
class Words:
    def __init__(self, rows: list[str]):
        vocab: dict[str, int] = {}
        ids = array("q")
        lens = array("q")
        sd = vocab.setdefault
        for r in rows:
            t = [w.lower() for w in WORD.findall(r)]
            lens.append(len(t))
            ids.extend([sd(w, len(vocab)) for w in t])
        self.V = len(vocab)
        self.n = len(rows)
        self.ids = np.frombuffer(ids, dtype=np.int64).copy()
        self.lens = np.frombuffer(lens, dtype=np.int64).copy()
        self.off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(self.lens, out=self.off[1:])

    def sel(self, idx) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return self.ids[np.repeat(m, self.lens)]

    def types(self, idx=None) -> int:
        return int(np.unique(self.ids if idx is None else self.sel(idx)).size)

    def tokens(self, idx=None) -> int:
        return int(self.lens.sum() if idx is None else self.lens[np.asarray(idx, dtype=np.int64)].sum())

    def types_at(self, idx, budget: int):
        order = sorted(int(i) for i in idx)
        random.Random(0).shuffle(order)
        seen = np.zeros(self.V, dtype=bool)
        got = 0
        for i in order:
            a, b = int(self.off[i]), int(self.off[i + 1])
            take = min(b - a, budget - got)
            seen[self.ids[a:a + take]] = True
            got += take
            if got >= budget:
                return int(seen.sum())
        return None                                     # fewer than `budget` tokens


def rep_zero_and_values(W: Words) -> np.ndarray:
    """Per row: (# of the row's 8-grams whose count over the whole parent is > 1) / (# 8-grams);
    0 for rows under 8 tokens. Exact: 8-gram identity by composing token ids."""
    N = W.ids.size

    def combine(a: np.ndarray, shift: int) -> np.ndarray:
        b = np.full(N, -1, dtype=np.int64)
        b[:N - shift] = a[shift:]
        key = a * (int(a.max()) + 2) + (b + 1)
        _, inv = np.unique(key, return_inverse=True)
        return inv.astype(np.int64).ravel()

    g8 = combine(combine(combine(W.ids, 1), 2), 4)
    row_of = np.repeat(np.arange(W.n, dtype=np.int64), W.lens)
    pos = np.arange(N, dtype=np.int64)
    valid = pos + 8 <= W.off[row_of + 1]
    gv = g8[valid]
    cnt = np.bincount(gv)
    rep_hit = (cnt[gv] > 1).astype(np.int64)
    hits = np.bincount(row_of[valid], weights=rep_hit, minlength=W.n)
    ngr = np.maximum(W.lens - 7, 0)
    rep = np.zeros(W.n, dtype=np.float64)
    nz = ngr > 0
    rep[nz] = hits[nz] / ngr[nz]
    return rep


def match_random(hist: dict, lens: list[int], eligible: list[int], seed: int) -> list[int]:
    buckets: dict[int, list[int]] = {}
    for i in eligible:
        buckets.setdefault(lens[i], []).append(i)
    for L, n in hist.items():
        if len(buckets.get(L, [])) < n:
            raise SystemExit(f"short bucket at length {L}")
    rng = random.Random(seed)
    kept: list[int] = []
    for L in sorted(hist):
        kept += rng.sample(buckets[L], hist[L])
    return sorted(kept)


def ranks_first_occurrence(W: Words, rows) -> np.ndarray:
    """Frequency rank over `rows` (ascending), 0 = most frequent, ties to first occurrence."""
    flat = W.sel(rows)
    cnt = np.bincount(flat, minlength=W.V)
    first = np.full(W.V, np.iinfo(np.int64).max, dtype=np.int64)
    u, fi = np.unique(flat, return_index=True)
    first[u] = fi
    present = np.nonzero(cnt)[0]
    order = present[np.lexsort((first[present], -cnt[present]))]
    rank = np.full(W.V, -1, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def row_max_rank(W: Words, rows, rank: np.ndarray) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.int64)
    for j, i in enumerate(rows):
        a, b = int(W.off[i]), int(W.off[i + 1])
        out[j] = int(rank[W.ids[a:b]].max()) if b > a else -1
    return out


def piece_stats(c: np.ndarray, special: np.ndarray, countable: int, unk: list[int]) -> dict:
    real = c.copy()
    real[special] = 0
    d = {"piece_tokens": int(c.sum()), "unk_tokens": int(c[unk].sum())}
    for k in KS:
        d[f"types_k{k}"] = int((real >= k).sum())
    d["coverage_pct"] = 100.0 * d["types_k1"] / countable
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sums", type=Path, required=True, help="e06data_SHA256SUMS")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--skip-pretrain", action="store_true")
    a = ap.parse_args()
    D = a.data
    res: dict = {"interpreter": {"python": sys.version.split()[0], "unidata": unicodedata.unidata_version,
                                 "numpy": np.__version__, "sentencepiece": spm.__version__, "threads": a.threads}}

    # ---- inputs: sha256 against the list ----
    listed = {}
    for ln in a.sums.read_text().splitlines():
        h, f = ln.split(None, 1)
        listed[f.strip().removeprefix("e06data/")] = h
    used = [f"phase0_e05/{n}.{e}" for n in E05 for e in ("en", "fr")] + \
           [f"e05_src/{n}.{e}" for n in ("ft_topk", "ft_random") for e in ("en", "fr", "source")] + \
           ["vocab/R0.npy", "vocab/A_narrow_42_58.npy", "spm/spm_enfr_v1_fixed.model"]
    if not a.skip_pretrain:
        used += ["v1_fixed/train.clean.en", "v1_fixed/train.clean.fr"]
    res["inputs"] = {}
    for f in used:
        h = sha256(D / f)
        res["inputs"][f] = {"sha256": h, "matches_list": h == listed.get(f)}
        if h != listed.get(f):
            raise SystemExit(f"sha256 mismatch: {f}")
    log("sha256: all", len(used), "inputs match")

    # ---- the SentencePiece model ----
    sp = spm.SentencePieceProcessor()
    sp.Load(str(D / "spm/spm_enfr_v1_fixed.model"))
    V = int(sp.get_piece_size())
    pm = parse_model(D / "spm/spm_enfr_v1_fixed.model")
    types = [t for _, _, t in pm["pieces"]]
    lib = {"control": [i for i in range(V) if sp.is_control(i)], "unknown": [i for i in range(V) if sp.is_unknown(i)],
           "unused": [i for i in range(V) if sp.is_unused(i)], "byte": [i for i in range(V) if sp.is_byte(i)]}
    mine = {"control": [i for i, t in enumerate(types) if t == 3], "unknown": [i for i, t in enumerate(types) if t == 2],
            "unused": [i for i, t in enumerate(types) if t == 5], "byte": [i for i, t in enumerate(types) if t == 6]}
    scores = [s for _, s, _ in pm["pieces"]]
    normal_ids = [i for i, t in enumerate(types) if t == 1]
    special = np.zeros(V, dtype=bool)
    special[lib["control"] + lib["unknown"]] = True
    countable = int((~special).sum())
    unk = lib["unknown"]
    res["spm"] = {"piece_size": V, "parsed_pieces": len(pm["pieces"]),
                  "parsed_type_counts": {TYPE_NAMES.get(t, str(t)): c for t, c in collections.Counter(types).items()},
                  "parser_agrees_with_library": mine == lib and [sp.id_to_piece(i) for i in range(V)] == [p for p, _, _ in pm["pieces"]],
                  "special_ids": {i: sp.id_to_piece(i) for i in lib["control"] + lib["unknown"]},
                  "user_defined_ids": [i for i, t in enumerate(types) if t == 4],
                  "countable": countable,
                  "normal_scores_strictly_decreasing": all(scores[normal_ids[j]] > scores[normal_ids[j + 1]] for j in range(len(normal_ids) - 1)),
                  "normal_scores_first_last": [scores[normal_ids[0]], scores[normal_ids[1]], scores[normal_ids[-1]]],
                  "first_normal_id": normal_ids[0], "n_normal": len(normal_ids),
                  "trainer_spec_fields": {k: v if len(json.dumps(v)) < 400 else str(v)[:400] for k, v in pm["trainer_spec_fields"].items()}}
    log("spm:", res["spm"]["parsed_type_counts"], "specials", res["spm"]["special_ids"], "countable", countable)

    # ---- read every set ----
    rows: dict[str, tuple[list[str], list[str]]] = {}
    res["files"] = {}
    for n in E05:
        en, ie = read_lf(D / f"phase0_e05/{n}.en")
        fr, ifr = read_lf(D / f"phase0_e05/{n}.fr")
        assert len(en) == len(fr)
        rows[n] = (en, fr)
        res["files"][n] = {"en": ie, "fr": ifr}
    for n in ("ft_topk", "ft_random"):
        en, ie = read_lf(D / f"e05_src/{n}.en")
        fr, ifr = read_lf(D / f"e05_src/{n}.fr")
        lab, il = read_lf(D / f"e05_src/{n}.source")
        assert len(en) == len(fr) == len(lab)
        rows[n] = (en, fr)
        rows[n + "_lab"] = (lab, lab)
        res["files"][n] = {"en": ie, "fr": ifr, "source": {"lines": il["lines"]}}
    log("read all sets")

    # ---- word tokens of ft_random (en, fr) and of every E0.5 file (en; fr for D12) ----
    RW_e, RW_f = Words(rows["ft_random"][0]), Words(rows["ft_random"][1])
    TW_e, TW_f = Words(rows["ft_topk"][0]), Words(rows["ft_topk"][1])
    FW = {n: (Words(rows[n][0]), Words(rows[n][1])) for n in E05}
    log("word-tokenised")
    rlab = rows["ft_random_lab"][0]
    rlen = RW_e.lens.tolist()

    # ---- R0 / A_narrow: what the indices are ----
    R0 = np.load(D / "vocab/R0.npy", allow_pickle=False)
    AN = np.load(D / "vocab/A_narrow_42_58.npy", allow_pickle=False)
    chk: dict = {}
    for nm, ix in (("R0", R0), ("A_narrow", AN)):
        chk[nm] = {"dtype": str(ix.dtype), "shape": list(ix.shape), "min": int(ix.min()), "max": int(ix.max()),
                   "strictly_ascending": bool(np.all(np.diff(ix) > 0)), "all_un": all(rlab[i] == "un" for i in ix.tolist())}
    hist_div = collections.Counter(FW["ft_un_div"][0].lens.tolist())
    hist_rep = collections.Counter(FW["ft_un_rep"][0].lens.tolist())
    chk["ft_un_div_and_ft_un_rep_share_histogram"] = hist_div == hist_rep
    for nm, ix in (("R0", R0), ("A_narrow", AN)):
        il = ix.tolist()
        chk[nm]["hist_equals_ft_un_div"] = collections.Counter(rlen[i] for i in il) == hist_div
        chk[nm]["hist_equals_ft_un_div_if_read_one_line_later"] = collections.Counter(rlen[i + 1] for i in il if i + 1 < len(rlen)) == hist_div
        chk[nm]["hist_equals_ft_un_div_if_read_one_line_earlier"] = collections.Counter(rlen[i - 1] for i in il if i >= 1) == hist_div
    rep_r = rep_zero_and_values(RW_e)
    log("rep_rand computed")
    ru0 = [i for i in range(len(rlab)) if rlab[i] == "un" and rep_r[i] == 0.0]
    chk["RU0"] = len(ru0)
    ru0set = set(ru0)
    for nm, ix in (("R0", R0), ("A_narrow", AN)):
        chk[nm]["all_rep_rand_zero"] = bool(np.all(rep_r[ix] == 0.0))
        chk[nm]["subset_of_RU0"] = all(i in ru0set for i in ix.tolist())
    r0_rebuilt = match_random(hist_div, rlen, ru0, 42)
    chk["R0_rebuilt_seed42_equal"] = r0_rebuilt == R0.tolist()
    un_rows = [i for i in range(len(rlab)) if rlab[i] == "un"]
    chk["ft_random_un_rows"] = len(un_rows)
    ur_rebuilt = match_random(hist_div, rlen, un_rows, 42)
    en_r, fr_r = rows["ft_random"]
    chk["ft_un_random_rebuilt_line_for_line"] = ([en_r[i] for i in ur_rebuilt] == rows["ft_un_random"][0]
                                                  and [fr_r[i] for i in ur_rebuilt] == rows["ft_un_random"][1])
    hist_halves = collections.Counter(FW["ft_topk_div"][0].lens.tolist())
    chk["ft_topk_halves_share_histogram"] = hist_halves == collections.Counter(FW["ft_topk_rep"][0].lens.tolist())
    rs_rebuilt = match_random(hist_halves, rlen, list(range(len(rlen))), 42)
    chk["ft_random_small_rebuilt_line_for_line"] = ([en_r[i] for i in rs_rebuilt] == rows["ft_random_small"][0]
                                                     and [fr_r[i] for i in rs_rebuilt] == rows["ft_random_small"][1])
    # A_narrow's capped pool (ranks over RU0, ties to first occurrence; caps 42,000 / 58,000)
    rk_e, rk_f = ranks_first_occurrence(RW_e, ru0), ranks_first_occurrence(RW_f, ru0)
    me, mf = row_max_rank(RW_e, ru0, rk_e), row_max_rank(RW_f, ru0, rk_f)
    capped = [i for i, x, y in zip(ru0, me.tolist(), mf.tolist()) if x < 42000 and y < 58000]
    cset = set(capped)
    chk["RU0_source_types"] = int((rk_e >= 0).sum())
    chk["capped_pool_42000_58000"] = len(capped)
    chk["A_narrow_inside_capped_pool"] = all(i in cset for i in AN.tolist())
    chk["A_narrow_shared_with_R0"] = len(set(AN.tolist()) & set(R0.tolist()))
    res["row_checks"] = chk
    log("row checks:", json.dumps(chk))

    # ---- set definitions (index into a parent's encodings, or a whole file) ----
    idx_sets = {"R0": R0.tolist(), "A_narrow": AN.tolist()}

    # ---- word types (D11, D12, source at 5.6M) ----
    wt = {}
    for n in E05:
        we, wf = FW[n]
        allr = list(range(we.n))
        wt[n] = {"rows": we.n, "source_words": we.tokens(), "source_word_types_D11": we.types(),
                 "french_words": wf.tokens(), "french_types_at_6500000_D12": wf.types_at(allr, 6_500_000),
                 "source_types_at_5600000": we.types_at(allr, 5_600_000)}
    for n, ix in idx_sets.items():
        wt[n] = {"rows": len(ix), "source_words": RW_e.tokens(ix), "source_word_types_D11": RW_e.types(ix),
                 "french_words": RW_f.tokens(ix), "french_types_at_6500000_D12": RW_f.types_at(ix, 6_500_000),
                 "source_types_at_5600000": RW_e.types_at(ix, 5_600_000)}
    for n, we, wf in (("ft_topk", TW_e, TW_f), ("ft_random", RW_e, RW_f)):
        allr = list(range(we.n))
        wt[n] = {"rows": we.n, "source_words": we.tokens(), "source_word_types_D11": we.types(),
                 "french_words": wf.tokens(), "french_types_at_6500000_D12": wf.types_at(allr, 6_500_000),
                 "source_types_at_5600000": we.types_at(allr, 5_600_000)}
    res["word_types"] = wt
    log("word types done")
    del TW_e, TW_f

    # ---- piece encodings ----
    P = {}
    for n in E05 + ("ft_topk", "ft_random"):
        P[n] = (Pieces(sp, rows[n][0], a.threads), Pieces(sp, rows[n][1], a.threads))
    log("piece-encoded all sets")
    # does strip() change any encoding?  encode the raw rows wherever strip changes the text
    strip_diff = {}
    for n in E05 + ("ft_topk", "ft_random"):
        dd = 0
        for side in (0, 1):
            ch = [r for r in rows[n][side] if r != r.strip()]
            if ch:
                a1 = sp.encode(ch, out_type=int)
                a2 = sp.encode([r.strip() for r in ch], out_type=int)
                dd += sum(1 for x, y in zip(a1, a2) if x != y)
        strip_diff[n] = dd
    res["rows_whose_ids_change_with_strip"] = strip_diff

    def counts_of(n):
        if n in idx_sets:
            E, F = P["ft_random"]
            return E.counts(idx_sets[n], V), F.counts(idx_sets[n], V)
        E, F = P[n]
        return E.counts(None, V), F.counts(None, V)

    def kept_rows(n):
        if n in idx_sets:
            E, F = P["ft_random"]
            ix = np.asarray(idx_sets[n], dtype=np.int64)
            ne = np.array([bool(rows["ft_random"][0][i].strip()) and bool(rows["ft_random"][1][i].strip()) for i in ix.tolist()])
            k = (E.lens[ix] + 2 <= MAXTOK) & (F.lens[ix] + 2 <= MAXTOK) & ne
            return int(len(ix)), int(k.sum())
        E, F = P[n]
        ne = np.array([bool(x.strip()) and bool(y.strip()) for x, y in zip(*rows[n])])
        k = (E.lens + 2 <= MAXTOK) & (F.lens + 2 <= MAXTOK) & ne
        return int(E.n), int(k.sum())

    C = {}
    cov = {}
    for n in ORDER:
        ce, cf = counts_of(n)
        C[n] = (ce, cf)
        r, k = kept_rows(n)
        cov[n] = {"rows": r, "rows_kept_by_trainer": k,
                  "source": piece_stats(ce, special, countable, unk), "target": piece_stats(cf, special, countable, unk)}
    res["coverage"] = cov
    log("coverage:", {n: (round(cov[n]["source"]["coverage_pct"], 2), round(cov[n]["target"]["coverage_pct"], 2)) for n in ORDER})

    # ---- matched piece budget (Random(0) shuffle, first B pieces) ----
    small_e = min(cov[n]["source"]["piece_tokens"] for n in ORDER)
    small_f = min(cov[n]["target"]["piece_tokens"] for n in ORDER)
    Be, Bf = (small_e // 100000) * 100000, (small_f // 100000) * 100000
    if Be == small_e:
        Be -= 100000
    if Bf == small_f:
        Bf -= 100000
    mb = {"budget_source": Be, "budget_target": Bf, "smallest_source_pieces": small_e, "smallest_target_pieces": small_f}
    for n in ORDER:
        if n in idx_sets:
            E, F = P["ft_random"]
            ix = idx_sets[n]
        else:
            E, F = P[n]
            ix = list(range(E.n))
        ce, cf = E.first_budget_counts(ix, Be, V), F.first_budget_counts(ix, Bf, V)
        mb[n] = {"source_types": None if ce is None else piece_stats(ce, special, countable, unk)["types_k1"],
                 "target_types": None if cf is None else piece_stats(cf, special, countable, unk)["types_k1"]}
        for s in ("source", "target"):
            mb[n][s + "_coverage_pct"] = None if mb[n][s + "_types"] is None else 100.0 * mb[n][s + "_types"] / countable
    res["coverage_at_matched_pieces"] = mb

    # ---- redraws: R0's rule and ft_un_random's rule, seeds 1-5 ----
    E, F = P["ft_random"]
    rd = {}
    for nm, elig in (("R0_rule", ru0), ("ft_un_random_rule", un_rows)):
        vals = []
        for seed in (1, 2, 3, 4, 5):
            ix = match_random(hist_div, rlen, elig, seed)
            vals.append((piece_stats(E.counts(ix, V), special, countable, unk)["types_k1"],
                         piece_stats(F.counts(ix, V), special, countable, unk)["types_k1"]))
        se, sf = [v[0] for v in vals], [v[1] for v in vals]
        rd[nm] = {"source_types": se, "target_types": sf,
                  "sd_sample_types": [statistics.stdev(se), statistics.stdev(sf)],
                  "sd_population_types": [statistics.pstdev(se), statistics.pstdev(sf)],
                  "sd_sample_pp": [100 * statistics.stdev(se) / countable, 100 * statistics.stdev(sf) / countable],
                  "sd_population_pp": [100 * statistics.pstdev(se) / countable, 100 * statistics.pstdev(sf) / countable]}
    res["redraws"] = rd
    log("redraws:", json.dumps(rd))

    # ---- pretraining corpus: piece token counts per side (all rows; trainer-kept rows) ----
    if not a.skip_pretrain:
        pre = {}
        acc = {k: np.zeros(V, dtype=np.int64) for k in ("en_all", "fr_all", "en_kept", "fr_kept")}
        nlines = ncr = nempty = nover = 0
        with open(D / "v1_fixed/train.clean.en", "rb") as fe, open(D / "v1_fixed/train.clean.fr", "rb") as ff:
            while True:
                be, bf = list(itertools.islice(fe, 200000)), list(itertools.islice(ff, 200000))
                if not be and not bf:
                    break
                if len(be) != len(bf):
                    raise SystemExit("pretraining sides differ in line count")
                se = [x.decode("utf-8").removesuffix("\n") for x in be]
                sf = [x.decode("utf-8").removesuffix("\n") for x in bf]
                nlines += len(se)
                ncr += sum(1 for x, y in zip(se, sf) if "\r" in x or "\r" in y)
                se, sf = [x.strip() for x in se], [x.strip() for x in sf]
                ee = sp.encode(se, out_type=int, num_threads=a.threads)
                ef = sp.encode(sf, out_type=int, num_threads=a.threads)
                le = np.fromiter((len(x) for x in ee), dtype=np.int64, count=len(ee))
                lf = np.fromiter((len(x) for x in ef), dtype=np.int64, count=len(ef))
                fe_ = np.fromiter(itertools.chain.from_iterable(ee), dtype=np.int32)
                ff_ = np.fromiter(itertools.chain.from_iterable(ef), dtype=np.int32)
                acc["en_all"] += np.bincount(fe_, minlength=V)
                acc["fr_all"] += np.bincount(ff_, minlength=V)
                ne = np.fromiter((bool(x) and bool(y) for x, y in zip(se, sf)), dtype=bool, count=len(se))
                nempty += int((~ne).sum())
                keep = ne & (le + 2 <= MAXTOK) & (lf + 2 <= MAXTOK)
                nover += int((ne & ~keep).sum())
                acc["en_kept"] += np.bincount(fe_[np.repeat(keep, le)], minlength=V)
                acc["fr_kept"] += np.bincount(ff_[np.repeat(keep, lf)], minlength=V)
        pre["lines"], pre["lines_with_cr"], pre["pairs_with_empty_side"], pre["pairs_over_max"] = nlines, ncr, nempty, nover
        for k, c in acc.items():
            pre[k] = {"piece_tokens": int(c.sum()), "unk_tokens": int(c[unk].sum()),
                      "types_seen_countable": int(((c > 0) & ~special).sum())}
        log("pretraining:", json.dumps(pre))
        unc = {}
        for n in ORDER:
            ce, cf = C[n]
            d = {}
            for side, cs, key in (("source", ce, "en"), ("target", cf, "fr")):
                for variant in ("all", "kept"):
                    pc = acc[f"{key}_{variant}"].copy()
                    tot_with_special = int(pc.sum())
                    pc_ns = pc.copy()
                    pc_ns[special] = 0
                    absent = (cs == 0) & ~special
                    side_vocab = (pc > 0) & ~special
                    d[f"{side}_{variant}"] = {
                        "uncovered_share_pct_excl_special": 100.0 * int(pc_ns[absent].sum()) / int(pc_ns.sum()),
                        "uncovered_share_pct_incl_unk_in_denominator": 100.0 * int(pc_ns[absent].sum()) / tot_with_special,
                        "coverage_of_side_vocab_pct": 100.0 * int(((cs > 0) & side_vocab).sum()) / int(side_vocab.sum()),
                        "set_types_outside_side_vocab": int(((cs > 0) & ~special & ~side_vocab).sum())}
            unc[n] = d
        res["pretraining"] = pre
        res["token_weighted_and_side_vocab"] = unc

    # ---- gaps ----
    def g(a_, b_, f):
        x, y = f(a_), f(b_)
        return {"a": x, "b": y, "diff": x - y, "rel_pct": 100.0 * (x - y) / y}

    contrasts = (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("A_narrow", "R0"),
                 ("A_narrow", "ft_un_div"), ("ft_un_div", "R0"))
    gaps = {}
    for a_, b_ in contrasts:
        gaps[f"{a_} - {b_}"] = {
            "src_cov_pp": g(a_, b_, lambda n: cov[n]["source"]["coverage_pct"]),
            "tgt_cov_pp": g(a_, b_, lambda n: cov[n]["target"]["coverage_pct"]),
            "src_types_k20": g(a_, b_, lambda n: cov[n]["source"]["types_k20"]),
            "tgt_types_k20": g(a_, b_, lambda n: cov[n]["target"]["types_k20"]),
            "src_word_types_D11": g(a_, b_, lambda n: wt[n]["source_word_types_D11"]),
            "fr_types_D12": g(a_, b_, lambda n: wt[n]["french_types_at_6500000_D12"])}
    res["gaps"] = gaps
    # share of the P gap (R0 vs ft_un_div) that A_narrow covers
    sh = {}
    for lab_, f in (("src_pieces", lambda n: cov[n]["source"]["types_k1"]), ("tgt_pieces", lambda n: cov[n]["target"]["types_k1"]),
                    ("src_words", lambda n: wt[n]["source_word_types_D11"]), ("fr_words", lambda n: wt[n]["french_types_at_6500000_D12"])):
        sh[lab_] = (f("R0") - f("A_narrow")) / (f("R0") - f("ft_un_div"))
    res["A_narrow_share_of_P_gap_closed"] = sh
    res["od1c_window_2pct"] = {s: [cov["ft_un_rep"][s]["types_k1"] * 0.98, cov["ft_un_rep"][s]["types_k1"] * 1.02] for s in ("source", "target")}
    res["seconds"] = time.time() - T0
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check_results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    log("written", a.out / "check_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
