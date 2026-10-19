#!/usr/bin/env python3
"""E0.6: the new UN sets of phase0/prereg_drafts.md "E0.6", built exactly, or nothing at all.

The rules are the draft's ("Definitions shared by every build", "How each new set is built",
"Inputs are pinned") as made exact by phase0/next/e06_build_spec.md; every PROPOSAL there is
implemented as written and is a parameter wherever the draft leaves a choice open. This
script only executes them.

  dry run      e06_build.py --dry-run --dir DR --src-dir S --e05-dir E --pretrain-en P --decisions J
                            --pretrain-fr PF --qe-tsv Q --news-en N --ref-lex L
  production   e06_build.py --dir D --src-dir S --e05-dir E --pretrain-en P --decisions J
                            (--expect-sha256-from DR/dry_run_manifest.json | --skip-expected-sha256)
                            [--pretrain-fr PF] [--qe-tsv Q] [--news-en N] [--ref-lex L]
  feasibility  e06_build.py --feasibility --out F --src-dir S --e05-dir E --pretrain-en P
                            [--decisions J] [--pretrain-fr PF] [--qe-tsv Q] [--news-en N] [--ref-lex L]
  measure      e06_build.py --measure --from B --out M --src-dir S --spm-model SPM
                            [--newstest-en T] [--mt-repo MT] [--max-seq-len 256]
  reproduce    e06_build.py --reproduce-check --ref-r0 R0.npy --ref-a-narrow A_narrow_42_58.npy
                            --src-dir S --e05-dir E --pretrain-en P
  verify       e06_build.py --verify --dir D

  S  ft_topk.{en,fr,source}, ft_random.{en,fr,source}, heldout_un.{en,fr}  (~/mt_local/phase0_final/e05_src)
  E  E0.5's finished data directory, read only (its manifest and the three anchor sets)
  P  the v1.1 pretraining corpus, English side (train.clean.en); PF its French side
  J  the decisions file: one key per OPEN DECISION (spec section 7.1); absent or "OPEN" is unset
  L  the vocab memo's lex.pkl, to compare the rebuilt Dice lexicon with entry by entry

DRY RUN is the production build with the registered decisions, on the Mac (spec A8): it
needs every descriptive input, so the registered table has no "n.m." cell, and it writes
DR/<set>.en/.fr, DR/index/<set>.npy, DR/heldout_un.*, NOT_FOR_TRAINING and, LAST,
DR/dry_run_manifest.json ("for_training": false). It never writes a manifest.json, so
nothing trains on it and --verify refuses it. It records the interpreter's Unicode tables
beside od10_unidata instead of requiring them, so the dry run and the box read ONE
decisions file; the box build is gated by reproducing the dry run's sha256 instead.

PRODUCTION (the box) builds the sets the decisions call for and writes D/<set>.en,
D/<set>.fr, D/index/<set>.npy, D/heldout_un.* and, LAST, D/manifest.json with
"e06_complete": true. It STOPs unless every file reproduces the registered dry run's
sha256 (--expect-sha256-from); --skip-expected-sha256 is for the tests, and --verify
refuses a build made with it. A set-defining decision that is unset (or set to an option no
rule exists for) is REFUSED: nothing is written, exit 2, every blocked set named. Any failed
check is a STOP: nothing is written, exit 1, and D keeps its previous contents.

FEASIBILITY builds every option of every set-defining decision, for measurement only. It
writes FEASIBILITY_*.npy index files, a NOT_FOR_TRAINING file and feasibility_report.json
("for_training": false); never a .en/.fr file and never a manifest.json, so nothing can
train on it, and --verify refuses it.

MEASURE (OD 16, spec section 9) reads a finished build, dry run or feasibility directory B
(every file checked against its recorded sha256) and writes only M/measure_report.json:
SentencePiece piece counts of every set the way the trainer tokenises (spec A28), and,
with --newstest-en, newstest2014 coverage. It never writes into B and nothing it measures
feeds back into a set.

REPRODUCE-CHECK builds R0 and A_narrow (caps 42,000 / 58,000, seed 42, top-band rows kept,
one draw) from the pinned inputs and compares them row for row with the scratch index files
of the vocab memo. It writes nothing; exit 1 on any difference.

FAIL-CLOSED. Inputs must have their pinned sha256 (or --allow-unpinned-src, for the test
fixtures only; any real use is a protocol deviation and is recorded). An output directory
that is an input directory, or that holds another experiment's manifest, is a STOP before
anything is read. Before any new set is built, E0.5's UN split is recomputed from the pinned
inputs and must reproduce E0.5's three anchor sets byte for byte. Every file is written under
a temporary name and renamed only after every set is built and every check has passed; files
of the previous build that this one no longer writes, the old manifest and every trainer
cache (*.cached_*.npz) are removed before the first rename, and the new manifest is written
last.

Exit codes: 0 built / verified / reproduced / measured; 1 STOP (a check failed or an input
is missing); 2 REFUSED (a decision is unset or names an option with no defined rule);
64 usage error (the command line itself is wrong).
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import io
import json
import os
import pickle
import platform
import random
import re
import sys
import unicodedata
from array import array
from fractions import Fraction
from pathlib import Path

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)      # must stay identical to phase0/e04_split.py and e05_build.py
WS = re.compile(r"\s+")
DIGIT = re.compile(r"\d")
SEED = 42
SOURCE = "un"
UNSEEN = 10 ** 7                           # the rank of a word never seen in pretraining (vocab/a5.py)
NO_CAP = None
EX_STOP, EX_REFUSED, EX_USAGE = 1, 2, 64   # 64 = EX_USAGE, so a wrong command line is never read as "REFUSED"

# ---- file names: what marks a directory as which kind of output ----------------------
MANIFEST = "manifest.json"                 # a production build: the only kind anything trains on
DRY_MANIFEST = "dry_run_manifest.json"     # a dry run: never named manifest.json, so nothing trains on it
FEAS_REPORT = "feasibility_report.json"
MEASURE_REPORT = "measure_report.json"
NOT_FOR_TRAINING = "NOT_FOR_TRAINING"
DATA_FILE = re.compile(r"(?:index/)?ft_[a-z0-9_]+\.(?:en|fr|npy)|heldout_un\.(?:en|fr)")   # what a build writes
OTHER_DONE = re.compile(r"e\d+_complete")  # e05_complete, ...: another experiment's completion flag

# ---- pinned inputs (spec section 3.1) ------------------------------------------------
PINNED_SRC = {                             # = phase0/e05_build.py PINNED_SRC
    "ft_topk.en": "787339d08e0c47d9960cb263757edafd844ec80a45875ac083845a02a20ff2b2",
    "ft_topk.fr": "22263da18eaf62cb7deea7438068ce0cb00ce3849c1a83f7e0c55008c3d78797",
    "ft_topk.source": "a4ad45860c92eae6b512c4a7178fe78fc92f00d4b9551d00a6637c80079244ca",
    "ft_random.en": "937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0",
    "ft_random.fr": "382e0de574feaaa323939d73a24736b762c76a8b55a0ffd258d64b15e5303e29",
    "ft_random.source": "fd675a1948ab4e30dce05aae4c85d2432aec99549c05fb794903a49e9b0b6923",
}
PINNED_HELDOUT = {                         # = phase0/e05_build.py PINNED_HELDOUT
    "heldout_un.en": "f70e725eae9972899b6129f6acf123ed5421021b2f5ba81bb165a6928e858bbc",
    "heldout_un.fr": "7daf654fa37cca653cbc9724c77ee721384a406e3ca3e6469a2a847b66b729ab",
}
PINNED_E05_MANIFEST = "979dafd76344444c56cad9afc52e0ec7ba7ebb74c73115cbeb237583419b8eb2"
PINNED_ANCHORS = {                         # E0.5's manifest, the run-of-record files
    "ft_un_div.en": "51d9ec1157c58591a7b25b2a1bde52e9dc93342a34c55e55547f0e3857ba94da",
    "ft_un_div.fr": "1957872adf3f96b8e7fe551eeeae42bbd38f03811511e013f9017b65b954a7e1",
    "ft_un_rep.en": "940cde0f051d6b473c55a6771af64900a20c9c52d66e1a4772e18a5bf9cafe68",
    "ft_un_rep.fr": "9373be1fc7afb86a1b5ebf76d7cb567bde0ef349443ca9c3c6a87c175185edbd",
    "ft_un_random.en": "80fa6aea32f342cc43adec8e66063f8f08c20b6e3b66a60b0bfd7baf3e35c59c",
    "ft_un_random.fr": "1b54a882d989153697417b1f4b2300b81500acac258e29c542776d39a32094c0",
}
PINNED_PRETRAIN_EN = "7230adc6c823d4df51407e445b1181ada450363dfe43507885c9f529d49b6b6e"
PINNED_PRETRAIN_FR = "b0a6ba91eeba2b4bdf28e8a73816a2bdd7ccbddd5f889fcc7d3f3f0a52dad7e2"
PINNED_REFERENCE = {                       # phase0/next/SHA256SUMS, vocab/
    "R0": "22eeb53dd3cb621831939cb41a5b9280045dbc5f6b1ce48686234479ac354dc9",
    "A_narrow": "e9fac76975bb83e2a4b12ed913c333212cc4a70c440346278c12ad7c0bd7cfa8",
}
PINNED_LEX = "7570fcb81cfc1870be21a5cd6d04c088847ef9887e54604a8c938c248207e3ac"   # SHA256SUMS, vocab/lex.pkl
PINNED_SPM = "d1c10ea80fb95984f3f240b3579c87a16313b77aa9bc103ca0d6a0e06672a561"   # spm_enfr_v1_fixed.model (spec 3.1)
REFERENCE_CAPS = (42000, 58000)
# wrote the scratch files: /usr/local/bin/python3, an x86_64 build that runs under Rosetta on
# the M3 Max (platform.machine() == "x86_64", not arm64), numpy 2.4.5
REFERENCE_INTERPRETER = {"python": "3.14.3", "unidata_version": "16.0.0", "machine": "x86_64"}

# Counts the pinned inputs must give (spec sections 4-6). Checked only when every input is
# the pinned one: a fixture or an unpinned input gives other counts by design.
EXPECTED_ON_PINNED = {
    "diverse UN candidates": 486_475, "repetitive UN candidates": 227_406,
    "RU0 (UN rows of ft_random with rep_rand = 0)": 450_709, "rows of each anchor": 214_489,
    "A_narrow capped pool at 42,000 / 58,000 (top-band rows kept)": 402_010,
    "repetitive candidates after OD 4(a) dedup": 207_561, "diverse candidates after OD 4(a) dedup": 486_223,
    "pretraining lines": 9_312_233,
}

# ---- set names on disk (spec 2.4): e03_collect.py accepts only ft_[a-z0-9_]+ ----------
SET_FILE = {"ft_un_div": "ft_un_div", "ft_un_rep": "ft_un_rep", "ft_un_random": "ft_un_random",
            "R0": "ft_un_r0", "A_narrow": "ft_un_a_narrow", "rep_dedup": "ft_un_rep_dedup",
            "div_dd": "ft_un_div_dd", "div_narrow": "ft_un_div_narrow", "div_narrow_dd": "ft_un_div_narrow_dd"}
ANCHORS = ("ft_un_div", "ft_un_rep", "ft_un_random")
ANCHOR_FILES = tuple(f"{n}.{e}" for n in ANCHORS for e in ("en", "fr"))
ALL_SETS = tuple(SET_FILE)

# ---- decisions (spec 7.1/7.2) ---------------------------------------------------------
OD2_TIE = {"a": "larger_Ke_then_larger_Kf", "b": "larger_Ke_then_larger_Kf",
           "c": "larger_k", "c2": "larger_k_then_smaller_lambda"}
A17_RULE = "lowercase_w+_tokens_unicode_digits_to_0"
# Rule (c) is underspecified in the draft (spec A15); choosing it registers these exact
# definitions, which q_orders / q_point / tie_u implement. The decisions file must repeat
# the string, so its sha256 names them.
OD2C_DEFINITION = {
    "c": "spec 6.5(c) / A15: per length bucket, rows ordered by (pretraining maxrank, u, row); "
         "m_L = h_L + (k*(n_L-h_L))//K of them kept; u drawn by G8a, the h_L rows drawn by G8b; seed 42",
    "c2": "spec 6.5 c2 / A15: per length bucket, rows ordered by ((1-lambda)*log1p(maxrank_en) + "
          "lambda*log1p(maxrank_fr), u, row); m_L = h_L + (k*(n_L-h_L))//K of them kept; u drawn by G8a, "
          "the h_L rows drawn by G8b; seed 42",
}
NARROW = ("div_narrow", "div_narrow_dd")
DEDUP = ("rep_dedup", "div_dd", "div_narrow_dd")
DECISIONS = {      # key: (default, the sets it defines)
    "od1_coverage": (None, NARROW), "od2_rule": (None, NARROW), "od2_grid": (None, NARROW),
    "od2_tie": (None, NARROW), "tolerance": (None, NARROW), "od2c_definition": (None, NARROW),
    "od3_caps": (None, ("A_narrow",)), "od3_caps_for_reduced_pool": (None, ("A_narrow",)),
    "od4_duplicate": (None, DEDUP), "od4_norm_rule": (None, DEDUP),
    "od4_recompute_rep": (None, DEDUP), "od5": (None, NARROW), "od6_topband": (None, ("R0", "A_narrow")),
    "od7_exposure": (None, ("A_narrow",)), "od10_unidata": ("15.0.0", ALL_SETS),
    "od12_draws": (None, ("R0", "A_narrow")), "od15_narrowness": (None, ALL_SETS),
    "od18_register_v": (None, NARROW), "fr_budget_214489": (6_500_000, ALL_SETS),
    "fr_budget_198333": (None, DEDUP), "seeds": ([42, 1, 2, 3, 4], ("R0", "A_narrow")),
}
# needed only under some options: od4_norm_rule under od4_duplicate = "b", od2c_definition
# under od2_rule = "c"/"c2", od3_caps_for_reduced_pool under od6_topband = "remove"
OPTIONAL_KEYS = {"od4_norm_rule", "od2c_definition", "od3_caps_for_reduced_pool"}
RECORDED_ONLY = {"od8_steps_198333", "od9_multiplicity", "od11_secondary", "od13_nrc",
                 "od14_q2_reference", "od16_checks", "od17_lr"}
FEASIBILITY_ONLY = {"od3_report_grid"}
FEAS_FR_BUDGET_198333 = 6_000_000          # the draft's proposal (L199-200), feasibility only
LEX_SAMPLE, LEX_SEED, LEX_RANKS, LEX_DICE = 1_500_000, 7, (100, 20000), 0.15   # vocab/dice.py

RNG_REGISTER = [
    {"id": "G1", "step": "UN split (anchor check)", "seed": 42,
     "draws": "strat_match(div_cand, rep_cand): per length ascending, diverse sample then repetitive sample"},
    {"id": "G2", "step": "ft_un_random (anchor check)", "seed": 42,
     "draws": "match_random over the ascending UN rows of ft_random at ft_un_div's histogram"},
    {"id": "G3", "step": "R0", "seed": "42 (OD 12 per_seed: each seed)", "draws": "match_random over RU0 at ft_un_div's histogram"},
    {"id": "G4", "step": "A_narrow tie keys", "seed": "42 (OD 12 per_seed: each seed)",
     "draws": "one .random() per capped-pool row, ascending, before any matching"},
    {"id": "G5", "step": "rep_dedup / div_dd", "seed": 42, "draws": "strat_match over the deduplicated lists, diverse first"},
    {"id": "G6", "step": "div_narrow OD 2(a), each grid point", "seed": "42, fresh at each point",
     "draws": "nn_match tie keys over that point's capped candidates"},
    {"id": "G7", "step": "div_narrow OD 2(b), each grid point", "seed": "42, fresh at each point",
     "draws": "match_random over that point's capped candidates at the target histogram"},
    {"id": "G8a", "step": "div_narrow OD 2(c)/(c2) tie keys", "seed": "42, once",
     "draws": "one .random() per diverse candidate, ascending"},
    {"id": "G8b", "step": "div_narrow OD 2(c)/(c2), each grid point", "seed": "42, fresh at each point",
     "draws": "per length ascending, sample(E_L ascending, h_L)"},
    {"id": "G9", "step": "French types at budget B", "seed": "0, fresh for each measurement",
     "draws": "one shuffle of the set's ascending index list"},
    {"id": "G10", "step": "div_narrow_dd", "seed": "as G6-G8", "draws": "as G6-G8 at rep_dedup's histogram"},
]


class Stop(Exception):
    """A check failed: nothing is written (exit 1)."""


# ======================================================================================
# shared definitions (spec section 4)
# ======================================================================================
def read_lines(p: Path) -> list[str]:
    # LF is the only line break; never splitlines(), which also splits on U+2028/NEL
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def toks(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


def tokenize(lines: list[str]) -> list[list[str]]:
    # interned: the same tokens as toks(), stored once each
    intern = sys.intern
    return [[intern(w.lower()) for w in WORD.findall(s)] for s in lines]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gram_key(gram: str) -> bytes:
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


def rep_within(parent_toks: list[list[str]]) -> list[float]:
    """E0.4's statistic as phase0/e04_split.py and e05_build.py compute it: per row, the
    fraction of its 8-grams whose count over the whole list is above one; rows under 8
    tokens -> 0."""
    grams = [[gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in parent_toks]
    counts = collections.Counter()
    for g in grams:
        counts.update(g)
    return [sum(1 for k in g if counts[k] > 1) / len(g) if g else 0.0 for g in grams]


def norm_line(s: str) -> str:
    """vocab/freq.py's normalisation: NFKC, casefold, whitespace runs to one space, stripped."""
    return WS.sub(" ", unicodedata.normalize("NFKC", s).casefold()).strip()


def nkey(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(norm_line(s).encode(), digest_size=8).digest(), "little")


def h64(src: str, tgt: str) -> int:
    """phase0/e03_build_controls.py's exact-pair key."""
    return int.from_bytes(hashlib.blake2b(f"{src}\x00{tgt}".encode("utf-8"), digest_size=8).digest(), "little")


def a17_key(src: str) -> str:
    """OD 4(b) under the spec's PROPOSAL A17: the lowercased \\w+ tokens, every Unicode digit -> 0."""
    return " ".join(DIGIT.sub("0", w) for w in toks(src))


def median_check(rep: list[float]) -> float:
    n = len(rep)
    med = rep[sorted(range(n), key=rep.__getitem__)[n // 2]]      # e05_build.py line 240
    if med != 0.0:
        raise Stop(f"the median of rep over ft_topk is {med!r}, not 0.0: the draft's 'rep = 0' "
                   f"candidates are E0.5's 'rep <= median' only when the median is 0.0")
    return med


def match_random(hist, pool_len, eligible, what: str, seed: int = SEED) -> tuple[list[int], float]:
    """e05_build.py match_random: exactly hist[L] rows of every length L, at random within the
    ascending bucket, lengths ascending, one Random(seed). Any short bucket is a STOP."""
    buckets: dict[int, list[int]] = {}
    for i in eligible:
        buckets.setdefault(pool_len[i], []).append(i)
    short = {L: (n, len(buckets.get(L, []))) for L, n in hist.items() if len(buckets.get(L, [])) < n}
    if short:
        lines = "".join(f"\n  length {L}: need {short[L][0]:,}, have {short[L][1]:,}" for L in sorted(short))
        raise Stop(f"{what} is short in {len(short)} length bucket(s) (total shortfall "
                   f"{sum(n - h for n, h in short.values()):,} rows):{lines}")
    rng = random.Random(seed)
    kept: list[int] = []
    for L in sorted(hist):
        kept += rng.sample(buckets[L], hist[L])
    return sorted(kept), min(len(buckets[L]) / hist[L] for L in hist)


def strat_match(a_rows, b_rows, lens, seed: int = SEED) -> tuple[list[int], list[int]]:
    """e05_build.py lines 255-268: per length ascending, k = min(|A_L|, |B_L|) rows of each,
    A (the diverse side) drawn first; an empty bucket draws nothing."""
    by: dict[str, dict[int, list[int]]] = {"a": {}, "b": {}}
    for nm, idx in (("a", a_rows), ("b", b_rows)):
        for i in idx:
            by[nm].setdefault(lens[i], []).append(i)
    rng = random.Random(seed)
    ka: list[int] = []
    kb: list[int] = []
    for L in sorted(set(by["a"]) | set(by["b"])):
        x, y = by["a"].get(L, []), by["b"].get(L, [])
        k = min(len(x), len(y))
        if k:
            ka += rng.sample(x, k)
            kb += rng.sample(y, k)
    return sorted(ka), sorted(kb)


def ranks_of(counter: collections.Counter, what: str) -> dict[str, int]:
    """Rank 0 = most frequent; Counter.most_common's stable sort keeps first occurrence
    among equal counts (the draft's tie rule)."""
    order = [w for w, _ in counter.most_common()]
    if len(order) >= UNSEEN:
        raise Stop(f"{what} has {len(order):,} word types: unseen words (rank {UNSEEN:,}) would collide with real ranks")
    return {w: r for r, w in enumerate(order)}


def maxrank_of(tok_lists, rank: dict[str, int]) -> list[int]:
    """vocab/a5.py: the rank of the row's rarest word; unseen -> 10**7; empty row -> 0."""
    return [max((rank.get(w, UNSEEN) for w in t), default=0) for t in tok_lists]


class Log1p:
    """float(np.log1p(np.int64(v))), as vocab/a8_lib.py evaluates it (a numpy scalar call)."""

    def __init__(self):
        self.cache: dict[int, float] = {}

    def __call__(self, v) -> float:
        v = int(v)
        r = self.cache.get(v)
        if r is None:
            r = self.cache[v] = float(np.log1p(np.int64(v)))
        return r


def nn_match(t_idx, t_len, t_key, p_idx, p_len, p_key, seed: int, what: str, lg: Log1p,
             stats: dict | None = None) -> list[int]:
    """vocab/a8_lib.py nn_match, unchanged (spec D10). Within each target length every target
    row takes the unused pool row with the nearest log1p(maxrank):
      - one Random(seed).random() per pool row, ascending, all before any matching (also for
        pool rows at lengths with no targets);
      - pool rows sorted by (key, tie key, index);
      - targets processed in decreasing |v - median pool key| (stable: ascending target order);
      - bisect_left, nearest unused neighbour, the lower one on equal distance."""
    rng = random.Random(seed)
    tb: dict[int, list[float]] = {}
    for i in t_idx:
        tb.setdefault(t_len[i], []).append(lg(t_key[i]))
    pb: dict[int, list[tuple]] = {}
    for i in p_idx:
        pb.setdefault(p_len[i], []).append((lg(p_key[i]), rng.random(), i))
    short = {L: (len(v), len(pb.get(L, []))) for L, v in tb.items() if len(pb.get(L, [])) < len(v)}
    if short:
        lines = "".join(f"\n  length {L}: need {short[L][0]:,}, have {short[L][1]:,}" for L in sorted(short))
        raise Stop(f"{what}: the pool is smaller than its targets in {len(short)} length bucket(s):{lines}")
    out: list[int] = []
    near_dist = near_order = 0
    for L, vals in tb.items():
        pool = sorted(pb[L])
        keys = [p[0] for p in pool]
        used = [False] * len(pool)
        med = np.median(keys)
        order = sorted(vals, key=lambda v: -abs(v - med))
        if stats is not None:
            ds = [abs(v - med) for v in order]
            near_order += sum(1 for x, y, dx, dy in zip(order, order[1:], ds, ds[1:]) if x != y and abs(dx - dy) < 1e-9)
        for v in order:
            j = bisect.bisect_left(keys, v)
            lo, hi = j - 1, j
            while lo >= 0 and used[lo]:
                lo -= 1
            while hi < len(pool) and used[hi]:
                hi += 1
            c = [x for x in (lo, hi) if 0 <= x < len(pool)]
            k = min(c, key=lambda x: abs(keys[x] - v))
            if stats is not None and len(c) == 2 and abs(abs(keys[lo] - v) - abs(keys[hi] - v)) < 1e-9:
                near_dist += 1
            used[k] = True
            out.append(pool[k][2])
    if stats is not None:
        stats["near_tie_neighbour_choices"] = near_dist
        stats["near_tie_target_order"] = near_order
    return sorted(int(i) for i in out)


def pool_order(rows, etoks, ftoks) -> dict:
    """Spec D7 (vocab/a12.py lines 6-9): source and French frequencies over the pool, rows
    ascending; ties to first occurrence inside the pool. Each row's largest rank on each side
    (-1 for an empty side), so that caps(Ke, Kf) is a comparison."""
    pe, pf = collections.Counter(), collections.Counter()
    for i in rows:
        pe.update(etoks[i])
        pf.update(ftoks[i])
    re_ = {w: r for r, (w, _) in enumerate(pe.most_common())}
    rf_ = {w: r for r, (w, _) in enumerate(pf.most_common())}
    me = np.array([max((re_[w] for w in etoks[i]), default=-1) for i in rows], dtype=np.int64)
    mf = np.array([max((rf_[w] for w in ftoks[i]), default=-1) for i in rows], dtype=np.int64)
    return {"rows": list(rows), "me": me, "mf": mf, "types_e": len(re_), "types_f": len(rf_)}


def cap_mask(po: dict, ke, kf) -> np.ndarray:
    m = np.ones(len(po["rows"]), dtype=bool)
    if ke is not None:
        m &= po["me"] < ke
    if kf is not None:
        m &= po["mf"] < kf
    return m


def caps(po: dict, ke, kf) -> list[int]:
    """The pool rows whose source words are all among the ke most frequent and whose French
    words are all among the kf most frequent (None: no cap); ascending."""
    rows = po["rows"]
    return [rows[j] for j in np.nonzero(cap_mask(po, ke, kf))[0]]


def dedup_first(rows, key) -> tuple[list[int], list[int]]:
    """Walk rows ascending, keep a row unless its key occurred earlier in this list."""
    seen, kept, removed = set(), [], []
    for i in rows:
        k = key(i)
        if k in seen:
            removed.append(i)
        else:
            seen.add(k)
            kept.append(i)
    return kept, removed


def within(dev: Fraction, tol: Fraction) -> bool:
    return dev <= tol                      # spec D14: inclusive


def rel_dev(x, target) -> Fraction:
    if target == 0:
        raise Stop("a coverage target is 0; the relative deviation is undefined")
    return Fraction(abs(Fraction(x) - Fraction(target))) / Fraction(target)


def tie_key(rule: str, p: dict) -> tuple:
    inf = float("inf")
    if rule in ("a", "b"):
        return (-(inf if p["Ke"] is None else p["Ke"]), -(inf if p["Kf"] is None else p["Kf"]))
    if rule == "c":
        return (-p["k"],)
    return (-p["k"], p["lambda"])


def choose(points: list[dict], measures: list[str], targets: dict, rule: str):
    """The feasible point with the smallest max relative deviation over `measures`; ties by
    the registered tie rule (OD2_TIE). -> (point, dev) or (None, None)."""
    cands = []
    for p in points:
        if not p.get("feasible") or any(p.get(m) is None for m in measures):
            continue
        dev = max(rel_dev(p[m], targets[m]) for m in measures)
        cands.append((dev, tie_key(rule, p), p))
    if not cands:
        return None, None
    cands.sort(key=lambda c: (c[0], c[1]))
    return cands[0][2], cands[0][0]


def npy_bytes(idx) -> bytes:
    """An int64 .npy file (format 1.0) with a fixed header, so its sha256 does not depend on
    the numpy version that writes it."""
    a = np.ascontiguousarray(np.asarray(idx, dtype="<i8"))
    header = "{'descr': '<i8', 'fortran_order': False, 'shape': (%d,), }" % a.shape[0]
    pad = 64 - (10 + len(header) + 1) % 64
    header = (header + " " * (pad % 64) + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + a.tobytes()


def fr(x) -> dict:
    return {"float": float(x), "exact": f"{x.numerator}/{x.denominator}"} if isinstance(x, Fraction) else x


class _PlainUnpickler(pickle.Unpickler):
    """vocab/dice.py pickles a dict of str -> [(str, float), ...]: no class is needed to read
    it, so none is allowed (a pickle that names one is refused, not executed)."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"names {module}.{name}; lex.pkl may hold only dicts, lists, tuples, "
                                     f"strings and floats")


def load_lex(p: Path) -> dict:
    try:
        lex = _PlainUnpickler(io.BytesIO(p.read_bytes())).load()
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        raise Stop(f"cannot read the reference lexicon {p}: {e}")
    if not (isinstance(lex, dict) and all(
            isinstance(w, str) and isinstance(tr, list) and tr and all(
                isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str) and isinstance(t[1], float) for t in tr)
            for w, tr in lex.items())):
        raise Stop(f"{p} is not a vocab/dice.py lexicon (a dict of word -> [(French word, Dice), ...])")
    return lex


def compare_lex(built: dict, ref: dict) -> dict:
    """Entry by entry. literalness depends only on each entry's SET of translations (any 5-character
    prefix hit counts), so "other translations" is the count that can move a literalness value;
    an entry whose translations agree but come in another order (a third-place tie broken
    differently) cannot."""
    kb, kr = set(built), set(ref)
    both = sorted(kb & kr)
    other_set = [w for w in both if {f for f, _ in built[w]} != {f for f, _ in ref[w]}]
    other_order = [w for w in both if w not in set(other_set) and [f for f, _ in built[w]] != [f for f, _ in ref[w]]]
    dmax = 0.0
    for w in both:
        rb = dict(built[w])
        for f, dv in ref[w]:
            if f in rb:
                dmax = max(dmax, abs(rb[f] - dv))
    return {"entries_built": len(kb), "entries_lex_pkl": len(kr), "only_in_built": len(kb - kr),
            "only_in_lex_pkl": len(kr - kb), "entries_with_other_translations": len(other_set),
            "entries_with_same_translations_in_another_order": len(other_order),
            "max_abs_dice_difference_on_shared_translations": dmax,
            "identical": not (kb ^ kr) and not other_set and not other_order and dmax == 0.0,
            "examples": {"only_in_built": sorted(kb - kr)[:5], "only_in_lex_pkl": sorted(kr - kb)[:5],
                         "other_translations": other_set[:5]}}


# ======================================================================================
# fast measures over token ids (checked against the literal definitions at the end)
# ======================================================================================
class Side:
    """One side of one parent as a CSR of token ids, for the measures every grid point needs."""

    def __init__(self, tok_lists, vocab: dict[str, int]):
        self.n = len(tok_lists)
        self.lens = np.fromiter((len(t) for t in tok_lists), dtype=np.int64, count=self.n)
        self.off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(self.lens, out=self.off[1:])
        sd = vocab.setdefault
        self.ids = np.fromiter((sd(w, len(vocab)) for t in tok_lists for w in t), dtype=np.int32,
                               count=int(self.off[-1]))

    def _sel(self, idx) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return self.ids[np.repeat(m, self.lens)]

    def tokens(self, idx) -> int:
        return int(self.lens[np.asarray(idx, dtype=np.int64)].sum()) if len(idx) else 0

    def types(self, idx) -> int:
        return int(np.unique(self._sel(idx)).size) if len(idx) else 0

    def counts(self, idx, nvocab: int) -> np.ndarray:
        return np.bincount(self._sel(idx), minlength=nvocab)

    def types_at(self, idx, budget: int):
        """vocab/common.py types_at: distinct tokens among the first `budget` tokens of the rows
        in a Random(0) shuffle of the ascending index list; None if the set has <= budget tokens."""
        order = sorted(int(i) for i in idx)
        random.Random(0).shuffle(order)
        o = np.asarray(order, dtype=np.int64)
        ln = self.lens[o]
        cum = np.cumsum(ln)
        if not len(cum) or cum[-1] <= budget:
            return None
        r = int(np.searchsorted(cum, budget)) + 1          # rows needed to reach `budget` tokens
        o, ln = o[:r], ln[:r]
        start = np.repeat(self.off[o] - (np.cumsum(ln) - ln), ln)
        gidx = start + np.arange(int(ln.sum()), dtype=np.int64)
        return int(np.unique(self.ids[gidx[:budget]]).size)


def py_types(idx, tok_lists) -> int:
    return len({w for i in idx for w in tok_lists[i]})


def py_types_at(idx, tok_lists, budget: int):
    """vocab/common.py types_at, literally."""
    rng = random.Random(0)
    idx = sorted(int(i) for i in idx)
    rng.shuffle(idx)
    seen, n = set(), 0
    for i in idx:
        for w in tok_lists[i]:
            if n >= budget:
                return len(seen)
            seen.add(w)
            n += 1
    return None


# ======================================================================================
# decisions
# ======================================================================================
def _caps_value(v) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool) and v > 0)


def parse_grid(rule: str, g) -> tuple[dict | None, str | None]:
    if not isinstance(g, dict):
        return None, "must be an object"
    if rule in ("a", "b"):
        ke, kf = g.get("Ke"), g.get("Kf")
        if not (isinstance(ke, list) and ke and all(_caps_value(x) for x in ke)
                and isinstance(kf, list) and kf and all(_caps_value(x) for x in kf)):
            return None, 'needs "Ke" and "Kf": non-empty lists of positive integers or null (no cap)'
        if len(set(ke)) != len(ke) or len(set(kf)) != len(kf):
            return None, "has a repeated cap"
        return {"Ke": list(ke), "Kf": list(kf)}, None
    k = g.get("K")
    if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1):
        return None, 'needs "K", a positive integer'
    if rule == "c":
        return {"K": k}, None
    lam = g.get("lambda")
    if not (isinstance(lam, list) and lam and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                   and 0 <= x <= 1 for x in lam) and len(set(lam)) == len(lam)):
        return None, 'needs "lambda", a non-empty list of distinct numbers in [0, 1]'
    return {"K": k, "lambda": [float(x) for x in lam]}, None


def default_grid(rule: str, po: dict | None = None) -> dict:
    """The spec's PROPOSAL grids (A14, A15, 6.5 c2), used by --feasibility when J gives none."""
    if rule in ("a", "b"):
        te, tf = po["types_e"], po["types_f"]
        return {"Ke": [1000 * j for j in range(1, -(-te // 1000) + 1)] + [None],
                "Kf": [1000 * j for j in range(1, -(-tf // 1000) + 1)] + [None]}
    if rule == "c":
        return {"K": 200}
    return {"K": 200, "lambda": [round(0.05 * j, 2) for j in range(21)]}


def resolve_decisions(raw, production: bool) -> tuple[dict, list[str], list[str]]:
    """-> (resolved values, unset problems, invalid/refused problems). Production requires
    every key without a default; --feasibility reads only the numeric parameters."""
    unset, bad = [], []
    if not isinstance(raw, dict):
        return {}, [], ["the decisions file is not a JSON object"]
    known = set(DECISIONS) | RECORDED_ONLY | FEASIBILITY_ONLY
    for k in raw:
        if k not in known and not k.startswith("_"):
            bad.append(f"{k}: not a decision key (typo?); known keys: {', '.join(sorted(known))}")
    val = {}
    for k, (default, _) in DECISIONS.items():
        v = raw.get(k, None)
        val[k] = default if v is None or v == "OPEN" else v
        if (v is None or v == "OPEN") and default is None and k not in OPTIONAL_KEYS and production:
            unset.append(k)

    def blocks(k):
        return ", ".join(DECISIONS[k][1])

    def opt(k, options):
        v = val[k]
        if v is not None and v not in options:
            bad.append(f"{k} = {v!r}: options are {', '.join(map(repr, options))} (blocks {blocks(k)})")
            val[k] = None
        return val[k]

    od1 = opt("od1_coverage", ("a", "b", "c", "d"))
    od2 = opt("od2_rule", ("a", "b", "c", "c2"))
    od4 = opt("od4_duplicate", ("a", "b"))
    od5 = opt("od5", ("a", "b", "c"))
    od6 = opt("od6_topband", ("keep", "remove"))
    od7 = opt("od7_exposure", ("accept", "match"))
    opt("od12_draws", ("one", "per_seed"))
    od15 = opt("od15_narrowness", ("type_count", "news_coverage"))
    for k in ("od4_recompute_rep", "od18_register_v", "od3_caps_for_reduced_pool"):
        if val[k] is not None and not isinstance(val[k], bool):
            bad.append(f"{k} = {val[k]!r}: must be true or false (blocks {blocks(k)})")
            val[k] = None
    if val["tolerance"] is not None:
        try:
            if isinstance(val["tolerance"], bool):
                raise ValueError
            t = Fraction(str(val["tolerance"]))
            if t < 0:
                raise ValueError
            val["tolerance"] = t
        except (ValueError, ZeroDivisionError):
            bad.append(f"tolerance = {val['tolerance']!r}: must be a non-negative number (blocks {blocks('tolerance')})")
            val["tolerance"] = None
    elif not production:
        val["tolerance"] = Fraction(1, 50)  # --feasibility only: the draft's +-2%, to mark points; never a build
    for k in ("fr_budget_214489", "fr_budget_198333"):
        v = val[k]
        if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
            bad.append(f"{k} = {v!r}: must be a positive integer (blocks {blocks(k)})")
            val[k] = None
    s = val["seeds"]
    if not (isinstance(s, list) and s and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in s)
            and len(set(s)) == len(s) and 42 in s):
        bad.append(f"seeds = {s!r}: distinct non-negative integers including 42 (blocks R0, A_narrow)")
        val["seeds"] = None
    if not isinstance(val["od10_unidata"], str):
        bad.append(f"od10_unidata = {val['od10_unidata']!r}: must be a Unicode version string")
    c3 = val["od3_caps"]
    if c3 == "retune":
        if production:
            bad.append("od3_caps = 'retune' is REFUSED: no grid, tolerance or objective for re-tuning is "
                       "defined (spec 7.2, OD 3) (blocks A_narrow)")
        val["od3_caps"] = None
    elif c3 is not None and not (isinstance(c3, list) and len(c3) == 2 and all(_caps_value(x) for x in c3)):
        bad.append(f"od3_caps = {c3!r}: [Ke, Kf] (positive integers or null) or 'retune' (blocks A_narrow)")
        val["od3_caps"] = None
    grids = val["od2_grid"]
    parsed = {}
    if grids is not None:
        if not isinstance(grids, dict):
            bad.append("od2_grid: an object keyed by rule ('a', 'b', 'c', 'c2') (blocks div_narrow, div_narrow_dd)")
        else:
            for r, g in grids.items():
                if r not in OD2_TIE:
                    bad.append(f"od2_grid.{r}: not a rule ('a', 'b', 'c', 'c2')")
                    continue
                pg, err = parse_grid(r, g)
                if err:
                    bad.append(f"od2_grid.{r} {err} (blocks div_narrow, div_narrow_dd)")
                else:
                    parsed[r] = pg
    val["od2_grid"] = parsed if grids is not None else None
    rg = raw.get("od3_report_grid")
    if rg is not None:
        pg, err = parse_grid("a", rg)
        if err:
            bad.append(f"od3_report_grid {err}")
        val["od3_report_grid"] = pg
    else:
        val["od3_report_grid"] = None
    for k in RECORDED_ONLY:
        if k in raw:
            val[k] = raw[k]
    if not production:
        return val, [], bad

    # ---- production: options that have no defined rule are refused (spec 7.2) ----
    if od1 == "c":
        bad.append("od1_coverage = 'c' is REFUSED: the piece-counting rule (spec A28) is not registered, and set "
                   "construction has no SentencePiece path (--measure counts the pieces of finished sets only) "
                   "(blocks div_narrow, div_narrow_dd)")
    if od1 == "d":
        bad.append("od1_coverage = 'd' is REFUSED: no tolerance for news coverage is registered "
                   "(spec A29) (blocks div_narrow, div_narrow_dd)")
    extra_unset = []
    if od2 is not None:
        if val["od2_tie"] is not None and val["od2_tie"] != OD2_TIE[od2]:
            bad.append(f"od2_tie = {val['od2_tie']!r}: the only tie rule implemented for od2_rule = {od2!r} "
                       f"is {OD2_TIE[od2]!r} (spec 6.5) (blocks div_narrow, div_narrow_dd)")
        if val["od2_grid"] is not None and od2 not in val["od2_grid"]:
            bad.append(f"od2_grid has no entry for od2_rule = {od2!r} (blocks div_narrow, div_narrow_dd)")
        c_def = val["od2c_definition"]
        if od2 in OD2C_DEFINITION:
            if c_def is None or c_def == "OPEN":
                extra_unset.append(f"od2c_definition: unset; od2_rule = {od2!r} is defined only by spec 6.5 / A15, "
                                   f"so the decisions file must register it as {OD2C_DEFINITION[od2]!r} "
                                   f"(blocks div_narrow, div_narrow_dd)")
            elif c_def != OD2C_DEFINITION[od2]:
                bad.append(f"od2c_definition = {c_def!r}: the only definition implemented for od2_rule = {od2!r} "
                           f"is {OD2C_DEFINITION[od2]!r} (blocks div_narrow, div_narrow_dd)")
        elif c_def is not None:
            bad.append(f"od2c_definition is set, but od2_rule = {od2!r} is not a q-family rule (c, c2): "
                       f"the decisions file contradicts itself (blocks div_narrow, div_narrow_dd)")
        g = (val["od2_grid"] or {}).get(od2)
        if od1 == "a" and od2 in ("a", "b") and g is not None and g["Kf"] != [None]:
            bad.append(f"od1_coverage = 'a' with od2_rule = {od2!r} is REFUSED unless od2_grid.{od2}.Kf is [null]: "
                       f"under (a) the French cap has no objective, so it would be chosen only by the source "
                       f"deviation (spec A12: no French cap, search Ke only) (blocks div_narrow, div_narrow_dd)")
    if od6 == "remove" and val["od3_caps_for_reduced_pool"] is not True:
        bad.append("od6_topband = 'remove' is REFUSED unless od3_caps_for_reduced_pool = true registers that "
                   "od3_caps were decided for the pool without the top-band rows: OD 3's 'keep' is defined by "
                   "reproducing A_narrow_42_58.npy, which describes the full pool only (spec A21) (blocks A_narrow)")
    if od6 == "keep" and val["od3_caps_for_reduced_pool"] is True:
        bad.append("od3_caps_for_reduced_pool = true contradicts od6_topband = 'keep' (the caps then apply to the "
                   "full pool) (blocks A_narrow)")
    if od4 == "b" and val["od4_norm_rule"] != A17_RULE:
        bad.append(f"od4_duplicate = 'b' is REFUSED unless od4_norm_rule = {A17_RULE!r} registers the "
                   f"masking rule (spec A17) (blocks {', '.join(DEDUP)})")
    if val["od4_recompute_rep"] is True:
        bad.append("od4_recompute_rep = true is REFUSED: the parent of a recomputation is undefined "
                   "(spec A18) (blocks rep_dedup, div_dd, div_narrow_dd)")
    if od5 == "c" and val["od18_register_v"] is True:
        bad.append("od5 = 'c' with od18_register_v = true is REFUSED: V needs div_narrow, which 'c' "
                   "does not build (spec A20)")
    if od7 == "match":
        bad.append("od7_exposure = 'match' is REFUSED: no exposure-matching rule is defined (blocks A_narrow)")
    if od15 == "news_coverage":
        bad.append("od15_narrowness = 'news_coverage' is REFUSED: the extra arm it needs has no defined rule")
    unset = [f"{k}: unset (blocks {blocks(k)})" for k in unset] + extra_unset
    return val, unset, bad


def decisions_record(val: dict) -> dict:
    out = {}
    for k, v in val.items():
        if isinstance(v, Fraction):
            out[k] = fr(v)
        else:
            out[k] = v
    return out


# ======================================================================================
# the build
# ======================================================================================
class Parent:
    def __init__(self, name: str, s: Path, ve: dict, vf: dict):
        self.name = name
        self.src, self.tgt, self.lab = (read_lines(s / f"{name}.{e}") for e in ("en", "fr", "source"))
        if not len(self.src) == len(self.tgt) == len(self.lab):
            raise Stop(f"{name}: .en has {len(self.src):,} lines, .fr {len(self.tgt):,}, .source {len(self.lab):,}")
        self.n = len(self.src)
        self.toks, self.ftoks = tokenize(self.src), tokenize(self.tgt)
        self.lens = [len(t) for t in self.toks]
        self.rep = rep_within(self.toks)
        self.E, self.F = Side(self.toks, ve), Side(self.ftoks, vf)
        self.maxrank: list[int] = []
        self.maxrank_fr: list[int] = []


class Builder:
    def __init__(self, a, val: dict, mode: str):
        self.a, self.val, self.mode = a, val, mode
        self.lg = Log1p()
        self.checks: list[dict] = []
        self.inputs: dict[str, dict] = {}
        self.unpinned: list[str] = []
        self.pre_hashes = None
        self.pre_counts_fr = None
        self.nc = None                     # news-commentary reference counts (--news-en)
        self.ho_src = self.ho_tgt = self.ho = None
        self.pinned = False
        self._po: dict[str, dict] = {}
        self.ve: dict[str, int] = {}
        self.vf: dict[str, int] = {}

    # ---- checks ----
    def ok(self, name: str, value=None):
        self.checks.append({"check": name, "ok": True, **({} if value is None else {"value": value})})

    def expect(self, what: str, got: int):
        if not self.pinned:
            return
        want = EXPECTED_ON_PINNED[what]
        if got != want:
            raise Stop(f"{what}: {got:,}, but the pinned inputs give {want:,} (spec sections 4-6). The "
                       f"interpreter's Unicode tables or the code depart from the spec's definitions")
        self.ok(f"{what} = {want:,} (pinned inputs)")

    # ---- inputs and pins ----
    def pin(self, name: str, path: Path, want: str | None):
        if not path.is_file():
            raise Stop(f"MISSING input: {path}")
        have = sha256_file(path)
        self.inputs[name] = {"sha256": have, "pinned_sha256": want,
                             "pinned": None if want is None else have == want}
        if want is not None and have != want:
            self.unpinned.append(name)
        return have

    def check_pins(self, need_fr: bool, refs: dict | None = None):
        a, s, e = self.a, self.a.src_dir, self.a.e05_dir
        for f, h in {**PINNED_SRC, **PINNED_HELDOUT}.items():
            self.pin(f, s / f, h)
        self.pin("E0.5 manifest.json", e / "manifest.json", PINNED_E05_MANIFEST)
        for f, h in PINNED_ANCHORS.items():
            self.pin(f"E0.5 {f}", e / f, h)
        self.pin("train.clean.en", a.pretrain_en, PINNED_PRETRAIN_EN)
        if need_fr:
            self.pin("train.clean.fr", a.pretrain_fr, PINNED_PRETRAIN_FR)
        for nm, p in (refs or {}).items():
            self.pin(f"reference {nm}", p, PINNED_REFERENCE[nm])
        if self.mode != "reproduce":
            for nm, p in (("v2_scored.tsv", a.qe_tsv), ("news-commentary .en", a.news_en)):
                if p is not None:
                    self.pin(nm, p, None)          # "to record" (spec 3.1)
            if getattr(a, "ref_lex", None) is not None:
                self.pin("lex.pkl", a.ref_lex, PINNED_LEX)
        if self.unpinned and not a.allow_unpinned_src:
            raise Stop("these inputs are not the pinned ones (spec 3.1):" + "".join(
                f"\n  {f}  sha256 {self.inputs[f]['sha256']}" for f in self.unpinned))
        self.pinned = not self.unpinned
        if self.pinned:
            self.ok("every input has its pinned sha256")

    # ---- reading, rep, pretraining ranks, the UN split ----
    def prepare(self, want_hashes: bool, need_fr_ranks: bool):
        a = self.a
        self.T = Parent("ft_topk", a.src_dir, self.ve, self.vf)
        self.R = Parent("ft_random", a.src_dir, self.ve, self.vf)
        T, R = self.T, self.R
        self.median = median_check(T.rep)
        self.ok("median rep over ft_topk is 0.0 (spec D4)")
        self.div_cand = [i for i in range(T.n) if T.lab[i] == SOURCE and T.rep[i] == 0.0]
        self.rep_cand = [i for i in range(T.n) if T.lab[i] == SOURCE and T.rep[i] > 0.0]
        self.ru0 = [i for i in range(R.n) if R.lab[i] == SOURCE and R.rep[i] == 0.0]
        self.expect("diverse UN candidates", len(self.div_cand))
        self.expect("repetitive UN candidates", len(self.rep_cand))
        self.expect("RU0 (UN rows of ft_random with rep_rand = 0)", len(self.ru0))
        if not self.div_cand or not self.rep_cand:
            raise Stop(f"no '{SOURCE}' rows on both sides of the median: nothing to compare")
        self.top_pairs = set(zip(T.src, T.tgt))
        if len(self.top_pairs) != T.n:
            raise Stop(f"ft_topk holds {T.n - len(self.top_pairs):,} repeated (en, fr) pairs: rows cannot be "
                       f"identified by their pair (spec D15)")
        # pretraining ranks (spec D6): one Counter, file order, single process
        C = collections.Counter()
        hs = array("Q") if want_hashes else None
        nlines = 0
        with open(a.pretrain_en, encoding="utf-8", newline="\n") as f:
            for ln in f:
                s = ln.rstrip("\n")
                C.update(toks(s))
                if hs is not None:
                    hs.append(nkey(s))
                nlines += 1
        self.pre_lines = nlines
        self.expect("pretraining lines", nlines)
        self.pre_counts = C
        self.rank = ranks_of(C, "train.clean.en")
        self.ok("pretraining word types below 10**7", len(self.rank))
        if hs is not None:
            self.pre_hashes = np.unique(np.frombuffer(hs, dtype=np.uint64))
        T.maxrank, R.maxrank = maxrank_of(T.toks, self.rank), maxrank_of(R.toks, self.rank)
        if need_fr_ranks:
            self.french_ranks()
        self.un_split()

    def french_ranks(self):
        if self.pre_counts_fr is None:
            CF = collections.Counter()
            n = 0
            with open(self.a.pretrain_fr, encoding="utf-8", newline="\n") as f:
                for ln in f:
                    CF.update(toks(ln.rstrip("\n")))
                    n += 1
            if n != self.pre_lines:
                raise Stop(f"train.clean.fr has {n:,} lines, train.clean.en {self.pre_lines:,}")
            self.pre_counts_fr = CF
            self.rank_fr = ranks_of(CF, "train.clean.fr")
            self.T.maxrank_fr = maxrank_of(self.T.ftoks, self.rank_fr)

    def un_split(self):
        T, R, e = self.T, self.R, self.a.e05_dir
        kd, kr = strat_match(self.div_cand, self.rep_cand, T.lens)          # G1
        self.kd, self.kr = kd, kr
        self.un_hist = collections.Counter(T.lens[i] for i in kd)
        if collections.Counter(T.lens[i] for i in kr) != self.un_hist:
            raise Stop("the two UN halves do not share one histogram")
        self.rnd_un = [i for i in range(R.n) if R.lab[i] == SOURCE]
        ku, self.slack_un_random = match_random(self.un_hist, R.lens, self.rnd_un,
                                                f"the '{SOURCE}' rows of ft_random")            # G2
        self.ku = ku
        self.expect("rows of each anchor", len(kd))
        # the byte check: every anchor file must equal E0.5's (spec 6.1)
        try:
            man = json.loads((e / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise Stop(f"cannot read E0.5's manifest {e / 'manifest.json'}: {ex}")
        if man.get("e05_complete") is not True:
            raise Stop(f"{e / 'manifest.json'} has no \"e05_complete\": true: not a finished E0.5 build")
        recorded = man.get("sha256", {})
        built = {"ft_un_div": (T, kd), "ft_un_rep": (T, kr), "ft_un_random": (R, ku)}
        bad = []
        for f in ANCHOR_FILES:
            nm, ext = f.rsplit(".", 1)
            P, idx = built[nm]
            rows = P.src if ext == "en" else P.tgt
            h = hashlib.sha256()
            for i in idx:
                h.update((rows[i] + "\n").encode("utf-8"))
            mine = h.hexdigest()
            on_disk = self.inputs[f"E0.5 {f}"]["sha256"]
            if recorded.get(f) != on_disk:
                bad.append(f"{f}: E0.5's file no longer has the sha256 its manifest records")
            elif mine != on_disk:
                bad.append(f"{f}: rebuilt {mine[:16]}..., E0.5's {on_disk[:16]}...")
        if bad:
            raise Stop("the UN split recomputed from the inputs does not reproduce E0.5's anchors byte for byte "
                       "(the build departs from the definition, or this interpreter's Unicode tables differ "
                       "from the box's):\n  " + "\n  ".join(bad))
        self.ok("E0.5's UN split recomputed in ascending order reproduces all six anchor files byte for byte")

    # ---- the sets ----
    def r0_pool(self, od6: str) -> list[int]:
        if od6 == "keep":
            return list(self.ru0)
        R = self.R
        return [i for i in self.ru0 if (R.src[i], R.tgt[i]) not in self.top_pairs]

    def build_r0(self, od6: str, seed: int) -> tuple[list[int], dict]:
        pool = self.r0_pool(od6)
        idx, slack = match_random(self.un_hist, self.R.lens, pool, f"R0's pool (od6 = {od6})", seed)   # G3
        return idx, {"pool_rows": len(pool), "tightest_bucket_slack": slack, "seed": seed, "od6_topband": od6}

    def build_a_narrow(self, od6: str, cap: list, seed: int, stats: dict | None = None) -> tuple[list[int], dict]:
        T, R = self.T, self.R
        pool0 = self.r0_pool(od6)
        if od6 not in self._po:                # D7 depends on the pool only, not on the caps or the seed
            self._po[od6] = pool_order(pool0, R.toks, R.ftoks)
        po = self._po[od6]
        pool = caps(po, cap[0], cap[1])
        if od6 == "keep" and list(cap) == list(REFERENCE_CAPS):
            self.expect("A_narrow capped pool at 42,000 / 58,000 (top-band rows kept)", len(pool))
        idx = nn_match(self.kd, T.lens, T.maxrank, pool, R.lens, R.maxrank, seed,
                       f"A_narrow (caps {cap[0]} / {cap[1]}, od6 = {od6})", self.lg, stats)       # G4
        return idx, {"pool_rows": len(pool0), "pool_source_types": po["types_e"], "pool_french_types": po["types_f"],
                     "capped_pool_rows": len(pool), "caps": list(cap), "seed": seed, "od6_topband": od6}

    def build_dedup(self, od4: str) -> tuple[list[int], list[int], dict]:
        T = self.T
        key = (lambda i: T.src[i]) if od4 == "a" else (lambda i: a17_key(T.src[i]))
        rep_k, rep_rm = dedup_first(self.rep_cand, key)
        div_k, div_rm = dedup_first(self.div_cand, key)
        long_rm = [i for i in div_rm if T.lens[i] >= 8]
        info = {"od4_duplicate": od4, "repetitive_candidates": len(self.rep_cand),
                "repetitive_after_dedup": len(rep_k), "repetitive_removed": len(rep_rm),
                "diverse_after_dedup": len(div_k), "diverse_removed": len(div_rm),
                "diverse_removed_lengths": dict(sorted(collections.Counter(T.lens[i] for i in div_rm).items())),
                "diverse_removed_with_8_or_more_tokens": len(long_rm)}
        if od4 == "a":
            self.expect("repetitive candidates after OD 4(a) dedup", len(rep_k))
            self.expect("diverse candidates after OD 4(a) dedup", len(div_k))
            if long_rm and self.mode in ("production", "dry_run"):
                raise Stop(f"OD 4(a) removed {len(long_rm):,} diverse rows of 8 or more tokens: the draft's argument "
                           f"(L248-252) that step 2 cannot change step 3 is false on these inputs")
        div_dd, rep_dd = strat_match(div_k, rep_k, T.lens)                  # G5
        return rep_dd, div_dd, info

    # ---- div_narrow / div_narrow_dd: grids (spec 6.5) ----
    def narrow_setup(self, target_rows: list[int], budget: int) -> dict:
        T = self.T
        h = collections.Counter(T.lens[i] for i in target_rows)
        by: dict[int, list[int]] = {}
        for i in self.div_cand:
            by.setdefault(T.lens[i], []).append(i)
        nL = {L: len(by.get(L, [])) for L in h}
        short = [L for L in h if nL[L] < h[L]]
        if short:
            raise Stop(f"the diverse candidates are short of the target histogram at lengths {sorted(short)}")
        forced = sorted(L for L in h if nL[L] == h[L])
        targets = {"src_types": T.E.types(target_rows), "fr_types": T.F.types_at(target_rows, budget)}
        if targets["fr_types"] is None:
            raise Stop(f"the coverage target set has {T.F.tokens(target_rows):,} French tokens, not more than "
                       f"the budget {budget:,}")
        if self.nc is not None:
            targets["nc_cov5"] = self.cover(T, target_rows, self.nc, 5)
        po = pool_order(self.div_cand, T.toks, T.ftoks)
        fset = set(i for L in forced for i in by[L])
        pos = {i: j for j, i in enumerate(po["rows"])}
        fe = max((int(po["me"][pos[i]]) for i in fset), default=-1) + 1
        ff = max((int(po["mf"][pos[i]]) for i in fset), default=-1) + 1
        return {"target_rows": target_rows, "h": h, "by": by, "nL": nL, "budget": budget, "targets": targets,
                "po": po, "lens_arr": np.asarray([T.lens[i] for i in po["rows"]], dtype=np.int64),
                "forced": {"lengths": forced, "rows": sum(h[L] for L in forced),
                           "smallest_caps_keeping_every_forced_row": {"Ke": fe, "Kf": ff},
                           "candidate_pool_source_types": po["types_e"], "candidate_pool_french_types": po["types_f"]}}

    def tie_u(self) -> dict[int, float]:
        if not hasattr(self, "_u"):
            rng = random.Random(SEED)                                          # G8a
            self._u = {i: rng.random() for i in self.div_cand}
        return self._u

    def measure(self, idx: list[int], budget: int) -> dict:
        T = self.T
        m = {"rows": len(idx), "src_types": T.E.types(idx), "fr_tokens": T.F.tokens(idx),
             "fr_types": T.F.types_at(idx, budget)}
        if self.nc is not None:
            m["nc_cov5"] = self.cover(T, idx, self.nc, 5)
        return m

    def grid_ab(self, ns: dict, rule: str, grid: dict) -> list[dict]:
        T = self.T
        h, po = ns["h"], ns["po"]
        maxL = max(max(h), int(ns["lens_arr"].max()) if len(ns["lens_arr"]) else 0)
        pts = []
        for ke in grid["Ke"]:
            for kf in grid["Kf"]:
                mask = cap_mask(po, ke, kf)
                cnt = np.bincount(ns["lens_arr"][mask], minlength=maxL + 1)
                short = [L for L in sorted(h) if cnt[L] < h[L]]
                p = {"Ke": ke, "Kf": kf, "pool_rows": int(mask.sum()), "feasible": not short}
                if short:
                    p.update({"short_buckets": len(short), "shortfall": int(sum(h[L] - cnt[L] for L in short)),
                              "first_short_length": short[0]})
                else:
                    p.update(self.measure(self.ab_point(ns, rule, ke, kf), ns["budget"]))
                pts.append(p)
        return pts

    def ab_point(self, ns: dict, rule: str, ke, kf, stats: dict | None = None) -> list[int]:
        T = self.T
        pool = caps(ns["po"], ke, kf)
        if rule == "a":                                                       # G6
            return nn_match(ns["target_rows"], T.lens, T.maxrank, pool, T.lens, T.maxrank, SEED,
                            f"div_narrow (a) at Ke={ke} Kf={kf}", self.lg, stats)
        idx, _ = match_random(ns["h"], T.lens, pool, f"div_narrow (b) at Ke={ke} Kf={kf}")   # G7
        return idx

    def q_orders(self, ns: dict, lam) -> dict[int, list[int]]:
        T, u, lg = self.T, self.tie_u(), self.lg
        if lam is None:                                                       # rule (c)
            key = lambda i: (T.maxrank[i], u[i], i)
        else:                                                                 # rule (c2)
            a, b = 1 - lam, lam
            key = lambda i: (a * lg(T.maxrank[i]) + b * lg(T.maxrank_fr[i]), u[i], i)
        return {L: sorted(ns["by"][L], key=key) for L in ns["h"]}

    @staticmethod
    def q_point(ns: dict, orders: dict, k: int, K: int) -> list[int]:
        rng = random.Random(SEED)                                             # G8b
        out: list[int] = []
        h, nL = ns["h"], ns["nL"]
        for L in sorted(h):
            m = h[L] + (k * (nL[L] - h[L])) // K
            out += rng.sample(sorted(orders[L][:m]), h[L])
        return sorted(out)

    def grid_q(self, ns: dict, rule: str, grid: dict) -> list[dict]:
        K = grid["K"]
        lams = [None] if rule == "c" else grid["lambda"]
        pts = []
        for lam in lams:
            orders = self.q_orders(ns, lam)
            for k in range(K + 1):
                p = {"k": k, "q": k / K, "feasible": True}
                if lam is not None:
                    p["lambda"] = lam
                p.update(self.measure(self.q_point(ns, orders, k, K), ns["budget"]))
                pts.append(p)
        return pts

    def grid(self, ns: dict, rule: str, grid: dict) -> list[dict]:
        return self.grid_ab(ns, rule, grid) if rule in ("a", "b") else self.grid_q(ns, rule, grid)

    def rebuild_point(self, ns: dict, rule: str, grid: dict, p: dict, stats: dict | None = None) -> list[int]:
        if rule in ("a", "b"):
            return self.ab_point(ns, rule, p["Ke"], p["Kf"], stats)
        return self.q_point(ns, self.q_orders(ns, p.get("lambda")), p["k"], grid["K"])

    # ---- descriptives (spec section 9) ----
    def cover(self, P: Parent, idx, ref: dict, k: int) -> float:
        cnt = P.E.counts(idx, len(self.ve))
        ids, n, tot = ref["ids"], ref["n"], ref["tot"]
        return 100 * int(n[cnt[ids] >= k].sum()) / tot if tot else 0.0

    def ref_counts(self, tok_iter) -> dict:
        C = collections.Counter()
        for t in tok_iter:
            C.update(t)
        tot = sum(C.values())
        known = [(self.ve[w], c) for w, c in C.items() if w in self.ve]
        return {"ids": np.asarray([i for i, _ in known], dtype=np.int64),
                "n": np.asarray([c for _, c in known], dtype=np.int64), "tot": tot, "types": len(C)}

    def load_heldout(self):
        if self.ho_src is None:
            a = self.a
            self.ho_src, self.ho_tgt = read_lines(a.src_dir / "heldout_un.en"), read_lines(a.src_dir / "heldout_un.fr")
            if len(self.ho_src) != len(self.ho_tgt):
                raise Stop("heldout_un.en and heldout_un.fr differ in line count")
            self.ho = self.ref_counts(toks(x) for x in self.ho_src)

    def descriptive_inputs(self):
        a = self.a
        self.load_heldout()
        self.qe = None
        if a.qe_tsv is not None:
            self.qe = self.load_qe(a.qe_tsv)
        self.pair_hashes = self.lex = self.ref_lex = self.lex_compare = None
        if a.pretrain_fr is not None:
            self.french_ranks()
            self.pretrain_pairs_and_lexicon()
            if getattr(a, "ref_lex", None) is not None:
                self.ref_lex = load_lex(a.ref_lex)
                self.lex_compare = compare_lex(self.lex, self.ref_lex)
                self.ok("the rebuilt Dice lexicon was compared with lex.pkl entry by entry", self.lex_compare)

    def descriptives_record(self) -> dict:
        """Which of the draft's "Measured on the sets" columns this build could fill."""
        have = {"qe (--qe-tsv)": self.qe is not None, "literalness (--pretrain-fr)": self.lex is not None,
                "pair_in_pretraining (--pretrain-fr)": self.pair_hashes is not None,
                "news_coverage (--news-en)": self.nc is not None,
                "lexicon compared with lex.pkl (--ref-lex)": self.lex_compare is not None}
        return {"complete": all(have.values()), "missing": [k for k, v in have.items() if not v]}

    def load_qe(self, path: Path) -> dict:
        """vocab/qe_join.py: QE per parent row, joined on the exact en + TAB + fr bytes; the
        maximum over conflicting duplicate scores."""
        keys: dict[bytes, list] = {}
        arr = {}
        for P in (self.T, self.R):
            arr[P.name] = np.full(P.n, np.nan)
            for i, (x, y) in enumerate(zip(P.src, P.tgt)):
                keys.setdefault(x.encode("utf-8") + b"\t" + y.encode("utf-8"), []).append((P.name, i))
        conflicts = 0
        with open(path, "rb", buffering=1 << 24) as f:
            for ln in f:
                sc, _, rest = ln.partition(b"\t")
                v = keys.get(rest.rstrip(b"\n"))
                if v is None:
                    continue
                s = float(sc)
                for name, i in v:
                    cur = arr[name][i]
                    if not np.isnan(cur) and cur != s:
                        conflicts += 1
                    arr[name][i] = s if np.isnan(cur) else max(cur, s)
        top = arr["ft_topk"]
        thr = float(np.nanmin(top)) if np.any(~np.isnan(top)) else None
        return {"arr": arr, "threshold": thr, "conflicting_duplicate_scores": conflicts,
                "matched": {k: int((~np.isnan(v)).sum()) for k, v in arr.items()}}

    def pretrain_pairs_and_lexicon(self):
        """Exact-pair exposure (e03_build_controls.py h64) and the Dice lexicon of vocab/dice.py
        (1.5M pairs drawn with Random(7); ranks 100-19,999 of each side, no isdigit() words;
        Dice >= 0.15, top 3). Without scipy: ties at the third place are broken by French
        word order of first appearance in the rank list, which may differ from lex.pkl."""
        a = self.a
        pp = array("Q")
        with open(a.pretrain_en, encoding="utf-8", newline="\n") as fe, \
                open(a.pretrain_fr, encoding="utf-8", newline="\n") as ff:
            for x, y in zip(fe, ff):
                pp.append(h64(x.rstrip("\n"), y.rstrip("\n")))
        self.pair_hashes = np.unique(np.frombuffer(pp, dtype=np.uint64))
        del pp
        lo, hi = LEX_RANKS
        ve = [w for w, _ in self.pre_counts.most_common(hi)[lo:] if not w.isdigit()]
        vf = [w for w, _ in self.pre_counts_fr.most_common(hi)[lo:] if not w.isdigit()]
        ie = {w: i for i, w in enumerate(ve)}
        jf = {w: j for j, w in enumerate(vf)}
        n = self.pre_lines
        keep = set(random.Random(LEX_SEED).sample(range(n), min(LEX_SAMPLE, n)))
        NF = max(len(vf), 1)
        ce = np.zeros(len(ve), dtype=np.int64)
        cf = np.zeros(len(vf), dtype=np.int64)
        run_c = np.zeros(0, dtype=np.int64)
        run_n = np.zeros(0, dtype=np.int64)
        buf: list[np.ndarray] = []
        nbuf = 0

        def merge(run_c, run_n, extra):
            if not extra:
                return run_c, run_n
            c = np.concatenate([run_c] + extra)
            w = np.concatenate([run_n] + [np.ones(len(x), dtype=np.int64) for x in extra])
            u, inv = np.unique(c, return_inverse=True)
            return u, np.bincount(inv.ravel(), weights=w, minlength=len(u)).astype(np.int64)

        with open(a.pretrain_en, encoding="utf-8", newline="\n") as fe, \
                open(a.pretrain_fr, encoding="utf-8", newline="\n") as ff:
            for k, (x, y) in enumerate(zip(fe, ff)):
                if k not in keep:
                    continue
                E = np.asarray(sorted({ie[w] for w in toks(x) if w in ie}), dtype=np.int64)
                F = np.asarray(sorted({jf[w] for w in toks(y) if w in jf}), dtype=np.int64)
                ce[E] += 1
                cf[F] += 1
                if len(E) and len(F):
                    buf.append((E[:, None] * NF + F[None, :]).ravel())
                    nbuf += len(buf[-1])
                    if nbuf > 20_000_000:
                        run_c, run_n = merge(run_c, run_n, buf)
                        buf, nbuf = [], 0
        run_c, run_n = merge(run_c, run_n, buf)
        lex = {}
        starts = np.searchsorted(run_c, np.arange(len(ve) + 1, dtype=np.int64) * NF)
        thr = np.float32(LEX_DICE)
        for i in range(len(ve)):
            s, e = starts[i], starts[i + 1]
            if s == e:
                continue
            js = run_c[s:e] - i * NF
            cs = run_n[s:e].astype(np.float32)
            d = (np.float32(2) * cs) / (np.float32(ce[i]) + cf[js].astype(np.float32))
            o = np.argsort(-d, kind="stable")[:3]
            tr = [(vf[js[q]], float(d[q])) for q in o if d[q] >= thr]
            if tr:
                lex[ve[i]] = tr
        self.lex = lex

    def literalness(self, P: Parent, idx, lex: dict | None = None) -> float | None:
        """vocab/arms.py lexhit: mean over rows with a lexicon-covered source token of the share
        of covered tokens with a translation whose 5-character prefix is in the French row."""
        lex = self.lex if lex is None else lex
        out = []
        for i in idx:
            fp = {w[:5] for w in P.ftoks[i]}
            cov = hit = 0
            for w in P.toks[i]:
                trs = lex.get(w)
                if trs:
                    cov += 1
                    hit += any(f[:5] in fp for f, _ in trs)
            if cov:
                out.append(hit / cov)
        return float(np.mean(out)) if out else None

    def describe(self, parent: str, idx: list[int], budget: int) -> dict:
        P = self.T if parent == "ft_topk" else self.R
        k = len(idx)
        ls = sorted(P.lens[i] for i in idx)
        srcs = [P.src[i] for i in idx]
        d = {"parent": parent, "rows": k, "len_mean": sum(ls) / k, "len_median": ls[k // 2],
             "length_histogram": {str(L): c for L, c in sorted(collections.Counter(ls).items())},
             "sources": dict(sorted(collections.Counter(P.lab[i] for i in idx).items(), key=lambda x: -x[1])),
             "source_words": P.E.tokens(idx), "source_word_types": P.E.types(idx),
             "french_tokens": P.F.tokens(idx), "french_budget": budget,
             "french_word_types_at_budget": P.F.types_at(idx, budget)}
        stat = "rep (over ft_topk)" if parent == "ft_topk" else "rep_rand (over ft_random)"
        d["parent_repeated_8gram_share"] = {"statistic": stat, "mean": sum(P.rep[i] for i in idx) / k,
                                            "rows_above_zero": sum(1 for i in idx if P.rep[i] > 0) / k}
        wr = rep_within([P.toks[i] for i in idx])
        d["within_set_repeated_8gram_share"] = {"mean": sum(wr) / k, "rows_above_zero": sum(1 for x in wr if x > 0) / k}
        d["duplicate_source_fraction"] = 1 - len(set(srcs)) / k
        d["duplicate_source_fraction_a17"] = 1 - len({a17_key(x) for x in srcs}) / k
        tb = sum(1 for i in idx if (P.src[i], P.tgt[i]) in self.top_pairs)
        d["top_band_rows_exact_pair"] = {"rows": tb, "share": tb / k}
        if self.qe is not None:
            q = self.qe["arr"][parent][np.asarray(idx, dtype=np.int64)]
            d["qe"] = {"mean": float(np.nanmean(q)) if np.any(~np.isnan(q)) else None,
                       "unmatched_rows": int(np.isnan(q).sum()),
                       "share_at_or_above_top_band_threshold": (float(np.mean(q[~np.isnan(q)] >= self.qe["threshold"]))
                                                                if self.qe["threshold"] is not None and np.any(~np.isnan(q)) else None)}
        else:
            d["qe"] = "n.m.: no --qe-tsv"
        d["literalness"] = self.literalness(P, idx) if self.lex is not None else "n.m.: no --pretrain-fr"
        if self.ref_lex is not None:          # the same measure over the vocab memo's own lexicon
            d["literalness_with_lex_pkl"] = self.literalness(P, idx, self.ref_lex)
        if self.pre_hashes is not None:
            keys = np.asarray([nkey(x) for x in srcs], dtype=np.uint64)
            d["source_in_pretraining"] = float(np.isin(keys, self.pre_hashes).mean())
        if self.pair_hashes is not None:
            keys = np.asarray([h64(P.src[i], P.tgt[i]) for i in idx], dtype=np.uint64)
            d["pair_in_pretraining"] = float(np.isin(keys, self.pair_hashes).mean())
        else:
            d["pair_in_pretraining"] = "n.m.: no --pretrain-fr"
        d["news_coverage"] = ({f"k{kk}": self.cover(P, idx, self.nc, kk) for kk in (1, 5, 20)}
                              if self.nc is not None else "n.m.: no --news-en")
        d["heldout_un_coverage_k5"] = self.cover(P, idx, self.ho, 5)
        hs, ht = set(self.ho_src), set(self.ho_tgt)
        d["heldout_un_shared_lines"] = {"source": sum(1 for x in set(srcs) if x in hs),
                                        "target": sum(1 for x in {P.tgt[i] for i in idx} if x in ht)}
        return d

    # ---- per-set checks (spec 6.2-6.5) and cross-set invariants (6.7): any failure is a STOP ----
    def check_sets(self, sets: dict, hist_dd: collections.Counter):
        """sets: file name -> {draft, parent, idx, budget}. Needs descriptive_inputs() (heldout_un)."""
        T, R = self.T, self.R
        for fname, s in sets.items():
            P = T if s["parent"] == "ft_topk" else R
            idx = s["idx"]
            if len(set(idx)) != len(idx) or list(idx) != sorted(idx):
                raise Stop(f"{fname}: index list is not strictly ascending")
            if any(P.lab[i] != SOURCE for i in idx):
                raise Stop(f"{fname}: holds rows that are not '{SOURCE}' in {s['parent']}")
            hist = collections.Counter(P.lens[i] for i in idx)
            want_h = hist_dd if s["draft"] in DEDUP else self.un_hist
            if hist != want_h:
                raise Stop(f"{fname}: its length histogram differs from the one it must share")
            if s["draft"] in ("R0", "A_narrow") and any(R.rep[i] > 0 for i in idx):
                raise Stop(f"{fname}: a row has rep_rand > 0")
            if s["draft"] in ("div_narrow", "div_narrow_dd", "div_dd") and any(T.rep[i] != 0.0 for i in idx):
                raise Stop(f"{fname}: a row has rep > 0")
            if s["draft"] == "rep_dedup" and any(T.rep[i] <= 0 for i in idx):
                raise Stop(f"{fname}: a row has rep = 0")
            if s["draft"] not in ANCHORS and len({P.src[i] for i in idx}) != len(idx):
                raise Stop(f"{fname}: has duplicate source lines")
            if P.F.tokens(idx) <= s["budget"]:
                raise Stop(f"{fname}: {P.F.tokens(idx):,} French tokens, not more than its budget {s['budget']:,} "
                           f"(the French-type measure is undefined)")
        dd = [len(s["idx"]) for s in sets.values() if s["draft"] in ("rep_dedup", "div_dd")]
        if len(set(dd)) > 1:
            raise Stop("rep_dedup and div_dd differ in size")
        self.ok("every set: ascending, UN only, its shared histogram, no duplicate source (new sets), "
                "French tokens above budget")
        self.load_heldout()
        hs, ht = set(self.ho_src), set(self.ho_tgt)
        for fname, s in sets.items():
            P = T if s["parent"] == "ft_topk" else R
            ns_ = sum(1 for i in s["idx"] if P.src[i] in hs)
            nt_ = sum(1 for i in s["idx"] if P.tgt[i] in ht)
            if ns_ or nt_:
                raise Stop(f"{fname}: shares {ns_:,} source and {nt_:,} target lines with heldout_un (spec 6.7)")
        self.ok("no set shares a source or target line with heldout_un")
        # the fast measures against their literal definitions (spec D11, D12)
        for fname, s in sets.items():
            P = T if s["parent"] == "ft_topk" else R
            if (P.E.types(s["idx"]) != py_types(s["idx"], P.toks)
                    or P.F.types_at(s["idx"], s["budget"]) != py_types_at(s["idx"], P.ftoks, s["budget"])):
                raise Stop(f"{fname}: the fast type counts disagree with the literal definitions")
        self.ok("fast source/French type counts equal the literal definitions on every set")


# ======================================================================================
# modes
# ======================================================================================
class Writer:
    """Temporary files beside their final names; published only by publish()."""

    def __init__(self, d: Path):
        self.d = d
        self.tmp: list[tuple[Path, Path]] = []
        self.sha: dict[str, str] = {}
        self.order: list[str] = []
        self.made_dirs: list[Path] = []
        self.removed: list[str] = []

    def _tmp(self, rel: str) -> tuple[Path, Path]:
        final = self.d / rel
        if not final.parent.is_dir():
            final.parent.mkdir(parents=True)
            self.made_dirs.append(final.parent)
        return final.parent / f".{final.name}.tmp", final

    def put(self, rel: str, data: bytes):
        t, final = self._tmp(rel)
        t.write_bytes(data)
        self.tmp.append((t, final))
        self.sha[rel] = hashlib.sha256(data).hexdigest()
        self.order.append(rel)

    def put_rows(self, rel: str, rows: list[str], idx):
        t, final = self._tmp(rel)
        h = hashlib.sha256()
        with open(t, "w", encoding="utf-8", newline="\n") as f:
            for i in idx:
                f.write(rows[i] + "\n")
                h.update((rows[i] + "\n").encode("utf-8"))
        self.tmp.append((t, final))
        self.sha[rel] = h.hexdigest()
        self.order.append(rel)

    def publish(self, last: str, last_bytes_fn, remove_first: Path | None = None, stale=(), drop_caches=False):
        """In this order: the files of the previous build that this one does not write (while the old
        manifest still lists them, so an interruption leaves a build --verify refuses); the old
        manifest; every trainer cache (TranslationDataset loads <set>.en.cached_<len>.npz by its name
        alone, so a cache of replaced text would train the old set); the renames; the new manifest."""
        keep = set(self.order) | {last}
        for rel in stale:
            if rel not in keep and DATA_FILE.fullmatch(rel) and (self.d / rel).is_file():
                (self.d / rel).unlink()
                self.removed.append(rel)
        if remove_first is not None and remove_first.exists():
            remove_first.unlink()            # an interrupted publish then leaves no manifest at all
        if drop_caches:
            for c in sorted(self.d.glob("*.cached_*.npz")):
                c.unlink()
                self.removed.append(c.name)
        for t, final in self.tmp:
            os.replace(t, final)
        self.tmp.clear()
        t, final = self._tmp(last)
        t.write_bytes(last_bytes_fn())
        self.tmp.append((t, final))
        os.replace(t, final)
        self.tmp.clear()

    def cleanup(self):
        for t, _ in self.tmp:
            t.unlink(missing_ok=True)
        self.tmp.clear()
        for p in reversed(self.made_dirs):
            try:
                p.rmdir()
            except OSError:
                pass


# ---- what a directory holds, checked before anything is read or written ------------------
def read_json_object(p: Path) -> dict:
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Stop(f"{p} cannot be read as JSON ({e}): not an E0.6 file, so it is left untouched; choose another directory")
    if not isinstance(v, dict):
        raise Stop(f"{p} is not a JSON object: not an E0.6 file, so it is left untouched; choose another directory")
    return v


def is_e06_manifest(man: dict, dry: bool) -> bool:
    """A finished E0.6 production manifest (dry = False) or dry-run manifest (dry = True), and
    nothing else: no other experiment's completion flag."""
    w = man.get("written")
    return (isinstance(man.get("e06"), dict) and man.get("e06_complete") is True
            and not any(OTHER_DONE.fullmatch(k) for k in man if k != "e06_complete")
            and (man.get("dry_run") is True) == dry and man.get("for_training") is (not dry)
            and isinstance(w, list) and all(isinstance(x, str) for x in w))


def guard_dir(a, d: Path, kind: str) -> list[str]:
    """kind: "production", "dry_run" or "feasibility". STOP if d is an input directory, or holds
    another kind of output or a manifest that is not this kind's E0.6 manifest (which is then
    left untouched, never deleted). -> the data files the previous build of this kind in d lists."""
    rd = d.resolve()
    named = {"--src-dir": a.src_dir, "--e05-dir": a.e05_dir}
    for flag in ("pretrain_en", "pretrain_fr", "qe_tsv", "news_en", "ref_lex"):
        p = getattr(a, flag, None)
        if p is not None:
            named[f"the directory of --{flag.replace('_', '-')}"] = Path(p).parent
    for what, p in named.items():
        if p is not None and Path(p).resolve() == rd:
            raise Stop(f"the output directory {d} is {what}: nothing is ever written into an input directory "
                       f"(E0.5's manifest.json and anchor files are pinned inputs)")
    if not d.exists():
        return []
    if not d.is_dir():
        raise Stop(f"{d} exists and is not a directory")
    feas = (d / FEAS_REPORT).exists() or any(d.glob("FEASIBILITY_*.npy"))
    if kind == "feasibility":
        for f in (MANIFEST, DRY_MANIFEST):
            if (d / f).exists():
                raise Stop(f"{d} holds a {f}: feasibility output is never written beside a build")
        return []
    if kind == "production":
        if feas or (d / NOT_FOR_TRAINING).exists() or (d / DRY_MANIFEST).exists():
            raise Stop(f"{d} holds --feasibility or --dry-run output (NOT_FOR_TRAINING): a production build never "
                       f"shares its directory")
        own, dry = MANIFEST, False
    else:
        if feas or (d / MANIFEST).exists():
            raise Stop(f"{d} holds a production build (manifest.json) or --feasibility output: a dry run never "
                       f"shares its directory")
        own, dry = DRY_MANIFEST, True
    if not (d / own).exists():
        return []
    man = read_json_object(d / own)
    if not is_e06_manifest(man, dry):
        raise Stop(f"{d / own} is not an E0.6 {'dry-run' if dry else 'production'} manifest (it needs an \"e06\" "
                   f"record, \"e06_complete\": true and no other experiment's flag such as \"e05_complete\"): it is "
                   f"left untouched and nothing is written; choose another --dir")
    return [x for x in man["written"] if DATA_FILE.fullmatch(x)]


def load_decisions(p: Path | None):
    if p is None:
        return None, None
    try:
        b = p.read_bytes()
        return json.loads(b.decode("utf-8")), hashlib.sha256(b).hexdigest()
    except (OSError, ValueError) as e:
        raise Stop(f"cannot read the decisions file {p}: {e}")


def interpreter() -> dict:
    return {"python": sys.version.split()[0], "unidata_version": unicodedata.unidata_version,
            "numpy": np.__version__, "machine": platform.machine()}


def load_expected(path: Path, dec_sha: str | None) -> dict:
    """The registered dry run (spec A8), read and checked before anything is built."""
    try:
        blob = path.read_bytes()
        man = json.loads(blob.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise Stop(f"cannot read the dry run's manifest {path}: {e}")
    if not (isinstance(man, dict) and is_e06_manifest(man, dry=True) and isinstance(man.get("sha256"), dict)):
        raise Stop(f"{path} is not a finished E0.6 dry run's manifest ({DRY_MANIFEST}, \"dry_run\": true, "
                   f"\"e06_complete\": true)")
    theirs = man["e06"].get("decisions_sha256")
    if theirs != dec_sha:
        raise Stop(f"{path} was built from the decisions file with sha256 {theirs}, this build from {dec_sha}: "
                   f"the box builds from the registered decisions file only")
    return {"path": path, "sha256": hashlib.sha256(blob).hexdigest(),
            "table": {k: v for k, v in man["sha256"].items() if DATA_FILE.fullmatch(k)},
            "built_with": man["e06"].get("built_with")}


def compare_expected(exp: dict, have: dict):
    mine = {k: v for k, v in have.items() if DATA_FILE.fullmatch(k)}
    want = exp["table"]
    diff = sorted(f for f in set(want) | set(mine) if want.get(f) != mine.get(f))
    if diff:
        raise Stop(f"the build does not reproduce the dry run {exp['path']} byte for byte (spec A8): " + ", ".join(diff))


def refused(unset: list[str], bad: list[str], head: str) -> int:
    print(f"REFUSED: {head}:")
    for x in unset + bad:
        print(f"  {x}")
    print("Nothing written.")
    return EX_REFUSED


def build_production(a, dry: bool = False) -> int:
    kind = "dry_run" if dry else "production"
    d = a.dir
    stale = guard_dir(a, d, kind)          # before anything is read
    raw, dec_sha = load_decisions(a.decisions)
    val, unset, bad = resolve_decisions(raw if raw is not None else {}, production=True)
    if raw is None:
        unset = ["the decisions file (--decisions): none given, so every decision is unset"] + unset
    if unset or bad:
        return refused(unset, bad, "the decisions file does not settle every set-defining decision (spec 7.1-7.2)")
    exp = load_expected(a.expect_sha256_from, dec_sha) if a.expect_sha256_from is not None else None
    od2 = val["od2_rule"]
    if od2 == "c2" and a.pretrain_fr is None:
        raise Stop("od2_rule = 'c2' orders rows by the French pretraining rank: --pretrain-fr is a construction "
                   "input then (pinned b0a6ba91...)")
    uni = {"interpreter": unicodedata.unidata_version, "od10_unidata": val["od10_unidata"],
           "equal": unicodedata.unidata_version == val["od10_unidata"]}
    if not uni["equal"] and not dry:
        raise Stop(f"this interpreter's Unicode tables are {unicodedata.unidata_version}, the registered build's "
                   f"{val['od10_unidata']} (od10_unidata): \\w+ tokens, and so every set, could differ")
    b = Builder(a, val, kind)
    b.check_pins(need_fr=a.pretrain_fr is not None)
    if uni["equal"]:
        b.ok(f"interpreter unidata {unicodedata.unidata_version} = od10_unidata")
    b.prepare(want_hashes=True, need_fr_ranks=(od2 == "c2"))
    T, R = b.T, b.R
    if a.news_en is not None:
        b.nc = b.ref_counts(toks(x) for x in read_lines(a.news_en))
    B, Bdd = val["fr_budget_214489"], val["fr_budget_198333"]
    sets: dict[str, dict] = {}            # file name -> {draft, parent, idx, budget, build}

    def add(draft, fname, parent, idx, budget, info):
        sets[fname] = {"draft": draft, "parent": parent, "idx": idx, "budget": budget, "build": info}

    add("ft_un_div", "ft_un_div", "ft_topk", b.kd, B, {"rule": "E0.5's UN split (anchor)"})
    add("ft_un_rep", "ft_un_rep", "ft_topk", b.kr, B, {"rule": "E0.5's UN split (anchor)"})
    add("ft_un_random", "ft_un_random", "ft_random", b.ku, B,
        {"rule": "E0.5's matched random UN draw (anchor)", "tightest_bucket_slack": b.slack_un_random})
    seeds = [SEED] if val["od12_draws"] == "one" else list(val["seeds"])
    od6 = val["od6_topband"]
    for s in seeds:
        sfx = "" if val["od12_draws"] == "one" else f"_d{s}"
        idx, info = b.build_r0(od6, s)
        add("R0", SET_FILE["R0"] + sfx, "ft_random", idx, B, info)
        st = {}
        idx, info = b.build_a_narrow(od6, val["od3_caps"], s, st)
        info["nn_match"] = st
        if od6 == "remove":
            info["note"] = ("od6 = remove: the caps are counted over the reduced pool and were registered for it "
                            "(od3_caps_for_reduced_pool); A_narrow_42_58.npy does not describe this set (spec A21)")
        add("A_narrow", SET_FILE["A_narrow"] + sfx, "ft_random", idx, B, info)
    rep_dd, div_dd, dinfo = b.build_dedup(val["od4_duplicate"])
    add("rep_dedup", SET_FILE["rep_dedup"], "ft_topk", rep_dd, Bdd, dinfo)
    add("div_dd", SET_FILE["div_dd"], "ft_topk", div_dd, Bdd, dinfo)
    grids = {}
    measures = ["src_types"] if val["od1_coverage"] == "a" else ["src_types", "fr_types"]
    rule, grid, tol = od2, val["od2_grid"][od2], val["tolerance"]
    want = []
    if val["od5"] in ("a", "b"):
        want.append(("div_narrow", b.kr, B))
    if val["od5"] in ("b", "c"):
        want.append(("div_narrow_dd", rep_dd, Bdd))
    for draft, target_rows, budget in want:
        ns = b.narrow_setup(target_rows, budget)
        pts = b.grid(ns, rule, grid)
        best, dev = choose(pts, measures, ns["targets"], rule)
        table = {"rule": rule, "grid": grid, "measures": measures, "targets": ns["targets"],
                 "tolerance": fr(tol), "forced": ns["forced"], "points": pts}
        grids[SET_FILE[draft]] = table
        if best is None:
            raise Stop(f"{draft}: no grid point of rule ({rule}) is feasible and measurable "
                       f"({len(pts)} points; forced rows {ns['forced']['rows']:,} in lengths {ns['forced']['lengths']})")
        if not within(dev, tol):
            raise Stop(f"{draft}: the best point of rule ({rule}) is {float(dev):.4%} from the target on "
                       f"{'/'.join(measures)}, outside the tolerance {float(tol):.2%} "
                       f"({ {k: best[k] for k in best if k in ('Ke', 'Kf', 'k', 'lambda')} })")
        st = {}
        idx = b.rebuild_point(ns, rule, grid, best, st if rule == "a" else None)
        m = b.measure(idx, budget)
        if any(m[x] != best[x] for x in ("src_types", "fr_types")):
            raise Stop(f"{draft}: rebuilding the chosen point did not reproduce its measured coverage")
        info = {"rule": rule, "chosen": {k: best[k] for k in best if k in ("Ke", "Kf", "k", "q", "lambda")},
                "max_relative_deviation": fr(dev), "measures": measures, "targets": ns["targets"],
                "forced": ns["forced"], **({"nn_match": st} if st else {})}
        if rule in OD2C_DEFINITION:
            info["definition"] = OD2C_DEFINITION[rule]
        add(draft, SET_FILE[draft], "ft_topk", idx, budget, info)

    # every set's checks first (cheap), then the descriptive inputs (QE join, lexicon: minutes)
    b.check_sets(sets, collections.Counter(T.lens[i] for i in rep_dd))
    b.descriptive_inputs()
    desc = b.descriptives_record()
    if dry and not desc["complete"]:
        raise Stop(f"a dry run fills every column of the registered table, but these are missing: {', '.join(desc['missing'])}")
    descr = {fname: {"draft_name": s["draft"], "build": s["build"], **b.describe(s["parent"], s["idx"], s["budget"])}
             for fname, s in sets.items()}
    overlaps = overlap_table(b, sets)

    w = Writer(d)
    try:
        for fname, s in sets.items():
            P = T if s["parent"] == "ft_topk" else R
            w.put_rows(f"{fname}.en", P.src, s["idx"])
            w.put_rows(f"{fname}.fr", P.tgt, s["idx"])
        for fname, s in sets.items():
            w.put(f"index/{fname}.npy", npy_bytes(s["idx"]))
        for f in ("heldout_un.en", "heldout_un.fr"):
            w.put(f, (a.src_dir / f).read_bytes())
        if dry:
            w.put(NOT_FOR_TRAINING, ("This directory is an e06_build.py --dry-run: the registered numbers and sha256 "
                                     "come from it, but nothing trains on it. The run of record is built on the box, "
                                     "with --expect-sha256-from this directory's dry_run_manifest.json.\n").encode("utf-8"))
        if exp is not None:
            compare_expected(exp, w.sha)
            b.ok(f"every set, index and heldout_un file reproduces the sha256 of the dry run {exp['path'].name} "
                 f"(that manifest's sha256 {exp['sha256']})")
            exp_rec = {"matched": True, "dry_run_manifest": exp["path"].name, "dry_run_manifest_sha256": exp["sha256"],
                       "dry_run_built_with": exp["built_with"]}
        elif dry:
            exp_rec = None
        else:
            exp_rec = {"matched": False, "skipped": "--skip-expected-sha256: not tied to a registered dry run "
                                                    "(tests only, spec A8); --verify refuses this build"}
        last = DRY_MANIFEST if dry else MANIFEST
        written = list(w.order) + [last]
        man = {"written": written, "sha256": dict(w.sha),
               "heldout_sets": {"heldout_un": {"n": len(b.ho_src)}},
               "sets": {fname: {"draft_name": s["draft"], "parent": s["parent"], "index_file": f"index/{fname}.npy",
                                "rows": len(s["idx"])} for fname, s in sets.items()},
               "for_training": not dry, **({"dry_run": True} if dry else {}),
               "e06": {"decisions": decisions_record(val), "decisions_sha256": dec_sha,
                       "inputs": b.inputs, "src_pinned": b.pinned, "unpinned_inputs": b.unpinned,
                       "allow_unpinned_src": bool(a.allow_unpinned_src), "built_with": interpreter(),
                       "unidata": uni, "expected_sha256": exp_rec, "descriptives": desc,
                       "set_names": {k: v for k, v in SET_FILE.items()},
                       "median_rep": b.median,
                       "pools": {"diverse_candidates": len(b.div_cand), "repetitive_candidates": len(b.rep_cand),
                                 "RU0": len(b.ru0), "RU0_rows_in_top_band": len(b.ru0) - len(b.r0_pool("remove")),
                                 "ft_random_un_rows": len(b.rnd_un), "pretraining_lines": b.pre_lines,
                                 "pretraining_word_types": len(b.rank)},
                       "sets": descr, "overlaps": overlaps, "grids": grids, "checks": b.checks,
                       "rng_register": RNG_REGISTER,
                       "qe": ({k: v for k, v in b.qe.items() if k != "arr"} if b.qe is not None else None),
                       "lexicon_entries": len(b.lex) if b.lex is not None else None,
                       "lexicon_vs_lex_pkl": b.lex_compare},
               "e06_complete": True}
        blob = (json.dumps(man, indent=1) + "\n").encode("utf-8")
        w.publish(last, lambda: blob, remove_first=d / last, stale=stale, drop_caches=True)
    finally:
        w.cleanup()                       # after a STOP or a crash: no temporary file is left behind
    for fname, s in descr.items():
        print(f"{fname:<24} {s['rows']:>8,} rows | {s['source_word_types']:>7,} src types | "
              f"{s['french_word_types_at_budget'] or 0:>7,} fr types @{s['french_budget']:,} | "
              f"dup {100 * s['duplicate_source_fraction']:.2f}% | rep {s['parent_repeated_8gram_share']['mean']:.3f}")
    if w.removed:
        print(f"removed before publishing (previous build's files it no longer writes, trainer caches): "
              f"{', '.join(w.removed)}")
    if b.unpinned:
        print(f"NOTE: built from UNPINNED inputs ({', '.join(b.unpinned)}); recorded in the manifest.")
    if not desc["complete"]:
        print(f"NOTE: descriptives incomplete (n.m.): {', '.join(desc['missing'])}")
    if b.lex_compare is not None and not b.lex_compare["identical"]:
        c = b.lex_compare
        print(f"NOTE: the rebuilt lexicon differs from lex.pkl: {c['only_in_built']} entries only rebuilt, "
              f"{c['only_in_lex_pkl']} only in lex.pkl, {c['entries_with_other_translations']} with other translations, "
              f"{c['entries_with_same_translations_in_another_order']} reordered")
    if not uni["equal"]:
        print(f"NOTE: dry run on Unicode {uni['interpreter']}, the box's od10_unidata is {uni['od10_unidata']}; the box "
              f"build STOPs unless it reproduces this dry run's sha256 (spec A8)")
    if exp_rec is not None and not exp_rec["matched"]:
        print("NOTE: --skip-expected-sha256: not tied to a dry run; --verify refuses this build")
    print(f"{'DRY RUN (not for training)' if dry else 'built'}: {len(written)} files in {d}"
          + (f"; register the sha256 of {d / DRY_MANIFEST}" if dry else ""))
    return 0


def overlap_table(b: Builder, sets: dict) -> dict:
    T, R = b.T, b.R
    pairs, srcs = {}, {}
    for fname, s in sets.items():
        P = T if s["parent"] == "ft_topk" else R
        pairs[fname] = {(P.src[i], P.tgt[i]) for i in s["idx"]}
        srcs[fname] = {P.src[i] for i in s["idx"]}
    names = list(sets)
    out = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            p, q = names[x], names[y]
            out[f"{p} & {q}"] = {"exact_pairs": len(pairs[p] & pairs[q]), "source_lines": len(srcs[p] & srcs[q])}
    return out


def build_feasibility(a) -> int:
    F = a.out
    if a.dir is not None and F.resolve() == a.dir.resolve():
        raise Stop("--out is the production directory --dir: feasibility output never shares it")
    guard_dir(a, F, "feasibility")          # before anything is read
    raw, dec_sha = load_decisions(a.decisions)
    val, _, bad = resolve_decisions(raw if raw is not None else {}, production=False)
    if bad:
        return refused([], bad, "the decisions file's parameters are invalid")
    b = Builder(a, val, "feasibility")
    b.check_pins(need_fr=a.pretrain_fr is not None)
    b.prepare(want_hashes=True, need_fr_ranks=a.pretrain_fr is not None)
    T, R = b.T, b.R
    if a.news_en is not None:
        b.nc = b.ref_counts(toks(x) for x in read_lines(a.news_en))
    B = val["fr_budget_214489"]
    Bdd = val["fr_budget_198333"] or FEAS_FR_BUDGET_198333
    cap0 = val["od3_caps"] or list(REFERENCE_CAPS)
    seeds = val["seeds"] or [42, 1, 2, 3, 4]
    b.descriptive_inputs()
    files: dict[str, tuple] = {}          # FEASIBILITY_ file -> (parent, idx, budget)
    results: dict[str, dict] = {}

    def keep(tag: str, parent: str, idx, budget: int, info: dict):
        fn = f"FEASIBILITY_{tag}.npy"
        files[fn] = (parent, idx, budget)
        results[tag] = {"file": fn, "build": info, **b.describe(parent, idx, budget)}

    def r0(od6, s, tag):
        idx, info = b.build_r0(od6, s)
        keep(tag, "ft_random", idx, B, info)

    def an(od6, s, tag):
        st = {}
        idx, info = b.build_a_narrow(od6, cap0, s, st)
        keep(tag, "ft_random", idx, B, {**info, "nn_match": st})

    anchors = {}
    for n, p, idx in (("ft_un_div", "ft_topk", b.kd), ("ft_un_rep", "ft_topk", b.kr), ("ft_un_random", "ft_random", b.ku)):
        fn = f"FEASIBILITY_{n}__anchor.npy"           # for --measure: the contrasts' anchor sides
        files[fn] = (p, idx, B)
        anchors[n] = {"file": fn, **b.describe(p, idx, B)}
    for od6 in ("keep", "remove"):
        for s in seeds:
            for nm, fn in (("ft_un_r0", r0), ("ft_un_a_narrow", an)):
                tag = f"{nm}__od6-{od6}__seed{s}"
                try:
                    fn(od6, s, tag)
                except Stop as e:
                    results[tag] = {"failed": str(e)}

    # OD 3: the cap report, measurement only (spec 7.2: +-4,000 around the caps in steps of 1,000),
    # over the full pool and over the pool without top-band rows, for which OD 6 'remove' must
    # re-decide the caps (spec A21)
    def around(c):
        return [c + 1000 * j for j in range(-4, 5) if c + 1000 * j > 0] if c is not None else [None]
    rg = val["od3_report_grid"] or {"Ke": around(cap0[0]), "Kf": around(cap0[1])}
    od3 = {}
    for od6 in ("keep", "remove"):
        od3[od6] = []
        for ke in rg["Ke"]:
            for kf in rg["Kf"]:
                try:
                    idx, info = b.build_a_narrow(od6, [ke, kf], SEED)
                    od3[od6].append({"Ke": ke, "Kf": kf, "feasible": True, "capped_pool_rows": info["capped_pool_rows"],
                                     "src_types": R.E.types(idx), "fr_types": R.F.types_at(idx, B)})
                except Stop as e:
                    od3[od6].append({"Ke": ke, "Kf": kf, "feasible": False, "why": str(e).split("\n")[0]})
    dedup = {}
    for od4 in ("a", "b"):
        try:
            rep_dd, div_dd, info = b.build_dedup(od4)
            dedup[od4] = (rep_dd, div_dd)
            keep(f"ft_un_rep_dedup__od4-{od4}", "ft_topk", rep_dd, Bdd, info)
            keep(f"ft_un_div_dd__od4-{od4}", "ft_topk", div_dd, Bdd, info)
        except Stop as e:
            results[f"ft_un_rep_dedup__od4-{od4}"] = {"failed": str(e)}
    grids = {}
    od1s = {"a": ["src_types"], "b": ["src_types", "fr_types"]}
    if b.nc is not None:
        od1s["d"] = ["nc_cov5"]
    targets_list = [("ft_un_div_narrow", b.kr, B)]
    if "a" in dedup:
        targets_list.append(("ft_un_div_narrow_dd", dedup["a"][0], Bdd))
    for fname, target_rows, budget in targets_list:
        try:
            ns = b.narrow_setup(target_rows, budget)
        except Stop as e:
            results[fname] = {"failed": str(e)}
            continue
        grids[fname] = {"targets": ns["targets"], "forced": ns["forced"], "budget": budget, "rules": {}}
        for rule in ("a", "b", "c", "c2"):
            if rule == "c2" and a.pretrain_fr is None:
                grids[fname]["rules"][rule] = "n.m.: c2 needs --pretrain-fr (the French pretraining ranks)"
                continue
            grid = (val["od2_grid"] or {}).get(rule) or default_grid(rule, ns["po"])
            pts = b.grid(ns, rule, grid)
            choice = {}
            for od1, ms in od1s.items():
                cand = pts
                if od1 == "a" and rule in ("a", "b"):         # spec A12: no French cap under OD 1(a)
                    cand = [p for p in pts if p["Kf"] is None]
                best, dev = choose(cand, ms, ns["targets"], rule)
                if best is None:
                    choice[od1] = {"feasible_point": False,
                                   **({"note": "OD 1(a): only Kf = null points (spec A12)"} if cand is not pts else {})}
                    continue
                tag = f"{fname}__od2-{rule}__od1-{od1}"
                choice[od1] = {"chosen": {k: best[k] for k in best if k in ("Ke", "Kf", "k", "q", "lambda")},
                               "max_relative_deviation": fr(dev),
                               "within_tolerance": within(dev, val["tolerance"]), "file": f"FEASIBILITY_{tag}.npy"}
                idx = b.rebuild_point(ns, rule, grid, best)
                keep(tag, "ft_topk", idx, budget, {"rule": rule, "od1": od1, **choice[od1]})
                results[tag]["overlap_with_ft_un_div"] = len(set(idx) & set(b.kd))
            choice["c"] = "n.m.: OD 1(c) needs the registered piece-counting rule (spec A28); see --measure for piece counts"
            if "d" not in od1s:
                choice["d"] = "n.m.: no --news-en"
            grids[fname]["rules"][rule] = {"grid": grid, "tie_rule": OD2_TIE[rule], "points": pts, "by_od1": choice}

    w = Writer(F)
    try:
        for fn, (parent, idx, budget) in files.items():
            w.put(fn, npy_bytes(idx))
        w.put(NOT_FOR_TRAINING, ("This directory is e06_build.py --feasibility output: measurement only, never "
                                 "for training. Production sets come from a production build.\n").encode("utf-8"))
        rep = {"for_training": False, "decisions": "all options", "parameters_from": a.decisions.name if a.decisions else None,
               "decisions_sha256": dec_sha, "parameters": decisions_record(val), "tolerance_used": fr(val["tolerance"]),
               "fr_budget_198333_used": Bdd, "od3_caps_used": cap0, "seeds_used": seeds,
               "inputs": b.inputs, "src_pinned": b.pinned, "unpinned_inputs": b.unpinned,
               "built_with": interpreter(), "checks": b.checks, "descriptives": b.descriptives_record(),
               "lexicon_vs_lex_pkl": b.lex_compare, "anchors": anchors,
               "results": results, "od3_cap_report": od3, "grids": grids, "rng_register": RNG_REGISTER,
               "files": {fn: w.sha[fn] for fn in w.order}}
        blob = (json.dumps(rep, indent=1) + "\n").encode("utf-8")
        w.publish(FEAS_REPORT, lambda: blob)
    finally:
        w.cleanup()
    print(f"feasibility (NOT FOR TRAINING): {len(files)} index files and {FEAS_REPORT} in {F}")
    return 0


# ======================================================================================
# --measure: OD 16's piece-level check and the newstest2014 coverage (spec section 9)
# ======================================================================================
CONTRASTS = (("N", "R0", "A_narrow"), ("S", "A_narrow", "ft_un_div"),
             ("C1-beyond-coverage", "div_narrow", "ft_un_rep"), ("C1-beyond-duplicates", "div_dd", "rep_dedup"),
             ("C1-beyond-both", "div_narrow_dd", "rep_dedup"), ("P", "R0", "ft_un_div"),
             ("Q1'", "ft_un_div", "ft_un_rep"), ("Q2'", "ft_un_random", "ft_un_div"), ("V", "ft_un_div", "div_narrow"))
# Machine_translation/src/data/dataset.py, which the measures mirror (spec A28's third point)
TRAINER_LINES = {"strip": "src_line = src_line.strip()",
                 "encode": "sp.encode(src_lines, add_bos=True, add_eos=True, out_type=int)",
                 "length filter": "if len(src_ids) > max_tokens or len(tgt_ids) > max_tokens:"}
PROFILE = (("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), ("21-100", 21, 100), (">100", 101, None))


def load_index(p: Path) -> list[int]:
    try:
        a = np.load(p, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise Stop(f"cannot load {p}: {e}")
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise Stop(f"{p} is not a 1-d integer index file")
    return [int(x) for x in a]


def load_measured(B: Path) -> dict:
    """A finished build, dry run or feasibility directory: every file it lists must still have its
    recorded sha256. -> kind, the record's sha256, the inputs it recorded, and its sets."""
    rec_file = next((f for f in (MANIFEST, DRY_MANIFEST, FEAS_REPORT) if (B / f).is_file()), None)
    if rec_file is None:
        raise Stop(f"{B} holds no {MANIFEST}, {DRY_MANIFEST} or {FEAS_REPORT}: nothing to measure")
    blob = (B / rec_file).read_bytes()
    try:
        rec = json.loads(blob.decode("utf-8"))
    except ValueError as e:
        raise Stop(f"cannot read {B / rec_file}: {e}")
    sets = {}
    if rec_file in (MANIFEST, DRY_MANIFEST):
        if not (isinstance(rec, dict) and is_e06_manifest(rec, dry=rec_file == DRY_MANIFEST)):
            raise Stop(f"{B / rec_file} is not a finished E0.6 manifest")
        table = {x: rec.get("sha256", {}).get(x) for x in rec["written"][:-1]}
        for name, info in rec["sets"].items():
            sets[name] = {"parent": info["parent"], "draft": info["draft_name"], "file": info["index_file"]}
        inputs, kind = rec["e06"]["inputs"], ("dry_run" if rec_file == DRY_MANIFEST else "production")
    else:
        if not (isinstance(rec, dict) and rec.get("for_training") is False and isinstance(rec.get("files"), dict)):
            raise Stop(f"{B / rec_file} is not a feasibility report")
        table = rec["files"]
        for name, r in list(rec.get("anchors", {}).items()) + list(rec.get("results", {}).items()):
            if isinstance(r, dict) and "file" in r and "parent" in r:
                sets[name] = {"parent": r["parent"], "draft": None, "file": r["file"]}
        inputs, kind = rec["inputs"], "feasibility"
    bad = sorted(x for x, h in table.items() if not (B / x).is_file() or sha256_file(B / x) != h)
    bad += sorted(s["file"] for s in sets.values() if s["file"] not in table)
    if bad:
        raise Stop(f"{B}: changed since it was built (sha256 differs, missing, or not listed): " + ", ".join(bad))
    for s in sets.values():
        s["idx"] = load_index(B / s["file"])
    return {"kind": kind, "record": rec_file, "record_sha256": hashlib.sha256(blob).hexdigest(),
            "inputs": inputs, "sets": sets}


def encode_pieces(sp, lines: list[str], chunk: int = 20000) -> Side:
    """The trainer's view of each row (dataset.py): line.strip(), then the model's defaults with no
    sampling; BOS/EOS are control ids and are added by the trainer around these pieces."""
    s = Side.__new__(Side)
    lens, parts = [], []
    for k in range(0, len(lines), chunk):
        enc = sp.encode([x.strip() for x in lines[k:k + chunk]], out_type=int)
        lens += [len(e) for e in enc]
        parts.append(np.fromiter((t for e in enc for t in e), dtype=np.int32))
    s.n = len(lines)
    s.lens = np.asarray(lens, dtype=np.int64)
    s.off = np.zeros(s.n + 1, dtype=np.int64)
    np.cumsum(s.lens, out=s.off[1:])
    s.ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    return s


def piece_measures(side: Side, idx, nvocab: int, special: np.ndarray, unk: list[int], countable: int) -> dict:
    cnt = side.counts(idx, nvocab) if len(idx) else np.zeros(nvocab, dtype=np.int64)
    real = cnt.copy()
    real[special] = 0                     # spec A28: control and unknown ids never count as types
    nz = real[real > 0]
    return {"piece_tokens": int(cnt.sum()), "unknown_tokens": int(cnt[unk].sum()) if unk else 0,
            "piece_types": int(nz.size), "vocab_share": nz.size / countable if countable else None,
            "profile": {nm: int(((nz >= lo) & (True if hi is None else nz <= hi)).sum()) for nm, lo, hi in PROFILE}}


def trainer_check(mt: Path) -> dict:
    p = mt / "src" / "data" / "dataset.py"
    try:
        s = p.read_text(encoding="utf-8")
    except OSError as e:
        raise Stop(f"cannot read the trainer's dataset code {p}: {e}")
    missing = [k for k, v in TRAINER_LINES.items() if v not in s]
    if missing:
        raise Stop(f"{p} no longer tokenises the way these measures mirror (missing: {', '.join(missing)}): "
                   f"spec A28 must be re-checked before the piece counts mean anything")
    return {"file": str(p), "sha256": sha256_file(p), "lines_found": TRAINER_LINES}


def measure_mode(a) -> int:
    Bd, M = a.from_dir, a.out
    rB, rM = Bd.resolve(), M.resolve()
    if rM == rB or rB in rM.parents:
        raise Stop("--out is --from or inside it: the measures never write into the build they read (OD 16)")
    if a.src_dir is not None and a.src_dir.resolve() == rM:
        raise Stop("--out is --src-dir: nothing is ever written into an input directory")
    for f in (MANIFEST, DRY_MANIFEST, FEAS_REPORT, NOT_FOR_TRAINING):
        if (M / f).exists():
            raise Stop(f"{M} holds {f}: the measures are written only into a directory of their own")
    rec = load_measured(Bd)
    try:
        import sentencepiece as spm
    except ImportError:
        raise Stop("sentencepiece is not importable in this interpreter (on the Mac, ~/anaconda3/envs/py310/bin/python "
                   "has it): the piece-level measures need it (OD 16)")
    inputs, unpinned = {}, []

    def pin(name, path, want):
        if not path.is_file():
            raise Stop(f"MISSING input: {path}")
        have = sha256_file(path)
        inputs[name] = {"sha256": have, "pinned_sha256": want, "pinned": None if want is None else have == want}
        return have

    if pin("spm model", a.spm_model, PINNED_SPM) != PINNED_SPM:
        unpinned.append("spm model")
    parents = sorted({s["parent"] for s in rec["sets"].values()})
    for name in parents:
        for e in ("en", "fr"):
            f = f"{name}.{e}"
            want = (rec["inputs"].get(f) or {}).get("sha256")
            have = pin(f, a.src_dir / f, want)
            if have != want:
                raise Stop(f"{a.src_dir / f} is not the file {Bd} was built from (sha256 {have[:16]}..., recorded "
                           f"{str(want)[:16]}...)")
    if unpinned and not a.allow_unpinned_src:
        raise Stop("these inputs are not the pinned ones (spec 3.1): " + ", ".join(
            f"{f} sha256 {inputs[f]['sha256']}" for f in unpinned))
    if a.newstest_en is not None:
        pin("newstest2014 .en", a.newstest_en, None)          # to record (spec 3.1)
    trainer = trainer_check(a.mt_repo) if a.mt_repo is not None else "not checked (no --mt-repo)"
    sp = spm.SentencePieceProcessor()
    sp.Load(str(a.spm_model))
    nvocab = int(sp.get_piece_size())
    special = np.array([bool(sp.is_control(i)) or bool(sp.is_unknown(i)) for i in range(nvocab)], dtype=bool)
    unk = [i for i in range(nvocab) if sp.is_unknown(i)]
    countable = int((~special).sum())
    L = a.max_seq_len
    enc, cr, words, wvocab = {}, {}, {}, {}
    for name in parents:
        src, tgt = read_lines(a.src_dir / f"{name}.en"), read_lines(a.src_dir / f"{name}.fr")
        if len(src) != len(tgt):
            raise Stop(f"{name}: .en and .fr differ in line count")
        enc[name] = (encode_pieces(sp, src), encode_pieces(sp, tgt))
        # the trainer opens its files in text mode (universal newlines): a row holding "\r" would be
        # read as two rows there, and every later row misaligned
        cr[name] = np.fromiter((("\r" in x) or ("\r" in y) for x, y in zip(src, tgt)), dtype=bool, count=len(src))
        if a.newstest_en is not None:
            words[name] = Side(tokenize(src), wvocab)
    nt = None
    if a.newstest_en is not None:
        C = collections.Counter()
        lines = read_lines(a.newstest_en)
        for x in lines:
            C.update(toks(x))
        known = [(wvocab[w], c) for w, c in C.items() if w in wvocab]
        nt = {"ids": np.asarray([i for i, _ in known], dtype=np.int64), "n": np.asarray([c for _, c in known], dtype=np.int64),
              "tot": sum(C.values()), "lines": len(lines), "types": len(C)}
    out = {}
    for name, s in rec["sets"].items():
        E, F = enc[s["parent"]]
        idx = s["idx"]
        ia = np.asarray(idx, dtype=np.int64)
        kept = ia[(E.lens[ia] + 2 <= L) & (F.lens[ia] + 2 <= L)] if len(ia) else ia
        d = {"parent": s["parent"], "draft_name": s["draft"], "rows": len(idx), "rows_kept_by_trainer": int(len(kept)),
             "rows_over_max_seq_len": int(len(ia) - len(kept)), "rows_with_carriage_return": int(cr[s["parent"]][ia].sum()),
             "source": piece_measures(E, idx, nvocab, special, unk, countable),
             "target": piece_measures(F, idx, nvocab, special, unk, countable),
             "source_piece_tokens_kept": E.tokens(kept), "target_piece_tokens_kept": F.tokens(kept)}
        if nt is not None:
            cnt = words[s["parent"]].counts(idx, len(wvocab))
            d["newstest2014_coverage"] = {f"k{k}": 100 * int(nt["n"][cnt[nt["ids"]] >= k].sum()) / nt["tot"] if nt["tot"] else 0.0
                                          for k in (1, 5, 20)}
        else:
            d["newstest2014_coverage"] = "n.m.: no --newstest-en"
        out[name] = d
    contrasts = []
    if rec["kind"] != "feasibility":
        by = collections.defaultdict(list)
        for name, s in rec["sets"].items():
            by[s["draft"]].append(name)
        sfx = lambda n: (re.search(r"_d\d+$", n) or [None])[0]
        for cname, da, db in CONTRASTS:
            for fa in by.get(da, []):
                for fb in by.get(db, []):
                    if sfx(fa) and sfx(fb) and sfx(fa) != sfx(fb):
                        continue
                    row = {"contrast": cname, "a": fa, "b": fb}
                    for side in ("source", "target"):
                        ta, tb = out[fa][f"{side}_piece_tokens_kept"], out[fb][f"{side}_piece_tokens_kept"]
                        row[f"{side}_piece_tokens_kept"] = [ta, tb]
                        row[f"{side}_imbalance"] = (ta - tb) / tb if tb else None
                        row[f"more_{side}_tokens"] = fa if ta > tb else fb if tb > ta else "equal"
                        row[f"{side}_piece_types"] = [out[fa][side]["piece_types"], out[fb][side]["piece_types"]]
                    contrasts.append(row)
    spv = getattr(spm, "__version__", "unknown")
    report = {"for_training": False,
              "what": "OD 16: descriptive measures of finished sets; nothing here defines, tunes or changes a set",
              "from": str(Bd), "from_kind": rec["kind"], "from_record": rec["record"],
              "from_record_sha256": rec["record_sha256"], "inputs": inputs, "unpinned_inputs": unpinned,
              "allow_unpinned_src": bool(a.allow_unpinned_src), "built_with": {**interpreter(), "sentencepiece": spv},
              "trainer_tokenisation": trainer, "max_seq_len": L,
              "spm": {"vocab_size": nvocab, "countable_ids": countable, "control_or_unknown_ids": int(special.sum())},
              "definitions": {
                  "rows": "each row as the trainer reads it: line.strip(), the pinned model's defaults, no sampling (spec A28)",
                  "piece_types": "distinct ids over all rows of the set, control and unknown ids excluded; vocab_share = "
                                 "piece_types / countable_ids",
                  "profile": "piece types by their frequency in the set",
                  "piece_tokens_kept": "pieces (BOS/EOS excluded) of the rows the trainer keeps: both sides within "
                                       "max_seq_len with BOS and EOS",
                  "imbalance": "(a - b) / b over piece_tokens_kept",
                  "rows_with_carriage_return": "rows holding \\r, which the trainer's text-mode read would split",
                  "newstest2014_coverage": "arms.cover: % of newstest2014 English \\w+ tokens whose type occurs at least "
                                           "k times in the set's source (descriptive only; no test set defines a set)"},
              "newstest2014": ({"lines": nt["lines"], "tokens": nt["tot"], "types": nt["types"]} if nt else None),
              "sets": out, "contrasts": contrasts if rec["kind"] != "feasibility" else
              "n.m.: a feasibility directory holds every option; contrasts are for a build or dry run"}
    blob = (json.dumps(report, indent=1) + "\n").encode("utf-8")
    w = Writer(M)
    try:
        w.publish(MEASURE_REPORT, lambda: blob)
    finally:
        w.cleanup()
    for name, d in out.items():
        print(f"{name:<44} {d['rows']:>8,} rows | pieces src {d['source']['piece_types']:>6,} tgt "
              f"{d['target']['piece_types']:>6,} types | over {L}: {d['rows_over_max_seq_len']:,} | CR rows "
              f"{d['rows_with_carriage_return']:,}")
    ncr = sum(d["rows_with_carriage_return"] for d in out.values())
    if ncr:
        print(f"WARNING: {ncr:,} set rows hold a carriage return; the trainer's text-mode read would misalign them")
    print(f"measured (NOT FOR TRAINING): {len(out)} sets -> {M / MEASURE_REPORT}")
    return 0


def reproduce_check(a) -> int:
    me = interpreter()
    if not a.allow_other_interpreter and any(me[k] != v for k, v in REFERENCE_INTERPRETER.items()):
        raise Stop(f"the scratch index files were written by Python {REFERENCE_INTERPRETER['python']} with Unicode "
                   f"{REFERENCE_INTERPRETER['unidata_version']} on {REFERENCE_INTERPRETER['machine']} "
                   f"(/usr/local/bin/python3); this is Python {me['python']} with Unicode {me['unidata_version']} on "
                   f"{me['machine']}, whose \\w+ tokens or floats can differ (spec 3.2, A31)")
    if tuple(a.ref_caps) != REFERENCE_CAPS and not a.allow_unpinned_src:
        raise Stop(f"--ref-caps other than {REFERENCE_CAPS} is for the test fixtures only (--allow-unpinned-src)")
    b = Builder(a, {}, "reproduce")
    refs = {"R0": a.ref_r0, "A_narrow": a.ref_a_narrow}
    b.check_pins(need_fr=False, refs=refs)
    b.prepare(want_hashes=False, need_fr_ranks=False)
    r0, _ = b.build_r0("keep", SEED)
    an, info = b.build_a_narrow("keep", list(a.ref_caps), SEED)
    fails = []
    for nm, mine in (("R0", r0), ("A_narrow", an)):
        try:
            ref = np.load(refs[nm], allow_pickle=False)
        except (OSError, ValueError) as e:
            fails.append(f"{nm}: cannot load {refs[nm]}: {e}")
            continue
        mine = np.asarray(mine, dtype=np.int64)
        if ref.ndim != 1 or not np.issubdtype(ref.dtype, np.integer):
            fails.append(f"{nm}: {refs[nm].name} is not a 1-d integer array ({ref.dtype}, shape {ref.shape})")
        elif not np.array_equal(mine, ref):
            n = min(len(mine), len(ref))
            first = next((j for j in range(n) if mine[j] != ref[j]), n)
            only_b, only_r = np.setdiff1d(mine, ref).size, np.setdiff1d(ref, mine).size
            fails.append(f"{nm} DIFFERS from {refs[nm].name}: built {len(mine):,} rows, reference {len(ref):,}; "
                         f"first difference at position {first:,}; {only_b:,} rows only in the build, "
                         f"{only_r:,} only in the reference")
        else:
            print(f"{nm}: REPRODUCED row for row ({len(mine):,} rows) from {refs[nm].name}"
                  + (f"  [capped pool {info['capped_pool_rows']:,} rows]" if nm == "A_narrow" else ""))
    print(f"interpreter: {json.dumps(me)}; inputs pinned: {b.pinned}; caps {tuple(a.ref_caps)}; seed {SEED}")
    if fails:
        print("NOT REPRODUCED:\n  " + "\n  ".join(fails))
        return 1
    return 0


def verify(d: Path, allow_test: bool = False) -> int:
    """0 only if d is a finished production E0.6 build, made here from the pinned inputs and tied to the
    registered dry run, with nothing changed since and no trainer cache older than the build;
    1 with a reason. allow_test (--allow-unpinned-src) accepts the test fixtures' builds."""
    if (d / NOT_FOR_TRAINING).exists() or (d / FEAS_REPORT).exists() or (d / DRY_MANIFEST).exists():
        what = "a --dry-run" if (d / DRY_MANIFEST).exists() else "--feasibility output"
        print(f"NOT BUILT: {d} holds {what} (NOT_FOR_TRAINING), which is never for training")
        return 1
    mp = d / MANIFEST
    try:
        man = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"NOT BUILT: {mp}: {e}")
        return 1
    if not isinstance(man, dict) or man.get("e06_complete") is not True or man.get("for_training") is not True:
        print(f"NOT BUILT: {mp} has no \"e06_complete\": true (an interrupted build, or not an E0.6 build)")
        return 1
    written, want = man.get("written", []), man.get("sha256", {})
    if (not isinstance(written, list) or not isinstance(want, dict) or not written or written[-1] != MANIFEST
            or len(set(written)) != len(written) or set(want) != set(written) - {MANIFEST}):
        print("NOT BUILT: the manifest's file list and its sha256 table disagree")
        return 1
    bad = [w for w in written[:-1] if not (d / w).is_file() or sha256_file(d / w) != want[w]]
    if bad:
        print("CHANGED SINCE THE BUILD (sha256 differs from the manifest, or missing): " + ", ".join(bad))
        return 1
    e = man.get("e06") if isinstance(man.get("e06"), dict) else {}
    if not allow_test:
        if e.get("src_pinned") is not True or e.get("allow_unpinned_src") is not False:
            print("NOT FOR TRAINING: built from inputs other than the pinned ones (--allow-unpinned-src is for the "
                  "test fixtures); rebuild from the pinned inputs")
            return 1
        x = e.get("expected_sha256")
        if not (isinstance(x, dict) and x.get("matched") is True):
            print("NOT FOR TRAINING: built without reproducing the registered dry run (--skip-expected-sha256); "
                  "rebuild with --expect-sha256-from (spec A8)")
            return 1
    here = interpreter()
    bw = e.get("built_with") if isinstance(e.get("built_with"), dict) else {}
    other = [f"{k} {bw.get(k)!r} (here {here[k]!r})" for k in ("python", "unidata_version", "machine") if bw.get(k) != here[k]]
    if other:
        print(f"BUILT ELSEWHERE: {mp} was built with " + ", ".join(other) + ": the run of record is built where it "
              "trains (OD 10); rebuild it here")
        return 1
    t_man = mp.stat().st_mtime_ns
    stale = sorted(c.name for c in d.glob("*.cached_*.npz") if c.stat().st_mtime_ns < t_man)
    if stale:
        print("STALE CACHE: older than the build's manifest, so made from text the build replaced (the trainer "
              "loads <set>.en.cached_<len>.npz by its name alone): " + ", ".join(stale) + "; delete them")
        return 1
    print(f"built and intact: {len(want)} files in {d}")
    return 0


class Args(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, which is this script's REFUSED; a wrong command line
    exits 64 instead, so a wrapper never reads a typo as an unset decision."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: usage error: {message}", file=sys.stderr)
        raise SystemExit(EX_USAGE)


def main(argv=None) -> int:
    ap = Args(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, help="production or dry-run output directory (or the build to --verify)")
    ap.add_argument("--src-dir", type=Path, help="ft_topk.*, ft_random.* (.en .fr .source) and heldout_un.*")
    ap.add_argument("--e05-dir", type=Path, help="E0.5's finished data directory (read only)")
    ap.add_argument("--pretrain-en", type=Path, help="v1.1 pretraining corpus, English (train.clean.en)")
    ap.add_argument("--pretrain-fr", type=Path, help="its French side: exact-pair exposure, literalness; "
                                                     "a construction input under od2_rule = c2")
    ap.add_argument("--qe-tsv", type=Path, help="v2_scored.tsv, for the QE descriptives")
    ap.add_argument("--news-en", type=Path, help="news-commentary v9 English, for the news-coverage descriptives")
    ap.add_argument("--ref-lex", type=Path, help="the vocab memo's lex.pkl (pinned 7570fcb8...): the rebuilt Dice "
                                                 "lexicon is compared with it entry by entry (needs --pretrain-fr)")
    ap.add_argument("--decisions", type=Path, help="the decisions file (spec 7.1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="the production build on the Mac, not for training: every descriptive input required, "
                         "writes dry_run_manifest.json (spec A8)")
    ap.add_argument("--expect-sha256-from", type=Path,
                    help="the registered dry run's dry_run_manifest.json: STOP unless every file reproduces its "
                         "sha256 (spec A8); required for a production build")
    ap.add_argument("--skip-expected-sha256", action="store_true",
                    help="production without a dry run to reproduce: tests only; recorded, and --verify refuses it")
    ap.add_argument("--verify", action="store_true", help="exit 0 only if --dir is a complete, unchanged build")
    ap.add_argument("--feasibility", action="store_true", help="build every option, for measurement only")
    ap.add_argument("--measure", action="store_true", help="OD 16 piece counts (and newstest2014 coverage) of a "
                                                           "finished build, dry run or feasibility directory")
    ap.add_argument("--from", dest="from_dir", type=Path, help="--measure: the directory to measure (read only)")
    ap.add_argument("--spm-model", type=Path, help="--measure: spm_enfr_v1_fixed.model (pinned d1c10ea8...)")
    ap.add_argument("--newstest-en", type=Path, help="--measure: newstest2014 English, one sentence per line "
                                                     "(descriptive only; sha256 recorded)")
    ap.add_argument("--mt-repo", type=Path, help="--measure: the trainer's repository, to check that it still "
                                                 "tokenises the way the measures mirror (spec A28)")
    ap.add_argument("--max-seq-len", type=int, default=256, help="--measure: the trainer's max_seq_len (default 256)")
    ap.add_argument("--out", type=Path, help="--feasibility or --measure output directory")
    ap.add_argument("--reproduce-check", action="store_true",
                    help="compare R0 and A_narrow with the scratch index files; writes nothing")
    ap.add_argument("--ref-r0", type=Path)
    ap.add_argument("--ref-a-narrow", type=Path)
    ap.add_argument("--ref-caps", type=int, nargs=2, default=list(REFERENCE_CAPS), metavar=("KE", "KF"),
                    help="A_narrow's caps for --reproduce-check (default 42000 58000; others for tests only)")
    ap.add_argument("--allow-unpinned-src", action="store_true",
                    help="build from inputs other than the pinned ones, or --verify such a build (tests; any real "
                         "use is a protocol deviation and is recorded)")
    ap.add_argument("--allow-other-interpreter", action="store_true",
                    help="--reproduce-check on an interpreter other than the one that wrote the scratch files (tests)")
    a = ap.parse_args(argv)
    modes = sum(bool(x) for x in (a.verify, a.feasibility, a.reproduce_check, a.dry_run, a.measure))
    if modes > 1:
        ap.error("--verify, --dry-run, --feasibility, --measure and --reproduce-check exclude one another")
    if a.verify:
        if a.dir is None:
            ap.error("--verify needs --dir")
        return verify(a.dir, allow_test=a.allow_unpinned_src)
    if a.allow_other_interpreter and not a.reproduce_check:
        ap.error("--allow-other-interpreter applies to --reproduce-check only (production uses od10_unidata)")
    if a.measure:
        need = {"--from": a.from_dir, "--out": a.out, "--src-dir": a.src_dir, "--spm-model": a.spm_model}
    else:
        need = {"--src-dir": a.src_dir, "--e05-dir": a.e05_dir, "--pretrain-en": a.pretrain_en}
        if a.feasibility:
            need["--out"] = a.out
        elif a.reproduce_check:
            need.update({"--ref-r0": a.ref_r0, "--ref-a-narrow": a.ref_a_narrow})
        else:
            need["--dir"] = a.dir
            if a.dry_run:
                need.update({"--pretrain-fr": a.pretrain_fr, "--qe-tsv": a.qe_tsv, "--news-en": a.news_en,
                             "--ref-lex": a.ref_lex})
    lacking = [k for k, v in need.items() if v is None]
    if lacking:
        ap.error("missing " + ", ".join(lacking) + (" (a dry run fills every column of the registered table)"
                                                     if a.dry_run else ""))
    production = not (a.measure or a.feasibility or a.reproduce_check)
    if a.skip_expected_sha256 and (not production or a.dry_run or a.expect_sha256_from is not None):
        ap.error("--skip-expected-sha256 applies to a production build without --expect-sha256-from only")
    if production and not a.dry_run and a.expect_sha256_from is None and not a.skip_expected_sha256:
        ap.error("a production build must reproduce the registered dry run: give --expect-sha256-from "
                 f"<dry run>/{DRY_MANIFEST} (spec A8), or --skip-expected-sha256 in the tests")
    if a.expect_sha256_from is not None and not production:
        ap.error("--expect-sha256-from applies to a production build or a dry run only")
    if a.ref_lex is not None and a.pretrain_fr is None and not a.measure:
        ap.error("--ref-lex needs --pretrain-fr (the lexicon it is compared with is rebuilt from it)")
    try:
        if a.measure:
            return measure_mode(a)
        if a.feasibility:
            return build_feasibility(a)
        if a.reproduce_check:
            return reproduce_check(a)
        return build_production(a, dry=a.dry_run)
    except Stop as e:
        print(f"STOP: {e}\nNothing written.")
        return EX_STOP


if __name__ == "__main__":
    raise SystemExit(main())
