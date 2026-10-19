#!/usr/bin/env python3
"""WMT2027_PLAN.md §3.2 item 3: Chimoto et al.'s rival explanation, measured.

Wrong-language and length-ratio contamination in every E0.5 set, in R0 / A_narrow, and (extra)
in the E0.6 feasibility candidate sets. Descriptive CPU analysis, not a registered result.

FAIL-CLOSED. Every input is checked against its pinned sha256 before anything is read, and the
anchor index files must reproduce E0.5's set files line for line.

Rules (README.md states them with their sources):
  LID      py3langid 0.4.0's naive-Bayes model over all its labels (140 labels in 142 model
           columns: sr and uz each have two script columns, which the tool merges), top-1 label
           on the raw line, through the PUBLIC API: LanguageIdentifier.from_model_file(MODEL_FILE,
           norm_probs=True).classify(line) -> (label, confidence).
           A row is wrong-language if its English side's top-1 is not 'en' or its French side's
           top-1 is not 'fr'. Sensitivity: the same with the flagged side's top-1 confidence
           >= 0.9 ("confident"), and a two-language decision (en vs fr only, the same public
           classify() on an identifier restricted by set_languages(['en', 'fr']): the English
           side is labelled fr, or the French side en: swapped or copied).
  CONF     The 0.9 rule uses ONE probability: the confidence that py3langid 0.4.0's classify()
           returns with norm_probs=True. In 0.4.0 that is NOT the plain naive-Bayes posterior:
           _decide() divides the raw NB log-scores s by sqrt(B), B = byte length of the line
           after the tool's own normalisation (lower-cased if the line is all upper case, NFC,
           UTF-8), then takes the softmax, then adds each duplicate script column (sr, uz) into
           the first: p = softmax(s / sqrt(B)), aliases summed. Sensitivity only
           ("lid_any_conf_T1"): the plain posterior softmax(s) (temperature 1, aliases summed),
           i.e. what langid.py's norm_probs used to return.
  RATIO    French/English length ratio r = n_fr / n_en in whitespace tokens (str.split(), the
           unit of the pool's own cleaning rule). Headline: outlier if r > 1.5 or r < 1/1.5
           (a symmetric fixed band, Moses clean-corpus-n.perl's -ratio test at 1.5).
           Sensitivity: thresholds 1.3 / 1.5 / 2.0 in three units (whitespace tokens, the base
           model's SentencePiece pieces, characters), and Tukey fences on log r taken from
           ft_random (all rows, and its UN rows).
  L0       the pool's own cleaning rule (phase0/rebuild_corpus.py: 3-200 whitespace tokens per
           side, en/fr token ratio in [0.5, 2.0], Latin-script share >= 0.9 per side). Every row
           of the pool passed it, so 0 violations are expected; it is checked, not assumed.

Two interpreters (the E0.6 reference venv must not get py3langid, and the py3langid venv has no
sentencepiece), so the SentencePiece unit comes from a first stage:
  1. ~/e06env/venv-arm64/bin/python -I spm_lengths.py --data ~/e06data --out spm_lengths.npz
  2. ~/cpu_items/item3_contamination/venv/bin/python -I item3_contamination.py --data ~/e06data \
         --feas ~/e06runs/feasibility --out ~/cpu_items/item3_contamination \
         --spm-lengths spm_lengths.npz --workers 5
Stage 2 checks every file sha256 recorded by stage 1 against its own pins and fails closed if a
string's piece count differs between the files it occurs in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PINNED = {   # ~/mt_local/phase0_next/e06data_SHA256SUMS
    "e05_src/ft_topk.en": "787339d08e0c47d9960cb263757edafd844ec80a45875ac083845a02a20ff2b2",
    "e05_src/ft_topk.fr": "22263da18eaf62cb7deea7438068ce0cb00ce3849c1a83f7e0c55008c3d78797",
    "e05_src/ft_topk.source": "a4ad45860c92eae6b512c4a7178fe78fc92f00d4b9551d00a6637c80079244ca",
    "e05_src/ft_random.en": "937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0",
    "e05_src/ft_random.fr": "382e0de574feaaa323939d73a24736b762c76a8b55a0ffd258d64b15e5303e29",
    "e05_src/ft_random.source": "fd675a1948ab4e30dce05aae4c85d2432aec99549c05fb794903a49e9b0b6923",
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
    "phase0_e05/heldout_un.en": "f70e725eae9972899b6129f6acf123ed5421021b2f5ba81bb165a6928e858bbc",
    "phase0_e05/heldout_un.fr": "7daf654fa37cca653cbc9724c77ee721384a406e3ca3e6469a2a847b66b729ab",
    "phase0_e05/manifest.json": "979dafd76344444c56cad9afc52e0ec7ba7ebb74c73115cbeb237583419b8eb2",
    "vocab/R0.npy": "22eeb53dd3cb621831939cb41a5b9280045dbc5f6b1ce48686234479ac354dc9",
    "vocab/A_narrow_42_58.npy": "e9fac76975bb83e2a4b12ed913c333212cc4a70c440346278c12ad7c0bd7cfa8",
    "spm/spm_enfr_v1_fixed.model": "d1c10ea80fb95984f3f240b3579c87a16313b77aa9bc103ca0d6a0e06672a561",
}
# sha256 of ~/e06runs/feasibility/feasibility_report.json (= the Mac's copy in
# ~/mt_local/phase0_next/e06_feasibility/); the .npy files are checked against its "files" map.
FEAS_REPORT_SHA = "9c73852f8f7ac801d0e9e74bcdf8b8011748cdd5efdc9fbbe69e8af48c05202b"

E05_FILE_SETS = ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div",
                 "ft_random_small", "heldout_un"]
FILE_PARENT = {"ft_un_rep": "ft_topk", "ft_un_div": "ft_topk", "ft_un_random": "ft_random",
               "ft_topk_rep": "ft_topk", "ft_topk_div": "ft_topk", "ft_random_small": "ft_random",
               "heldout_un": None}
ANCHOR_FILE = {"ft_un_div": "FEASIBILITY_ft_un_div__anchor.npy",
               "ft_un_rep": "FEASIBILITY_ft_un_rep__anchor.npy",
               "ft_un_random": "FEASIBILITY_ft_un_random__anchor.npy"}
VOCAB_SETS = {"R0": ("vocab/R0.npy", "ft_random", "FEASIBILITY_ft_un_r0__od6-keep__seed42.npy"),
              "A_narrow": ("vocab/A_narrow_42_58.npy", "ft_random",
                           "FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy")}
CONTRASTS = [   # (a, b): reported as a - b
    ("ft_un_rep", "ft_un_div", "C1 / E0.5 Q1: repetitive vs diverse UN half"),
    ("ft_un_div", "ft_un_random", "C2 / E0.5 Q2: diverse UN half vs random UN"),
    ("R0", "ft_un_div", "E0.6 P (sign as R0 - ft_un_div)"),
    ("A_narrow", "R0", "E0.6 N (A_narrow - R0)"),
    ("ft_topk_rep", "ft_topk_div", "E0.4 halves (all sources)"),
    ("ft_topk_div", "ft_random_small", "E0.4 diverse half vs random (all sources)"),
    ("ft_un_rep", "feas:ft_un_div_narrow__od2-c2__od1-b",
     "E0.6 C1-beyond-coverage, feasibility draw (OD 2 c2, OD 1 b)"),
    ("feas:ft_un_rep_dedup__od4-a", "feas:ft_un_div_dd__od4-a",
     "E0.6 C1-beyond-duplicates, feasibility draw (OD 4 a)"),
]
RATIO_T = [1.3, 1.5, 2.0]
RATIO_FRAC = {1.3: (13, 10), 1.5: (3, 2), 2.0: (2, 1)}
UNITS = ["ws", "spm", "char"]
HEAD_T = 1.5
CONF_P = 0.9
SEED = 42
WORD = re.compile(r"\w+", re.UNICODE)                 # e04_split.py / e05_build.py token rule
NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ſ]")            # rebuild_corpus.py latin_ratio()
LEN_BUCKETS = [(0, 7), (8, 15), (16, 25), (26, 40), (41, 10**9)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def die(msg):
    sys.exit(f"ERROR: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(p: Path) -> list[str]:
    """Lines split on '\\n' only (U+2028 etc. stay inside lines)."""
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]


def latin_ratio(s: str) -> float:
    if not s:
        return 0.0
    return (len(s) - NON_LATIN.subn("", s)[1]) / len(s)


# ---- worker: LID + lengths for a chunk of unique strings -------------------------------------
# py3langid 0.4.0 (read from the installed package): LanguageIdentifier.from_model_file(MODEL_FILE,
# norm_probs=...), .classify(text) -> (label, conf), .rank(text), .set_languages(langs),
# .nb_classes (142 columns, 'sr' and 'uz' twice), .labels (140 distinct), ._alias_pairs.
# The headline values come from the public classify(); the private _encode/_raw_score (the raw
# NB log-scores s, the same call classify() makes) give only the sensitivity quantities.
_FULL = None     # norm_probs=True, all labels   (headline label + confidence)
_TWO = None      # norm_probs=True, set_languages(['en', 'fr'])
_COLS = None     # per-label column lists, for summing aliases


def _make_identifiers():
    from py3langid.langid import LanguageIdentifier, MODEL_FILE
    full = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    two = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    two.set_languages(["en", "fr"])
    if two.nb_classes != ["en", "fr"]:
        raise RuntimeError(f"set_languages(['en','fr']) gave {two.nb_classes}")
    return full, two


def _init():
    global _FULL, _TWO, _COLS
    _FULL, _TWO = _make_identifiers()
    cols = {}
    for i, c in enumerate(_FULL.nb_classes):
        cols.setdefault(c, []).append(i)
    _COLS = [cols[c] for c in _FULL.labels]


def _work(args):
    k, strs = args
    n = len(strs)
    lab_ix = {c: i for i, c in enumerate(_FULL.labels)}
    i_en, i_fr = _FULL.nb_classes.index("en"), _FULL.nb_classes.index("fr")
    top = np.empty(n, np.int16)          # index into labels (public classify, norm_probs=True)
    topp = np.empty(n, np.float64)       # its confidence (the 0.9 rule's probability)
    two = np.empty(n, np.int8)           # 0 = en, 1 = fr (public classify, restricted)
    top_raw = np.empty(n, np.int16)      # label of argmax s (norm_probs=False semantics)
    topp_t1 = np.empty(n, np.float64)    # softmax(s) of the headline label, aliases summed
    topp_re = np.empty(n, np.float64)    # softmax(s / sqrt(B)) recomputed in float64 (check)
    lp_en = np.empty(n, np.float64)
    lp_fr = np.empty(n, np.float64)
    nofeat = np.zeros(n, bool)
    for j, s in enumerate(strs):
        lab, conf = _FULL.classify(s)
        lab2, _ = _TWO.classify(s)
        top[j] = lab_ix[lab]
        topp[j] = float(conf)
        two[j] = 0 if lab2 == "en" else 1
        b = _FULL._encode(s)
        raw = np.asarray(_FULL._raw_score(b), dtype=np.float64)
        nofeat[j] = bool(raw.max() == raw.min())
        top_raw[j] = lab_ix[_FULL.nb_classes[int(np.argmax(raw))]]
        lp_en[j], lp_fr[j] = raw[i_en], raw[i_fr]
        q = np.exp(raw - raw.max())
        q /= q.sum()
        topp_t1[j] = float(q[_COLS[top[j]]].sum())
        z = raw / math.sqrt(len(b) or 1)
        q = np.exp(z - z.max())
        q /= q.sum()
        topp_re[j] = float(q[_COLS[top[j]]].sum())
    n_ws = np.fromiter((len(s.split()) for s in strs), np.int32, n)
    n_char = np.fromiter((len(s.strip()) for s in strs), np.int32, n)
    n_w = np.fromiter((len(WORD.findall(s)) for s in strs), np.int32, n)
    lat = np.fromiter((latin_ratio(s.strip()) for s in strs), np.float64, n)
    return (k, top, topp, two, top_raw, topp_t1, topp_re, lp_en, lp_fr, nofeat,
            n_ws, n_char, n_w, lat)


_FIELDS = [("top", np.int16), ("topp", np.float64), ("two", np.int8), ("top_raw", np.int16),
           ("topp_t1", np.float64), ("topp_re", np.float64), ("lp_en", np.float64),
           ("lp_fr", np.float64), ("nofeat", bool), ("n_ws", np.int32), ("n_char", np.int32),
           ("n_w", np.int32), ("lat", np.float64)]


def classify_all(strs: list[str], workers: int, tag: str) -> dict:
    chunk = 4000
    jobs = [(k, strs[k:k + chunk]) for k in range(0, len(strs), chunk)]
    n = len(strs)
    out = {name: np.empty(n, dt) for name, dt in _FIELDS}
    seen = np.zeros(n, bool)
    t0 = time.time()
    done = 0
    with get_context("spawn").Pool(workers, initializer=_init) as pool:
        for k, *arrs in pool.imap_unordered(_work, jobs):
            m = len(arrs[0])
            if seen[k:k + m].any():
                die(f"{tag}: chunk {k} returned twice")
            seen[k:k + m] = True
            for (name, _), a in zip(_FIELDS, arrs):
                out[name][k:k + m] = a
            done += m
            if done % 200000 < chunk:
                log(f"  {tag}: {done:,}/{n:,} ({done / max(1e-9, time.time() - t0):,.0f}/s)")
    if not seen.all():
        die(f"{tag}: {int((~seen).sum())} strings were not classified")
    log(f"  {tag}: {n:,} strings in {time.time() - t0:.0f}s")
    return out


# ---- per-set statistics -----------------------------------------------------------------------
def share(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def set_flags(en_ids, fr_ids, EN, FR, ckey_en, ckey_fr, ci, fences):
    """Boolean flag arrays for one set's rows."""
    f = {}
    et, ft = EN["top"][en_ids], FR["top"][fr_ids]
    f["en_wrong"] = et != ci["en"]
    f["fr_wrong"] = ft != ci["fr"]
    f["lid_any"] = f["en_wrong"] | f["fr_wrong"]
    f["en_wrong_conf"] = f["en_wrong"] & (EN["topp"][en_ids] >= CONF_P)
    f["fr_wrong_conf"] = f["fr_wrong"] & (FR["topp"][fr_ids] >= CONF_P)
    f["lid_any_conf"] = f["en_wrong_conf"] | f["fr_wrong_conf"]
    # sensitivity: the same 0.9 rule on the plain NB posterior softmax(s) (temperature 1)
    f["lid_any_conf_T1"] = ((f["en_wrong"] & (EN["topp_t1"][en_ids] >= CONF_P))
                            | (f["fr_wrong"] & (FR["topp_t1"][fr_ids] >= CONF_P)))
    # sensitivity: top-1 label under norm_probs=False semantics (argmax of s, aliases max-folded)
    f["lid_any_rawlabel"] = (EN["top_raw"][en_ids] != ci["en"]) | (FR["top_raw"][fr_ids] != ci["fr"])
    # en-vs-fr only: public classify() of the identifier restricted by set_languages(['en','fr'])
    f["en_side_as_fr_2way"] = EN["two"][en_ids] == 1
    f["fr_side_as_en_2way"] = FR["two"][fr_ids] == 0
    f["swap_or_copy_2way"] = f["en_side_as_fr_2way"] | f["fr_side_as_en_2way"]
    f["same_top1_both_sides"] = et == ft
    f["identical_sides"] = ckey_en[en_ids] == ckey_fr[fr_ids]
    for u in UNITS:
        ne = EN[f"n_{u}"][en_ids].astype(np.float64)
        nf = FR[f"n_{u}"][fr_ids].astype(np.float64)
        zero = (ne == 0) | (nf == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(zero, np.nan, nf / ne)
        ne_i = EN[f"n_{u}"][en_ids].astype(np.int64)
        nf_i = FR[f"n_{u}"][fr_ids].astype(np.int64)
        for t in RATIO_T:
            num, den = RATIO_FRAC[t]          # exact: r > t  <=>  nf*den > num*ne
            longf = ~zero & (nf_i * den > num * ne_i)
            shortf = ~zero & (ne_i * den > num * nf_i)
            f[f"ratio_{u}_{t}"] = zero | longf | shortf
            f[f"ratio_{u}_{t}_fr_long"] = longf
            f[f"ratio_{u}_{t}_fr_short"] = shortf
        lr = np.log(np.where(zero, 1.0, r))
        for ref, (lo, hi) in fences[u].items():
            f[f"ratio_{u}_tukey_{ref}"] = zero | (lr < lo) | (lr > hi)
    f["either_headline"] = f["lid_any"] | f[f"ratio_ws_{HEAD_T}"]
    # L0: the pool's own cleaning rule (rebuild_corpus.py); violations expected to be 0
    ne, nf = EN["n_ws"][en_ids], FR["n_ws"][fr_ids]
    with np.errstate(divide="ignore", invalid="ignore"):
        r0 = ne / np.maximum(nf, 1)
    f["L0_violation"] = ((ne < 3) | (nf < 3) | (ne > 200) | (nf > 200) | (r0 < 0.5) | (r0 > 2.0)
                         | (EN["lat"][en_ids] < 0.9) | (FR["lat"][fr_ids] < 0.9))
    return f


def ratio_summary(en_ids, fr_ids, EN, FR):
    out = {}
    for u in UNITS:
        ne = EN[f"n_{u}"][en_ids].astype(np.float64)
        nf = FR[f"n_{u}"][fr_ids].astype(np.float64)
        ok = (ne > 0) & (nf > 0)
        r = nf[ok] / ne[ok]
        q = np.quantile(r, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]) if len(r) else [float("nan")] * 7
        out[u] = {"mean": float(r.mean()) if len(r) else None,
                  "q0.1%": float(q[0]), "q1%": float(q[1]), "q5%": float(q[2]), "median": float(q[3]),
                  "q95%": float(q[4]), "q99%": float(q[5]), "q99.9%": float(q[6]),
                  "min": float(r.min()) if len(r) else None, "max": float(r.max()) if len(r) else None,
                  "en_max": int(ne.max()) if len(ne) else None,
                  "fr_max": int(nf.max()) if len(nf) else None,
                  "zero_length_rows": int((~ok).sum())}
    return out


def top_langs(ids, TAB, classes, mask, k=10):
    c = Counter(TAB["top"][ids][mask].tolist())
    return [[classes[i], n] for i, n in c.most_common(k)]


def summarise(name, en_ids, fr_ids, EN, FR, ckey_en, ckey_fr, ci, classes, fences, src_labels=None):
    f = set_flags(en_ids, fr_ids, EN, FR, ckey_en, ckey_fr, ci, fences)
    n = len(en_ids)
    s = {"rows": n,
         "counts": {k: int(v.sum()) for k, v in f.items()},
         "shares": {k: share(v) for k, v in f.items()},
         "en_side_top1_when_wrong": top_langs(en_ids, EN, classes, f["en_wrong"]),
         "fr_side_top1_when_wrong": top_langs(fr_ids, FR, classes, f["fr_wrong"]),
         "length_ratio_fr_over_en": ratio_summary(en_ids, fr_ids, EN, FR)}
    # by source length (lowercased \w+ tokens, the project's length measure)
    L = EN["n_w"][en_ids]
    by = {}
    for lo, hi in LEN_BUCKETS:
        m = (L >= lo) & (L <= hi)
        if m.sum():
            by[f"{lo}-{hi if hi < 10**9 else 'max'}"] = {
                "rows": int(m.sum()), "lid_any": share(f["lid_any"][m]),
                f"ratio_ws_{HEAD_T}": share(f[f"ratio_ws_{HEAD_T}"][m]),
                "en_wrong": share(f["en_wrong"][m]), "fr_wrong": share(f["fr_wrong"][m])}
    s["by_source_length"] = by
    if src_labels is not None:
        bs = {}
        for lab in sorted(set(src_labels.tolist())):
            m = src_labels == lab
            bs[str(lab)] = {"rows": int(m.sum()), "lid_any": share(f["lid_any"][m]),
                            "en_wrong": share(f["en_wrong"][m]), "fr_wrong": share(f["fr_wrong"][m]),
                            f"ratio_ws_{HEAD_T}": share(f[f"ratio_ws_{HEAD_T}"][m])}
        s["by_source_label"] = bs
    return s, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "e06data"))
    ap.add_argument("--feas", default=str(Path.home() / "e06runs" / "feasibility"))
    ap.add_argument("--out", default=str(Path.home() / "cpu_items" / "item3_contamination"))
    ap.add_argument("--spm-lengths", required=True,
                    help="spm_lengths.npz from spm_lengths.py (reference interpreter)")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.workers > 5:
        die("at most 5 workers (6 processes with the parent)")
    data, feas, out = Path(a.data), Path(a.feas), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 1. pins ----
    bad = []
    for rel, h in PINNED.items():
        p = data / rel
        got = sha256_file(p) if p.is_file() else "MISSING"
        if got != h:
            bad.append((rel, got))
    rep_p = feas / "feasibility_report.json"
    got = sha256_file(rep_p) if rep_p.is_file() else "MISSING"
    if got != FEAS_REPORT_SHA:
        bad.append(("feasibility_report.json", got))
    if bad:
        die(f"sha256 mismatch: {bad}")
    report = json.loads(rep_p.read_text())
    feas_files = report["files"]
    for fn, h in feas_files.items():
        if not fn.endswith(".npy"):
            continue
        g = sha256_file(feas / fn)
        if g != h:
            die(f"{fn}: sha256 {g} != report's {h}")
    log(f"sha256: {len(PINNED)} pinned inputs, feasibility_report.json and "
        f"{sum(1 for f in feas_files if f.endswith('.npy'))} feasibility .npy files match")

    import py3langid
    from py3langid.langid import MODEL_DIR, MODEL_FILE
    if py3langid.__version__ != "0.4.0":
        die(f"py3langid {py3langid.__version__}, expected 0.4.0")
    model_path = MODEL_DIR / MODEL_FILE
    ident, _two = _make_identifiers()
    classes = list(ident.labels)            # 140 distinct labels; top-1 indices point here
    columns = list(ident.nb_classes)        # 142 model columns
    ci = {"en": classes.index("en"), "fr": classes.index("fr")}
    log(f"py3langid {py3langid.__version__}, {len(classes)} labels in {len(columns)} columns "
        f"(aliases {[(columns[i], columns[j]) for i, j in ident._alias_pairs]}), model "
        f"{model_path} sha256 {sha256_file(model_path)}")

    # SentencePiece piece counts (stage 1, reference interpreter): same files, same bytes
    with np.load(a.spm_lengths, allow_pickle=False) as z:
        SPM = {k: z[k] for k in z.files}
    spm_meta = json.loads(str(SPM.pop("__meta__")))
    if spm_meta["model_sha256"] != PINNED["spm/spm_enfr_v1_fixed.model"]:
        die("spm_lengths.npz was made with another SentencePiece model")
    for rel, h in spm_meta["files_sha256"].items():
        if PINNED.get(rel) != h:
            die(f"spm_lengths.npz: {rel} sha256 {h} != pin {PINNED.get(rel)}")
    log(f"spm_lengths.npz: sentencepiece {spm_meta['sentencepiece']}, {len(SPM)} files, "
        f"sha256s match the pins")

    # ---- 2. read text ----
    P = {}
    for par in ("ft_topk", "ft_random"):
        en = read_lines(data / "e05_src" / f"{par}.en")
        fr = read_lines(data / "e05_src" / f"{par}.fr")
        sr = read_lines(data / "e05_src" / f"{par}.source")
        if not (len(en) == len(fr) == len(sr)):
            die(f"{par}: line counts differ {len(en)} {len(fr)} {len(sr)}")
        P[par] = (en, fr, np.array(sr, dtype=object))
        log(f"{par}: {len(en):,} rows")
    F = {}
    for s in E05_FILE_SETS:
        en = read_lines(data / "phase0_e05" / f"{s}.en")
        fr = read_lines(data / "phase0_e05" / f"{s}.fr")
        if len(en) != len(fr):
            die(f"{s}: .en {len(en)} != .fr {len(fr)}")
        F[s] = (en, fr)
        log(f"{s}: {len(en):,} rows")

    # ---- 3. unique string tables ----
    def table(side):
        idx = {}
        for par in ("ft_topk", "ft_random"):
            for x in P[par][side]:
                if x not in idx:
                    idx[x] = len(idx)
        for s in E05_FILE_SETS:
            for x in F[s][side]:
                if x not in idx:
                    idx[x] = len(idx)
        strs = [None] * len(idx)
        for x, i in idx.items():
            strs[i] = x
        return idx, strs
    en_idx, en_strs = table(0)
    fr_idx, fr_strs = table(1)
    log(f"unique strings: en {len(en_strs):,}, fr {len(fr_strs):,}")
    # identical-sides key: the stripped string, shared namespace across both sides
    ck = {}
    ckey_en = np.fromiter((ck.setdefault(s.strip(), len(ck)) for s in en_strs), np.int64, len(en_strs))
    ckey_fr = np.fromiter((ck.setdefault(s.strip(), len(ck)) for s in fr_strs), np.int64, len(fr_strs))
    del ck

    checks = []
    # ---- 4. LID + lengths ----
    EN = classify_all(en_strs, a.workers, "en")
    FR = classify_all(fr_strs, a.workers, "fr")
    # SentencePiece counts: every line of every file, mapped through the unique tables
    for side, idx, T in ((0, en_idx, EN), (1, fr_idx, FR)):
        ext = "en" if side == 0 else "fr"
        n_spm = np.full(len(idx), -1, np.int64)
        srcs = [(f"e05_src/{par}.{ext}", P[par][side]) for par in ("ft_topk", "ft_random")] + \
               [(f"phase0_e05/{s}.{ext}", F[s][side]) for s in E05_FILE_SETS]
        for rel, seq in srcs:
            cnt = SPM.get(rel)
            if cnt is None or len(cnt) != len(seq):
                die(f"spm_lengths.npz: {rel} missing or wrong length")
            ids = np.fromiter((idx[x] for x in seq), np.int64, len(seq))
            prev = n_spm[ids]
            clash = (prev >= 0) & (prev != cnt)
            if clash.any():
                die(f"spm_lengths.npz: {int(clash.sum())} strings of {rel} have another count elsewhere")
            n_spm[ids] = cnt
        if (n_spm < 0).any():
            die(f"{ext}: {int((n_spm < 0).sum())} strings without a piece count")
        T["n_spm"] = n_spm.astype(np.int32)
    checks.append({"check": "SentencePiece counts from spm_lengths.npz: all 18 text files' sha256 = pins; "
                            "every string has one count across all files it occurs in", "ok": True,
                   "sentencepiece": spm_meta["sentencepiece"]})

    # in-process cross-check: a fresh identifier pair, no multiprocessing, against the
    # reassembled per-string arrays (labels and confidences must be bit-identical)
    rng = random.Random(SEED)
    full, two = _make_identifiers()
    api_mis = Counter()
    for side, strs, T in (("en", en_strs, EN), ("fr", fr_strs, FR)):
        for i in rng.sample(range(len(strs)), 3000):
            lab, p = full.classify(strs[i])
            if lab != classes[T["top"][i]]:
                api_mis[f"{side}_top1"] += 1
            if float(p) != T["topp"][i]:
                api_mis[f"{side}_topp"] += 1
            lab2, _ = two.classify(strs[i])
            if lab2 != ("en", "fr")[T["two"][i]]:
                api_mis[f"{side}_2way"] += 1
    if any(api_mis.values()):
        die(f"reassembled LID differs from a fresh public classify(): {dict(api_mis)}")
    log("public-API re-run: 6,000 strings, top-1, confidence and two-language label identical")
    checks.append({"check": "per-string arrays = a fresh py3langid classify() (full and "
                            "set_languages(['en','fr'])) on 3,000 random unique strings per side; "
                            "label, confidence (exact float) and 2-way label", "ok": True})
    # consistency of the headline quantities with the raw scores (reported, not gating)
    cons = {}
    for side, T in (("en", EN), ("fr", FR)):
        two_raw = (T["lp_fr"] > T["lp_en"]).astype(np.int8)      # tie -> en (en column first)
        cons[side] = {
            "strings": int(len(T["top"])),
            "no_features": int(T["nofeat"].sum()),
            "label_norm_probs_true_vs_false_differ": int((T["top"] != T["top_raw"]).sum()),
            "two_way_label_vs_raw_en_fr_comparison_differ": int((T["two"] != two_raw).sum()),
            "max_abs_conf_minus_float64_recomputation": float(np.abs(T["topp"] - T["topp_re"]).max()),
            "conf_ge_0.9": int((T["topp"] >= CONF_P).sum()),
            "conf_T1_ge_0.9": int((T["topp_t1"] >= CONF_P).sum()),
        }
    log(f"raw-score consistency: {cons}")
    checks.append({"check": "headline LID vs raw NB scores (reported, not gating)", "ok": True,
                   "by_side": cons})

    # ---- 5. fences (Tukey, on log r) from ft_random ----
    def ids_of(par, rows):
        en, fr, _ = P[par]
        return (np.fromiter((en_idx[en[i]] for i in rows), np.int64, len(rows)),
                np.fromiter((fr_idx[fr[i]] for i in rows), np.int64, len(rows)))
    nR = len(P["ft_random"][0])
    rnd_all = ids_of("ft_random", range(nR))
    top_all = ids_of("ft_topk", range(len(P["ft_topk"][0])))
    ALL = {"ft_random": rnd_all, "ft_topk": top_all}
    un_rows = np.flatnonzero(P["ft_random"][2] == "un")
    rnd_un = (rnd_all[0][un_rows], rnd_all[1][un_rows])
    fences = {u: {} for u in UNITS}
    for u in UNITS:
        for ref, (ei, fi) in (("ftrandom_all", rnd_all), ("ftrandom_un", rnd_un)):
            ne = EN[f"n_{u}"][ei].astype(np.float64)
            nf = FR[f"n_{u}"][fi].astype(np.float64)
            ok = (ne > 0) & (nf > 0)
            lr = np.log(nf[ok] / ne[ok])
            q1, q3 = np.quantile(lr, [0.25, 0.75])
            fences[u][ref] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))
    log(f"Tukey fences (log r): {fences}")

    # ---- 6. sets ----
    sets = {}       # name -> (en_ids, fr_ids, src_labels or None, meta)
    pair_rows = {}
    for par in ("ft_topk", "ft_random"):
        pen, pfr, _ = P[par]
        d = {}
        for i in range(len(pen)):
            d.setdefault((pen[i], pfr[i]), i)
        pair_rows[par] = d
    for s in E05_FILE_SETS:
        en, fr = F[s]
        ei = np.fromiter((en_idx[x] for x in en), np.int64, len(en))
        fi = np.fromiter((fr_idx[x] for x in fr), np.int64, len(fr))
        lab = None
        par = FILE_PARENT[s]
        if par is not None:
            psrc = P[par][2]
            pair_row = pair_rows[par]
            rows = [pair_row.get((en[i], fr[i]), -1) for i in range(len(en))]
            miss = sum(1 for r in rows if r < 0)
            checks.append({"check": f"every {s} pair occurs in {par}", "ok": miss == 0,
                           "missing": miss})
            if miss:
                die(f"{s}: {miss} pairs not in {par}")
            lab = psrc[np.array(rows)]
        sets[s] = (ei, fi, lab, {"source": f"phase0_e05/{s}.{{en,fr}}", "parent": par})
    for s, fn in ANCHOR_FILE.items():
        idx = np.load(feas / fn)
        par = report["anchors"][s]["parent"]
        pen, pfr, _ = P[par]
        en, fr = F[s]
        same = (len(idx) == len(en) and all(pen[j] == en[i] and pfr[j] == fr[i]
                                             for i, j in enumerate(idx.tolist())))
        checks.append({"check": f"{fn} indexes {par} and reproduces phase0_e05/{s} line for line",
                       "ok": bool(same)})
        if not same:
            die(f"{fn} does not reproduce {s}")
    for name, (rel, par, twin) in VOCAB_SETS.items():
        same = PINNED[rel] == feas_files[twin]
        checks.append({"check": f"{rel} is byte-identical to {twin}", "ok": same})
        if not same:
            die(f"{rel} != {twin}")
        idx = np.load(data / rel).astype(np.int64)
        if idx.min() < 0 or idx.max() >= nR or len(np.unique(idx)) != len(idx):
            die(f"{rel}: bad indices")
        lab = P[par][2][idx]
        sets[name] = (rnd_all[0][idx], rnd_all[1][idx], lab,
                      {"source": f"{rel} (rows of {par})", "parent": par})
    feas_sets = {}
    for sec in ("anchors", "results"):
        for k, v in report[sec].items():
            fn = v.get("file")
            if not fn:
                continue
            idx = np.load(feas / fn).astype(np.int64)
            par = v["parent"]
            if len(idx) != v["rows"]:
                die(f"{fn}: {len(idx)} rows, report says {v['rows']}")
            base = ALL[par]
            feas_sets[f"feas:{k}"] = (base[0][idx], base[1][idx], P[par][2][idx],
                                      {"source": f"{fn} (rows of {par})", "parent": par})
    for par in ("ft_topk", "ft_random"):
        allr = ALL[par]
        sets[par] = (allr[0], allr[1], P[par][2], {"source": f"e05_src/{par}.{{en,fr}}", "parent": None})
        un = np.flatnonzero(P[par][2] == "un")
        sets[f"{par}[un]"] = (allr[0][un], allr[1][un], P[par][2][un],
                              {"source": f"UN-labelled rows of {par}", "parent": par})

    results, flags = {}, {}
    for name, (ei, fi, lab, meta) in list(sets.items()) + list(feas_sets.items()):
        s, f = summarise(name, ei, fi, EN, FR, ckey_en, ckey_fr, ci, classes, fences, lab)
        s.update(meta)
        results[name] = s
        if not name.startswith("feas:"):
            flags[name] = f
        log(f"{name:48s} rows {s['rows']:>8,}  lid_any {s['shares']['lid_any']:.4%}  "
            f"ratio_ws_{HEAD_T} {s['shares'][f'ratio_ws_{HEAD_T}']:.4%}  "
            f"L0 {s['counts']['L0_violation']}")

    # ---- 7. contrasts ----
    contrasts = []
    keys = ["lid_any", "en_wrong", "fr_wrong", "lid_any_conf", "lid_any_conf_T1", "lid_any_rawlabel",
            "swap_or_copy_2way", "identical_sides", "same_top1_both_sides"] + \
           [f"ratio_{u}_{t}" for u in UNITS for t in RATIO_T] + \
           [f"ratio_{u}_tukey_{r}" for u in UNITS for r in ("ftrandom_all", "ftrandom_un")] + \
           ["either_headline", "L0_violation"]
    for x, y, what in CONTRASTS:
        rx, ry = results[x], results[y]
        row = {"a": x, "b": y, "what": what, "measures": {}}
        for k in keys:
            px, py = rx["shares"][k], ry["shares"][k]
            nx, ny = rx["rows"], ry["rows"]
            sd = math.sqrt(px * (1 - px) / nx + py * (1 - py) / ny)
            row["measures"][k] = {"a_share": px, "b_share": py, "a_count": rx["counts"][k],
                                  "b_count": ry["counts"][k], "diff_pp": 100 * (px - py),
                                  "ratio": (px / py) if py > 0 else None,
                                  "binomial_sd_of_diff_pp": 100 * sd}
        contrasts.append(row)

    # ---- 8. examples ----
    def ex_row(setname, i):
        ei, fi = sets[setname][0][i], sets[setname][1][i]
        f = flags[setname]
        en_s, fr_s = en_strs[ei], fr_strs[fi]
        why = [k for k in ("en_wrong", "fr_wrong", f"ratio_ws_{HEAD_T}_fr_long",
                           f"ratio_ws_{HEAD_T}_fr_short", "identical_sides") if f[k][i]]
        return {"set": setname, "line": int(i) + 1, "flags": why,
                "en_top1": [classes[EN["top"][ei]], round(float(EN["topp"][ei]), 4)],
                "fr_top1": [classes[FR["top"][fi]], round(float(FR["topp"][fi]), 4)],
                "en_ws": int(EN["n_ws"][ei]), "fr_ws": int(FR["n_ws"][fi]),
                "ratio_ws": round(FR["n_ws"][fi] / max(1, EN["n_ws"][ei]), 3),
                "en": en_s, "fr": fr_s}
    rng = random.Random(SEED)
    headline_examples = []
    for setname, k in (("ft_un_rep", 7), ("ft_un_div", 7), ("ft_un_random", 6)):
        cand = np.flatnonzero(flags[setname]["either_headline"]).tolist()
        for i in sorted(rng.sample(cand, min(k, len(cand)))):
            headline_examples.append(ex_row(setname, i))
    more = {}
    rng = random.Random(SEED)
    for setname in ("ft_un_rep", "ft_un_div", "ft_un_random", "R0", "A_narrow",
                    "ft_topk_rep", "ft_topk_div", "ft_random_small"):
        more[setname] = {}
        for k in ("en_wrong", "fr_wrong", f"ratio_ws_{HEAD_T}_fr_long", f"ratio_ws_{HEAD_T}_fr_short",
                  "identical_sides"):
            cand = np.flatnonzero(flags[setname][k]).tolist()
            more[setname][k] = [ex_row(setname, i) for i in sorted(rng.sample(cand, min(10, len(cand))))]

    res = {
        "item": "WMT2027_PLAN.md §3.2 item 3 (Chimoto et al.'s rival explanation)",
        "descriptive_only": True,
        "rules": {
            "lid": "py3langid 0.4.0 public API: LanguageIdentifier.from_model_file(MODEL_FILE, "
                   "norm_probs=True).classify(raw line) -> (label, conf), all 140 labels; wrong if "
                   "en side's label != 'en' or fr side's label != 'fr'",
            "lid_conf": f"flagged side's classify() confidence >= {CONF_P}, where (py3langid 0.4.0, "
                        "norm_probs=True) conf = softmax(s / sqrt(B))[label], s = raw naive-Bayes "
                        "log-scores, B = UTF-8 byte length after the tool's normalisation (lower-case "
                        "if all upper, NFC), duplicate script columns (sr, uz) summed into one",
            "lid_conf_T1 (sensitivity)": f"the same >= {CONF_P} rule on softmax(s)[label] (temperature "
                                         "1, aliases summed), the plain NB posterior",
            "lid_rawlabel (sensitivity)": "label = argmax of s (norm_probs=False semantics)",
            "two_way": "en vs fr only: classify() of a norm_probs=True identifier after "
                       "set_languages(['en','fr'])",
            "ratio_headline": f"r = n_fr/n_en in whitespace tokens (str.split()); outlier if r > {HEAD_T} "
                              f"or r < 1/{HEAD_T}",
            "ratio_units": {"ws": "str.split() tokens", "spm": "pieces of spm_enfr_v1_fixed.model "
                            "(encode, no BOS/EOS)", "char": "len(line.strip())"},
            "ratio_thresholds": RATIO_T,
            "tukey_fences_log_r": fences,
            "L0": "rebuild_corpus.py: 3..200 whitespace tokens per side, en/fr ratio in [0.5, 2.0], "
                  "Latin share >= 0.9 per side",
            "length_buckets": "lowercased \\w+ source tokens",
        },
        "environment": {"python": sys.version.split()[0], "machine": platform.machine(),
                        "numpy": np.__version__,
                        "sentencepiece (stage 1, reference interpreter)": spm_meta["sentencepiece"],
                        "stage1_python": spm_meta["python"],
                        "interpreter": sys.executable,
                        "py3langid": py3langid.__version__,
                        "py3langid_wheel_sha256": "c25bd038147951ba3a89efe9ec12cc7d2116cb6f311ff8af84ade398a8bc0616",
                        "py3langid_model": str(model_path),
                        "py3langid_model_sha256": sha256_file(model_path),
                        "n_labels": len(classes), "labels": classes, "n_model_columns": len(columns),
                        "unidata": unicodedata.unidata_version, "workers": a.workers},
        "inputs_sha256": PINNED, "feasibility_report_sha256": FEAS_REPORT_SHA,
        "checks": checks,
        "unique_strings": {"en": len(en_strs), "fr": len(fr_strs)},
        "sets": results,
        "contrasts": contrasts,
        "examples_headline_20": headline_examples,
        "examples_by_flag": more,
        "seconds": round(time.time() - t_start, 1),
    }
    tmp = out / "results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, out / "results.json")
    # per-string flags for the independent check
    np.savez_compressed(out / "per_string.npz", en_top=EN["top"], en_topp=EN["topp"],
                        en_topp_t1=EN["topp_t1"], en_two=EN["two"],
                        fr_top=FR["top"], fr_topp=FR["topp"], fr_topp_t1=FR["topp_t1"],
                        fr_two=FR["two"], en_ws=EN["n_ws"], fr_ws=FR["n_ws"])
    log(f"wrote {out / 'results.json'} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
