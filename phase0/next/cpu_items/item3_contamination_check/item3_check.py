#!/usr/bin/env python3
"""Independent check of WMT2027_PLAN.md §3.2 item 3 (Chimoto et al.'s rival explanation):
wrong-language and length-ratio contamination per set. Descriptive only.

Written from the definitions in the item3_contamination README; it imports nothing from the
builder's scripts. Two stages so that at most 3 compute processes run at any time:

  item3_check.py prep  --sums e06data_SHA256SUMS   pins, unique strings -> work/lid_in_{0,1,2}.txt,
                                                   per-string lengths -> work/lengths.npz
  (run.sh: 3 x nl_lid on the three chunks; nl_lid_check on a 6,000-string sample)
  item3_check.py stats --sums e06data_SHA256SUMS   flags per set, contrasts, examples -> results.json

LANGUAGE ID. The builder's py3langid 0.4.0 was never installed on the Studio (its run was
blocked), and nothing is downloaded here. The language-ID numbers therefore come from Apple's
NaturalLanguage framework (NLLanguageRecognizer, the OS's own model; nl_lid.swift). Same rules as
the builder's definition, different tool: top-1 label on the raw line; a row is wrong-language if
the English side's top-1 is not 'en' or the French side's is not 'fr'; "conf" = the flagged
side's top-1 probability >= 0.9; "2way" = the recognizer restricted to {en, fr} labels the English
side 'fr' or the French side 'en'.

LENGTH RATIO (exact integers). r = n_fr / n_en; outlier if r > t or r < 1/t, t in {1.3, 1.5, 2.0};
units: str.split() tokens (headline, t = 1.5), sentencepiece pieces of spm_enfr_v1_fixed.model,
len(line.strip()) characters; a zero length on either side counts as an outlier. Tukey fences on
log r (np.quantile, linear) from ft_random, all rows and UN rows.

L0 (the pool's own cleaning rule, phase0/rebuild_corpus.py): 3..200 whitespace tokens per side,
en/fr token ratio in [0.5, 2.0], Latin share (ASCII or U+00C0..U+017F, of the stripped line)
>= 0.9 per side. Expected 0 violations; counted.
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
from collections import Counter
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = HOME / "e06data"
FEAS = HOME / "e06runs" / "feasibility"
OUT = HOME / "cpu_items" / "item3_contamination_check"
WORK = OUT / "work"
NCHUNK = 3
FEAS_REPORT_SHA = "9c73852f8f7ac801d0e9e74bcdf8b8011748cdd5efdc9fbbe69e8af48c05202b"
E05_SETS = ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div",
            "ft_random_small", "heldout_un"]
E05_PARENT = {"ft_un_rep": "ft_topk", "ft_un_div": "ft_topk", "ft_un_random": "ft_random",
              "ft_topk_rep": "ft_topk", "ft_topk_div": "ft_topk", "ft_random_small": "ft_random",
              "heldout_un": None}
USED = ([f"e05_src/{p}.{x}" for p in ("ft_topk", "ft_random") for x in ("en", "fr", "source")]
        + [f"phase0_e05/{s}.{x}" for s in E05_SETS for x in ("en", "fr")]
        + ["vocab/R0.npy", "vocab/A_narrow_42_58.npy", "spm/spm_enfr_v1_fixed.model"])
VOCAB = {"R0": ("vocab/R0.npy", "FEASIBILITY_ft_un_r0__od6-keep__seed42.npy"),
         "A_narrow": ("vocab/A_narrow_42_58.npy", "FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy")}
THRESH = {"1.3": (13, 10), "1.5": (3, 2), "2.0": (2, 1)}     # t = num / den
UNITS = ("ws", "spm", "char")
CONF = 0.9
WORD = re.compile(r"\w+")
# Latin-share table: deleting the allowed code points leaves the non-Latin ones
_ALLOWED = dict.fromkeys(list(range(0x80)) + list(range(0xC0, 0x180)))
CONTRASTS = [
    ("ft_un_rep", "ft_un_div", "requested: repetitive vs diverse UN half (E0.5 Q1)"),
    ("ft_un_div", "ft_un_random", "requested: diverse UN half vs random UN (E0.5 Q2)"),
    ("R0", "ft_un_div", "extra: E0.6 P"),
    ("A_narrow", "R0", "extra: E0.6 N"),
    ("ft_topk_rep", "ft_topk_div", "extra: E0.4 halves, all sources"),
    ("ft_topk_div", "ft_random_small", "extra: E0.4 diverse half vs random, all sources"),
    ("ft_un_rep", "feas:ft_un_div_narrow__od2-c2__od1-b", "extra: rep vs div_narrow (OD 2 c2, OD 1 b)"),
    ("feas:ft_un_rep_dedup__od4-a", "feas:ft_un_div_dd__od4-a", "extra: rep_dedup vs div_dd (OD 4 a)"),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fail(msg):
    log("FAIL:", msg)
    sys.exit(1)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_lines(p: Path) -> list[str]:
    """LF is the only separator (U+2028, CR, NEL stay inside the line)."""
    b = p.read_bytes()
    if not b.endswith(b"\n"):
        fail(f"{p}: no final LF")
    return b[:-1].decode("utf-8").split("\n")


def check_pins(sums: Path):
    pins = {}
    for ln in sums.read_text().splitlines():
        if not ln.strip():
            continue
        h, rel = ln.split(None, 1)
        rel = rel.strip()
        if not rel.startswith("e06data/"):
            fail(f"unexpected SUMS line {ln!r}")
        pins[rel[len("e06data/"):]] = h
    got = {}
    for rel in USED:
        if rel not in pins:
            fail(f"{rel}: no pin in {sums}")
        g = sha256(DATA / rel)
        if g != pins[rel]:
            fail(f"{rel}: sha256 {g} != pin {pins[rel]}")
        got[rel] = g
    g = sha256(FEAS / "feasibility_report.json")
    if g != FEAS_REPORT_SHA:
        fail(f"feasibility_report.json sha256 {g}")
    rep = json.loads((FEAS / "feasibility_report.json").read_text())
    listed = sorted(f for f in rep["files"] if f.endswith(".npy"))
    on_disk = sorted(p.name for p in FEAS.glob("FEASIBILITY_*.npy"))
    if listed != on_disk:
        fail(f"feasibility .npy on disk {len(on_disk)} != listed {len(listed)}")
    for fn in listed:
        if sha256(FEAS / fn) != rep["files"][fn]:
            fail(f"{fn}: sha256 differs from the report")
    log(f"pins: {len(got)} inputs, feasibility_report.json, {len(listed)} feasibility .npy: all match")
    return got, rep


def load_text():
    P = {}
    for par in ("ft_topk", "ft_random"):
        en, fr, src = (text_lines(DATA / "e05_src" / f"{par}.{x}") for x in ("en", "fr", "source"))
        if not (len(en) == len(fr) == len(src)):
            fail(f"{par}: line counts {len(en)} {len(fr)} {len(src)}")
        P[par] = (en, fr, src)
    F = {}
    for s in E05_SETS:
        en, fr = (text_lines(DATA / "phase0_e05" / f"{s}.{x}") for x in ("en", "fr"))
        if len(en) != len(fr):
            fail(f"{s}: {len(en)} != {len(fr)}")
        F[s] = (en, fr)
    return P, F


def unique_table(P, F):
    """Both sides pooled; deterministic order (ft_topk en, fr, ft_random en, fr, then the set files)."""
    idx: dict[str, int] = {}
    U: list[str] = []
    seqs = [P["ft_topk"][0], P["ft_topk"][1], P["ft_random"][0], P["ft_random"][1]]
    for s in E05_SETS:
        seqs += [F[s][0], F[s][1]]
    for seq in seqs:
        for x in seq:
            if x not in idx:
                idx[x] = len(U)
                U.append(x)
    return idx, U


def stage_prep(a):
    import sentencepiece as spm
    check_pins(Path(a.sums))
    P, F = load_text()
    idx, U = unique_table(P, F)
    n = len(U)
    log(f"{n:,} unique strings (both sides pooled)")
    WORK.mkdir(parents=True, exist_ok=True)
    bounds = [round(k * n / NCHUNK) for k in range(NCHUNK + 1)]
    shas = []
    for k in range(NCHUNK):
        p = WORK / f"lid_in_{k}.txt"
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for x in U[bounds[k]:bounds[k + 1]]:
                f.write(x)
                f.write("\n")
        shas.append(sha256(p))
    odd = Counter()
    for x in U:
        for ch, name in (("\r", "CR"), (" ", "U+2028"), ("\x85", "NEL"), ("\t", "TAB"),
                         ("\xa0", "NBSP")):
            if ch in x:
                odd[name] += 1
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "spm" / "spm_enfr_v1_fixed.model"))
    n_spm = np.empty(n, np.int32)
    B = 100_000
    for i in range(0, n, B):
        enc = sp.encode(U[i:i + B], num_threads=1)
        n_spm[i:i + len(enc)] = [len(e) for e in enc]
    n_ws = np.fromiter((len(x.split()) for x in U), np.int32, n)
    n_char = np.fromiter((len(x.strip()) for x in U), np.int32, n)
    n_w = np.fromiter((len(WORD.findall(x)) for x in U), np.int32, n)
    n_strip = np.fromiter((len(x.strip()) for x in U), np.int64, n)
    n_nonlatin = np.fromiter((len(x.strip().translate(_ALLOWED)) for x in U), np.int64, n)
    lat = np.where(n_strip > 0, (n_strip - n_nonlatin) / np.maximum(n_strip, 1), 0.0)
    np.savez(WORK / "lengths.npz", n_spm=n_spm, n_ws=n_ws, n_char=n_char, n_w=n_w, lat=lat,
             bounds=np.array(bounds))
    (WORK / "prep.json").write_text(json.dumps(
        {"unique_strings": n, "bounds": bounds, "chunk_sha256": shas,
         "strings_containing": dict(odd)}, indent=1))
    log(f"prep done: chunks {bounds}, odd characters {dict(odd)}")


def read_lid(n: int):
    labs: dict[str, int] = {}

    def code(s):
        if s not in labs:
            labs[s] = len(labs)
        return labs[s]
    dom = np.empty(n, np.int16)
    top = np.empty(n, np.int16)
    ptop = np.empty(n, np.float64)
    pen = np.empty(n, np.float64)
    pfr = np.empty(n, np.float64)
    dom2 = np.empty(n, np.int16)
    i = 0
    for k in range(NCHUNK):
        with open(WORK / f"lid_out_{k}.tsv", encoding="utf-8") as f:
            for ln in f:
                c = ln.rstrip("\n").split("\t")
                if len(c) != 8:
                    fail(f"lid_out_{k}.tsv: bad row {ln!r}")
                dom[i] = code(c[0])
                top[i] = code(c[1])
                ptop[i] = float(c[2])
                pen[i] = float(c[3])
                pfr[i] = float(c[4])
                dom2[i] = code(c[5])
                i += 1
    if i != n:
        fail(f"LID output rows {i:,} != unique strings {n:,}")
    names = [None] * len(labs)
    for s, j in labs.items():
        names[j] = s
    return {"dom": dom, "top": top, "ptop": ptop, "pen": pen, "pfr": pfr, "dom2": dom2}, names


def stage_stats(a):
    t0 = time.time()
    pins, rep = check_pins(Path(a.sums))
    P, F = load_text()
    idx, U = unique_table(P, F)
    n = len(U)
    prep = json.loads((WORK / "prep.json").read_text())
    if prep["unique_strings"] != n:
        fail("unique table differs from prep")
    for k in range(NCHUNK):
        if sha256(WORK / f"lid_in_{k}.txt") != prep["chunk_sha256"][k]:
            fail(f"lid_in_{k}.txt changed since prep")
    # the chunk files must hold exactly U, in order
    j = 0
    for k in range(NCHUNK):
        for x in text_lines(WORK / f"lid_in_{k}.txt"):
            if x != U[j]:
                fail(f"lid_in_{k}.txt row {j} differs from the unique table")
            j += 1
    if j != n:
        fail("chunk files do not cover the unique table")
    with np.load(WORK / "lengths.npz") as z:
        Ln = {k: z[k] for k in z.files}
    for k in ("n_spm", "n_ws", "n_char", "n_w", "lat"):
        if len(Ln[k]) != n:
            fail(f"lengths.npz {k} has {len(Ln[k])} rows")
    LID, names = read_lid(n)
    C = {s: names.index(s) if s in names else -1 for s in ("en", "fr", "nil")}
    log(f"LID labels seen: {len(names)}; en={C['en']} fr={C['fr']}")
    checks = []
    dis = int((LID["dom"] != LID["top"]).sum())
    checks.append({"check": "dominantLanguage == argmax of languageHypotheses, all unique strings",
                   "mismatches": dis, "ok": dis == 0})
    # stripped-string key for identical sides
    sk: dict[str, int] = {}
    skey = np.fromiter((sk.setdefault(x.strip(), len(sk)) for x in U), np.int64, n)
    del sk

    def ids(seq):
        return np.fromiter((idx[x] for x in seq), np.int64, len(seq))

    S = {}     # name -> dict(e, f, src, parent, note)
    PID = {}
    for par in ("ft_topk", "ft_random"):
        en, fr, src = P[par]
        e, f = ids(en), ids(fr)
        srcarr = np.array(src)
        PID[par] = (e, f, srcarr)
        S[par] = {"e": e, "f": f, "src": srcarr, "from": f"e05_src/{par}"}
        un = np.flatnonzero(srcarr == "un")
        S[f"{par}[un]"] = {"e": e[un], "f": f[un], "src": srcarr[un], "from": f"UN rows of {par}"}
    # anchors reproduce E0.5's UN files line for line
    for s in ("ft_un_div", "ft_un_rep", "ft_un_random"):
        meta = rep["anchors"][s]
        par = meta["parent"]
        ix = np.load(FEAS / meta["file"]).astype(np.int64)
        pen, pfr, psrc = P[par]
        en, fr = F[s]
        ok = len(ix) == len(en) and all(pen[r] == en[i] and pfr[r] == fr[i] for i, r in enumerate(ix.tolist()))
        lab = set(psrc[r] for r in ix.tolist())
        checks.append({"check": f"{meta['file']} rows of {par} = phase0_e05/{s} line for line",
                       "ok": bool(ok), "source_labels": sorted(lab)})
        if not ok:
            fail(f"anchor {s}")
    # source labels for the set files via (en, fr) -> labels in the parent
    plab = {}
    for par in ("ft_topk", "ft_random"):
        d: dict = {}
        en, fr, src = P[par]
        for i in range(len(en)):
            d.setdefault((en[i], fr[i]), set()).add(src[i])
        plab[par] = d
    for s in E05_SETS:
        en, fr = F[s]
        par = E05_PARENT[s]
        srcarr = None
        note = {}
        if par:
            labs_ = [plab[par].get((en[i], fr[i])) for i in range(len(en))]
            miss = sum(1 for x in labs_ if x is None)
            amb = sum(1 for x in labs_ if x is not None and len(x) > 1)
            note = {"pairs_not_in_parent": miss, "pairs_with_several_labels_in_parent": amb}
            checks.append({"check": f"every phase0_e05/{s} pair occurs in {par}", "ok": miss == 0, **note})
            if miss:
                fail(f"{s}: {miss} pairs not in {par}")
            srcarr = np.array([sorted(x)[0] if len(x) == 1 else "+".join(sorted(x)) for x in labs_])
        S[s] = {"e": ids(en), "f": ids(fr), "src": srcarr, "from": f"phase0_e05/{s}", **note}
    nR = len(P["ft_random"][0])
    for name, (rel, twin) in VOCAB.items():
        same = pins[rel] == rep["files"][twin]
        checks.append({"check": f"{rel} sha256 == {twin} (report)", "ok": same})
        if not same:
            fail(f"{rel} != {twin}")
        ix = np.load(DATA / rel).astype(np.int64)
        if ix.min() < 0 or ix.max() >= nR or len(set(ix.tolist())) != len(ix):
            fail(f"{rel}: bad indices")
        e, f, src = PID["ft_random"]
        S[name] = {"e": e[ix], "f": f[ix], "src": src[ix], "from": f"{rel} (rows of ft_random)"}
    for sec in ("anchors", "results"):
        for k, v in rep[sec].items():
            if not v.get("file"):
                continue
            ix = np.load(FEAS / v["file"]).astype(np.int64)
            if len(ix) != v["rows"]:
                fail(f"{v['file']}: {len(ix)} rows, report {v['rows']}")
            e, f, src = PID[v["parent"]]
            S[f"feas:{k}"] = {"e": e[ix], "f": f[ix], "src": src[ix],
                              "from": f"{v['file']} (rows of {v['parent']})"}

    # Tukey fences on log r from ft_random
    fences = {}
    for u in UNITS:
        nn = Ln[f"n_{u}"]
        for ref in ("ft_random", "ft_random[un]"):
            e, f = S[ref]["e"], S[ref]["f"]
            ne, nf = nn[e].astype(np.float64), nn[f].astype(np.float64)
            ok = (ne > 0) & (nf > 0)
            lr = np.log(nf[ok] / ne[ok])
            q1, q3 = np.quantile(lr, [0.25, 0.75])
            fences[f"{u}|{ref}"] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))

    def flags(e, f):
        g = {}
        de, df = LID["dom"][e], LID["dom"][f]
        g["en_wrong"] = de != C["en"]
        g["fr_wrong"] = df != C["fr"]
        g["lid_any"] = g["en_wrong"] | g["fr_wrong"]
        g["en_wrong_conf"] = g["en_wrong"] & (LID["ptop"][e] >= CONF)
        g["fr_wrong_conf"] = g["fr_wrong"] & (LID["ptop"][f] >= CONF)
        g["lid_any_conf"] = g["en_wrong_conf"] | g["fr_wrong_conf"]
        g["en_nil"] = de == C["nil"]
        g["fr_nil"] = df == C["nil"]
        g["en_side_fr_2way"] = LID["dom2"][e] == C["fr"]
        g["fr_side_en_2way"] = LID["dom2"][f] == C["en"]
        g["swap_or_copy_2way"] = g["en_side_fr_2way"] | g["fr_side_en_2way"]
        g["same_top1_both_sides"] = de == df
        g["identical_sides"] = skey[e] == skey[f]
        for u in UNITS:
            ne = Ln[f"n_{u}"][e].astype(np.int64)
            nf = Ln[f"n_{u}"][f].astype(np.int64)
            zero = (ne == 0) | (nf == 0)
            for t, (num, den) in THRESH.items():
                lng = ~zero & (nf * den > num * ne)
                sht = ~zero & (ne * den > num * nf)
                g[f"ratio_{u}_{t}"] = zero | lng | sht
                g[f"ratio_{u}_{t}_fr_long"] = lng
                g[f"ratio_{u}_{t}_fr_short"] = sht
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(np.where(zero, 1, nf) / np.where(zero, 1, ne))
            for ref in ("ft_random", "ft_random[un]"):
                lo, hi = fences[f"{u}|{ref}"]
                g[f"ratio_{u}_tukey_{ref}"] = zero | (lr < lo) | (lr > hi)
        g["either_headline"] = g["lid_any"] | g["ratio_ws_1.5"]
        ne, nf = Ln["n_ws"][e].astype(np.int64), Ln["n_ws"][f].astype(np.int64)
        g["L0_violation"] = ((ne < 3) | (nf < 3) | (ne > 200) | (nf > 200) | (2 * ne < nf) | (ne > 2 * nf)
                             | (Ln["lat"][e] < 0.9) | (Ln["lat"][f] < 0.9))
        g["over_200_ws_either_side"] = (ne > 200) | (nf > 200)
        return g

    def summary(name, d):
        e, f = d["e"], d["f"]
        g = flags(e, f)
        rows = len(e)
        out = {"rows": rows, "from": d["from"],
               "counts": {k: int(v.sum()) for k, v in g.items()},
               "shares": {k: float(v.mean()) for k, v in g.items()}}
        for side, ix, w in (("en", e, g["en_wrong"]), ("fr", f, g["fr_wrong"])):
            c = Counter(LID["dom"][ix][w].tolist())
            out[f"{side}_side_top1_when_wrong"] = [[names[k], v] for k, v in c.most_common(10)]
        ne, nf = Ln["n_ws"][e].astype(np.float64), Ln["n_ws"][f].astype(np.float64)
        r = nf / np.maximum(ne, 1)
        out["ratio_ws_fr_over_en"] = {q: float(np.quantile(r, float(q))) for q in
                                      ("0.001", "0.01", "0.05", "0.5", "0.95", "0.99", "0.999")}
        out["max_ws_tokens"] = {"en": int(ne.max()), "fr": int(nf.max())}
        # by source length (\w+ tokens of the English side)
        L = Ln["n_w"][e]
        by = {}
        for lo, hi in ((0, 10), (11, 20), (21, 30), (31, 40), (41, 10 ** 6)):
            m = (L >= lo) & (L <= hi)
            if m.any():
                by[f"{lo}-{hi if hi < 10 ** 6 else 'max'}"] = {
                    "rows": int(m.sum()), "lid_any": float(g["lid_any"][m].mean()),
                    "ratio_ws_1.5": float(g["ratio_ws_1.5"][m].mean())}
        out["by_source_w_tokens"] = by
        if d.get("src") is not None:
            bs = {}
            for lab_ in sorted(set(d["src"].tolist())):
                m = d["src"] == lab_
                bs[lab_] = {"rows": int(m.sum()), "lid_any": float(g["lid_any"][m].mean()),
                            "en_wrong": float(g["en_wrong"][m].mean()),
                            "fr_wrong": float(g["fr_wrong"][m].mean()),
                            "ratio_ws_1.5": float(g["ratio_ws_1.5"][m].mean())}
            out["by_source_label"] = bs
        for k in ("pairs_not_in_parent", "pairs_with_several_labels_in_parent"):
            if k in d:
                out[k] = d[k]
        return out, g

    res_sets, G = {}, {}
    for name, d in S.items():
        s_, g = summary(name, d)
        res_sets[name] = s_
        G[name] = g
        log(f"{name:46s} {s_['rows']:>8,}  lid_any {s_['shares']['lid_any']:.4%}  "
            f"ratio_ws_1.5 {s_['shares']['ratio_ws_1.5']:.4%}  L0 {s_['counts']['L0_violation']}")

    keys = (["lid_any", "en_wrong", "fr_wrong", "lid_any_conf", "swap_or_copy_2way",
             "same_top1_both_sides", "identical_sides"]
            + [f"ratio_{u}_{t}" for u in UNITS for t in THRESH]
            + [f"ratio_{u}_tukey_{r}" for u in UNITS for r in ("ft_random", "ft_random[un]")]
            + ["either_headline", "L0_violation"])
    contrasts = []
    for x, y, what in CONTRASTS:
        A, B_ = res_sets[x], res_sets[y]
        m = {}
        for k in keys:
            pa, pb = A["shares"][k], B_["shares"][k]
            sd = math.sqrt(pa * (1 - pa) / A["rows"] + pb * (1 - pb) / B_["rows"])
            m[k] = {"a_count": A["counts"][k], "b_count": B_["counts"][k],
                    "a_pct": 100 * pa, "b_pct": 100 * pb, "diff_pp": 100 * (pa - pb),
                    "binomial_sd_diff_pp": 100 * sd}
        contrasts.append({"a": x, "b": y, "what": what, "a_rows": A["rows"], "b_rows": B_["rows"],
                          "measures": m})

    # 20 examples, seed 42: 7 / 7 / 6 from the three UN sets, rows flagged by either headline rule
    rng = random.Random(42)
    examples = []
    for s, k in (("ft_un_rep", 7), ("ft_un_div", 7), ("ft_un_random", 6)):
        cand = np.flatnonzero(G[s]["either_headline"]).tolist()
        for i in sorted(rng.sample(cand, min(k, len(cand)))):
            e, f = int(S[s]["e"][i]), int(S[s]["f"][i])
            why = [w for w in ("en_wrong", "fr_wrong", "ratio_ws_1.5_fr_long", "ratio_ws_1.5_fr_short",
                               "identical_sides") if G[s][w][i]]
            examples.append({"set": s, "line": i + 1, "why": why,
                             "en_top1": [names[LID["dom"][e]], round(float(LID["ptop"][e]), 4)],
                             "fr_top1": [names[LID["dom"][f]], round(float(LID["ptop"][f]), 4)],
                             "en_ws": int(Ln["n_ws"][e]), "fr_ws": int(Ln["n_ws"][f]),
                             "en": U[e], "fr": U[f]})
    # sample-API cross-check results (run.sh writes them)
    xc = None
    p = WORK / "xcheck_out.tsv"
    if p.exists():
        sample = json.loads((WORK / "xcheck_ids.json").read_text())
        labs2 = [ln.rstrip("\n") for ln in open(p, encoding="utf-8")]
        mism = [int(i) for i, l2 in zip(sample, labs2) if names[LID["dom"][i]] != l2]
        xc = {"strings": len(sample), "rows_returned": len(labs2), "label_mismatches": len(mism),
              "ok": len(labs2) == len(sample) and not mism}
        checks.append({"check": "nl_lid dominant label == NLLanguageRecognizer.dominantLanguage(for:) "
                                "(class method, separate binary) on 6,000 random unique strings", **xc})
    import sentencepiece
    res = {
        "item": "WMT2027_PLAN.md §3.2 item 3: independent check (descriptive only)",
        "lid_tool": "Apple NaturalLanguage NLLanguageRecognizer (macOS system framework), NOT py3langid",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "sentencepiece": sentencepiece.__version__, "machine": platform.machine(),
                        "macos": platform.mac_ver()[0], "lid_labels_seen": names},
        "inputs_sha256": pins, "feasibility_report_sha256": FEAS_REPORT_SHA,
        "unique_strings": n, "prep": prep,
        "checks": checks, "tukey_fences_log_r": fences,
        "sets": res_sets, "contrasts": contrasts, "examples_20": examples,
        "seconds": round(time.time() - t0, 1),
    }
    tmp = OUT / "results.json.tmp"
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT / "results.json")
    log(f"wrote {OUT / 'results.json'} in {time.time() - t0:.0f}s")


def stage_sample(a):
    """6,000 random unique-string ids (seed 42) for the class-method cross-check."""
    prep = json.loads((WORK / "prep.json").read_text())
    n = prep["unique_strings"]
    ids_ = sorted(random.Random(42).sample(range(n), 6000))
    U = []
    for k in range(NCHUNK):
        U += text_lines(WORK / f"lid_in_{k}.txt")
    with open(WORK / "xcheck_in.txt", "w", encoding="utf-8", newline="\n") as f:
        for i in ids_:
            f.write(U[i] + "\n")
    (WORK / "xcheck_ids.json").write_text(json.dumps(ids_))
    log("wrote 6,000-string cross-check sample")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "sample", "stats"])
    ap.add_argument("--sums", required=True)
    a = ap.parse_args()
    {"prep": stage_prep, "sample": stage_sample, "stats": stage_stats}[a.stage](a)
