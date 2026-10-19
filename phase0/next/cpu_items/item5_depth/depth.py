#!/usr/bin/env python3
"""WMT2027_PLAN §3.2 item 5: depth descriptives for W2 along the QE ranking of the scored pool.

Descriptive CPU analysis for the author's decisions, not a registered result.

Stages (run in this order; each writes under --work and the next one reads it):
  textsha  one sequential pass over v2_scored.tsv: sha256 of the whole file, of its source and
           target columns re-joined as files (must equal the fixed corpus's train.clean.{en,fr}
           sha256, which ties every TSV row to the corpus row the provenance labels index), and
           E0.3's blake2b first/last-16MiB fingerprint.
  pass1    score, pair hash and held-out reservation flags for every row (E0.3's pass 1).
  rank     E0.3's ranking: pool = pool_mask_reused & not reserved; order = stable argsort of
           -float32(score) (ties by row index); top-k then exact-pair dedup, first copy kept in
           score order. Checks the top 1,000,000 against the E0.3 manifest and ft_topk files.
  pass2    for every dedup row in the deepest subset: lowercased \\w+ tokens (both sides), source
           8-gram keys (E0.4's blake2b-64), raw-source hash.
  reduce   per depth: rep > 0 share (rep within that depth's dedup subset), duplicate rates,
           source composition, word types (whole set and at matched token counts).

Usage: depth.py <stage> [--work DIR] [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from array import array
from pathlib import Path

import numpy as np

HOME = Path.home()
DEF_TSV = HOME / "e06data" / "v2_scored.tsv"
DEF_E05 = HOME / "e06data" / "e05_src"
DEF_WORK = HOME / "cpu_items" / "item5_depth"

WORD = re.compile(r"\w+", re.UNICODE)          # e04_split.py / e05_build.py token rule
FRACS = [0.0013, 0.003, 0.01, 0.06, 0.10, 0.12, 0.25, 0.30, 0.50]
BAND_K = 1_000_000                             # E0.3's n_ft: the band itself (about 6.007%)
N_PART = 16                                    # 8-gram key partitions (top 4 bits)
UNMATCHED = 255
CORPUS_SHA = {"en": "c8cc665c8fc9971bf689bb02d7409bc76bbb489cb7b7dab6b35d13778863e655",
              "fr": "e4f5301af4c49b8cd91ff81807db08116a7da94f952ba0ee3962e82c0618421a"}
FP_CHUNK = 16 * 1024 * 1024


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def die(msg):
    sys.exit(f"ERROR: {msg}")


# ---- E0.3's keys (phase0/e03_build_controls.py) -------------------------------------
def h64(src: str, tgt: str) -> int:
    d = hashlib.blake2b(f"{src}\x00{tgt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def toks(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


def boundaries(path: Path, n: int) -> list[int]:
    size = path.stat().st_size
    b = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()
            b.append(f.tell())
    b.append(size)
    if b != sorted(b):
        die("chunk boundaries not monotone")
    return b


def iter_lines(path: Path, start: int, end: int):
    with open(path, "rb", buffering=1 << 24) as f:
        f.seek(start)
        pos = start
        for line in f:
            if pos >= end:
                break
            pos += len(line)
            yield line


def fingerprint(path: Path) -> dict:
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
        f.seek(max(0, size - FP_CHUNK))
        tail = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
    return {"size": size, "blake2b_first_16MiB": head, "blake2b_last_16MiB": tail}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---- stage: textsha ------------------------------------------------------------------
def stage_textsha(a):
    tsv = Path(a.tsv)
    whole, en, fr = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n = bad = 0
    last_nl = True
    with open(tsv, "rb", buffering=1 << 24) as f:
        for line in f:
            whole.update(line)
            p = line.split(b"\t")
            if len(p) != 3:
                bad += 1
            en.update(p[1])
            en.update(b"\n")
            last_nl = line.endswith(b"\n")
            fr.update(p[2] if last_nl else p[2] + b"\n")
            n += 1
            if n % 5_000_000 == 0:
                log(f"textsha {n:,}")
    out = {"rows": n, "rows_not_3_fields": bad, "last_line_has_newline": last_nl,
           "sha256_tsv": whole.hexdigest(), "sha256_en_column": en.hexdigest(),
           "sha256_fr_column": fr.hexdigest(),
           "en_column_equals_fixed_corpus": en.hexdigest() == CORPUS_SHA["en"],
           "fr_column_equals_fixed_corpus": fr.hexdigest() == CORPUS_SHA["fr"],
           "fingerprint": fingerprint(tsv)}
    Path(a.work, "work", "textsha.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out, indent=1))


# ---- stage: pass1 --------------------------------------------------------------------
def p1_worker(args):
    i, tsv, start, end, work, held = args
    hp, hs, ht = held
    sc, ph, code = array("d"), array("Q"), array("B")
    ckpt = array("q")                          # byte offset of every 100,000th row in the chunk
    nbad = 0
    pos = start
    for k, line in enumerate(iter_lines(Path(tsv), start, end)):
        if k % 100_000 == 0:
            ckpt.append(pos)
        pos += len(line)
        parts = line.decode("utf-8").rstrip("\n").split("\t")
        if len(parts) < 3:
            raise RuntimeError(f"chunk {i} line {k}: {len(parts)} fields")
        if len(parts) != 3:
            nbad += 1
        sc.append(float(parts[0]))
        s, t = parts[1], parts[2]
        h = h64(s, t)
        ph.append(h)
        c = 0
        if h in hp:
            c |= 1
        if norm(s) in hs:
            c |= 2
        if norm(t) in ht:
            c |= 4
        code.append(c)
    np.savez(Path(work, "work", f"p1_{i}.npz"), score=np.frombuffer(sc, np.float64),
             pair_h=np.frombuffer(ph, np.uint64), code=np.frombuffer(code, np.uint8),
             ckpt=np.frombuffer(ckpt, np.int64))
    return i, len(sc), nbad


def read_lines(p: Path) -> list[str]:
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def stage_pass1(a):
    import multiprocessing as mp
    tsv = Path(a.tsv)
    he, hf = read_lines(Path(a.e05, "heldout_un.en")), read_lines(Path(a.e05, "heldout_un.fr"))
    if len(he) != 2000 or len(hf) != 2000:
        die("heldout_un is not 2000 rows")
    held = ({h64(s, t) for s, t in zip(he, hf)}, {norm(s) for s in he}, {norm(t) for t in hf})
    b = boundaries(tsv, a.workers)
    jobs = [(i, str(tsv), b[i], b[i + 1], a.work, held) for i in range(a.workers)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = sorted(pool.map(p1_worker, jobs))
    meta = {"boundaries": b, "rows_per_chunk": [r[1] for r in res],
            "rows_not_3_fields": sum(r[2] for r in res),
            "heldout_keys": {"pairs": len(held[0]), "norm_sources": len(held[1]),
                             "norm_targets": len(held[2])}}
    Path(a.work, "work", "pass1.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


# ---- stage: rank ---------------------------------------------------------------------
def load_p1(a):
    meta = json.loads(Path(a.work, "work", "pass1.json").read_text())
    parts = [np.load(Path(a.work, "work", f"p1_{i}.npz")) for i in range(len(meta["rows_per_chunk"]))]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("score", "pair_h", "code")}
    # absolute byte offsets of every 100,000th row (global row index = 100,000 * j within chunk)
    ck = []
    off = 0
    for p, n in zip(parts, meta["rows_per_chunk"]):
        rows = off + 100_000 * np.arange(len(p["ckpt"]), dtype=np.int64)
        ck.append(np.stack([rows, p["ckpt"]], 1))
        off += n
    return meta, cat, np.concatenate(ck)


def stage_rank(a):
    inp = Path(a.work, "inputs")
    man = json.loads(Path(inp, "controls_manifest.json").read_text())
    meta, cat, ckpt = load_p1(a)
    score64, pair_h, code = cat["score"], cat["pair_h"], cat["code"]
    n = len(score64)
    lab = np.load(inp / "provenance_labels.npy", allow_pickle=False)
    mask = np.load(inp / "pool_mask_reused.npy", allow_pickle=False)
    rep = json.loads(Path(inp, "provenance_report.json").read_text())
    names = rep["sources"]
    chk = {}
    chk["rows"] = {"tsv": n, "labels": len(lab), "mask": len(mask),
                   "manifest_n_pool_total": man["n_pool_total"]}
    if not (n == len(lab) == len(mask) == man["n_pool_total"]):
        die(f"row counts disagree: {chk['rows']}")
    if mask.dtype != np.bool_:
        die("mask not bool")
    scores = score64.astype(np.float32)                 # E0.3: np.asarray(array('d'), float32)
    nan = np.isnan(scores)
    chk["nan_in_mask"] = int((nan & mask).sum())
    chk["nan_outside_mask"] = int((nan & ~mask).sum())
    if chk["nan_in_mask"]:
        die("NaN scores inside the pool mask")
    res = {"rows_reserved_total": int((code != 0).sum()),
           "exact_pair": int(((code & 1) != 0).sum()),
           "extra_by_source": int((((code & 2) != 0) & ((code & 1) == 0)).sum()),
           "extra_by_target": int((((code & 4) != 0) & ((code & 3) == 0)).sum())}
    want = {k: man["reservation"][k] for k in res}
    chk["reservation"] = {"measured": res, "e03_manifest": want, "equal": res == want}
    if res != want:
        die(f"reservation differs from E0.3: {res} vs {want}")
    reserved = code != 0
    chk["reserved_inside_mask"] = int((reserved & mask).sum())
    avail = np.flatnonzero(mask & ~reserved)
    N = len(avail)
    log(f"pool {mask.sum():,} masked, {N:,} after reservation")
    order = avail[np.argsort(-scores[avail], kind="stable")]

    # ---- the E0.3 band check: top 1,000,000 -> 926,999 unique, row for row ------------
    top = order[:BAND_K]
    _, first = np.unique(pair_h[top], return_index=True)
    topk = top[np.sort(first)]
    s = scores[topk]
    cnt = {nm: int((lab[topk] == c).sum()) for c, nm in enumerate(names)}
    cnt["unmatched"] = int((lab[topk] == UNMATCHED).sum())
    e = man["ft_topk"]
    band = {"n_unique": int(len(topk)), "duplicates_removed": int(BAND_K - len(topk)),
            "qe_min": float(s.min()), "qe_max": float(s.max()), "qe_mean_f32": float(s.mean()),
            "source_counts": cnt}
    band["equal_to_manifest"] = {
        "n_unique": band["n_unique"] == man["n_unique_per_set"],
        "duplicates_removed": band["duplicates_removed"] == man["topk_duplicates_removed"],
        "qe_min": band["qe_min"] == e["qe_min"], "qe_max": band["qe_max"] == e["qe_max"],
        "qe_mean_abs_diff": abs(band["qe_mean_f32"] - e["qe_mean"]),
        "source_counts": cnt == e["source_counts"]}
    fe, ff = read_lines(Path(a.e05, "ft_topk.en")), read_lines(Path(a.e05, "ft_topk.fr"))
    fh = np.fromiter((h64(x, y) for x, y in zip(fe, ff)), np.uint64, len(fe))
    band["ft_topk_files_row_for_row"] = bool(len(fh) == len(topk) and np.array_equal(fh, pair_h[topk]))
    src_lab = read_lines(Path(a.e05, "ft_topk.source"))
    code_of = {nm: c for c, nm in enumerate(names)} | {"unmatched": UNMATCHED}
    sl = np.array([code_of[x] for x in src_lab], np.uint8)
    band["ft_topk_source_labels_row_for_row"] = bool(len(sl) == len(topk) and np.array_equal(sl, lab[topk]))
    del fe, ff, fh, src_lab, sl
    chk["band"] = band
    log(json.dumps(band))
    if not (band["ft_topk_files_row_for_row"] and band["ft_topk_source_labels_row_for_row"]
            and band["equal_to_manifest"]["n_unique"]):
        die("the top 1,000,000 does not reproduce E0.3's ft_topk")

    # ---- depths ---------------------------------------------------------------------
    depths = [{"name": f"{f * 100:g}%", "frac": f, "k": int(round(f * N))} for f in FRACS]
    depths.append({"name": "E0.3 band", "frac": BAND_K / N, "k": BAND_K})
    depths.sort(key=lambda d: (d["k"], d["name"]))
    K = max(d["k"] for d in depths)
    ordK = order[:K]
    _, first = np.unique(pair_h[ordK], return_index=True)
    dd_pos = np.sort(first)                               # rank positions of first copies
    dd_rows = ordK[dd_pos]
    M = len(dd_rows)
    s_ord = scores[ordK]
    for d in depths:
        k = d["k"]
        d["n_dedup"] = int(np.searchsorted(dd_pos, k))
        d["qe_cut"] = float(s_ord[k - 1])
        d["qe_top"] = float(s_ord[0])
        tie = s_ord[k - 1]
        d["rows_tied_with_cut_score_in_pool"] = int((scores[avail] == tie).sum())
        d["rows_tied_with_cut_score_inside"] = int((s_ord[:k] == tie).sum())
        lr = lab[ordK[:k]]
        d["un_share_raw"] = float((lr == code_of["un"]).mean())
    pos_of_row = np.full(n, -1, np.int32)
    pos_of_row[dd_rows] = np.arange(M, dtype=np.int32)
    np.save(Path(a.work, "work", "pos_of_row.npy"), pos_of_row)
    np.savez(Path(a.work, "work", "rank.npz"), dd_rows=dd_rows, dd_pos=dd_pos,
             lab=lab[dd_rows], score=scores[dd_rows], ckpt=ckpt)
    out = {"N_ranked_pool": int(N), "pool_mask_true": int(mask.sum()), "checks": chk,
           "depths": depths, "M_dedup_rows_deepest": int(M), "label_names": names}
    Path(a.work, "work", "rank.json").write_text(json.dumps(out, indent=1))
    log(f"{len(depths)} depths, deepest k={K:,}, {M:,} dedup rows")


# ---- stage: pass2 --------------------------------------------------------------------
def p2_worker(args):
    i, tsv, row0, row1, byte0, work = args
    pos_of_row = np.load(Path(work, "work", "pos_of_row.npy"), mmap_mode="r")
    sub = np.asarray(pos_of_row[row0:row1])
    wd = Path(work, "work")
    occ_dir = wd / "occ"
    kbuf = [array("Q") for _ in range(N_PART)]
    qbuf = [array("i") for _ in range(N_PART)]
    kf = [open(occ_dir / f"k{p:02d}_w{i}.bin", "wb") for p in range(N_PART)]
    qf = [open(occ_dir / f"q{p:02d}_w{i}.bin", "wb") for p in range(N_PART)]
    tok_en, tok_fr = open(wd / f"tok_en_w{i}.bin", "wb"), open(wd / f"tok_fr_w{i}.bin", "wb")
    voc = {"en": {}, "fr": {}}
    fq = {"en": array("i"), "fr": array("i")}
    qs, len_en, len_fr, src_h = array("i"), array("i"), array("i"), array("Q")
    off_en = off_fr = 0
    offs_en, offs_fr = array("q"), array("q")
    blake = hashlib.blake2b
    nbuf = 0
    r = row0
    with open(tsv, "rb", buffering=1 << 24) as f:
        f.seek(byte0)
        for line in f:
            if r >= row1:
                break
            q = int(sub[r - row0])
            r += 1
            if q < 0:
                continue
            parts = line.decode("utf-8").rstrip("\n").split("\t")
            src, tgt = parts[1], parts[2]
            te, tf = toks(src), toks(tgt)
            qs.append(q)
            len_en.append(len(te))
            len_fr.append(len(tf))
            src_h.append(int.from_bytes(blake(src.encode("utf-8"), digest_size=8,
                                              person=b"item5src").digest(), "little"))
            for side, tl, fh in (("en", te, tok_en), ("fr", tf, tok_fr)):
                v, fqs = voc[side], fq[side]
                ids = array("I")
                for w in tl:
                    j = v.get(w)
                    if j is None:
                        j = len(v)
                        v[w] = j
                        fqs.append(q)
                    elif q < fqs[j]:
                        fqs[j] = q
                    ids.append(j)
                fh.write(ids.tobytes())
            offs_en.append(off_en)
            offs_fr.append(off_fr)
            off_en += len(te)
            off_fr += len(tf)
            for j in range(len(te) - 7):
                kk = int.from_bytes(blake(" ".join(te[j:j + 8]).encode("utf-8"), digest_size=8).digest(), "little")
                p = kk >> 60
                kbuf[p].append(kk)
                qbuf[p].append(q)
                nbuf += 1
            if nbuf >= 4_000_000:
                for p in range(N_PART):
                    kf[p].write(kbuf[p].tobytes()); qf[p].write(qbuf[p].tobytes())
                    kbuf[p], qbuf[p] = array("Q"), array("i")
                nbuf = 0
    if r != row1:
        raise RuntimeError(f"worker {i}: stopped at row {r} before {row1}")
    for p in range(N_PART):
        kf[p].write(kbuf[p].tobytes()); qf[p].write(qbuf[p].tobytes())
        kf[p].close(); qf[p].close()
    tok_en.close(); tok_fr.close()
    np.savez(wd / f"p2_w{i}.npz", q=np.frombuffer(qs, np.int32), len_en=np.frombuffer(len_en, np.int32),
             len_fr=np.frombuffer(len_fr, np.int32), src_h=np.frombuffer(src_h, np.uint64),
             off_en=np.frombuffer(offs_en, np.int64), off_fr=np.frombuffer(offs_fr, np.int64),
             fq_en=np.frombuffer(fq["en"], np.int32), fq_fr=np.frombuffer(fq["fr"], np.int32))
    for side in ("en", "fr"):
        words = sorted(voc[side], key=voc[side].get)
        (wd / f"vocab_{side}_w{i}.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    return i, len(qs)


def stage_pass2(a):
    import multiprocessing as mp
    wd = Path(a.work, "work")
    (wd / "occ").mkdir(exist_ok=True)
    for old in (wd / "occ").glob("*.bin"):
        old.unlink()
    rk = np.load(wd / "rank.npz")
    ckpt = rk["ckpt"]                                    # (row, byte offset) of every 100,000th row
    pos_of_row = np.load(wd / "pos_of_row.npy", mmap_mode="r")
    need = np.flatnonzero(np.asarray(pos_of_row) >= 0)
    hi = int(need.max()) + 1
    # split rows [0, hi) into workers with equal needed-row counts, at checkpoint rows
    targets = [need[len(need) * j // a.workers] for j in range(1, a.workers)]
    cut_rows = [0]
    for t in targets:
        j = int(np.searchsorted(ckpt[:, 0], t, side="right")) - 1
        cut_rows.append(int(ckpt[j, 0]))
    cut_rows.append(hi)
    cut_rows = sorted(set(cut_rows))
    byte_of = {int(r): int(b) for r, b in ckpt}
    jobs = [(i, a.tsv, cut_rows[i], cut_rows[i + 1], byte_of[cut_rows[i]], a.work)
            for i in range(len(cut_rows) - 1)]
    log(f"pass2: {len(need):,} rows in [0, {hi:,}), chunks at rows {cut_rows}")
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = sorted(pool.map(p2_worker, jobs))
    meta = {"chunks": [list(j[1:5]) for j in jobs], "rows_per_worker": [r[1] for r in res]}
    (wd / "pass2.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))


# ---- stage: reduce -------------------------------------------------------------------
def stage_reduce(a):
    wd = Path(a.work, "work")
    R = json.loads((wd / "rank.json").read_text())
    rk = np.load(wd / "rank.npz")
    p2 = json.loads((wd / "pass2.json").read_text())
    W = len(p2["rows_per_worker"])
    M = R["M_dedup_rows_deepest"]
    lab = rk["lab"]
    names = R["label_names"]
    un = names.index("un")
    depths = R["depths"]
    cuts = np.array([d["n_dedup"] for d in depths], np.int64)   # sorted ascending
    if not np.all(np.diff(cuts) >= 0):
        die("cuts not sorted")

    # assemble per-row arrays indexed by dedup rank q
    len_en = np.full(M, -1, np.int32); len_fr = np.full(M, -1, np.int32)
    src_h = np.zeros(M, np.uint64)
    wid = np.full(M, -1, np.int8)
    off_en = np.zeros(M, np.int64); off_fr = np.zeros(M, np.int64)
    P = []
    for w in range(W):
        z = np.load(wd / f"p2_w{w}.npz")
        q = z["q"]
        if (wid[q] >= 0).any():
            die("a row was processed twice")
        wid[q] = w
        len_en[q], len_fr[q], src_h[q] = z["len_en"], z["len_fr"], z["src_h"]
        off_en[q], off_fr[q] = z["off_en"], z["off_fr"]
        P.append(z)
    if (wid < 0).any():
        die(f"{int((wid < 0).sum())} rows missing from pass 2")
    log(f"assembled {M:,} rows")

    # ---- rep (E0.4) within each depth's dedup subset --------------------------------
    BIG = np.int64(M + 1)
    ngr = np.maximum(len_en.astype(np.int64) - 7, 0)
    rowmin = np.full(M, BIG, np.int64)
    rep_hist = np.zeros(len(cuts) + 1, np.float64)
    n_occ = 0
    for p in range(N_PART):
        ks = np.concatenate([np.fromfile(wd / "occ" / f"k{p:02d}_w{w}.bin", np.uint64) for w in range(W)])
        qs = np.concatenate([np.fromfile(wd / "occ" / f"q{p:02d}_w{w}.bin", np.int32) for w in range(W)]).astype(np.int64)
        n_occ += len(ks)
        o = np.lexsort((qs, ks))
        ks, qs = ks[o], qs[o]
        del o
        new = np.empty(len(ks), bool)
        new[0] = True
        new[1:] = ks[1:] != ks[:-1]
        starts = np.flatnonzero(new)
        sizes = np.diff(np.append(starts, len(ks)))
        pos2 = np.where(sizes >= 2, qs[np.minimum(starts + 1, len(ks) - 1)], BIG)
        pos2_occ = np.repeat(pos2, sizes)
        np.minimum.at(rowmin, qs, pos2_occ)
        u = np.maximum(qs, pos2_occ)
        b = np.searchsorted(cuts, u, side="right")
        rep_hist += np.bincount(b, weights=1.0 / ngr[qs], minlength=len(cuts) + 1)
        log(f"partition {p}: {len(ks):,} occurrences")
        del ks, qs, new, starts, sizes, pos2, pos2_occ, u, b
    if n_occ != int(ngr.sum()):
        die(f"8-gram occurrences {n_occ:,} != expected {int(ngr.sum()):,}")
    t = np.maximum(np.arange(M, dtype=np.int64), rowmin)        # row is rep>0 in prefix c iff t < c
    rep_sum = np.cumsum(rep_hist)[:len(cuts)]
    is_un = lab == un

    # ---- duplicate sources (e05_build.duplicate_source_fraction) ----------------------
    _, first_src = np.unique(src_h, return_index=True)
    first_src = np.sort(first_src)

    # ---- word types: whole set, by first occurrence ---------------------------------
    gvoc, l2g, gfq = {}, {}, {}
    for side in ("en", "fr"):
        g = {}
        maps = []
        for w in range(W):
            words = (wd / f"vocab_{side}_w{w}.txt").read_text(encoding="utf-8").split("\n")[:-1]
            m = np.empty(len(words), np.int64)
            for j, x in enumerate(words):
                m[j] = g.setdefault(x, len(g))
            maps.append(m)
        fqa = np.full(len(g), BIG, np.int64)
        for w in range(W):
            np.minimum.at(fqa, maps[w], P[w][f"fq_{side}"].astype(np.int64))
        gvoc[side], l2g[side], gfq[side] = len(g), maps, np.sort(fqa)
        del g
    log(f"vocab: en {gvoc['en']:,} fr {gvoc['fr']:,}")

    tok = {side: [np.memmap(wd / f"tok_{side}_w{w}.bin", np.uint32, "r") for w in range(W)]
           for side in ("en", "fr")}
    lens = {"en": len_en, "fr": len_fr}
    offs = {"en": off_en, "fr": off_fr}

    def tokens_total(side, n):
        return int(lens[side][:n].astype(np.int64).sum())

    def types_at(side, idx, B):
        """vocab/common.py types_at(): distinct words in the first B tokens of the rows in idx's order"""
        L = lens[side][idx].astype(np.int64)
        cum = np.cumsum(L)
        if cum[-1] < B:
            return None
        m = int(np.searchsorted(cum, B, side="left"))        # rows idx[0..m] reach B tokens
        take = idx[:m + 1]
        last_take = int(B - (cum[m - 1] if m else 0))
        ids = []
        for j, qq in enumerate(take):
            w = int(wid[qq]); o = int(offs[side][qq]); ln = int(lens[side][qq])
            if j == m:
                ln = last_take
            ids.append(l2g[side][w][np.asarray(tok[side][w][o:o + ln], np.int64)])
        allids = np.concatenate(ids)
        if len(allids) != B:
            die(f"types_at gathered {len(allids)} != {B}")
        return int(len(np.unique(allids)))

    # matched token budgets: B1 reachable by every depth (both sides), B2 by depths >= 1%,
    # B3 = E0.6's French budget (6,500,000) for depths that reach it
    def round_down_2sig(x):
        e = 10 ** (len(str(int(x))) - 2)
        return int(x // e * e)
    tot = {side: [tokens_total(side, int(c)) for c in cuts] for side in ("en", "fr")}
    B1 = round_down_2sig(min(min(tot["en"]), min(tot["fr"])))
    ge1 = [j for j, d in enumerate(depths) if d["frac"] >= 0.01]
    B2 = round_down_2sig(min(min(tot["en"][j] for j in ge1), min(tot["fr"][j] for j in ge1)))
    B3 = 6_500_000
    budgets = {"B1": B1, "B2": B2, "B3": B3}
    log(f"budgets {budgets}")

    out_depths = []
    for j, d in enumerate(depths):
        c = int(cuts[j])
        k = d["k"]
        lc = lab[:c]
        comp = {nm: int((lc == code).sum()) for code, nm in enumerate(names)}
        comp["unmatched"] = int((lc == UNMATCHED).sum())
        rp = t[:c] < c
        dd = dict(d)
        dd.update({
            "exact_pair_duplicate_rate_raw": 1 - c / k,
            "duplicate_source_rate_dedup": 1 - int(np.searchsorted(first_src, c)) / c,
            "un_share_dedup": comp["un"] / c,
            "source_counts_dedup": comp,
            "rep_positive_share": float(rp.mean()),
            "rep_positive_rows": int(rp.sum()),
            "rep_mean": float(rep_sum[j] / c),
            "un_rows_rep_positive": int((rp & is_un[:c]).sum()),
            "un_rows_rep_zero": int((~rp & is_un[:c]).sum()),
            "rep_positive_share_within_un": float(rp[is_un[:c]].mean()) if is_un[:c].any() else None,
            "src_len_mean": float(len_en[:c].mean()),
            "tokens_en": tot["en"][j], "tokens_fr": tot["fr"][j],
            "types_whole_set_en": int(np.searchsorted(gfq["en"], c)),
            "types_whole_set_fr": int(np.searchsorted(gfq["fr"], c)),
        })
        dd["types_matched"] = {}
        for seed in range(5):
            idx = list(range(c))
            random.Random(seed).shuffle(idx)
            idx = np.asarray(idx, np.int64)
            for bname, B in budgets.items():
                if seed and bname != "B1":
                    continue
                for side in ("en", "fr"):
                    dd["types_matched"].setdefault(f"{bname}_{side}", {})[f"seed{seed}"] = types_at(side, idx, B)
        out_depths.append(dd)
        log(f"{d['name']}: k={k:,} dedup={c:,} rep>0={rp.mean():.4f} UN={comp['un']/c:.4f} "
            f"dupsrc={dd['duplicate_source_rate_dedup']:.4f} B1en={dd['types_matched']['B1_en']['seed0']}")
    res = {"definition": {
               "pool": "pool_mask_reused.npy (16,646,992 rows that carry the paper's CometKiwi score; "
                       "E0.3 decision D1) minus E0.3's held-out reservation",
               "ranking": "stable argsort of -float32(score); ties broken by ascending row index (e03_build_controls.py)",
               "depth_k": "round(frac * N_ranked_pool) rows from the top; 'E0.3 band' is k = 1,000,000",
               "subset": "the top k rows, then exact (src, tgt) dedup, first copy kept in score order (E0.3's rule)",
               "rep": "E0.4 (e04_split.py): per row, fraction of its source 8-grams (lowercased \\w+, blake2b-64) "
                      "whose count over the depth's dedup subset is > 1; rows under 8 tokens rep = 0",
               "duplicate_source_rate": "1 - distinct raw source strings / rows, on the dedup subset (e05_build.py)",
               "exact_pair_duplicate_rate_raw": "1 - dedup rows / k",
               "un_share": "provenance label 'un' (prov_v2_fixed, e01 exact) on the dedup subset; un_share_raw on the top k",
               "types_matched": "vocab/common.py types_at(): rows of the dedup subset in score order, "
                                "random.Random(seed).shuffle, distinct lowercased \\w+ words in the first B tokens",
               "budgets": budgets},
           "N_ranked_pool": R["N_ranked_pool"], "pool_mask_true": R["pool_mask_true"],
           "M_dedup_rows_deepest": M, "eightgram_occurrences": n_occ,
           "vocab_deepest": gvoc, "checks": R["checks"], "depths": out_depths}
    ts = wd / "textsha.json"
    if ts.exists():
        res["checks"]["textsha"] = json.loads(ts.read_text())
    res["checks"]["pass1"] = json.loads((wd / "pass1.json").read_text())
    res["checks"]["pass2"] = p2
    res["python"] = sys.version.split()[0]
    res["unidata_version"] = unicodedata.unidata_version
    res["numpy"] = np.__version__
    Path(a.work, "item5_depth_results.json").write_text(json.dumps(res, indent=1))
    log("wrote item5_depth_results.json")


# ---- stage: sized (supplement) -------------------------------------------------------
SIZES = [20_621, 214_489, 926_999]   # the 0.13% subset's size, the E0.5/E0.6 set size, the band


def stage_sized(a):
    """rep > 0 share, mean rep, duplicate-source rate and UN share in fixed-size uniform random
    samples of each depth's dedup subset, rep computed within the sample. Within-subset rep and
    duplicate rates grow with subset size, so only these are comparable across depths."""
    wd = Path(a.work, "work")
    resp = Path(a.work, "item5_depth_results.json")
    res = json.loads(resp.read_text())
    R = json.loads((wd / "rank.json").read_text())
    rk = np.load(wd / "rank.npz")
    lab = rk["lab"]
    un = R["label_names"].index("un")
    p2 = json.loads((wd / "pass2.json").read_text())
    W = len(p2["rows_per_worker"])
    M = R["M_dedup_rows_deepest"]
    len_en = np.zeros(M, np.int32); src_h = np.zeros(M, np.uint64)
    for w in range(W):
        z = np.load(wd / f"p2_w{w}.npz")
        len_en[z["q"]], src_h[z["q"]] = z["len_en"], z["src_h"]
    ngr = np.maximum(len_en.astype(np.int64) - 7, 0)
    samples = []
    for j, d in enumerate(R["depths"]):
        c = d["n_dedup"]
        for n in SIZES:
            if n > c:
                continue
            for seed in ((0, 1, 2) if n == SIZES[0] else (0,)):
                q = np.sort(np.random.default_rng(seed).choice(c, n, replace=False))
                samples.append({"depth": d["name"], "n": n, "seed": seed, "q": q})
    for s in samples:
        s["slot"] = np.full(M, -1, np.int32)
        s["slot"][s["q"]] = np.arange(len(s["q"]), dtype=np.int32)
        s["pos"] = np.zeros(len(s["q"]), bool)
        s["rep_sum"] = 0.0
    log(f"{len(samples)} samples")
    for p in range(N_PART):
        ks = np.concatenate([np.fromfile(wd / "occ" / f"k{p:02d}_w{w}.bin", np.uint64) for w in range(W)])
        qs = np.concatenate([np.fromfile(wd / "occ" / f"q{p:02d}_w{w}.bin", np.int32) for w in range(W)])
        for s in samples:
            sl = s["slot"][qs]
            m = sl >= 0
            kk, ss = ks[m], sl[m]
            _, inv, cnt = np.unique(kk, return_inverse=True, return_counts=True)
            hit = cnt[inv] > 1
            s["pos"][ss[hit]] = True
            s["rep_sum"] += float((1.0 / ngr[s["q"][ss[hit]]]).sum())
        log(f"partition {p}")
    out = []
    for s in samples:
        q = s["q"]; n = len(q)
        isu = lab[q] == un
        out.append({"depth": s["depth"], "n": n, "seed": s["seed"],
                    "rep_positive_share": float(s["pos"].mean()),
                    "rep_mean": s["rep_sum"] / n,
                    "rep_positive_share_within_un": float(s["pos"][isu].mean()),
                    "duplicate_source_rate": 1 - len(np.unique(src_h[q])) / n,
                    "un_share": float(isu.mean()),
                    "src_len_mean": float(len_en[q].mean())})
    res["size_matched"] = {
        "definition": "uniform random sample without replacement of n rows from the depth's dedup subset "
                      "(numpy default_rng(seed).choice); rep (E0.4) and duplicate sources computed within "
                      "the sample; no length matching",
        "sizes": SIZES, "samples": out}
    resp.write_text(json.dumps(res, indent=1))
    log("added size_matched to item5_depth_results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["textsha", "pass1", "rank", "pass2", "reduce", "sized"])
    ap.add_argument("--tsv", default=str(DEF_TSV))
    ap.add_argument("--e05", default=str(DEF_E05))
    ap.add_argument("--work", default=str(DEF_WORK))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.workers > 4:
        die("at most 4 workers")
    Path(a.work, "work").mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
