#!/usr/bin/env python3
"""Independent re-derivation of WMT2027_PLAN 3.2 item 5 (depth descriptives), written for the
item5_depth check. It does not import or copy the builder's depth.py / verify.py.

Definitions are taken from the sources, not from the builder:
  pool      pool_mask_reused.npy (E0.3 decision D1) minus E0.3's held-out reservation, which is
            re-derived here from the TEXT of heldout_un.{en,fr}: every row whose exact (src, tgt)
            pair, NFKC+casefold+whitespace-collapsed source, or normalised target equals one of a
            held-out row (e03_build_controls.py, reservation block). Exact strings, no hashing.
  ranking   float32(score) descending, ties by ascending row index (E0.3: stable argsort of
            -score over avail in ascending row order). Implemented with np.lexsort.
  depth     k = round(f * N) top rows of the ranking; band = k = 1,000,000 (E0.3 n_ft)
  subset    top k, exact-pair dedup keeping the first copy in ranking order (E0.3).
            Pair identity = md5(src_bytes + TAB + tgt_bytes) (128-bit; TAB cannot occur in a field).
  rep       E0.4 (e04_split.py): source side, lowercased \\w+ tokens; per row the fraction of its
            8-grams whose count over the subset is > 1; rows under 8 tokens rep = 0.
            8-gram identity: a 64-bit splitmix-chained hash over global token ids (not blake2b).
            Computed directly per depth (sort + membership), not by a prefix trick.
  dup src   1 - distinct raw source strings / rows (e05_build.py lexicon()); source identity md5.
  types@B   vocab/common.py types_at(): the subset's rows in ranking order, random.Random(seed)
            shuffle, distinct lowercased \\w+ words among the first B tokens; en and fr separately.
  UN share  provenance label == index of 'un' in provenance_report.json 'sources'.

Stages: pass1 | rank | pass2 | reduce | sized      (at most 3 processes: parent + 2 workers)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import re
import sys
import time
import unicodedata
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HOME = Path.home()
TSV = HOME / "e06data" / "v2_scored.tsv"
E05 = HOME / "e06data" / "e05_src"
BASE = HOME / "cpu_items" / "item5_depth_check"
INP, WORK = BASE / "inputs", BASE / "work"
WORD = re.compile(r"\w+", re.UNICODE)
FRACS = [("0.13%", 0.0013), ("0.3%", 0.003), ("1%", 0.01), ("6%", 0.06), ("10%", 0.10),
         ("12%", 0.12), ("25%", 0.25), ("30%", 0.30), ("50%", 0.50)]
BAND = 1_000_000
UNMATCHED = 255
NPROC_CHILD = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def nrm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def lines_of(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [x[:-1] if x.endswith("\n") else x for x in f]


def pkey(sb: bytes, tb: bytes) -> bytes:
    return hashlib.md5(sb + b"\t" + tb).digest()


def run_parallel(fn, jobs):
    """job 0 in this process, the rest on NPROC_CHILD children (3 processes in all)."""
    with ProcessPoolExecutor(max_workers=NPROC_CHILD) as ex:
        futs = [ex.submit(fn, j) for j in jobs[1:]]
        r0 = fn(jobs[0])
        return [r0] + [f.result() for f in futs]


# ------------------------------------------------------------------ pass1
def p1(job):
    i, start, end = job
    he, hf = lines_of(E05 / "heldout_un.en"), lines_of(E05 / "heldout_un.fr")
    hp = set(zip(he, hf)); hs = {nrm(x) for x in he}; ht = {nrm(x) for x in hf}
    off, sc, fl, nf = array("q"), array("d"), array("B"), array("B")
    keys = bytearray()
    pos = start
    with open(TSV, "rb", buffering=1 << 24) as f:
        f.seek(start)
        while pos < end:
            line = f.readline()
            if not line:
                break
            off.append(pos)
            pos += len(line)
            body = line[:-1] if line.endswith(b"\n") else line
            parts = body.split(b"\t")
            if len(parts) < 3:
                raise SystemExit(f"row at byte {off[-1]}: {len(parts)} fields")
            nf.append(min(len(parts), 255))
            sc.append(float(parts[0]))
            sb, tb = parts[1], parts[2]
            keys += pkey(sb, tb)
            s, t = sb.decode("utf-8"), tb.decode("utf-8")
            c = 0
            if (s, t) in hp:
                c |= 1
            if nrm(s) in hs:
                c |= 2
            if nrm(t) in ht:
                c |= 4
            fl.append(c)
    if pos != end:
        raise SystemExit(f"chunk {i}: ended at {pos}, expected {end}")
    np.save(WORK / f"p1_off_{i}.npy", np.frombuffer(off, np.int64))
    np.save(WORK / f"p1_sc_{i}.npy", np.frombuffer(sc, np.float64))
    np.save(WORK / f"p1_fl_{i}.npy", np.frombuffer(fl, np.uint8))
    np.save(WORK / f"p1_nf_{i}.npy", np.frombuffer(nf, np.uint8))
    np.save(WORK / f"p1_key_{i}.npy", np.frombuffer(bytes(keys), np.uint64).reshape(-1, 2))
    return i, len(off), {"pairs": len(hp), "norm_src": len(hs), "norm_tgt": len(ht)}


def stage_pass1(a):
    size = TSV.stat().st_size
    b = [0]
    with open(TSV, "rb") as f:
        for j in (1, 2):
            f.seek(size * j // 3 - 1)
            f.readline()                       # boundary = first line start at or after the cut
            b.append(f.tell())
    b.append(size)
    jobs = [(i, b[i], b[i + 1]) for i in range(3)]
    log("pass1 chunks", jobs)
    res = run_parallel(p1, jobs)
    (WORK / "pass1.json").write_text(json.dumps({"bounds": b, "rows": [r[1] for r in res],
                                                 "heldout_keys": res[0][2]}))
    log("pass1 done", [r[1] for r in res], res[0][2])


# ------------------------------------------------------------------ rank
def fp16(p):
    size = p.stat().st_size
    C = 16 * 1024 * 1024
    with open(p, "rb") as f:
        h = hashlib.blake2b(f.read(C)).hexdigest()
        f.seek(max(0, size - C))
        t = hashlib.blake2b(f.read(C)).hexdigest()
    return {"size": size, "blake2b_first_16MiB": h, "blake2b_last_16MiB": t}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def stage_rank(a):
    meta = json.loads((WORK / "pass1.json").read_text())
    cat = lambda nm: np.concatenate([np.load(WORK / f"p1_{nm}_{i}.npy") for i in range(3)])
    off, sc64, fl, nf, key = cat("off"), cat("sc"), cat("fl"), cat("nf"), cat("key")
    n = len(off)
    man = json.loads((INP / "controls_manifest.json").read_text())
    lab = np.load(INP / "provenance_labels.npy", allow_pickle=False)
    mask = np.load(INP / "pool_mask_reused.npy", allow_pickle=False)
    rep = json.loads((INP / "provenance_report.json").read_text())
    names = rep["sources"]
    C = {}
    C["rows"] = {"tsv": n, "labels": int(len(lab)), "mask": int(len(mask)), "manifest": man["n_pool_total"]}
    assert n == len(lab) == len(mask) == man["n_pool_total"], C["rows"]
    C["rows_not_3_fields"] = int((nf != 3).sum())
    C["offsets_strictly_increasing"] = bool(np.all(np.diff(off) > 0))
    mi = man["inputs"]
    tfp, lfp, rfp = fp16(TSV), fp16(INP / "provenance_labels.npy"), fp16(INP / "provenance_report.json")
    C["fingerprints_equal_e03_manifest"] = {
        "qe_scores_tsv": all(tfp[k] == mi["qe_scores"][k] for k in tfp),
        "provenance_labels": all(lfp[k] == mi["provenance"][k] for k in lfp),
        "provenance_report": all(rfp[k] == mi["provenance_report"][k] for k in rfp),
        "pool_mask_sha256": sha(INP / "pool_mask_reused.npy") == man["pool_mask"]["sha256"],
        "pool_mask_n_true": int(mask.sum()) == man["pool_mask"]["n_true"]}
    s32 = sc64.astype(np.float32)
    C["nan_in_mask"] = int((np.isnan(s32) & mask).sum())
    assert C["nan_in_mask"] == 0
    resv = {"rows_reserved_total": int((fl != 0).sum()),
            "exact_pair": int(((fl & 1) > 0).sum()),
            "extra_by_source": int((((fl & 2) > 0) & ((fl & 1) == 0)).sum()),
            "extra_by_target": int((((fl & 4) > 0) & ((fl & 3) == 0)).sum())}
    C["reservation"] = {"mine": resv, "manifest": {k: man["reservation"][k] for k in resv}}
    C["reservation"]["equal"] = C["reservation"]["mine"] == C["reservation"]["manifest"]
    reserved = fl != 0
    C["reserved_inside_mask"] = int((reserved & mask).sum())
    avail = np.flatnonzero(mask & ~reserved)
    N = int(len(avail))
    log(f"N = {N:,}")
    o = np.lexsort((avail, -s32[avail]))
    order = avail[o]
    del o
    # sanity: non-increasing scores, ties ascending index
    so = s32[order]
    assert np.all(so[1:] <= so[:-1])
    tie = so[1:] == so[:-1]
    assert np.all(order[1:][tie] > order[:-1][tie])
    depths = [{"name": nm, "frac": f, "k": int(round(f * N))} for nm, f in FRACS]
    depths.append({"name": "band", "frac": BAND / N, "k": BAND})
    depths.sort(key=lambda d: d["k"])
    K = depths[-1]["k"]
    top = order[:K]
    kk = key[top]
    pos = np.arange(K, dtype=np.int64)
    srt = np.lexsort((pos, kk[:, 1], kk[:, 0]))
    ks = kk[srt]
    new = np.ones(K, bool)
    new[1:] = (ks[1:, 0] != ks[:-1, 0]) | (ks[1:, 1] != ks[:-1, 1])
    Dpos = np.sort(pos[srt][new])            # ranking positions of first copies
    D = top[Dpos]
    del kk, srt, ks, new
    for d in depths:
        k = d["k"]
        d["n_dedup"] = int(np.searchsorted(Dpos, k))
        d["exact_pair_dup_rate_raw"] = 1 - d["n_dedup"] / k
        d["qe_cut"] = float(so[k - 1])
        d["un_share_raw"] = float((lab[order[:k]] == names.index("un")).mean())
        d["ties_with_cut_in_pool"] = int((so == so[k - 1]).sum())
        d["ties_with_cut_inside"] = int((so[:k] == so[k - 1]).sum())
    C["qe_top"] = float(so[0])
    # band vs E0.3 manifest and ft_topk files
    nb = int(np.searchsorted(Dpos, BAND))
    bs = s32[D[:nb]]
    cnt = {nm: int((lab[D[:nb]] == c).sum()) for c, nm in enumerate(names)}
    cnt["unmatched"] = int((lab[D[:nb]] == UNMATCHED).sum())
    e = man["ft_topk"]
    band = {"n_unique": nb, "removed": BAND - nb, "qe_min": float(bs.min()), "qe_max": float(bs.max()),
            "qe_mean_f32": float(bs.mean()), "qe_mean_f64": float(bs.astype(np.float64).mean()),
            "source_counts": cnt}
    band["vs_manifest"] = {"n_unique": nb == man["n_unique_per_set"],
                           "removed": BAND - nb == man["topk_duplicates_removed"],
                           "qe_min": band["qe_min"] == e["qe_min"], "qe_max": band["qe_max"] == e["qe_max"],
                           "qe_mean_absdiff_f32": abs(band["qe_mean_f32"] - e["qe_mean"]),
                           "qe_mean_absdiff_f64": abs(band["qe_mean_f64"] - e["qe_mean"]),
                           "source_counts": cnt == e["source_counts"]}
    fe, ff = lines_of(E05 / "ft_topk.en"), lines_of(E05 / "ft_topk.fr")
    fk = np.frombuffer(b"".join(pkey(x.encode(), y.encode()) for x, y in zip(fe, ff)), np.uint64).reshape(-1, 2)
    band["ft_topk_pairs_row_for_row"] = bool(len(fk) == nb and np.array_equal(fk, key[D[:nb]]))
    code = {nm: c for c, nm in enumerate(names)} | {"unmatched": UNMATCHED}
    fs = np.array([code[x] for x in lines_of(E05 / "ft_topk.source")], np.uint8)
    band["ft_topk_source_row_for_row"] = bool(len(fs) == nb and np.array_equal(fs, lab[D[:nb]]))
    C["band"] = band
    log(json.dumps(band))
    ends = np.append(off[1:], TSV.stat().st_size)
    np.save(WORK / "D.npy", D)
    np.save(WORK / "D_off.npy", off[D]); np.save(WORK / "D_end.npy", ends[D])
    np.save(WORK / "D_key.npy", key[D]); np.save(WORK / "D_lab.npy", lab[D])
    np.save(WORK / "D_score.npy", s32[D])
    out = {"N": N, "pool_mask_true": int(mask.sum()), "M": int(len(D)), "names": names,
           "depths": depths, "checks": C,
           "qe_range_top50": [float(so[depths[-1]['k'] - 1]), float(so[0])]}
    (WORK / "rank.json").write_text(json.dumps(out, indent=1))
    log("rank done", len(D))


# ------------------------------------------------------------------ pass2
def p2(job):
    i, a, b = job
    off = np.load(WORK / "D_off.npy", mmap_mode="r")[a:b]
    end = np.load(WORK / "D_end.npy", mmap_mode="r")[a:b]
    dkey = np.load(WORK / "D_key.npy", mmap_mode="r")[a:b]
    voc = {"en": {}, "fr": {}}
    ids = {"en": array("I"), "fr": array("I")}
    ln = {"en": array("i"), "fr": array("i")}
    skeys = bytearray()
    pk = bytearray()
    with open(TSV, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for o, e in zip(off.tolist(), end.tolist()):
            line = mm[o:e]
            body = line[:-1] if line.endswith(b"\n") else line
            parts = body.split(b"\t")
            sb, tb = parts[1], parts[2]
            pk += pkey(sb, tb)
            skeys += hashlib.md5(sb).digest()
            for side, txt in (("en", sb), ("fr", tb)):
                v = voc[side]
                t = [w.lower() for w in WORD.findall(txt.decode("utf-8"))]
                ids[side].extend([v.setdefault(w, len(v)) for w in t])
                ln[side].append(len(t))
        mm.close()
    pkarr = np.frombuffer(bytes(pk), np.uint64).reshape(-1, 2)
    if not np.array_equal(pkarr, np.asarray(dkey)):
        raise SystemExit(f"worker {i}: pair keys differ from pass 1")
    for side in ("en", "fr"):
        np.save(WORK / f"p2_{side}_ids_{i}.npy", np.frombuffer(ids[side], np.uint32))
        np.save(WORK / f"p2_{side}_len_{i}.npy", np.frombuffer(ln[side], np.int32))
        words = sorted(voc[side], key=voc[side].get)
        (WORK / f"p2_{side}_voc_{i}.json").write_text(json.dumps(words, ensure_ascii=False))
    np.save(WORK / f"p2_skey_{i}.npy", np.frombuffer(bytes(skeys), np.uint64).reshape(-1, 2))
    return i, b - a


def stage_pass2(a):
    M = len(np.load(WORK / "D_off.npy", mmap_mode="r"))
    b = [0, M // 3, 2 * M // 3, M]
    jobs = [(i, b[i], b[i + 1]) for i in range(3)]
    log("pass2", jobs)
    res = run_parallel(p2, jobs)
    (WORK / "pass2.json").write_text(json.dumps({"bounds": b, "rows": [r[1] for r in res]}))
    log("pass2 done")


# ------------------------------------------------------------------ reduce helpers
def load_side(side):
    g = {}
    parts, lens = [], []
    for i in range(3):
        words = json.loads((WORK / f"p2_{side}_voc_{i}.json").read_text())
        lut = np.array([g.setdefault(w, len(g)) for w in words], np.uint32)
        loc = np.load(WORK / f"p2_{side}_ids_{i}.npy")
        parts.append(lut[loc] if len(loc) else loc)
        lens.append(np.load(WORK / f"p2_{side}_len_{i}.npy"))
    return np.concatenate(parts), np.concatenate(lens).astype(np.int64), len(g)


M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)
GOLD = np.uint64(0x9E3779B97F4A7C15)


def mix(z):
    z = z ^ (z >> np.uint64(30)); z = z * M1
    z = z ^ (z >> np.uint64(27)); z = z * M2
    return z ^ (z >> np.uint64(31))


def gram_hashes(ids, tokoff, ng):
    """64-bit key of every source 8-gram, rows in D order, row-major."""
    rows = np.flatnonzero(ng > 0)
    total = int(ng.sum())
    H = np.empty(total, np.uint64)
    ostart = np.concatenate([[0], np.cumsum(ng)])
    CH = 1_000_000                                  # rows per chunk
    with np.errstate(over="ignore"):
        for c0 in range(0, len(rows), CH):
            r = rows[c0:c0 + CH]
            cnt = ng[r]
            base = np.repeat(tokoff[r], cnt)
            within = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            st = base + within
            h = np.full(len(st), np.uint64(0x243F6A8885A308D3), np.uint64)
            for p in range(8):
                h = mix(h * GOLD + ids[st + p].astype(np.uint64) + np.uint64(1))
            H[ostart[r[0]]:ostart[r[0]] + len(h)] = h
    return H, ostart


def rep_flags(Hsub):
    """per occurrence: does its key occur more than once in Hsub?"""
    s = np.sort(Hsub)
    eq = s[1:] == s[:-1]
    dup = np.unique(s[1:][eq])
    del s, eq
    out = np.zeros(len(Hsub), bool)
    if len(dup) == 0:
        return out
    CH = 20_000_000
    for c0 in range(0, len(Hsub), CH):
        x = Hsub[c0:c0 + CH]
        j = np.searchsorted(dup, x)
        j[j == len(dup)] = 0
        out[c0:c0 + CH] = dup[j] == x
    return out


def first_B_types(ids, tokoff, lens, perm, B):
    L = lens[perm]
    cum = np.cumsum(L)
    if cum[-1] < B:
        return None
    m = int(np.searchsorted(cum, B))             # rows perm[0..m] reach B
    pieces = [ids[tokoff[r]:tokoff[r] + lens[r]] for r in perm[:m + 1].tolist()]
    allids = np.concatenate(pieces)[:B]
    assert len(allids) == B
    return int(len(np.unique(allids)))


# ------------------------------------------------------------------ reduce
def stage_reduce(a):
    R = json.loads((WORK / "rank.json").read_text())
    names, depths, M = R["names"], R["depths"], R["M"]
    un = names.index("un")
    lab = np.load(WORK / "D_lab.npy")
    isun = lab == un
    en, len_en, Ven = load_side("en")
    fr, len_fr, Vfr = load_side("fr")
    assert len(len_en) == M == len(len_fr)
    off_en = np.concatenate([[0], np.cumsum(len_en)[:-1]])
    off_fr = np.concatenate([[0], np.cumsum(len_fr)[:-1]])
    log(f"vocab en {Ven:,} fr {Vfr:,}; tokens en {len(en):,} fr {len(fr):,}")
    skey = np.concatenate([np.load(WORK / f"p2_skey_{i}.npy") for i in range(3)])
    # distinct sources per prefix: first occurrence in D order
    pos = np.arange(M, dtype=np.int64)
    o = np.lexsort((pos, skey[:, 1], skey[:, 0]))
    ks = skey[o]
    new = np.ones(M, bool)
    new[1:] = (ks[1:, 0] != ks[:-1, 0]) | (ks[1:, 1] != ks[:-1, 1])
    first_src = np.zeros(M, bool); first_src[o[new]] = True
    distinct_src_cum = np.cumsum(first_src)
    del o, ks, new
    # whole-set types per prefix: row of each word's first occurrence
    def first_rows(ids, offs):
        _, fi = np.unique(ids, return_index=True)
        return np.sort(np.searchsorted(offs, fi, side="right") - 1)
    fr_en, fr_fr = first_rows(en, off_en), first_rows(fr, off_fr)
    ng = np.maximum(len_en - 7, 0)
    H, ostart = gram_hashes(en, off_en, ng)
    log(f"8-gram occurrences {len(H):,}")
    np.save(WORK / "H.npy", H)
    np.save(WORK / "len_en.npy", len_en)
    out = []
    for d in depths:
        c = d["n_dedup"]
        m = int(ostart[c])
        flag = rep_flags(H[:m])
        rows = np.flatnonzero(ng[:c] > 0)
        nrep = np.zeros(c, np.int64)
        if len(rows):
            nrep[rows] = np.add.reduceat(flag.astype(np.int32), ostart[rows])
        rp = nrep > 0
        repv = np.where(ng[:c] > 0, nrep / np.maximum(ng[:c], 1), 0.0)
        u = isun[:c]
        r = dict(d)
        r.update({"rows": c,
                  "rep_pos_share": float(rp.mean()), "rep_pos_rows": int(rp.sum()),
                  "rep_mean": float(repv.mean()),
                  "un_rep_pos": int((rp & u).sum()), "un_rep_zero": int((~rp & u).sum()),
                  "rep_pos_share_un": float(rp[u].mean()),
                  "dup_src_rate": 1 - int(distinct_src_cum[c - 1]) / c,
                  "un_share": float(u.mean()), "un_rows": int(u.sum()),
                  "tokens_en": int(len_en[:c].sum()), "tokens_fr": int(len_fr[:c].sum()),
                  "src_len_mean": float(len_en[:c].mean()),
                  "types_whole_en": int(np.searchsorted(fr_en, c)),
                  "types_whole_fr": int(np.searchsorted(fr_fr, c)),
                  "types_at": {}})
        out.append(r)
        log(r["name"], c, f"rep>0 {r['rep_pos_share']:.4f} un {r['un_share']:.4f} dup {r['dup_src_rate']:.4f}")
        del flag
    # matched budgets
    tot_min = min(min(r["tokens_en"], r["tokens_fr"]) for r in out)
    budgets = {"B410k": 410_000, "B3.3M": 3_300_000, "B6.5M": 6_500_000}
    for r in out:
        c = r["rows"]
        for seed in range(5):
            perm = list(range(c))
            random.Random(seed).shuffle(perm)
            perm = np.asarray(perm, np.int64)
            for bn, B in budgets.items():
                if seed and bn != "B410k":
                    continue
                r["types_at"][f"{bn}_seed{seed}"] = {
                    "en": first_B_types(en, off_en, len_en, perm, B),
                    "fr": first_B_types(fr, off_fr, len_fr, perm, B)}
        log(r["name"], "types", r["types_at"]["B410k_seed0"])
    res = {"N": R["N"], "pool_mask_true": R["pool_mask_true"], "M": M, "checks": R["checks"],
           "qe_range_top50": R["qe_range_top50"], "min_tokens_any_depth_side": tot_min,
           "vocab_deepest": {"en": Ven, "fr": Vfr}, "eightgram_occ": int(len(H)),
           "depths": out, "python": sys.version.split()[0],
           "unidata": unicodedata.unidata_version, "numpy": np.__version__}
    (BASE / "check_results.json").write_text(json.dumps(res, indent=1))
    log("reduce done")


# ------------------------------------------------------------------ sized
def stage_sized(a):
    R = json.loads((BASE / "check_results.json").read_text())
    names = json.loads((WORK / "rank.json").read_text())["names"]
    un = names.index("un")
    lab = np.load(WORK / "D_lab.npy")
    H = np.load(WORK / "H.npy", mmap_mode="r")
    len_en = np.load(WORK / "len_en.npy")
    ng = np.maximum(len_en - 7, 0)
    ostart = np.concatenate([[0], np.cumsum(ng)])
    skey = np.concatenate([np.load(WORK / f"p2_skey_{i}.npy") for i in range(3)])
    sizes = [20_621, 214_489, 926_999]
    out = []
    for d in R["depths"]:
        c = d["rows"]
        for n in sizes:
            if n > c:
                continue
            for seed in ((0, 1, 2, 3, 4) if n == sizes[0] else (0, 1)):
                q = np.sort(np.random.default_rng(seed).choice(c, n, replace=False))
                cnt = ng[q]
                st = np.repeat(ostart[q], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
                h = np.asarray(H[st])
                flag = rep_flags(h)
                rows = np.flatnonzero(cnt > 0)
                starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                nrep = np.zeros(n, np.int64)
                if len(rows):
                    nrep[rows] = np.add.reduceat(flag.astype(np.int32), starts[rows])
                rp = nrep > 0
                repv = np.where(cnt > 0, nrep / np.maximum(cnt, 1), 0.0)
                u = lab[q] == un
                sk = skey[q]
                nd_exact = len({(int(x), int(y)) for x, y in sk})
                out.append({"depth": d["name"], "n": n, "seed": seed,
                            "rep_pos_share": float(rp.mean()), "rep_pos_share_un": float(rp[u].mean()),
                            "rep_mean": float(repv.mean()), "dup_src_rate": 1 - nd_exact / n,
                            "un_share": float(u.mean()), "src_len_mean": float(len_en[q].mean())})
        log("sized", d["name"])
    R["sized"] = out
    (BASE / "check_results.json").write_text(json.dumps(R, indent=1))
    log("sized done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pass1", "rank", "pass2", "reduce", "sized"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
