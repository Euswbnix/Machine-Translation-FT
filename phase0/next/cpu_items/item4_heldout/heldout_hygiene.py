#!/usr/bin/env python3
"""WMT2027_PLAN section 3.2 item 4: heldout_un hygiene. Descriptive only; nothing here feeds a set.

  heldout_hygiene.py inputs   --home H --out O               sha256 and E0.3 fingerprints of every input
  heldout_hygiene.py sets     --home H --out O               (a) heldout_un against every E0.6 candidate set
  heldout_hygiene.py pool     --home H --out O --workers 6   (b) heldout_un against v2_scored.tsv
  heldout_hygiene.py pretrain --home H --out O --workers 6   (b) heldout_un against train.clean.{en,fr}
  heldout_hygiene.py reduce   --out O                        O/results.json from O/parts/*.json

H is the Studio home: H/e06data (read only), H/e06runs/feasibility (read only). O is the item's
own directory; O/inputs holds the files copied there from the MacBook (e06data_SHA256SUMS,
E0.3's controls_manifest.json, provenance_labels.npy, provenance_report.json,
pool_mask_reused.npy). Every subcommand writes one O/parts/<name>.json; `reduce` joins them.

MATCHING, primary ("exact"). The matching of E0.5's second pre-flight (message of commit
0fb52a6: "no row of the in-domain test set shares a source or a target with any of the six
training sets") as phase0/e05_build.py and phase0/e06_build.py implement it: a file is read
as UTF-8 with LF as the only line break and the one trailing LF removed (read_lines); a
source line, a target line, or a pair (the source and target of one row) "occurs" in another
file when the identical string occurs there. For v2_scored.tsv the source and target are
fields 2 and 3 of `line.rstrip("\\n").split("\\t")`, as e03_build_controls.py split_scored().

MATCHING, secondary ("norm"). E0.3's reservation key, e03_build_controls.py norm(): NFKC,
casefold, whitespace runs collapsed to one space, stripped, compared side by side. E0.3
reserved every pool row sharing the exact pair, the normalised source or the normalised
target of a held-out row; the categories are reproduced here as e03 counted them.

Counts are given from both sides: "heldout rows" = how many of heldout_un's 2,000 rows have
their source / target / pair in the other file; "rows" = how many rows of the other file
have a source / target / pair that is one of heldout_un's; "distinct" = distinct strings.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIX = ("ft_topk_div", "ft_topk_rep", "ft_random_small", "ft_un_div", "ft_un_rep", "ft_un_random")
PARENTS = ("ft_topk", "ft_random")
FP_CHUNK = 16 * 1024 * 1024          # e03_build_controls.py fingerprint window
BLOCK = int(os.environ.get("HH_BLOCK", 64 * 1024 * 1024))   # bytes per read in the streaming scans (tests set it small)
SOURCE_UN = "un"
# what E0.3's controls_manifest.json and the rental bundle's SHA256SUMS record (inputs/)
BUNDLE_SHA = {"provenance_labels.npy": "c9174045438cee37eaa1e5f46fc56ad75e9a9e2d7131cbbcdeb7ba89cf148218",
              "provenance_report.json": "046a52fe343082a6422ed78083af15c1fc9124fa8d797c7e3941cf158c6299c4",
              "pool_mask_reused.npy": "dbb55f9b0812c2fe757f8e4340f7642e5d4c5a0d9e9cabf7c6af4d3fbae99d96"}


class Stop(Exception):
    pass


# ---- the definitions, copied from the scripts they belong to ---------------------------
def read_lines(p: Path) -> list[str]:
    """phase0/e05_build.py and e06_build.py read_lines(): LF is the only line break."""
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]


def norm(s: str) -> str:
    """phase0/e03_build_controls.py norm()."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(p: Path) -> dict:
    """phase0/e03_build_controls.py fingerprint() without the path."""
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
        f.seek(max(0, size - FP_CHUNK))
        tail = hashlib.blake2b(f.read(FP_CHUNK)).hexdigest()
    return {"size": size, "blake2b_first_16MiB": head, "blake2b_last_16MiB": tail}


def interpreter() -> dict:
    return {"python": sys.version.split()[0], "unidata_version": unicodedata.unidata_version,
            "numpy": np.__version__, "machine": platform.machine(), "node": platform.node()}


class Held:
    """heldout_un, exact strings and E0.3's normalised keys."""

    def __init__(self, d: Path):
        self.src, self.tgt = read_lines(d / "heldout_un.en"), read_lines(d / "heldout_un.fr")
        if len(self.src) != len(self.tgt):
            raise Stop("heldout_un.en and heldout_un.fr differ in line count")
        self.n = len(self.src)
        self.S, self.T = set(self.src), set(self.tgt)
        self.P = set(zip(self.src, self.tgt))
        self.nsrc, self.ntgt = [norm(x) for x in self.src], [norm(x) for x in self.tgt]
        self.NS, self.NT = set(self.nsrc), set(self.ntgt)

    def describe(self) -> dict:
        return {"rows": self.n, "distinct_sources": len(self.S), "distinct_targets": len(self.T),
                "distinct_pairs": len(self.P), "distinct_norm_sources": len(self.NS),
                "distinct_norm_targets": len(self.NT),
                "lines_with_tab": sum(1 for x in self.src + self.tgt if "\t" in x),
                "empty_lines": sum(1 for x in self.src + self.tgt if x == "")}

    def side(self, fs: set, ft: set, fp: set, fns: set, fnt: set) -> dict:
        """heldout rows whose source / target / pair is among the found strings."""
        return {"exact": {"source": sum(1 for x in self.src if x in fs),
                          "target": sum(1 for x in self.tgt if x in ft),
                          "pair": sum(1 for p in zip(self.src, self.tgt) if p in fp)},
                "norm": {"source": sum(1 for x in self.nsrc if x in fns),
                         "target": sum(1 for x in self.ntgt if x in fnt)},
                "distinct_exact": {"source": len(fs), "target": len(ft), "pair": len(fp)}}


def write_part(out: Path, name: str, obj: dict) -> None:
    d = out / "parts"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{name}.json.tmp"
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, d / f"{name}.json")


# ======================================================================================
# inputs
# ======================================================================================
def cmd_inputs(a) -> int:
    H, O = a.home, a.out
    want = {}
    for ln in (O / "inputs" / "e06data_SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, p = ln.split(None, 1)
            want[p.strip()] = h
    used = [f"e06data/e05_src/{n}.{e}" for n in PARENTS for e in ("en", "fr", "source")]
    used += [f"e06data/e05_src/heldout_un.{e}" for e in ("en", "fr")]
    used += [f"e06data/phase0_e05/{n}.{e}" for n in SIX for e in ("en", "fr")]
    used += [f"e06data/phase0_e05/heldout_un.{e}" for e in ("en", "fr")] + ["e06data/phase0_e05/manifest.json"]
    used += ["e06data/vocab/R0.npy", "e06data/vocab/A_narrow_42_58.npy", "e06data/v2_scored.tsv",
             "e06data/v1_fixed/train.clean.en", "e06data/v1_fixed/train.clean.fr"]
    rec = {"sha256": {}, "all_match": True}
    for p in used:
        t0 = time.time()
        got = sha256_file(H / p)
        ok = want.get(p) == got
        rec["sha256"][p] = {"sha256": got, "expected": want.get(p), "match": ok}
        rec["all_match"] &= ok
        print(f"{'ok ' if ok else 'BAD'} {p} {time.time() - t0:.1f}s", flush=True)
    # E0.3's own fingerprints of the files it read (controls_manifest.json "inputs")
    man = json.loads((O / "inputs" / "controls_manifest.json").read_text(encoding="utf-8"))
    fp = {}
    for key, p in (("qe_scores", H / "e06data/v2_scored.tsv"), ("pretrain_src", H / "e06data/v1_fixed/train.clean.en"),
                   ("pretrain_tgt", H / "e06data/v1_fixed/train.clean.fr"),
                   ("provenance", O / "inputs/provenance_labels.npy"), ("pool_mask", O / "inputs/pool_mask_reused.npy")):
        e = {k: v for k, v in man["inputs"][key].items() if k != "path"}
        g = fingerprint(p)
        fp[key] = {"file": str(p), "e03_path": man["inputs"][key]["path"], "match": g == e, "got": g}
    rec["e03_fingerprints"] = fp
    rec["bundle_sha256"] = {}
    for f, h in BUNDLE_SHA.items():
        g = sha256_file(O / "inputs" / f)
        rec["bundle_sha256"][f] = {"sha256": g, "expected": h, "match": g == h}
    rec["controls_manifest_sha256"] = sha256_file(O / "inputs" / "controls_manifest.json")
    rec["e06data_SHA256SUMS_sha256"] = sha256_file(O / "inputs" / "e06data_SHA256SUMS")
    rec["built_with"] = interpreter()
    write_part(O, "inputs", rec)
    bad = [k for k, v in fp.items() if not v["match"]] + [k for k, v in rec["bundle_sha256"].items() if not v["match"]]
    print(f"sha256 all match: {rec['all_match']}; fingerprint/bundle mismatches: {bad or 'none'}")
    return 0 if rec["all_match"] and not bad else 1


# ======================================================================================
# (a) the E0.6 candidate sets
# ======================================================================================
class ParentM:
    """A parent file (ft_topk or ft_random) with, per row, whether it touches heldout_un."""

    def __init__(self, d: Path, name: str, H: Held):
        self.name = name
        self.src, self.tgt, self.lab = (read_lines(d / f"{name}.{e}") for e in ("en", "fr", "source"))
        if not len(self.src) == len(self.tgt) == len(self.lab):
            raise Stop(f"{name}: .en/.fr/.source differ in line count")
        self.n = len(self.src)
        self.hit = {"source": np.fromiter((x in H.S for x in self.src), bool, self.n),
                    "target": np.fromiter((x in H.T for x in self.tgt), bool, self.n),
                    "pair": np.fromiter(((x, y) in H.P for x, y in zip(self.src, self.tgt)), bool, self.n),
                    "norm_source": np.fromiter((norm(x) in H.NS for x in self.src), bool, self.n),
                    "norm_target": np.fromiter((norm(x) in H.NT for x in self.tgt), bool, self.n)}
        self.un = np.fromiter((x == SOURCE_UN for x in self.lab), bool, self.n)


def measure_idx(P: ParentM, idx: np.ndarray, H: Held) -> dict:
    rows = {k: idx[v[idx]] for k, v in P.hit.items()}
    fs = {P.src[i] for i in rows["source"]}
    ft = {P.tgt[i] for i in rows["target"]}
    fp = {(P.src[i], P.tgt[i]) for i in rows["pair"]}
    fns = {norm(P.src[i]) for i in rows["norm_source"]}
    fnt = {norm(P.tgt[i]) for i in rows["norm_target"]}
    return {"rows": int(len(idx)), "set_rows_touching_heldout": {k: int(len(v)) for k, v in rows.items()},
            "heldout_rows_in_set": H.side(fs, ft, fp, fns, fnt),
            # = e06_build.py describe() "heldout_un_shared_lines": distinct set lines found in heldout_un
            "shared_lines_as_feasibility_report": {"source": len(fs), "target": len(ft)}}


def measure_strings(src: list[str], tgt: list[str], H: Held) -> dict:
    s_hit = [x in H.S for x in src]
    t_hit = [x in H.T for x in tgt]
    p_hit = [(x, y) in H.P for x, y in zip(src, tgt)]
    ns = [norm(x) for x in src]
    nt = [norm(x) for x in tgt]
    fs = {x for x, h in zip(src, s_hit) if h}
    ft = {x for x, h in zip(tgt, t_hit) if h}
    fp = {p for p, h in zip(zip(src, tgt), p_hit) if h}
    fns = {x for x in ns if x in H.NS}
    fnt = {x for x in nt if x in H.NT}
    return {"rows": len(src),
            "set_rows_touching_heldout": {"source": sum(s_hit), "target": sum(t_hit), "pair": sum(p_hit),
                                          "norm_source": sum(1 for x in ns if x in H.NS),
                                          "norm_target": sum(1 for x in nt if x in H.NT)},
            "heldout_rows_in_set": H.side(fs, ft, fp, fns, fnt),
            "shared_lines_as_feasibility_report": {"source": len(fs), "target": len(ft)}}


def load_index(p: Path, n_parent: int) -> np.ndarray:
    a = np.load(p, allow_pickle=False)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise Stop(f"{p}: not a 1-d integer index file")
    a = a.astype(np.int64)
    if len(a) and (a[0] < 0 or a[-1] >= n_parent or np.any(np.diff(a) <= 0)):
        raise Stop(f"{p}: not strictly ascending inside [0, {n_parent})")
    return a


def zero(m: dict) -> bool:
    h = m["heldout_rows_in_set"]
    return (not any(m["set_rows_touching_heldout"].values()) and not any(h["exact"].values())
            and not any(h["norm"].values()) and not any(m["shared_lines_as_feasibility_report"].values()))


def cmd_sets(a) -> int:
    H_, O = a.home, a.out
    src_dir, e05_dir, feas = H_ / "e06data/e05_src", H_ / "e06data/phase0_e05", a.feas
    t0 = time.time()
    H = Held(src_dir)
    H5 = Held(e05_dir)
    if (H5.src, H5.tgt) != (H.src, H.tgt):
        raise Stop("phase0_e05/heldout_un.* differ from e05_src/heldout_un.*")
    par = {n: ParentM(src_dir, n, H) for n in PARENTS}
    print(f"parents loaded and matched {time.time() - t0:.1f}s", flush=True)
    rep = json.loads((feas / "feasibility_report.json").read_text(encoding="utf-8"))
    rec = {"heldout_un": H.describe(), "feasibility_dir": str(feas),
           "feasibility_report_sha256": sha256_file(feas / "feasibility_report.json")}

    # the report's inputs must be the files measured here, or its indices point elsewhere
    inp = {}
    for f in [f"{n}.{e}" for n in PARENTS for e in ("en", "fr", "source")] + ["heldout_un.en", "heldout_un.fr"]:
        g = sha256_file(src_dir / f)
        inp[f] = {"sha256": g, "report": rep["inputs"][f]["sha256"], "match": g == rep["inputs"][f]["sha256"]}
    rec["report_inputs_match"] = inp
    if not all(v["match"] for v in inp.values()):
        raise Stop("the feasibility report was built from other input files")

    # map every FEASIBILITY_*.npy to its tag, parent, row count and report field
    ent = {}
    for nm, v in rep["anchors"].items():
        ent[v["file"]] = {"tag": nm, "kind": "anchor", "parent": v["parent"], "rows": v["rows"],
                          "report_field": v["heldout_un_shared_lines"]}
    failed = {}
    for tag, v in rep["results"].items():
        if "failed" in v:
            failed[tag] = v["failed"].split("\n")[0]
            continue
        ent[v["file"]] = {"tag": tag, "kind": "candidate", "parent": v["parent"], "rows": v["rows"],
                          "report_field": v["heldout_un_shared_lines"]}
    disk = sorted(p.name for p in feas.glob("FEASIBILITY_*.npy"))
    if sorted(ent) != disk:
        raise Stop(f"report and directory disagree: only in report {sorted(set(ent) - set(disk))}, "
                   f"only on disk {sorted(set(disk) - set(ent))}")
    rec["failed_options_no_file"] = failed
    sets = {}
    for fn in disk:
        e = ent[fn]
        g = sha256_file(feas / fn)
        if g != rep["files"][fn]:
            raise Stop(f"{fn}: sha256 {g} is not the report's {rep['files'][fn]}")
        P = par[e["parent"]]
        idx = load_index(feas / fn, P.n)
        if len(idx) != e["rows"]:
            raise Stop(f"{fn}: {len(idx)} rows, the report says {e['rows']}")
        m = measure_idx(P, idx, H)
        m.update({"tag": e["tag"], "kind": e["kind"], "parent": e["parent"], "sha256": g,
                  "un_rows": int(P.un[idx].sum()), "report_field": e["report_field"],
                  "agrees_with_report": m["shared_lines_as_feasibility_report"] == e["report_field"]})
        m["all_zero"] = zero(m)
        sets[fn] = m
    rec["feasibility_sets"] = sets
    rec["feasibility_summary"] = {"files": len(sets), "all_zero": sum(v["all_zero"] for v in sets.values()),
                                  "agree_with_report": sum(v["agrees_with_report"] for v in sets.values()),
                                  "by_parent": dict(collections.Counter(v["parent"] for v in sets.values())),
                                  "distinct_index_contents": len({v["sha256"] for v in sets.values()})}
    # the same files as the repo's record of the run (phase0/next/e06_feasibility/SHA256SUMS)
    rs = O / "inputs/feasibility_SHA256SUMS"
    if rs.exists():
        rsum = {}
        for ln in rs.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                h, f = ln.split(None, 1)
                rsum[f.strip()] = h
        here = {fn: v["sha256"] for fn, v in sets.items()} | {"feasibility_report.json": rec["feasibility_report_sha256"]}
        rec["repo_feasibility_SHA256SUMS"] = {"files_listed": len(rsum), "match": rsum == here,
                                              "differ": sorted(k for k in set(rsum) | set(here) if rsum.get(k) != here.get(k))}
    else:
        rec["repo_feasibility_SHA256SUMS"] = "n.m.: no inputs/feasibility_SHA256SUMS"

    # supplementary: the vocab memo's scratch index files (into ft_random), the six E0.5 files,
    # and the two parents themselves (every set above is a subset of one of them)
    sup = {}
    for f, twin in (("R0.npy", "FEASIBILITY_ft_un_r0__od6-keep__seed42.npy"),
                    ("A_narrow_42_58.npy", "FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy")):
        P = par["ft_random"]
        idx = load_index(H_ / "e06data/vocab" / f, P.n)
        m = measure_idx(P, idx, H)
        m.update({"parent": "ft_random", "un_rows": int(P.un[idx].sum()),
                  "identical_to": twin if np.array_equal(idx, load_index(feas / twin, P.n)) else None})
        m["all_zero"] = zero(m)
        sup[f"vocab/{f}"] = m
    for n in SIX:
        m = measure_strings(read_lines(e05_dir / f"{n}.en"), read_lines(e05_dir / f"{n}.fr"), H)
        m["all_zero"] = zero(m)
        sup[f"phase0_e05/{n}"] = m
    for n, P in par.items():
        for lab, idx in (("all rows", np.arange(P.n)), ("UN rows", np.flatnonzero(P.un))):
            m = measure_idx(P, idx, H)
            m["all_zero"] = zero(m)
            sup[f"{n} ({lab})"] = m
    rec["supplementary_sets"] = sup
    rec["built_with"] = interpreter()
    write_part(O, "sets", rec)
    s = rec["feasibility_summary"]
    print(f"(a) {s['files']} feasibility files: {s['all_zero']} share nothing with heldout_un; "
          f"{s['agree_with_report']} agree with the report's field; supplementary all zero: "
          f"{all(v['all_zero'] for v in sup.values())}  ({time.time() - t0:.1f}s)")
    return 0


# ======================================================================================
# streaming scans: byte ranges aligned to line starts, one worker per range
# ======================================================================================
def line_chunks(path: Path, n: int) -> list[tuple[int, int, int]]:
    size = path.stat().st_size
    b = [0]
    with open(path, "rb") as f:
        for k in range(1, n):
            f.seek(size * k // n)
            f.readline()
            b.append(max(f.tell(), b[-1]))
    b.append(size)
    return [(i, b[i], b[i + 1]) for i in range(n) if b[i] < b[i + 1]]


def iter_lines(path: Path, start: int, end: int):
    """Lines of path[start:end] (start at a line start), LF only, trailing LF removed."""
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start
        carry = b""
        while left > 0 or carry:
            buf = f.read(min(BLOCK, left)) if left > 0 else b""
            if left > 0 and not buf:
                raise Stop(f"{path} ended {left:,} bytes early (changed during the scan?)")
            left -= len(buf)
            buf = carry + buf
            if left > 0:
                cut = buf.rfind(b"\n") + 1
                buf, carry = buf[:cut], buf[cut:]
            else:
                carry = b""
            if not buf:
                continue
            lines = buf.decode("utf-8").split("\n")
            if lines[-1] == "":
                lines.pop()
            yield from lines


_H: Held | None = None


def _init(d: str) -> None:
    global _H
    _H = Held(Path(d))


def scan_tsv(task):
    path, cid, start, end = task
    H = _H
    n, hits, odd = 0, [], []
    for ln in iter_lines(Path(path), start, end):
        p = ln.split("\t")
        if len(p) != 3:
            odd.append((n, len(p)))
            if len(p) < 3:
                n += 1
                continue
        s, t = p[1], p[2]
        f = ((s, t) in H.P, s in H.S, t in H.T, norm(s) in H.NS, norm(t) in H.NT)
        if any(f):
            hits.append((n, p[0], s, t) + f)
        n += 1
    return cid, n, hits, odd


def scan_side(task):
    path, side, cid, start, end = task
    H = _H
    E, N = (H.S, H.NS) if side == "en" else (H.T, H.NT)
    n, hits = 0, []
    for ln in iter_lines(Path(path), start, end):
        e, m = ln in E, norm(ln) in N
        if e or m:
            hits.append((n, ln, e, m))
        n += 1
    return side, cid, n, hits


def run_pool(tasks, fn, workers: int, held_dir: Path):
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(held_dir),)) as ex:
        return list(ex.map(fn, tasks))


# ======================================================================================
# (b) the scored pool
# ======================================================================================
def cmd_pool(a) -> int:
    Hd, O = a.home / "e06data/e05_src", a.out
    tsv = a.home / "e06data/v2_scored.tsv"
    H = Held(Hd)
    t0 = time.time()
    chunks = line_chunks(tsv, a.workers * 4)
    res = sorted(run_pool([(str(tsv), c, s, e) for c, s, e in chunks], scan_tsv, a.workers, Hd))
    off, rows, odd = 0, [], []
    for _, n, hits, od in res:
        rows += [(off + h[0],) + h[1:] for h in hits]
        odd += [(off + r, k) for r, k in od]
        off += n
    print(f"pool scanned: {off:,} rows, {len(rows):,} touching heldout_un, {time.time() - t0:.0f}s", flush=True)
    rid = np.asarray([r[0] for r in rows], dtype=np.int64)
    F = np.asarray([r[4:] for r in rows], dtype=bool).reshape(-1, 5)
    ep, es, et, ns, nt = (F[:, j] for j in range(5))
    by_src, by_tgt = ns & ~ep, nt & ~ns & ~ep          # e03_build_controls.py l.392-394
    cat = {"exact_pair": int(ep.sum()), "exact_source": int(es.sum()), "exact_target": int(et.sum()),
           "exact_source_not_pair": int((es & ~ep).sum()), "exact_target_not_pair": int((et & ~ep).sum()),
           "exact_any": int((ep | es | et).sum()),
           "norm_source": int(ns.sum()), "norm_target": int(nt.sum()),
           "norm_source_not_exact_source": int((ns & ~es).sum()), "norm_target_not_exact_target": int((nt & ~et).sum())}
    e03 = {"exact_pair": int(ep.sum()), "extra_by_source": int(by_src.sum()), "extra_by_target": int(by_tgt.sum()),
           "rows_reserved_total": int((ep | ns | nt).sum())}
    # heldout side
    fs = {r[2] for r, f in zip(rows, es) if f}
    ft = {r[3] for r, f in zip(rows, et) if f}
    fp = {(r[2], r[3]) for r, f in zip(rows, ep) if f}
    fns = {norm(r[2]) for r, f in zip(rows, ns) if f}
    fnt = {norm(r[3]) for r, f in zip(rows, nt) if f}
    side = H.side(fs, ft, fp, fns, fnt)
    # how many pool copies each heldout pair has
    cp = collections.Counter((r[2], r[3]) for r, f in zip(rows, ep) if f)
    copies = collections.Counter(cp[p] for p in zip(H.src, H.tgt))
    rec = {"pool_rows": off, "rows_not_3_fields": len(odd), "rows_not_3_fields_first": odd[:10],
           "pool_rows_touching_heldout": cat, "e03_reservation_categories": e03,
           "heldout_rows_in_pool": side,
           "heldout_rows_by_pool_copies_of_their_pair": {str(k): v for k, v in sorted(copies.items())},
           "pool_rows_exact_pair_per_distinct_heldout_pair": {"min": min(cp.values()) if cp else 0,
                                                              "max": max(cp.values()) if cp else 0}}
    # per heldout row: pool rows with its exact pair / source / target (Studio-side file)
    cs_ = collections.Counter(r[2] for r, f in zip(rows, es) if f)
    ct_ = collections.Counter(r[3] for r, f in zip(rows, et) if f)
    per = [{"pair": cp[(s, t)], "source": cs_[s], "target": ct_[t]} for s, t in zip(H.src, H.tgt)]
    (O / "pool_per_heldout_row.json").write_text(json.dumps(per) + "\n", encoding="utf-8")
    rec["heldout_rows_with_no_pool_row"] = {k: sum(1 for p in per if p[k] == 0) for k in ("pair", "source", "target")}
    # the copy the runs were scored on: heldout_un after --max-src-tokens 254 (inputs/filtered_tests)
    ev = O / "inputs/filtered_tests"
    if (ev / "heldout_un.en").exists():
        fe, ff = read_lines(ev / "heldout_un.en"), read_lines(ev / "heldout_un.fr")
        j, dropped = 0, []
        for i, p in enumerate(zip(H.src, H.tgt)):
            if j < len(fe) and (fe[j], ff[j]) == p:
                j += 1
            else:
                dropped.append(i)
        if j != len(fe) or len(fe) != len(ff):
            raise Stop("inputs/filtered_tests/heldout_un.* is not heldout_un with rows removed")
        keep = [i for i in range(H.n) if i not in set(dropped)]
        rec["evaluated_copy"] = {"sha256": {f: sha256_file(ev / f) for f in ("heldout_un.en", "heldout_un.fr")},
                                 "rows": len(fe), "dropped_rows_0based": dropped,
                                 "dropped_rows_pool_rows": [per[i] for i in dropped],
                                 "kept_rows_with_exact_pair_in_pool": sum(1 for i in keep if per[i]["pair"] > 0),
                                 "kept_rows_with_exact_source_in_pool": sum(1 for i in keep if per[i]["source"] > 0),
                                 "kept_rows_with_exact_target_in_pool": sum(1 for i in keep if per[i]["target"] > 0)}
    else:
        rec["evaluated_copy"] = "n.m.: no inputs/filtered_tests/heldout_un.*"
    # sources and the reused-score mask of the rows found (E0.3's labels, inputs/)
    lab_p, mask_p = O / "inputs/provenance_labels.npy", O / "inputs/pool_mask_reused.npy"
    if lab_p.exists() and mask_p.exists():
        lab = np.load(lab_p, allow_pickle=False)
        mask = np.load(mask_p, allow_pickle=False)
        names = json.loads((O / "inputs/provenance_report.json").read_text(encoding="utf-8"))["sources"]
        if len(lab) != off or len(mask) != off:
            raise Stop(f"labels {len(lab):,} / mask {len(mask):,} rows, pool {off:,}")
        nm = lambda c: names[c] if c < len(names) else ("unmatched" if c == 255 else f"code_{c}")
        L, M = lab[rid], mask[rid]
        brk = {}
        for k, sel in (("exact_pair", ep), ("extra_by_source", by_src), ("extra_by_target", by_tgt),
                       ("exact_source_not_pair", es & ~ep), ("exact_target_not_pair", et & ~ep),
                       ("rows_reserved_total", ep | ns | nt)):
            brk[k] = {"sources": dict(sorted(collections.Counter(nm(int(c)) for c in L[sel]).items())),
                      "inside_reused_mask": int(M[sel].sum()), "rows": int(sel.sum())}
        rec["by_source_and_mask"] = brk
    else:
        rec["by_source_and_mask"] = "n.m.: no inputs/provenance_labels.npy or inputs/pool_mask_reused.npy"
    # QE score strings of the exact-pair rows (descriptive)
    sc = np.asarray([float(r[1]) for r, f in zip(rows, ep) if f])
    rec["exact_pair_rows_qe"] = ({"min": float(sc.min()), "median": float(np.median(sc)), "max": float(sc.max()),
                                  "nan": int(np.isnan(sc).sum())} if len(sc) else None)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["workers"] = a.workers
    rec["built_with"] = interpreter()
    np.save(O / "pool_hit_rows.npy", rid)
    np.save(O / "pool_hit_flags.npy", F)
    write_part(O, "pool", rec)
    print(json.dumps({"categories": cat, "e03": e03, "heldout_side": side}, indent=1))
    return 0


# ======================================================================================
# (b) the base's pretraining corpus
# ======================================================================================
def cmd_pretrain(a) -> int:
    Hd, O = a.home / "e06data/e05_src", a.out
    en, fr = a.home / "e06data/v1_fixed/train.clean.en", a.home / "e06data/v1_fixed/train.clean.fr"
    H = Held(Hd)
    t0 = time.time()
    w = max(1, a.workers // 2)
    tasks = [(str(en), "en", c, s, e) for c, s, e in line_chunks(en, w * 4)]
    tasks += [(str(fr), "fr", c, s, e) for c, s, e in line_chunks(fr, w * 4)]
    res = run_pool(tasks, scan_side, a.workers, Hd)
    out = {}
    for side in ("en", "fr"):
        off, hits = 0, []
        for _, c, n, h in sorted(r for r in res if r[0] == side):
            hits += [(off + x[0],) + x[1:] for x in h]
            off += n
        out[side] = (off, hits)
    n_en, n_fr = out["en"][0], out["fr"][0]
    if n_en != n_fr:
        raise Stop(f"train.clean.en has {n_en:,} lines, .fr {n_fr:,}")
    he, hf = out["en"][1], out["fr"][1]
    es_rows = {r: s for r, s, e, m in he if e}
    et_rows = {r: t for r, t, e, m in hf if e}
    pair_rows = [r for r in es_rows if r in et_rows and (es_rows[r], et_rows[r]) in H.P]
    fs, ft = set(es_rows.values()), set(et_rows.values())
    fp = {(es_rows[r], et_rows[r]) for r in pair_rows}
    fns = {norm(s) for r, s, e, m in he if m}
    fnt = {norm(t) for r, t, e, m in hf if m}
    rec = {"pretrain_rows": n_en,
           "pretrain_rows_touching_heldout": {"exact_pair": len(pair_rows), "exact_source": len(es_rows),
                                              "exact_target": len(et_rows),
                                              "norm_source": sum(1 for x in he if x[3]),
                                              "norm_target": sum(1 for x in hf if x[3])},
           "heldout_rows_in_pretrain": H.side(fs, ft, fp, fns, fnt),
           "first_hits": {"en": [(r, s[:120], e, m) for r, s, e, m in he[:10]],
                          "fr": [(r, t[:120], e, m) for r, t, e, m in hf[:10]]},
           "seconds": round(time.time() - t0, 1), "workers": a.workers, "built_with": interpreter()}
    write_part(O, "pretrain", rec)
    print(json.dumps({k: rec[k] for k in ("pretrain_rows", "pretrain_rows_touching_heldout", "heldout_rows_in_pretrain")}, indent=1))
    return 0


# ======================================================================================
# reduce
# ======================================================================================
def cmd_reduce(a) -> int:
    O = a.out
    parts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((O / "parts").glob("*.json"))}
    res = {"item": "WMT2027_PLAN section 3.2 item 4: heldout_un hygiene (descriptive only)",
           "matching": {"exact": "UTF-8, LF the only line break, one trailing LF removed; identical strings "
                                 "(e05_build.py / e06_build.py read_lines; E0.5's second pre-flight, commit 0fb52a6)",
                        "norm": "e03_build_controls.py norm(): NFKC, casefold, whitespace runs collapsed, stripped "
                                "(E0.3's reservation key)"},
           **parts}
    chk = O / "check"
    if chk.is_dir():
        res["independent_check_awk"] = {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(chk.glob("*.txt"))}
    (O / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {O / 'results.json'} from parts {sorted(parts)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("inputs", "sets", "pool", "pretrain", "reduce"))
    ap.add_argument("--home", type=Path, default=Path.home())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--feas", type=Path, help="feasibility directory (default HOME/e06runs/feasibility)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.workers < 1 or a.workers > 6:
        ap.error("--workers must be 1-6")
    a.feas = a.feas or a.home / "e06runs/feasibility"
    try:
        return {"inputs": cmd_inputs, "sets": cmd_sets, "pool": cmd_pool, "pretrain": cmd_pretrain,
                "reduce": cmd_reduce}[a.cmd](a)
    except Stop as e:
        print(f"STOP: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
