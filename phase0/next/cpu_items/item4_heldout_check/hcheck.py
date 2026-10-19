#!/usr/bin/env python3
"""Independent check of WMT2027_PLAN section 3.2 item 4 (heldout_un hygiene). Descriptive only.

Written from the sources (e03_build_controls.py, e05_build.py, e06_build.py, E0.3's
controls_manifest.json, feasibility_report.json), not from the builder's heldout_hygiene.py.
Runs on the Mac Studio from ~/cpu_items/item4_heldout_check/ (run.sh: at most 3 processes).

  hcheck.py inputs        sha256 of every e06data input used; E0.3's fingerprints; bundle sha256
  hcheck.py sets          (a) every FEASIBILITY_*.npy, R0/A_narrow, the six E0.5 sets, the parents
  hcheck.py pre           (b) train.clean.{en,fr}; also key arrays for re-deriving E0.3's draw
  hcheck.py pool K N      (b) byte chunk K of N of v2_scored.tsv; also key arrays
  hcheck.py evalcopy      the 1,999-row copy the runs were scored on; source piece lengths
  hcheck.py reduce        joins everything into results.json

Matching (definitions, from the sources):
  exact  read_lines() (e05_build.py l.72-75 / e06_build.py l.235-238): UTF-8, only LF splits,
         the trailing LF removed; identical strings. v2_scored.tsv: fields [1], [2] of
         line.rstrip("\\n").split("\\t") (e03 split_scored l.150-155, pass 1 l.209).
  norm   e03_build_controls.py norm() l.115-116: NFKC, casefold, whitespace runs collapsed, stripped.
Keys for the pool/pretraining joins are this script's own 64-bit keyed blake2b (not E0.3's h64/nh64),
so a reproduction of E0.3's counts and draw is not a re-run of E0.3's hashing.
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import os
import platform
import sys
import time
import unicodedata
from array import array
from pathlib import Path

import numpy as np

HOME = Path.home()
D = HOME / "e06data"
FEAS = HOME / "e06runs" / "feasibility"
W = Path(os.environ.get("HC_DIR", HOME / "cpu_items" / "item4_heldout_check"))
INP = W / "inputs"
WORK = W / "work"

PARENTS = ("ft_topk", "ft_random")
E05 = {"ft_topk_div": "ft_topk", "ft_topk_rep": "ft_topk", "ft_random_small": "ft_random",
       "ft_un_div": "ft_topk", "ft_un_rep": "ft_topk", "ft_un_random": "ft_random"}
KEY = b"item4-heldout-chk"
UN = "un"


class Fail(Exception):
    pass


def env() -> dict:
    return {"python": sys.version.split()[0], "unidata_version": unicodedata.unidata_version,
            "numpy": np.__version__, "machine": platform.machine(), "node": platform.node()}


def lines_of(p: Path) -> list[str]:
    """= read_lines(): split on LF bytes only, drop the empty piece after a final LF."""
    parts = Path(p).read_bytes().split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return [x.decode("utf-8") for x in parts]


def nkey(s: str) -> str:
    """E0.3's normalisation (e03_build_controls.py l.115-116)."""
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def hk(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8, key=KEY).digest(), "little")


def pkey(sb: bytes, tb: bytes) -> int:
    """key of a PAIR; the length prefix makes (src, tgt) -> bytes injective."""
    return hk(len(sb).to_bytes(5, "little") + sb + tb)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fp16(p: Path) -> dict:
    """E0.3's fingerprint() (l.128-139) without the path: size, blake2b of first/last 16 MiB."""
    size, c = os.path.getsize(p), 16 * 1024 * 1024
    with open(p, "rb") as f:
        a = hashlib.blake2b(f.read(c)).hexdigest()
        f.seek(max(0, size - c))
        b = hashlib.blake2b(f.read(c)).hexdigest()
    return {"size": size, "blake2b_first_16MiB": a, "blake2b_last_16MiB": b}


def sums(p: Path) -> dict:
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, name = ln.split(None, 1)
            out[name.strip()] = h
    return out


def dump(name: str, obj) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    tmp = WORK / f".{name}.json.tmp"
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, WORK / f"{name}.json")


class Held:
    def __init__(self, d: Path):
        self.src, self.tgt = lines_of(d / "heldout_un.en"), lines_of(d / "heldout_un.fr")
        if len(self.src) != len(self.tgt):
            raise Fail(f"{d}: heldout_un.en/.fr differ in line count")
        self.n = len(self.src)
        self.nsrc = [nkey(x) for x in self.src]
        self.ntgt = [nkey(x) for x in self.tgt]
        self.S, self.T = set(self.src), set(self.tgt)
        self.P = set(zip(self.src, self.tgt))
        self.NS, self.NT = set(self.nsrc), set(self.ntgt)

    def describe(self) -> dict:
        dup = [p for p, c in collections.Counter(zip(self.src, self.tgt)).items() if c > 1]
        return {"rows": self.n, "distinct_pairs": len(self.P), "distinct_sources": len(self.S),
                "distinct_targets": len(self.T), "distinct_norm_sources": len(self.NS),
                "distinct_norm_targets": len(self.NT),
                "rows_of_duplicated_pairs": [[i for i, p in enumerate(zip(self.src, self.tgt)) if p == q] for q in dup],
                "empty_lines": sum(1 for x in self.src + self.tgt if x == ""),
                "lines_with_tab": sum(1 for x in self.src + self.tgt if "\t" in x)}


def measure(S, T, NS, NT, H: Held) -> dict:
    """A set given as row lists (source, target, normalised source, normalised target)."""
    n = len(S)
    fs = [x in H.S for x in S]
    ft = [x in H.T for x in T]
    fp = [(a, b) in H.P for a, b in zip(S, T)]
    fns = [x in H.NS for x in NS]
    fnt = [x in H.NT for x in NT]
    anyf = sum(1 for i in range(n) if fs[i] or ft[i] or fp[i] or fns[i] or fnt[i])
    setS, setT, setP, setNS, setNT = set(S), set(T), set(zip(S, T)), set(NS), set(NT)
    return {
        "rows": n,
        "set_rows_with_heldout": {"exact_source": sum(fs), "exact_target": sum(ft), "exact_pair": sum(fp),
                                  "norm_source": sum(fns), "norm_target": sum(fnt), "any": anyf},
        # e06_build.py describe() l.1369-1371: DISTINCT set lines found among heldout_un's lines
        "distinct_shared_lines_as_report": {"source": len(setS & H.S), "target": len(setT & H.T)},
        "heldout_rows_found_in_set": {
            "exact_source": sum(1 for x in H.src if x in setS),
            "exact_target": sum(1 for x in H.tgt if x in setT),
            "exact_pair": sum(1 for p in zip(H.src, H.tgt) if p in setP),
            "norm_source": sum(1 for x in H.nsrc if x in setNS),
            "norm_target": sum(1 for x in H.ntgt if x in setNT)},
    }


def all_zero(m: dict) -> bool:
    return not (any(m["set_rows_with_heldout"].values()) or any(m["distinct_shared_lines_as_report"].values())
                or any(m["heldout_rows_found_in_set"].values()))


# ======================================================================================
def cmd_inputs() -> dict:
    want = sums(INP / "e06data_SHA256SUMS")
    used = [f"e06data/e05_src/{n}.{e}" for n in PARENTS for e in ("en", "fr", "source")]
    used += [f"e06data/e05_src/heldout_un.{e}" for e in ("en", "fr")]
    used += [f"e06data/phase0_e05/{n}.{e}" for n in E05 for e in ("en", "fr")]
    used += ["e06data/phase0_e05/heldout_un.en", "e06data/phase0_e05/heldout_un.fr", "e06data/phase0_e05/manifest.json",
             "e06data/vocab/R0.npy", "e06data/vocab/A_narrow_42_58.npy", "e06data/spm/spm_enfr_v1_fixed.model",
             "e06data/v1_fixed/train.clean.en", "e06data/v1_fixed/train.clean.fr", "e06data/v2_scored.tsv"]
    res = {"sha256_vs_e06data_SHA256SUMS": {}}
    for rel in used:
        g = sha256(HOME / rel)
        res["sha256_vs_e06data_SHA256SUMS"][rel] = {"got": g, "listed": want.get(rel), "ok": g == want.get(rel)}
    man = json.loads((INP / "controls_manifest.json").read_text(encoding="utf-8"))
    fpres = {}
    for key, p in (("qe_scores", D / "v2_scored.tsv"), ("pretrain_src", D / "v1_fixed/train.clean.en"),
                   ("pretrain_tgt", D / "v1_fixed/train.clean.fr"), ("provenance", INP / "provenance_labels.npy"),
                   ("provenance_report", INP / "provenance_report.json"), ("pool_mask", INP / "pool_mask_reused.npy")):
        e = {k: man["inputs"][key][k] for k in ("size", "blake2b_first_16MiB", "blake2b_last_16MiB")}
        g = fp16(p)
        fpres[key] = {"file": str(p), "e03_path": man["inputs"][key]["path"], "ok": g == e}
    res["e03_fingerprints"] = fpres
    bund = sums(INP / "bundle_SHA256SUMS")
    res["bundle_sha256"] = {f: {"got": sha256(INP / f), "listed": bund.get(f)} for f in
                            ("provenance_labels.npy", "provenance_report.json", "pool_mask_reused.npy")}
    for v in res["bundle_sha256"].values():
        v["ok"] = v["got"] == v["listed"]
    res["pool_mask_sha256_vs_manifest"] = res["bundle_sha256"]["pool_mask_reused.npy"]["got"] == man["pool_mask"]["sha256"]
    res["all_ok"] = (all(v["ok"] for v in res["sha256_vs_e06data_SHA256SUMS"].values())
                     and all(v["ok"] for v in fpres.values()) and all(v["ok"] for v in res["bundle_sha256"].values())
                     and res["pool_mask_sha256_vs_manifest"])
    res["env"] = env()
    return res


# ======================================================================================
class Par:
    def __init__(self, name: str):
        d = D / "e05_src"
        self.name = name
        self.src, self.tgt, self.lab = (lines_of(d / f"{name}.{e}") for e in ("en", "fr", "source"))
        if not (len(self.src) == len(self.tgt) == len(self.lab)):
            raise Fail(f"{name}: line counts differ")
        self.n = len(self.src)
        self.nsrc = [nkey(x) for x in self.src]
        self.ntgt = [nkey(x) for x in self.tgt]

    def rows(self, idx):
        return ([self.src[i] for i in idx], [self.tgt[i] for i in idx],
                [self.nsrc[i] for i in idx], [self.ntgt[i] for i in idx])


def load_idx(p: Path, n: int) -> tuple[list[int], dict]:
    a = np.load(p, allow_pickle=False)
    info = {"dtype": str(a.dtype), "ndim": int(a.ndim), "len": int(a.size)}
    ok = a.ndim == 1 and np.issubdtype(a.dtype, np.integer)
    if ok and a.size:
        b = a.astype(np.int64)
        ok = bool(b[0] >= 0 and b[-1] < n and np.all(b[1:] > b[:-1]))
    info["strictly_ascending_in_range"] = bool(ok)
    if not ok:
        raise Fail(f"{p}: not a strictly ascending 1-d integer index into [0, {n})")
    return a.astype(np.int64).tolist(), info


def cmd_sets() -> dict:
    t0 = time.time()
    H = Held(D / "e05_src")
    H2 = Held(D / "phase0_e05")
    res = {"heldout_un": H.describe(),
           "phase0_e05_heldout_identical_text": (H.src, H.tgt) == (H2.src, H2.tgt)}
    P = {n: Par(n) for n in PARENTS}
    res["parents_rows"] = {n: p.n for n, p in P.items()}
    rep = json.loads((FEAS / "feasibility_report.json").read_text(encoding="utf-8"))
    repo = sums(INP / "feasibility_SHA256SUMS")
    res["feasibility_report_sha256"] = {"got": sha256(FEAS / "feasibility_report.json"),
                                        "repo_SHA256SUMS": repo.get("feasibility_report.json")}
    # the report's inputs are the files read here
    ri = {}
    for f in [f"{n}.{e}" for n in PARENTS for e in ("en", "fr", "source")] + ["heldout_un.en", "heldout_un.fr"]:
        g = sha256(D / "e05_src" / f)
        ri[f] = {"got": g, "report": rep["inputs"][f]["sha256"], "ok": g == rep["inputs"][f]["sha256"]}
    res["report_inputs_are_these_files"] = all(v["ok"] for v in ri.values())
    res["report_inputs"] = ri

    meta, failed = {}, {}
    for tag, v in rep["anchors"].items():
        meta[v["file"]] = (tag, "anchor", v["parent"], v["rows"], v["heldout_un_shared_lines"])
    for tag, v in rep["results"].items():
        if v.get("file") is None:
            failed[tag] = str(v.get("failed", "no file")).split("\n")[0]
            continue
        meta[v["file"]] = (tag, "result", v["parent"], v["rows"], v["heldout_un_shared_lines"])
    on_disk = sorted(p.name for p in FEAS.glob("FEASIBILITY_*.npy"))
    res["feasibility_listing"] = {"npy_on_disk": len(on_disk), "report_entries_with_file": len(meta),
                                  "report_results_without_file": failed,
                                  "report_files_map_entries": len(rep["files"]),
                                  "report_files_map_not_npy": sorted(k for k in rep["files"] if not k.endswith(".npy")),
                                  "disk_equals_report": sorted(meta) == on_disk}
    per = {}
    for fn in on_disk:
        tag, kind, parent, rows, field = meta[fn]
        g = sha256(FEAS / fn)
        idx, info = load_idx(FEAS / fn, P[parent].n)
        m = measure(*P[parent].rows(idx), H)
        m.update({"tag": tag, "kind": kind, "parent": parent, "sha256": g,
                  "sha256_ok_report": g == rep["files"].get(fn), "sha256_ok_repo": g == repo.get(fn),
                  "rows_ok_report": len(idx) == rows, "index": info,
                  "un_rows": sum(1 for i in idx if P[parent].lab[i] == UN),
                  "report_field": field, "agrees_with_report_field": m["distinct_shared_lines_as_report"] == field})
        m["all_zero"] = all_zero(m)
        per[fn] = m
    res["feasibility_sets"] = per
    res["feasibility_summary"] = {
        "files": len(per), "all_zero": sum(v["all_zero"] for v in per.values()),
        "agree_with_report_field": sum(v["agrees_with_report_field"] for v in per.values()),
        "sha256_ok_report": sum(v["sha256_ok_report"] for v in per.values()),
        "sha256_ok_repo": sum(v["sha256_ok_repo"] for v in per.values()),
        "rows_ok_report": sum(v["rows_ok_report"] for v in per.values()),
        "all_rows_un": sum(v["un_rows"] == v["rows"] for v in per.values()),
        "by_parent": dict(collections.Counter(v["parent"] for v in per.values())),
        "distinct_index_contents": len({v["sha256"] for v in per.values()}),
        "repo_SHA256SUMS_npy_listed": sum(1 for k in repo if k.endswith(".npy"))}

    sup = {}
    for f, twin in (("R0.npy", "FEASIBILITY_ft_un_r0__od6-keep__seed42.npy"),
                    ("A_narrow_42_58.npy", "FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy")):
        idx, info = load_idx(D / "vocab" / f, P["ft_random"].n)
        m = measure(*P["ft_random"].rows(idx), H)
        m.update({"index": info, "same_bytes_as": twin if sha256(D / "vocab" / f) == sha256(FEAS / twin) else None})
        m["all_zero"] = all_zero(m)
        sup[f"vocab/{f}"] = m
    ppairs = {n: set(zip(p.src, p.tgt)) for n, p in P.items()}
    for n, parent in E05.items():
        s, t = lines_of(D / "phase0_e05" / f"{n}.en"), lines_of(D / "phase0_e05" / f"{n}.fr")
        if len(s) != len(t):
            raise Fail(f"{n}: line counts differ")
        m = measure(s, t, [nkey(x) for x in s], [nkey(x) for x in t], H)
        m["rows_whose_pair_is_in_parent"] = {"parent": parent, "rows": sum(1 for p in zip(s, t) if p in ppairs[parent])}
        m["all_zero"] = all_zero(m)
        sup[f"phase0_e05/{n}"] = m
    for n, p in P.items():
        m = measure(p.src, p.tgt, p.nsrc, p.ntgt, H)
        m["all_zero"] = all_zero(m)
        sup[f"{n} (all {p.n:,} rows)"] = m
    res["supplementary_sets"] = sup

    # positive controls: the same code must find what is planted
    ctrl = {"heldout_un_against_itself": measure(H.src, H.tgt, H.nsrc, H.ntgt, H)}
    R = P["ft_random"]
    js = list(range(0, 50, 5))
    S = R.src[:1000] + [H.src[j] for j in js] + [H.src[100], R.src[1]]
    T = R.tgt[:1000] + [H.tgt[j] for j in js] + [R.tgt[0], H.tgt[200] + "  "]
    m = measure(S, T, [nkey(x) for x in S], [nkey(x) for x in T], H)
    exp = {"exact_source": 11, "exact_target": 10, "exact_pair": 10, "norm_source": 11, "norm_target": 11, "any": 12}
    ctrl["planted"] = {"measured": m["set_rows_with_heldout"], "expected": exp, "ok": m["set_rows_with_heldout"] == exp}
    ctrl["self_ok"] = (ctrl["heldout_un_against_itself"]["set_rows_with_heldout"]["exact_pair"] == H.n
                       and ctrl["heldout_un_against_itself"]["heldout_rows_found_in_set"]["exact_pair"] == H.n)
    res["positive_controls"] = ctrl
    res["seconds"] = round(time.time() - t0, 1)
    res["env"] = env()
    return res


# ======================================================================================
def cmd_pre() -> dict:
    t0 = time.time()
    H = Held(D / "e05_src")
    en, fr = D / "v1_fixed/train.clean.en", D / "v1_fixed/train.clean.fr"
    kp, ks, kns, knt = array("Q"), array("Q"), array("Q"), array("Q")
    hits = []
    n = 0
    with open(en, "rb") as a, open(fr, "rb") as b:
        for x, y in itertools.zip_longest(a, b):
            if x is None or y is None:
                raise Fail(f"train.clean sides differ in length at line {n + 1}")
            if x.endswith(b"\n"):
                x = x[:-1]
            if y.endswith(b"\n"):
                y = y[:-1]
            s, t = x.decode("utf-8"), y.decode("utf-8")
            ns, nt = nkey(s), nkey(t)
            kp.append(pkey(x, y)); ks.append(hk(x)); kns.append(hk(ns.encode("utf-8"))); knt.append(hk(nt.encode("utf-8")))
            f = (s in H.S, t in H.T, (s, t) in H.P, ns in H.NS, nt in H.NT)
            if f[0] or f[1] or f[2] or f[3] or f[4]:
                hits.append([n, *f, s, t])
            n += 1
    U = {}
    for name, arr in (("pair", kp), ("src_exact", ks), ("src_norm", kns), ("tgt_norm", knt)):
        u = np.unique(np.frombuffer(arr, dtype=np.uint64))
        np.save(WORK / f"pre_{name}_u.npy", u)
        U[name] = u
    found = {"exact_source": {h[6] for h in hits if h[1]}, "exact_target": {h[7] for h in hits if h[2]},
             "exact_pair": {(h[6], h[7]) for h in hits if h[3]},
             "norm_source": {nkey(h[6]) for h in hits if h[4]}, "norm_target": {nkey(h[7]) for h in hits if h[5]}}
    res = {"rows": n, "distinct": {"pairs": int(len(U["pair"])), "sources_exact": int(len(U["src_exact"])),
                                   "sources_norm": int(len(U["src_norm"])), "targets_norm": int(len(U["tgt_norm"]))},
           "pretrain_rows_with_heldout": {"exact_source": sum(1 for h in hits if h[1]),
                                          "exact_target": sum(1 for h in hits if h[2]),
                                          "exact_pair": sum(1 for h in hits if h[3]),
                                          "norm_source": sum(1 for h in hits if h[4]),
                                          "norm_target": sum(1 for h in hits if h[5]), "any": len(hits)},
           "heldout_rows_found_in_pretrain": {
               "exact_source": sum(1 for x in H.src if x in found["exact_source"]),
               "exact_target": sum(1 for x in H.tgt if x in found["exact_target"]),
               "exact_pair": sum(1 for p in zip(H.src, H.tgt) if p in found["exact_pair"]),
               "norm_source": sum(1 for x in H.nsrc if x in found["norm_source"]),
               "norm_target": sum(1 for x in H.ntgt if x in found["norm_target"])},
           "first_hits": [h[:6] + [h[6][:150], h[7][:150]] for h in hits[:20]]}
    # heldout keys against the key arrays (a second path to the same zero)
    hp = np.array(sorted({pkey(s.encode(), t.encode()) for s, t in zip(H.src, H.tgt)}), dtype=np.uint64)
    hs = np.array(sorted({hk(x.encode()) for x in H.nsrc}), dtype=np.uint64)
    ht = np.array(sorted({hk(x.encode()) for x in H.ntgt}), dtype=np.uint64)
    res["heldout_keys_in_pretrain_keys"] = {"pair": int(np.isin(hp, U["pair"]).sum()),
                                            "norm_source": int(np.isin(hs, U["src_norm"]).sum()),
                                            "norm_target": int(np.isin(ht, U["tgt_norm"]).sum())}
    # positive control for this path: E0.5's "already in pretraining" (exact pairs, PROTOCOL.md E0.5 table)
    pc = {}
    for nm in E05:
        a = Path(D / "phase0_e05" / f"{nm}.en").read_bytes().split(b"\n")
        b = Path(D / "phase0_e05" / f"{nm}.fr").read_bytes().split(b"\n")
        if a and a[-1] == b"":
            a.pop()
        if b and b[-1] == b"":
            b.pop()
        k = np.fromiter((pkey(x, y) for x, y in zip(a, b)), dtype=np.uint64, count=len(a))
        pc[nm] = round(float(np.isin(k, U["pair"]).mean()) * 100, 1)
    res["positive_control_e05_pairs_in_pretraining_pct"] = {
        "measured": pc, "protocol_table": {"ft_topk_div": 52.9, "ft_topk_rep": 42.4, "ft_random_small": 58.5,
                                           "ft_un_div": 36.7, "ft_un_rep": 36.7, "ft_un_random": 37.9}}
    res["seconds"] = round(time.time() - t0, 1)
    res["env"] = env()
    return res


# ======================================================================================
def bounds(path: Path, k: int, N: int) -> int:
    size = path.stat().st_size
    if k <= 0:
        return 0
    if k >= N:
        return size
    with open(path, "rb") as f:
        f.seek(size * k // N)
        f.readline()
        return f.tell()


def cmd_pool(k: int, N: int) -> dict:
    t0 = time.time()
    H = Held(D / "e05_src")
    path = D / "v2_scored.tsv"
    start, end = bounds(path, k, N), bounds(path, k + 1, N)
    kp, ks, kns, knt = array("Q"), array("Q"), array("Q"), array("Q")
    hits, odd = [], []
    n, pos = 0, start
    with open(path, "rb") as f:
        f.seek(start)
        while pos < end:
            line = f.readline()
            if not line:
                raise Fail(f"{path} ended early")
            pos += len(line)
            if line.endswith(b"\n"):
                line = line[:-1]
            u = line.decode("utf-8")
            F = u.split("\t")
            if len(F) != 3:
                odd.append([n, len(F)])
                if len(F) < 3:
                    raise Fail(f"chunk {k} row {n}: {len(F)} fields (E0.3 would stop here too)")
            B = line.split(b"\t")
            s, t = F[1], F[2]
            ns, nt = nkey(s), nkey(t)
            kp.append(pkey(B[1], B[2])); ks.append(hk(B[1]))
            kns.append(hk(ns.encode("utf-8"))); knt.append(hk(nt.encode("utf-8")))
            f5 = (s in H.S, t in H.T, (s, t) in H.P, ns in H.NS, nt in H.NT)
            if f5[0] or f5[1] or f5[2] or f5[3] or f5[4]:
                hits.append([n, *f5, s, t])
            n += 1
    for name, arr in (("pair", kp), ("src_exact", ks), ("src_norm", kns), ("tgt_norm", knt)):
        np.save(WORK / f"pool_{k}_{name}.npy", np.frombuffer(arr, dtype=np.uint64))
    return {"chunk": k, "of": N, "start": start, "end": end, "rows": n, "hits": hits,
            "rows_not_3_fields": len(odd), "rows_not_3_fields_first": odd[:20],
            "seconds": round(time.time() - t0, 1), "env": env()}


# ======================================================================================
def cmd_evalcopy() -> dict:
    H = Held(D / "e05_src")
    ev = INP / "filtered_tests"
    fs, ft = lines_of(ev / "heldout_un.en"), lines_of(ev / "heldout_un.fr")
    kept, j = [], 0
    for i in range(H.n):
        if j < len(fs) and fs[j] == H.src[i] and ft[j] == H.tgt[i]:
            kept.append(i)
            j += 1
    dropped = sorted(set(range(H.n)) - set(kept))
    res = {"rows": [len(fs), len(ft)], "sha256": {f: sha256(ev / f) for f in ("heldout_un.en", "heldout_un.fr")},
           "is_heldout_with_rows_removed_in_order": j == len(fs) == len(ft), "dropped_rows_0based": dropped}
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(D / "spm/spm_enfr_v1_fixed.model"))
        L = [len(sp.encode(x)) for x in H.src]
        order = sorted(range(H.n), key=lambda i: -L[i])
        res["source_pieces"] = {"sentencepiece": spm.__version__, "dropped_rows_pieces": [L[i] for i in dropped],
                                "max_kept": max(L[i] for i in kept), "rows_over_254": [i for i in range(H.n) if L[i] > 254],
                                "rows_over_253": [i for i in range(H.n) if L[i] > 253],
                                "top5": [[i, L[i]] for i in order[:5]]}
    except Exception as exc:                                              # noqa: BLE001
        res["source_pieces"] = f"n.m.: {exc!r}"
    res["env"] = env()
    return res


# ======================================================================================
def cmd_reduce() -> dict:
    t0 = time.time()
    H = Held(D / "e05_src")
    man = json.loads((INP / "controls_manifest.json").read_text(encoding="utf-8"))
    hm = man["heldout_sets"]["heldout_un"]
    chunks = sorted(WORK.glob("pool_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in chunks]
    N = parts[0]["of"]
    if [p["chunk"] for p in parts] != list(range(N)):
        raise Fail(f"pool chunks present: {[p['chunk'] for p in parts]}")
    for a, b in zip(parts, parts[1:]):
        if a["end"] != b["start"]:
            raise Fail("pool chunks do not tile the file")
    if parts[0]["start"] != 0 or parts[-1]["end"] != (D / "v2_scored.tsv").stat().st_size:
        raise Fail("pool chunks do not cover the file")
    off, hits = 0, []
    for p in parts:
        hits += [[off + h[0]] + h[1:] for h in p["hits"]]
        off += p["rows"]
    n = off
    K = {nm: np.concatenate([np.load(WORK / f"pool_{c}_{nm}.npy") for c in range(N)])
         for nm in ("pair", "src_exact", "src_norm", "tgt_norm")}
    res = {"pool_rows": n, "pool_rows_e03_manifest": man["n_pool_total"],
           "rows_not_3_fields": sum(p["rows_not_3_fields"] for p in parts)}

    rid = np.array([h[0] for h in hits], dtype=np.int64)
    es, et, ep, ns, nt = (np.array([h[i] for h in hits], dtype=bool) for i in range(1, 6))
    res["pool_rows_with_heldout"] = {
        "exact_pair": int(ep.sum()), "exact_source": int(es.sum()), "exact_target": int(et.sum()),
        "exact_source_or_target": int((es | et).sum()), "exact_source_not_pair": int((es & ~ep).sum()),
        "exact_target_not_pair": int((et & ~ep).sum()),
        "norm_source": int(ns.sum()), "norm_target": int(nt.sum()),
        "norm_source_not_exact": int((ns & ~es).sum()), "norm_target_not_exact": int((nt & ~et).sum()),
        "any": int(len(hits))}
    cat = {"exact_pair": ep, "extra_by_source": ns & ~ep, "extra_by_target": nt & ~ns & ~ep}
    cat["rows_reserved_total"] = cat["exact_pair"] | cat["extra_by_source"] | cat["extra_by_target"]
    res["e03_reservation_categories"] = {
        "measured": {k: int(v.sum()) for k, v in cat.items()},
        "e03_manifest": man["reservation"] | {"heldout_un": {k: hm[k] for k in hm if k.startswith("reserved_rows")}}}
    # the same counts through the key arrays (second path)
    hp = np.unique(np.array([pkey(s.encode(), t.encode()) for s, t in zip(H.src, H.tgt)], dtype=np.uint64))
    hsx = np.unique(np.array([hk(x.encode()) for x in H.src], dtype=np.uint64))
    hs = np.unique(np.array([hk(x.encode()) for x in H.nsrc], dtype=np.uint64))
    ht = np.unique(np.array([hk(x.encode()) for x in H.ntgt], dtype=np.uint64))
    bp, bsx, bs, bt = np.isin(K["pair"], hp), np.isin(K["src_exact"], hsx), np.isin(K["src_norm"], hs), np.isin(K["tgt_norm"], ht)
    res["key_path"] = {"exact_pair": int(bp.sum()), "exact_source": int(bsx.sum()), "norm_source": int(bs.sum()),
                       "norm_target": int(bt.sum()), "extra_by_source": int((bs & ~bp).sum()),
                       "extra_by_target": int((bt & ~bs & ~bp).sum()), "total": int((bp | bs | bt).sum()),
                       "same_rows_as_string_path": bool(np.array_equal(np.flatnonzero(bp | bs | bt),
                                                                       np.sort(rid[ep | ns | nt])))}
    # the heldout side
    fsrc = {h[6] for h in hits if h[1]}
    ftgt = {h[7] for h in hits if h[2]}
    fpair = {(h[6], h[7]) for h in hits if h[3]}
    fns = {nkey(h[6]) for h in hits if h[4]}
    fnt = {nkey(h[7]) for h in hits if h[5]}
    res["heldout_rows_found_in_pool"] = {
        "exact_pair": sum(1 for p in zip(H.src, H.tgt) if p in fpair),
        "exact_source": sum(1 for x in H.src if x in fsrc), "exact_target": sum(1 for x in H.tgt if x in ftgt),
        "norm_source": sum(1 for x in H.nsrc if x in fns), "norm_target": sum(1 for x in H.ntgt if x in fnt)}
    cp = collections.Counter((h[6], h[7]) for h in hits if h[3])
    cps = collections.Counter(h[6] for h in hits if h[1])
    cpt = collections.Counter(h[7] for h in hits if h[2])
    per_pair = [cp[p] for p in zip(H.src, H.tgt)]
    res["pool_copies_per_heldout_row"] = {
        "pair": dict(sorted(collections.Counter(per_pair).items())),
        "pair_max": max(per_pair), "pair_rows_with_more_than_1": sum(1 for c in per_pair if c > 1),
        "source_max": max(cps[x] for x in H.src), "target_max": max(cpt[x] for x in H.tgt),
        "sum_over_distinct_pairs": sum(cp[p] for p in H.P)}
    dup = H.describe()["rows_of_duplicated_pairs"]
    res["duplicated_heldout_pair"] = [{"heldout_rows": r, "pool_copies": cp[(H.src[r[0]], H.tgt[r[0]])],
                                       "pool_rows": sorted(int(h[0]) for h in hits if h[3] and (h[6], h[7]) == (H.src[r[0]], H.tgt[r[0]]))}
                                      for r in dup]

    # labels and mask (E0.3's inputs)
    lab = np.load(INP / "provenance_labels.npy", allow_pickle=False)
    mask = np.load(INP / "pool_mask_reused.npy", allow_pickle=False)
    names = json.loads((INP / "provenance_report.json").read_text(encoding="utf-8"))["sources"]
    if len(lab) != n or len(mask) != n or mask.dtype != np.bool_:
        raise Fail("labels / mask do not describe this pool")
    nm = lambda c: names[c] if c < len(names) else ("unmatched" if c == 255 else f"code_{c}")
    res["touching_rows_by_source_and_mask"] = {
        k: {"rows": int(v.sum()), "inside_reused_mask": int(mask[rid[v]].sum()),
            "sources": dict(sorted(collections.Counter(nm(int(c)) for c in lab[rid[v]]).items()))}
        for k, v in list(cat.items()) + [("exact_source", es), ("exact_target", et), ("any", np.ones(len(rid), bool))]}
    mpair = {(h[6], h[7]) for h in hits if h[3] and mask[h[0]]}
    msrc = {h[6] for h in hits if h[1] and mask[h[0]]}
    mtgt = {h[7] for h in hits if h[2] and mask[h[0]]}
    res["heldout_rows_found_in_mask_rows"] = {
        "exact_pair": sum(1 for p in zip(H.src, H.tgt) if p in mpair),
        "exact_source": sum(1 for x in H.src if x in msrc), "exact_target": sum(1 for x in H.tgt if x in mtgt)}

    # E0.3's candidate pool and draw, re-derived (e03_build_controls.py l.272-299, 367-383)
    U = {nm_: np.load(WORK / f"pre_{nm_}_u.npy") for nm_ in ("pair", "src_exact", "src_norm", "tgt_norm")}
    in_pre = {"pair": np.isin(K["pair"], U["pair"]), "source": np.isin(K["src_norm"], U["src_norm"]),
              "target": np.isin(K["tgt_norm"], U["tgt_norm"])}
    pre_any = in_pre["pair"] | in_pre["source"] | in_pre["target"]
    res["scored_rows_in_pretrain"] = {"measured": {k: int(v.sum()) for k, v in in_pre.items()} | {"any": int(pre_any.sum())},
                                      "e03_manifest": man["pretrain"]["scored_rows_in_pretrain"]}
    un_code = names.index(UN)
    in_dom = lab == un_code
    cand = in_dom & mask
    pool = np.flatnonzero(cand & ~pre_any)
    res["heldout_candidates"] = {
        "measured": {"n_pool_eligible": int(mask.sum()), "label_rows": int(in_dom.sum()), "pool_rows": int(len(pool)),
                     "pool_rows_excluded_pretrain": int((cand & pre_any).sum()),
                     "pool_rows_excluded_mask": int((in_dom & ~mask).sum())},
        "e03_manifest": {"n_pool_eligible": man["n_pool_eligible"], "label_rows": hm["label_rows"],
                         "pool_rows": hm["pool_rows"], "pool_rows_excluded_pretrain": hm["pool_rows_excluded_pretrain"],
                         "pool_rows_excluded_mask": hm["pool_rows_excluded_mask"]}}
    rng = np.random.RandomState(man["seed"])
    pick = rng.choice(pool, man["n_heldout"], replace=False)
    txt = {int(h[0]): (h[6], h[7]) for h in hits if h[3]}
    same = [txt.get(int(r)) == (H.src[j], H.tgt[j]) for j, r in enumerate(pick)]
    by_pair = np.isin(K["pair"], np.unique(K["pair"][pick]))
    by_src = np.isin(K["src_norm"], np.unique(K["src_norm"][pick]))
    by_tgt = np.isin(K["tgt_norm"], np.unique(K["tgt_norm"][pick]))
    res["draw_rederived"] = {
        "seed": man["seed"], "n": int(len(pick)),
        "rows_equal_to_heldout_un_row_by_row": int(sum(same)),
        "first_mismatch": next((j for j, s in enumerate(same) if not s), None),
        "distinct_pairs_in_draw": int(len(np.unique(K["pair"][pick]))),
        "drawn_rows_in_pretrain_any": int(pre_any[pick].sum()),
        "reservation_from_drawn_rows": {"exact_pair": int(by_pair.sum()), "extra_by_source": int((by_src & ~by_pair).sum()),
                                        "extra_by_target": int((by_tgt & ~by_src & ~by_pair).sum()),
                                        "total": int((by_pair | by_src | by_tgt).sum())}}
    # PROTOCOL.md D4's figure (default picks, no pretraining exclusion): 793/2000 pairs, 826/2000 sources verbatim
    rng0 = np.random.RandomState(man["seed"])
    pick0 = rng0.choice(np.flatnonzero(cand), man["n_heldout"], replace=False)
    res["protocol_d4_default_draw"] = {
        "pairs_verbatim_in_pretrain": int(np.isin(K["pair"][pick0], U["pair"]).sum()),
        "sources_verbatim_in_pretrain": int(np.isin(K["src_exact"][pick0], U["src_exact"]).sum()),
        "protocol_text": "793/2000 pairs (826/2000 sources)",
        "note": "the D4 measurement's pool definition is not recorded; this assumes UN & reused mask, first rng use"}
    res["seconds"] = round(time.time() - t0, 1)
    res["env"] = env()
    return res


def main() -> int:
    cmd = sys.argv[1]
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        if cmd == "pool":
            k, N = int(sys.argv[2]), int(sys.argv[3])
            dump(f"pool_{k}", cmd_pool(k, N))
        elif cmd in ("inputs", "sets", "pre", "evalcopy", "reduce"):
            out = {"inputs": cmd_inputs, "sets": cmd_sets, "pre": cmd_pre, "evalcopy": cmd_evalcopy,
                   "reduce": cmd_reduce}[cmd]()
            dump(cmd, out)
        else:
            raise Fail(f"unknown subcommand {cmd}")
    except Fail as e:
        print(f"FAIL: {e}", flush=True)
        return 1
    print(f"{cmd} {' '.join(sys.argv[2:])}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
