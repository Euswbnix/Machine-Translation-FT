#!/usr/bin/env python3
"""Fixture test for heldout_hygiene.py and check_awk.sh: plants known overlaps of every kind
(exact pair, exact source / target under another partner, normalisation-only source / target,
a duplicated held-out pair, a 4-field TSV row) and compares every count with a brute-force
recount written here without the script's helpers. Tiny files; run before the real inputs.

  test_hygiene.py --tmp DIR        (DIR is created; nothing outside it is written)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAIL = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"))
    if not cond:
        FAIL.append(name)


def nrm(s):                                    # e03's normalisation, written out again
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def wl(p: Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in lines:
            f.write(x + "\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fp(p):
    b = Path(p).read_bytes()
    n = 16 * 1024 * 1024
    return {"size": len(b), "blake2b_first_16MiB": hashlib.blake2b(b[:n]).hexdigest(),
            "blake2b_last_16MiB": hashlib.blake2b(b[max(0, len(b) - n):]).hexdigest()}


def brute(src, tgt, hs, ht):
    """Every count, the slow way: set rows touching heldout_un and heldout rows found."""
    P = list(zip(hs, ht))
    rows = {"source": sum(x in hs for x in src), "target": sum(y in ht for y in tgt),
            "pair": sum((x, y) in P for x, y in zip(src, tgt)),
            "norm_source": sum(nrm(x) in [nrm(h) for h in hs] for x in src),
            "norm_target": sum(nrm(y) in [nrm(h) for h in ht] for y in tgt)}
    pairs = list(zip(src, tgt))
    ho = {"exact": {"source": sum(h in src for h in hs), "target": sum(h in tgt for h in ht),
                    "pair": sum(p in pairs for p in P)},
          "norm": {"source": sum(nrm(h) in [nrm(x) for x in src] for h in hs),
                   "target": sum(nrm(h) in [nrm(y) for y in tgt] for h in ht)}}
    shared = {"source": len({x for x in src if x in hs}), "target": len({y for y in tgt if y in ht})}
    return rows, ho, shared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    T = a.tmp
    if T.exists() and any(T.iterdir()):
        sys.exit(f"{T} is not empty")
    home, out = T / "home", T / "out"
    D = home / "e06data"

    hs = ["The Committee met.", "Item 3 of the agenda.", "Report of the Secretary-General",
          "Résumé  of  findings", "The Committee met."]
    ht = ["Le Comité s'est réuni.", "Point 3 de l'ordre du jour.", "Rapport du Secrétaire général",
          "Résumé des conclusions", "Le Comité s'est réuni."]
    wl(D / "e05_src/heldout_un.en", hs)
    wl(D / "e05_src/heldout_un.fr", ht)
    wl(D / "phase0_e05/heldout_un.en", hs)
    wl(D / "phase0_e05/heldout_un.fr", ht)

    top_s = [f"topk sentence {i}" for i in range(40)]
    top_t = [f"phrase topk {i}" for i in range(40)]
    top_l = ["un" if i % 3 else "europarl" for i in range(40)]
    top_s[5] = hs[1]                                   # exact source, other target
    top_t[9] = ht[2].upper()                           # norm-only target
    top_s[12], top_t[12] = hs[2], ht[2]                # exact pair
    rnd_s = [f"random sentence {i}" for i in range(40)]
    rnd_t = [f"phrase hasard {i}" for i in range(40)]
    rnd_l = ["un"] * 40
    rnd_s[7] = "Résumé of findings"                    # norm-only source (whitespace)
    for n, s_, t_, l_ in (("ft_topk", top_s, top_t, top_l), ("ft_random", rnd_s, rnd_t, rnd_l)):
        wl(D / f"e05_src/{n}.en", s_)
        wl(D / f"e05_src/{n}.fr", t_)
        wl(D / f"e05_src/{n}.source", l_)
    six = {"ft_topk_div": ("t", range(0, 6)), "ft_topk_rep": ("t", range(10, 16)), "ft_random_small": ("r", range(0, 8)),
           "ft_un_div": ("t", range(20, 26)), "ft_un_rep": ("t", range(26, 32)), "ft_un_random": ("r", range(10, 20))}
    for n, (p, rg) in six.items():
        S, Tt = (top_s, top_t) if p == "t" else (rnd_s, rnd_t)
        wl(D / f"phase0_e05/{n}.en", [S[i] for i in rg])
        wl(D / f"phase0_e05/{n}.fr", [Tt[i] for i in rg])
    (D / "phase0_e05/manifest.json").write_text("{}\n")

    F = home / "e06runs/feasibility"
    F.mkdir(parents=True)
    fsets = {"FEASIBILITY_ft_un_div__anchor.npy": ("ft_un_div", "anchor", "ft_topk", list(range(0, 10))),
             "FEASIBILITY_ft_un_rep__anchor.npy": ("ft_un_rep", "anchor", "ft_topk", list(range(10, 20))),
             "FEASIBILITY_ft_un_random__anchor.npy": ("ft_un_random", "anchor", "ft_random", list(range(0, 10))),
             "FEASIBILITY_ft_un_r0__od6-keep__seed42.npy": ("ft_un_r0__od6-keep__seed42", "c", "ft_random", list(range(20, 30))),
             "FEASIBILITY_ft_un_a_narrow__od6-keep__seed42.npy": ("ft_un_a_narrow__od6-keep__seed42", "c", "ft_random", list(range(30, 40))),
             "FEASIBILITY_ft_un_div_narrow__od2-a__od1-a.npy": ("ft_un_div_narrow__od2-a__od1-a", "c", "ft_topk", [1, 3, 9, 12, 30])}
    rep = {"anchors": {}, "results": {"ft_un_a_narrow__od6-remove__seed42": {"failed": "pool too small\nmore"}},
           "files": {}, "inputs": {}}
    for fn, (tag, kind, par, idx) in fsets.items():
        np.save(F / fn, np.asarray(idx, dtype=np.int64))
        S, Tt = (top_s, top_t) if par == "ft_topk" else (rnd_s, rnd_t)
        _, _, shared = brute([S[i] for i in idx], [Tt[i] for i in idx], hs, ht)
        e = {"file": fn, "parent": par, "rows": len(idx), "heldout_un_shared_lines": shared}
        (rep["anchors"] if kind == "anchor" else rep["results"])[tag] = e
        rep["files"][fn] = sha(F / fn)
    for f in [f"{n}.{e}" for n in ("ft_topk", "ft_random") for e in ("en", "fr", "source")] + ["heldout_un.en", "heldout_un.fr"]:
        rep["inputs"][f] = {"sha256": sha(D / "e05_src" / f)}
    (F / "feasibility_report.json").write_text(json.dumps(rep))
    (out / "inputs").mkdir(parents=True)
    (out / "inputs/feasibility_SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(F.iterdir()) if p.name != "NOT_FOR_TRAINING"))
    (D / "vocab").mkdir()
    np.save(D / "vocab/R0.npy", np.arange(20, 30, dtype=np.int64))
    np.save(D / "vocab/A_narrow_42_58.npy", np.asarray([30, 31, 32, 33, 34, 35, 36, 37, 38, 7], dtype=np.int64)[np.argsort([30, 31, 32, 33, 34, 35, 36, 37, 38, 7])])

    pool = [f"0.{i % 10}\tpool src {i}\tpool tgt {i}" for i in range(120)]
    plant = {3: (hs[0], ht[0]), 17: (hs[0], ht[0]), 40: (hs[0], ht[0]), 41: (hs[1], ht[1]), 42: (hs[2], ht[2]),
             43: (hs[3], ht[3]), 60: (hs[1], "autre"), 61: (hs[1], "autre 2"), 70: ("the committee met.", "x"),
             80: ("y", ht[2]), 90: ("z", ht[3].upper()), 100: ("Résumé of findings", "Résumé des conclusions")}
    for r, (s_, t_) in plant.items():
        pool[r] = f"0.9\t{s_}\t{t_}"
    pool[110] = "0.5\tfour\tfields\textra"
    wl(D / "v2_scored.tsv", pool)
    pre_s = [f"pre src {i}" for i in range(100)]
    pre_t = [f"pre tgt {i}" for i in range(100)]
    pre_s[50], pre_t[50] = hs[2], ht[2]
    pre_s[60] = hs[1].upper()
    pre_t[70] = ht[3]
    wl(D / "v1_fixed/train.clean.en", pre_s)
    wl(D / "v1_fixed/train.clean.fr", pre_t)

    labels = np.asarray([2 if i % 4 else 0 for i in range(120)], dtype=np.uint8)
    labels[100] = 255
    np.save(out / "inputs/provenance_labels.npy", labels)
    mask = np.ones(120, dtype=bool)
    mask[40] = False
    np.save(out / "inputs/pool_mask_reused.npy", mask)
    (out / "inputs/provenance_report.json").write_text(json.dumps({"sources": ["europarl", "commoncrawl", "un"]}))
    wl(out / "inputs/filtered_tests/heldout_un.en", [x for i, x in enumerate(hs) if i != 2])
    wl(out / "inputs/filtered_tests/heldout_un.fr", [x for i, x in enumerate(ht) if i != 2])
    sums = []
    for p in sorted(D.rglob("*")):
        if p.is_file():
            sums.append(f"{sha(p)}  {p.relative_to(home)}")
    (out / "inputs/e06data_SHA256SUMS").write_text("\n".join(sums) + "\n")
    man = {"inputs": {k: {"path": "/fixture", **fp(p)} for k, p in
                      (("qe_scores", D / "v2_scored.tsv"), ("pretrain_src", D / "v1_fixed/train.clean.en"),
                       ("pretrain_tgt", D / "v1_fixed/train.clean.fr"), ("provenance", out / "inputs/provenance_labels.npy"),
                       ("pool_mask", out / "inputs/pool_mask_reused.npy"))}}
    (out / "inputs/controls_manifest.json").write_text(json.dumps(man))

    env = {**os.environ, "HH_BLOCK": "64"}
    run = lambda *c: subprocess.run([a.python, str(HERE / "heldout_hygiene.py"), *c, "--home", str(home), "--out", str(out)],
                                    env=env, capture_output=True, text=True)
    r = run("inputs")
    part = json.loads((out / "parts/inputs.json").read_text())
    check("inputs: every e06data sha256 matches the SUMS file", part["all_match"], r.stdout[-500:])
    check("inputs: E0.3 fingerprints match", all(v["match"] for v in part["e03_fingerprints"].values()))
    check("inputs: the fixture's bundle files are (correctly) not the real ones", r.returncode == 1
          and not any(v["match"] for v in part["bundle_sha256"].values()))

    r = run("sets")
    check("sets: exit 0", r.returncode == 0, r.stdout[-800:] + r.stderr[-800:])
    s = json.loads((out / "parts/sets.json").read_text())
    for fn, (tag, kind, par, idx) in fsets.items():
        S, Tt = (top_s, top_t) if par == "ft_topk" else (rnd_s, rnd_t)
        rows, ho, shared = brute([S[i] for i in idx], [Tt[i] for i in idx], hs, ht)
        m = s["feasibility_sets"][fn]
        check(f"sets: {fn} set-side counts", m["set_rows_touching_heldout"] == rows, f"{m['set_rows_touching_heldout']} vs {rows}")
        check(f"sets: {fn} heldout-side counts", {k: m["heldout_rows_in_set"][k] for k in ("exact", "norm")} == ho,
              f"{m['heldout_rows_in_set']} vs {ho}")
        check(f"sets: {fn} agrees with the report field", m["agrees_with_report"] and m["shared_lines_as_feasibility_report"] == shared)
    check("sets: the repo's SHA256SUMS of the feasibility run matches", s["repo_feasibility_SHA256SUMS"]["match"],
          s["repo_feasibility_SHA256SUMS"])
    check("sets: the failed option is listed", list(s["failed_options_no_file"]) == ["ft_un_a_narrow__od6-remove__seed42"])
    check("sets: planted overlaps are found (not all zero)",
          s["feasibility_summary"]["all_zero"] == 2, s["feasibility_summary"])
    check("sets: R0 identical to its twin, A_narrow not",
          s["supplementary_sets"]["vocab/R0.npy"]["identical_to"] and s["supplementary_sets"]["vocab/A_narrow_42_58.npy"]["identical_to"] is None)
    for n, (p, rg) in six.items():
        S, Tt = (top_s, top_t) if p == "t" else (rnd_s, rnd_t)
        rows, ho, _ = brute([S[i] for i in rg], [Tt[i] for i in rg], hs, ht)
        m = s["supplementary_sets"][f"phase0_e05/{n}"]
        check(f"sets: E0.5 file {n}", m["set_rows_touching_heldout"] == rows
              and {k: m["heldout_rows_in_set"][k] for k in ("exact", "norm")} == ho, f"{m} vs {rows} {ho}")
    rows, ho, _ = brute(top_s, top_t, hs, ht)
    m = s["supplementary_sets"]["ft_topk (all rows)"]
    check("sets: whole ft_topk", m["set_rows_touching_heldout"] == rows, f"{m['set_rows_touching_heldout']} vs {rows}")

    r = run("pool", "--workers", "3")
    check("pool: exit 0", r.returncode == 0, r.stdout[-800:] + r.stderr[-800:])
    q = json.loads((out / "parts/pool.json").read_text())
    ps = [ln.split("\t") for ln in pool]
    src = [p[1] for p in ps]
    tgt = [p[2] for p in ps]
    ep = [(x, y) in list(zip(hs, ht)) for x, y in zip(src, tgt)]
    ns = [nrm(x) in [nrm(h) for h in hs] for x in src]
    nt = [nrm(y) in [nrm(h) for h in ht] for y in tgt]
    want = {"exact_pair": sum(ep), "extra_by_source": sum(n_ and not e for n_, e in zip(ns, ep)),
            "extra_by_target": sum(t_ and not n_ and not e for t_, n_, e in zip(nt, ns, ep))}
    want["rows_reserved_total"] = sum(want.values())
    check("pool: E0.3's reservation categories", q["e03_reservation_categories"] == want, f"{q['e03_reservation_categories']} vs {want}")
    check("pool: planted figures (6 exact-pair rows, 4 distinct heldout pairs, 5/5 heldout rows found)",
          want["exact_pair"] == 6 and q["heldout_rows_in_pool"]["exact"] == {"source": 5, "target": 5, "pair": 5}
          and q["heldout_rows_in_pool"]["distinct_exact"]["pair"] == 4, q["heldout_rows_in_pool"])
    rows, ho, _ = brute(src, tgt, hs, ht)
    pr = q["pool_rows_touching_heldout"]
    check("pool: exact and norm row counts", (pr["exact_pair"], pr["exact_source"], pr["exact_target"], pr["norm_source"], pr["norm_target"])
          == (rows["pair"], rows["source"], rows["target"], rows["norm_source"], rows["norm_target"]), f"{pr} vs {rows}")
    check("pool: rows and the 4-field row", q["pool_rows"] == 120 and q["rows_not_3_fields"] == 1)
    check("pool: copies per heldout pair", q["heldout_rows_by_pool_copies_of_their_pair"] == {"1": 3, "3": 2},
          q["heldout_rows_by_pool_copies_of_their_pair"])
    b = q["by_source_and_mask"]
    ec = q["evaluated_copy"]
    check("pool: the evaluated copy's dropped row is found, with its pool rows",
          ec["dropped_rows_0based"] == [2] and ec["dropped_rows_pool_rows"] == [{"pair": 1, "source": 1, "target": 2}]
          and ec["kept_rows_with_exact_pair_in_pool"] == 4, ec)
    check("pool: every heldout row has a pool row", q["heldout_rows_with_no_pool_row"] == {"pair": 0, "source": 0, "target": 0})
    check("pool: labels and mask of the exact-pair rows", b["exact_pair"]["inside_reused_mask"] == 5
          and b["exact_pair"]["sources"] == {"un": 5, "europarl": 1}, b["exact_pair"])

    r = run("pretrain", "--workers", "4")
    check("pretrain: exit 0", r.returncode == 0, r.stdout[-800:] + r.stderr[-800:])
    t = json.loads((out / "parts/pretrain.json").read_text())
    rows, ho, _ = brute(pre_s, pre_t, hs, ht)
    pt = t["pretrain_rows_touching_heldout"]
    check("pretrain: row counts", pt == {"exact_pair": rows["pair"], "exact_source": rows["source"], "exact_target": rows["target"],
                                         "norm_source": rows["norm_source"], "norm_target": rows["norm_target"]}, f"{pt} vs {rows}")
    check("pretrain: heldout-side counts", {k: t["heldout_rows_in_pretrain"][k] for k in ("exact", "norm")} == ho,
          f"{t['heldout_rows_in_pretrain']} vs {ho}")
    check("pretrain: planted figures", pt == {"exact_pair": 1, "exact_source": 1, "exact_target": 2, "norm_source": 2, "norm_target": 2}, pt)

    r = subprocess.run(["bash", str(HERE / "check_awk.sh")], env={**os.environ, "HOME_DIR": str(home), "OUT": str(out)},
                       capture_output=True, text=True)
    check("awk: exit 0", r.returncode == 0, r.stderr[-500:])
    aw = {p.name: p.read_text() for p in (out / "check").glob("*_awk.txt")}
    pr = q["pool_rows_touching_heldout"]
    hp = q["heldout_rows_in_pool"]["exact"]
    check("awk: pool line equals the script's exact counts",
          f"rows 120 | rows with a heldout source {pr['exact_source']}, target {pr['exact_target']}, pair {pr['exact_pair']} | "
          f"heldout rows (of 5) whose source {hp['source']}, target {hp['target']}, pair {hp['pair']} occurs" in aw["pool_awk.txt"]
          and "other than 3 fields 1" in aw["pool_awk.txt"], aw.get("pool_awk.txt"))
    he = t["heldout_rows_in_pretrain"]["exact"]
    check("awk: pretrain line equals the script's exact counts",
          f"rows 100 | rows with a heldout source {pt['exact_source']}, target {pt['exact_target']}, pair {pt['exact_pair']} | "
          f"heldout rows (of 5) whose source {he['source']}, target {he['target']}, pair {he['pair']} occurs" in aw["pretrain_awk.txt"],
          aw.get("pretrain_awk.txt"))
    m = s["supplementary_sets"]["ft_topk (all rows)"]
    check("awk: ft_topk line equals the script's exact counts",
          f"rows 40 | rows with a heldout source {m['set_rows_touching_heldout']['source']}, target {m['set_rows_touching_heldout']['target']}, "
          f"pair {m['set_rows_touching_heldout']['pair']}" in aw["ft_topk_awk.txt"], aw.get("ft_topk_awk.txt"))
    r = run("reduce")
    check("reduce: results.json written with the awk lines", r.returncode == 0
          and "independent_check_awk" in json.loads((out / "results.json").read_text()))
    print(f"\n{len(FAIL)} FAIL" if FAIL else "\nall checks pass")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
