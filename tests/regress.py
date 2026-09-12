#!/usr/bin/env python3
"""Self-contained regression for phase0/ and phase1/.

Every fixture is generated from scratch in a fresh temp directory, every script is
driven through its real CLI, and wherever a script could be wrong about ITSELF the
result is checked by independent recomputation, not by reading the script's own
"OK" message.

Why this file exists: the previous regression was ad-hoc shell reading fixtures
from a session temp directory. When that directory was cleaned, 13 of 17 checks
failed with "missing input" -- indistinguishable, on the pass/fail line, from real
breakage. A test that depends on state it does not create cannot tell you anything.

    python3 tests/regress.py            # exit 0 iff nothing FAILED
    python3 tests/regress.py --keep     # keep the fixture dir for inspection
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MT_REPO = Path("/Users/euswbnic/Machine_translation")
RESULTS: list[tuple[str, str, str]] = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))


def skip(name, why):
    RESULTS.append((name, "SKIP", why))
    print(f"  [SKIP] {name}  — {why}")


def run(args, env=None):
    r = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, **(env or {})})
    return r.returncode, r.stdout + r.stderr


# ---------------------------------------------------------------- fixtures
def make_cache(d: Path, n=200_000):
    rng = np.random.RandomState(0)
    src = np.clip(rng.lognormal(3.25, 0.62, n).astype(np.int64), 2, 256)
    tgt = np.clip((src * rng.normal(1.10, 0.13, n)).astype(np.int64), 2, 256)
    p = d / "cache.npz"
    np.savez(p, src_offsets=np.concatenate([[0], np.cumsum(src)]),
             tgt_offsets=np.concatenate([[0], np.cumsum(tgt)]))
    return p


def make_corpus(d: Path):
    """Constituent corpora + a cleaned corpus drawn from them (with duplicates)."""
    rng = random.Random(1)
    sizes = {"europarl": 3000, "commoncrawl": 4000, "un": 8000, "giga": 7000}
    qe_mean = {"europarl": 0.80, "commoncrawl": 0.62, "un": 0.86, "giga": 0.72}
    text2src, rows, specs = {}, [], []
    for lab, n in sizes.items():
        s_path, t_path = d / f"{lab}.en", d / f"{lab}.fr"
        with open(s_path, "w") as fs, open(t_path, "w") as ft:
            for i in range(n):
                s = f"{lab} source sentence {i} w{rng.randint(0, 10**9)}"
                t = f"{lab} phrase cible {i} m{rng.randint(0, 10**9)}"
                fs.write(s + "\n"); ft.write(t + "\n")
                text2src[(s, t)] = lab
                if rng.random() < 0.9:
                    rows.append((round(min(1.05, max(0.5, rng.gauss(qe_mean[lab], 0.06))), 4), s, t))
        specs.append(f"{lab}:{s_path}:{t_path}")
    rows += rng.sample(rows, int(len(rows) * 0.03))        # clean_data does not dedup
    rng.shuffle(rows)
    with open(d / "clean.en", "w") as fs, open(d / "clean.fr", "w") as ft, \
         open(d / "scores.tsv", "w") as fq:
        for sc, s, t in rows:
            fs.write(s + "\n"); ft.write(t + "\n"); fq.write(f"{sc}\t{s}\t{t}\n")
    return specs, text2src, len(rows)


def read_pairs(stem: Path):
    with open(f"{stem}.en") as a, open(f"{stem}.fr") as b:
        return [(x.rstrip("\n"), y.rstrip("\n")) for x, y in zip(a, b)]


DECIDE = {
    "go": ("""condition seed testset bleu
baseline - newstest2014 38.21
baseline - heldout_un 35.00
baseline - heldout_europarl 33.00
ft_topk 42 newstest2014 36.60
ft_topk 1 newstest2014 36.70
ft_topk 2 newstest2014 36.50
ft_topk 42 heldout_un 36.50
ft_topk 1 heldout_un 36.60
ft_topk 2 heldout_un 36.40
ft_topk 42 heldout_europarl 33.80
ft_topk 1 heldout_europarl 33.90
ft_topk 2 heldout_europarl 33.70
ft_random 42 newstest2014 37.90
ft_random 1 newstest2014 38.00
ft_random 2 newstest2014 37.80
""", 0, "VERDICT: GO"),
    "nogo": ("""condition seed testset bleu
baseline - newstest2014 38.21
baseline - heldout_un 35.00
baseline - heldout_europarl 33.00
ft_topk 42 newstest2014 36.60
ft_topk 1 newstest2014 36.70
ft_topk 2 newstest2014 36.50
ft_topk 42 heldout_un 34.50
ft_topk 1 heldout_un 34.60
ft_topk 2 heldout_un 34.40
ft_random 42 newstest2014 36.50
ft_random 1 newstest2014 36.70
ft_random 2 newstest2014 36.60
""", 1, "criterion 1: FAIL"),
    # fine-tuning IMPROVES news BLEU; top-k just improves less. Must NOT be GO.
    "falsego": ("""condition seed testset bleu
baseline - newstest2014 38.21
baseline - heldout_un 35.00
ft_topk 42 newstest2014 39.50
ft_topk 1 newstest2014 39.60
ft_topk 2 newstest2014 39.40
ft_topk 42 heldout_un 36.50
ft_topk 1 heldout_un 36.60
ft_topk 2 heldout_un 36.40
ft_random 42 newstest2014 40.80
ft_random 1 newstest2014 40.90
ft_random 2 newstest2014 40.70
""", 1, "does NOT degrade"),
    "noheld": ("""condition seed testset bleu
baseline - newstest2014 38.21
ft_topk 42 newstest2014 36.60
ft_topk 1 newstest2014 36.70
ft_topk 2 newstest2014 36.50
ft_random 42 newstest2014 37.90
ft_random 1 newstest2014 38.00
ft_random 2 newstest2014 37.80
""", 1, "NO DATA"),
    "junk": ("""condition seed testset bleu
baseline - newstest2014 38.21
ft_topk 42 newstest2014 bad
ft_random 42 newstest2014 37.9
""", 2, "NO DECISION"),
}


def trace_rows(cell, seeds, floor, amp, tau, up_t, up_k, tmax, noise, grid=50_000_000):
    out = []
    for seed in seeds:
        rng = np.random.RandomState(seed * 131 + sum(map(ord, cell)))
        t = grid
        while t <= tmax:
            ce = floor + amp * np.exp(-t / tau) + (up_k * (t - up_t) / 1e9 if t > up_t else 0)
            out.append(f"{cell} {seed} {t} {ce + rng.normal(0, noise):.5f}")
            t += grid
    return out


# ---------------------------------------------------------------- suites
def suite_counter():
    print("\n== token counter state machine ==")
    rc, out = run([ROOT / "phase0/test_token_counter.py"])
    check("test_token_counter invariants (incl. I6 max_steps regression)",
          rc == 0 and "ALL INVARIANTS HOLD" in out, out[-300:])


def suite_e02(d: Path, cache: Path):
    print("\n== e02 token accounting ==")
    sys.path.insert(0, str(ROOT / "phase0"))
    import e02_token_accounting as e02
    r = e02.measure(np.array([10, 10, 10]), np.array([7, 9, 11]), 10**6, 3)
    check("label count: cached lengths 7/9/11 -> 24 label tokens (not 27)",
          r["nonpad_tgt"] == 24, f"got {r['nonpad_tgt']}")
    js = d / "e02.json"
    rc, out = run([ROOT / "phase0/e02_token_accounting.py", "--cache", cache, "--seeds", "1",
                   "--json-out", js])
    check("e02 runs", rc == 0, out[-300:])
    if rc != 0:
        return
    j = json.load(open(js))
    base = j["configs"]["Base (24576/192)"]
    check("sentence cap binds on WMT-like lengths (>80% of Base batches)",
          base["sentence_capped_frac"] > 0.8, f"{base['sentence_capped_frac']:.3f}")
    # independent recomputation of the fatal e02:179 fix
    bad = []
    for (name, N, steps, mb, accum, hours), row in zip(e02.CELLS, j["cells"]):
        key = "Base (24576/192)" if mb == 24_576 else "Big (8192/64)"
        want = steps * j["configs"][key]["mean_nonpad_tgt_per_microbatch"]
        if abs(row["true_nonpad_tgt_tokens"] - want) > 1e-6 * want:
            bad.append(name)
    check("true tokens == steps x measured mean per micro-batch, every cell", not bad, str(bad))


def suite_e01_e03(d: Path):
    print("\n== e01 provenance -> e03 build_controls (real data path) ==")
    specs, text2src, n = make_corpus(d)
    args = [ROOT / "phase0/e01_provenance.py", "--clean-src", d / "clean.en",
            "--clean-tgt", d / "clean.fr", "--qe-scores", d / "scores.tsv", "--top-k", "2000",
            "--out", d / "prov"]
    for s in specs:
        args += ["--corpus", s]
    rc, out = run(args)
    check("e01 runs", rc == 0, out[-400:])
    if rc != 0:
        return
    rep = json.load(open(d / "prov_report.json"))
    labels = np.load(d / "prov_labels.npy")
    check("e01 match rate 100% on a pure-subset corpus", rep["match_rate"] == 1.0, str(rep["match_rate"]))
    check("e01 label rows == cleaned corpus rows", len(labels) == n, f"{len(labels)} vs {n}")
    pairs = read_pairs(d / "clean")
    wrong = sum(1 for i in range(0, n, 7) if rep["sources"][labels[i]] != text2src[pairs[i]])
    check("e01 labels agree with independent text->source lookup", wrong == 0, f"{wrong} mislabelled")

    ctrl = d / "ctrl"
    rc, out = run([ROOT / "phase0/e03_build_controls.py", "--qe-scores", d / "scores.tsv",
                   "--provenance", d / "prov_labels.npy", "--out-dir", ctrl,
                   "--n-ft", "4000", "--n-heldout", "300"])
    check("e03_build_controls runs (report order auto-read from e01)", rc == 0, out[-400:])
    if rc != 0:
        return
    held = {k: set(read_pairs(ctrl / f"heldout_{k}")) for k in ("un", "europarl")}
    ft = {k: set(read_pairs(ctrl / k)) for k in ("ft_topk", "ft_random", "ft_bottom")}
    leak = {k: len((held["un"] | held["europarl"]) & v) for k, v in ft.items()}
    check("NO held-out pair in any FT set — recomputed by TEXT, not trusted from the script",
          not any(leak.values()), str(leak))
    check("heldout_un really comes from UN, heldout_europarl from Europarl",
          all(text2src[p] == "un" for p in held["un"])
          and all(text2src[p] == "europarl" for p in held["europarl"]))
    check("ft_topk and ft_bottom are disjoint by text", not (ft["ft_topk"] & ft["ft_bottom"]))

    rc, _ = run([ROOT / "phase0/e03_build_controls.py", "--qe-scores", d / "scores.tsv",
                 "--provenance", d / "prov_labels.npy", "--out-dir", d / "c_small",
                 "--n-ft", "12000", "--n-heldout", "300"])
    check("hard-fails when pool < 2 x n_ft", rc == 1, f"rc={rc}")
    rc, _ = run([ROOT / "phase0/e03_build_controls.py", "--qe-scores", d / "scores.tsv",
                 "--provenance", d / "prov_labels.npy", "--sources", "europarl,un,commoncrawl,giga",
                 "--out-dir", d / "c_order", "--n-ft", "4000", "--n-heldout", "300"])
    check("hard-fails when --sources disagrees with e01's order", rc == 1, f"rc={rc}")
    low = dict(rep, match_rate=0.883)
    json.dump(low, open(d / "prov_low.json", "w"))
    rc, _ = run([ROOT / "phase0/e03_build_controls.py", "--qe-scores", d / "scores.tsv",
                 "--provenance", d / "prov_labels.npy", "--provenance-report", d / "prov_low.json",
                 "--out-dir", d / "c_low", "--n-ft", "4000", "--n-heldout", "300"])
    check("hard-fails when e01 match rate < 95%", rc == 1, f"rc={rc}")


def suite_run_matrix(d: Path):
    print("\n== e03_run_matrix ==")
    shim = d / "shim"; shim.mkdir()
    (shim / "yaml.py").write_text("import json\n"
                                  "def safe_load(f): return json.load(f)\n"
                                  "def safe_dump(o, f, **k): json.dump(o, f, indent=1)\n")
    base = {"model": {"d_model": 512},
            "training": {"batch_size": 24576, "accumulate_steps": 4, "max_steps": 115000,
                         "lr_scale": 1.0, "min_lr": 1e-5, "seed": 42, "early_stopping": True,
                         "patience": 5, "eval_interval": 2000, "eval_interval_min": 1000},
            "data": {"train_src": "x.en", "train_tgt": "x.fr", "valid_src": "v.en"},
            "checkpoint": {"dir": "ck", "keep_last": 5},
            "logging": {"swanlab": {"experiment": "sft"}}}
    json.dump(base, open(d / "base.yaml", "w"))
    out_dir = d / "matrix"
    rc, out = run([ROOT / "phase0/e03_run_matrix.py", "--base-config", d / "base.yaml",
                   "--out-dir", out_dir], env={"PYTHONPATH": str(shim)})
    check("e03_run_matrix runs", rc == 0, out[-300:])
    if rc != 0:
        return
    cfgs = sorted(out_dir.glob("ft_*.yaml"))
    check("12 configs generated", len(cfgs) == 12, str(len(cfgs)))
    tr = json.load(open(out_dir / "ft_topk_lr0.15.yaml"))["training"]
    check("early stopping disabled and eval grid pinned",
          tr["early_stopping"] is False and tr["eval_interval"] == tr["eval_interval_min"], str(tr))
    import re
    runs = []
    for sh in ("run_stage1.sh", "run_stage2.sh"):
        txt = (out_dir / sh).read_text()
        runs += [(m.group(1), m.group(2)) for m in
                 re.finditer(r"--config (\S+) .*?--suffix (\S+)", txt)]
        r = subprocess.run(["bash", "-n", str(out_dir / sh)], capture_output=True)
        check(f"{sh} is valid shell", r.returncode == 0)
    stage2_lr = [(c.replace("${LR}", "0.15"), s) for c, s in runs]
    check("no two runs share (config, suffix) -> no checkpoint overwrite",
          len(set(stage2_lr)) == len(stage2_lr), f"{len(stage2_lr)} runs")


def suite_decide(d: Path):
    print("\n== e03_decide (pre-registered gate) ==")
    for name, (text, want_rc, want_str) in DECIDE.items():
        p = d / f"{name}.tsv"; p.write_text(text)
        rc, out = run([ROOT / "phase0/e03_decide.py", "--results", p])
        check(f"decide/{name}: exit {want_rc} and says '{want_str}'",
              rc == want_rc and want_str in out, f"rc={rc}")
    rc, out = run([ROOT / "phase0/e03_decide.py", "--results", d / "nope.tsv"])
    check("decide/missing-file exits 2 (never read as NO-GO)", rc == 2, f"rc={rc}")


def suite_calibrate(d: Path, cache: Path):
    print("\n== phase1 calibrate_batch ==")
    js = d / "cal.json"
    rc, out = run([ROOT / "phase1/calibrate_batch.py", "--cache", cache, "--limit", "50000",
                   "--seeds", "1", "--json-out", js])
    check("calibrate_batch runs and finds a feasible pair", rc == 0, out[-300:])
    if rc != 0:
        return
    j = json.load(open(js))
    tol = j["tolerance"]
    check("both arms within tolerance and arm spread within tolerance",
          all(abs(a["err"]) <= tol for a in j["arms"].values()) and j["arm_spread"] <= tol)
    # independent recomputation of the reported tokens/micro-batch, in LABELS
    sys.path.insert(0, str(ROOT / "phase0"))
    from e02_token_accounting import build_batches
    z = np.load(cache)
    s_l = np.diff(z["src_offsets"]).astype(np.int64); t_l = np.diff(z["tgt_offsets"]).astype(np.int64)
    idx = np.random.RandomState(0).choice(len(s_l), 50000, replace=False)
    s_l, t_l = s_l[idx], t_l[idx]
    arms = {"Base": 24_576, "Big": 8_192}
    for arm, a in j["arms"].items():
        b = build_batches(s_l, t_l, arms[arm], a["max_sentences"], seed=0)
        labels = sum(int(t_l[np.asarray(x)].sum()) - len(x) for x in b) / len(b)
        check(f"{arm}: reported tgt/micro equals independent LABEL count",
              abs(labels - a["mean_tgt_per_micro"]) < 1e-6 * labels,
              f"reported {a['mean_tgt_per_micro']:.2f} vs recomputed {labels:.2f}")


def suite_converge(d: Path):
    print("\n== phase1 converge (decision script) ==")
    P = "--params", "enfr_base_cap=60000000,enfr_big_cap=209000000"
    pre = "--base-prefix", "enfr_base", "--big-prefix", "enfr_big"
    scen = {
        "invariant": ([*trace_rows("enfr_base_cap", (1, 2, 3), 3.20, 1.4, 1.5e9, 5e9, .03, 12e9, .004),
                       *trace_rows("enfr_big_cap", (1, 2, 3), 3.36, 1.6, 2.2e9, 7e9, .028, 12e9, .004)],
                      "INVARIANT across all 3"),
        "invert": ([*trace_rows("enfr_base_cap", (1, 2, 3), 3.30, 1.4, 1.2e9, 4e9, .035, 12e9, .004),
                    *trace_rows("enfr_big_cap", (1, 2, 3), 3.12, 2.2, 4.5e9, 9.5e9, .02, 12e9, .004)],
                   "INVERTS"),
        "truncated": ([*trace_rows("enfr_base_cap", (1, 2, 3), 3.20, 1.4, 1.5e9, 9e9, .03, 3e9, .004),
                       *trace_rows("enfr_big_cap", (1, 2, 3), 3.36, 1.6, 2.2e9, 9e9, .028, 3e9, .004)],
                      "NOT CONVERGED"),
        "null": ([*trace_rows("enfr_base_cap", (1, 2, 3), 3.200, 1.5, 2e9, 6e9, .03, 12e9, .012),
                  *trace_rows("enfr_big_cap", (1, 2, 3), 3.203, 1.5, 2e9, 6e9, .03, 12e9, .012)],
                 "NULL (PROTOCOL 6.3)"),
        "oneseed": ([*trace_rows("enfr_base_cap", (1,), 3.20, 1.5, 2e9, 6e9, .03, 12e9, .012),
                     *trace_rows("enfr_big_cap", (1,), 3.30, 1.5, 2e9, 6e9, .03, 12e9, .012)],
                    "NO VARIANCE ESTIMATE"),
    }
    for name, (rows, want) in scen.items():
        p = d / f"conv_{name}.tsv"
        p.write_text("cell seed tokens dev_ce\n" + "\n".join(rows) + "\n")
        rc, out = run([ROOT / "phase1/converge.py", "--traces", p, *P, *pre])
        check(f"converge/{name}: says '{want}'", rc == 0 and want in out, f"rc={rc}")

    # REGRESSION for the audit fatal: noise floor must be WITHIN-cell. Two base cells
    # far apart (between-cell spread 0.6), tiny seed noise, Big worse by 0.1 in both
    # regimes. Concatenated sd (~0.3) would call this NULL; within-cell sd (~0.001) must not.
    rows = []
    for cell, level in (("b_cap", 3.0), ("b_full", 3.6), ("g_cap", 3.1), ("g_full", 3.7)):
        for seed in (1, 2, 3):
            rng = np.random.RandomState(seed * 7 + len(cell))
            for i in range(10):
                ce = level + 0.02 * (i - 5) ** 2 + rng.normal(0, 0.001)
                rows.append(f"{cell} {seed} {(i + 1) * 50_000_000} {ce:.5f}")
    p = d / "conv_within.tsv"
    p.write_text("cell seed tokens dev_ce\n" + "\n".join(rows) + "\n")
    rc, out = run([ROOT / "phase1/converge.py", "--traces", p, "--base-prefix", "b_",
                   "--big-prefix", "g_"])
    line = next((l for l in out.splitlines() if l.startswith("(a) convergence")), "")
    check("noise floor is WITHIN-cell: a 0.1 effect under 0.6 between-cell spread is NOT null",
          "Big WORSE" in line, line.strip())
    rc, _ = run([ROOT / "phase1/converge.py", "--traces", d / "nope.tsv"])
    check("converge/missing-file exits 2", rc == 2, f"rc={rc}")


def suite_inventory_fetch(d: Path):
    print("\n== inventory + fetch ==")
    js = d / "inv.json"
    rc, out = run([ROOT / "phase0/inventory.py", ROOT, "--max-depth", "2", "--json-out", js])
    check("inventory runs", rc == 0, out[-300:])
    if rc == 0:
        repos = json.load(open(js))["repos"]
        lying = [p for p, s in repos.items() if s.get("errors") and s.get("dirty_files") == 0]
        check("a repo git cannot read is never reported as clean (dirty_files 0)", not lying, str(lying))

    box = d / "box"
    for rel, data in (("ckpt/averaged.pt", b"w"), ("ckpt/step_104000.pt", b"r"),
                      ("logs/sft.log", b"LR = 2.73e-04\n"), ("data/v2_scored.tsv", os.urandom(3_000_000)),
                      ("data/spm.model", b"s")):
        f = box / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(data)
    def e(prio, kind, rel):
        return {"prio": prio, "kind": kind, "path": str(box / rel),
                "size": (box / rel).stat().st_size, "mtime": "x"}
    inv = {"files": [e("P0", "checkpoint", "ckpt/averaged.pt"), e("P0", "checkpoint", "ckpt/step_104000.pt"),
                     e("P0", "log", "logs/sft.log"), e("P1", "qe", "data/v2_scored.tsv"),
                     e("P1", "spm", "data/spm.model")]}
    json.dump(inv, open(d / "fetch_inv.json", "w"))
    dest = d / "pulled"
    rc, out = run([ROOT / "phase0/fetch.py", "pull", "--host", "local", "--inventory", d / "fetch_inv.json",
                   "--prio", "P0,P1", "--max-file", "2M", "--dest", dest, "--go"])
    check("fetch pull (local) runs", rc == 0, out[-300:])
    if rc == 0:
        got = {p.name for p in dest.rglob("*") if p.is_file() and p.name != "FETCH_MANIFEST.json"}
        check("pulled exactly averaged.pt, sft.log, spm.model", got == {"averaged.pt", "sft.log", "spm.model"}, str(got))
        m = json.load(open(dest / "FETCH_MANIFEST.json"))
        check("skipped files are listed, not silently dropped",
              {Path(f["path"]).name for f in m["not_pulled"]} == {"step_104000.pt", "v2_scored.tsv"})


def suite_patch():
    print("\n== trainer patch ==")
    patch = ROOT / "phase0/trainer_token_accounting.patch"
    if not (MT_REPO / "src/training/trainer.py").exists():
        skip("patch applies", f"{MT_REPO} not present")
        return
    r = subprocess.run(["git", "apply", "--check", "-p1", str(patch)], cwd=MT_REPO,
                       capture_output=True, text=True)
    if r.returncode != 0 and "xcrun" in r.stderr:
        r = subprocess.run(["patch", "--dry-run", "-p1", "-i", str(patch)], cwd=MT_REPO,
                           capture_output=True, text=True)
        check("patch applies (patch --dry-run; git unusable in this env)", r.returncode == 0, r.stderr[-200:])
    else:
        check("patch applies (git apply --check)", r.returncode == 0, r.stderr[-200:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    d = Path(tempfile.mkdtemp(prefix="mt_regress_"))
    print(f"fixtures: {d}")
    try:
        cache = make_cache(d)
        suite_counter()
        suite_e02(d, cache)
        suite_e01_e03(d)
        suite_run_matrix(d)
        suite_decide(d)
        suite_calibrate(d, cache)
        suite_converge(d)
        suite_inventory_fetch(d)
        suite_patch()
    finally:
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)
    n = {s: sum(1 for _, st, _ in RESULTS if st == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\nPASS={n['PASS']} FAIL={n['FAIL']} SKIP={n['SKIP']}")
    for name, st, why in RESULTS:
        if st != "PASS":
            print(f"  {st}: {name}  {why}")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
