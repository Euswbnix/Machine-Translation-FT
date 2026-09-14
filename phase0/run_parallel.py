#!/usr/bin/env python3
"""Run the train.py lines of a generated E0.3 runner in parallel, one GPU per run.

run_stage1.sh / run_stage2.sh from e03_run_matrix.py execute their runs one after
another; on a rented multi-GPU box that leaves every GPU but one idle for the
whole experiment. This executes the SAME lines -- parsed from the runner, not
retyped, so it cannot drift from what the matrix defines -- with
CUDA_VISIBLE_DEVICES leased from a pool.

Resumable: a run whose <checkpoint.dir><suffix>/final.pt already exists is
skipped, so after a crash or a lost box you rerun the same command. A run counts
as done only if train.py exits 0 AND final.pt exists (an interrupted trainer
saves interrupted_step_*.pt and would otherwise look successful).

Stale-run guard: when the runner's matrix.json carries a controls_sha (the content
fingerprint of the control sets it was generated from, phase0/controls_fingerprint.py),
every launched run gets <ckdir>/controls_sha, and an existing final.pt is accepted only if
that marker equals matrix.json's value. Otherwise the command exits 2 listing the stale
runs -- a rebuilt control set must never be "resumed" onto models trained on the old one.
Before anything launches (also with --dry-run), the control-set directory itself must still
fingerprint to matrix.json's controls_sha (controls_fingerprint.check_controls), else exit 2.

Run the cache warm-up (phase0/warm_cache.py) first: parallel runs of the same condition
race on the trainer's non-atomic tokenisation cache.

    python phase0/run_parallel.py --runner configs/phase0/run_stage2.sh --lr-scale 0.15 \
        --gpus 4 --cwd ~/mt/Machine_translation --logdir logs/phase0/stage2
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controls_fingerprint import check_controls  # noqa: E402

LINE_RE = re.compile(r"^\s*python\s+train\.py\s")
MARKER = "controls_sha"


def load_cfg(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        return json.load(open(path))       # JSON is a YAML subset; used by the offline tests


def opt(tokens, name):
    return tokens[tokens.index(name) + 1] if name in tokens else None


def parse_runs(runner: Path, lr_scale, cwd: Path, python: str) -> list[dict]:
    """Runs defined by a generated runner: [{name, cmd, cfg_path, cfg, ckdir, final}]."""
    text = runner.read_text()
    if "${LR}" in text:
        if lr_scale is None:
            sys.exit(f"{runner.name} uses ${{LR}}; pass --lr-scale")
        m = re.search(r'VALID="([^"]*)"', text)
        if m and lr_scale not in m.group(1).split():
            sys.exit(f"lr_scale {lr_scale!r} is not in the runner's VALID list ({m.group(1)}); "
                     "config filenames use %g formatting, e.g. 0.15 not 0.150")
    runs = []
    for line in text.splitlines():
        if not LINE_RE.match(line):
            continue
        if lr_scale is not None:
            line = line.replace("${LR}", lr_scale)
        toks = shlex.split(line)
        toks[0] = python
        cfg_path = Path(opt(toks, "--config"))
        cfg_path = cfg_path if cfg_path.is_absolute() else cwd / cfg_path
        if not cfg_path.exists():
            sys.exit(f"config {cfg_path} referenced by the runner does not exist")
        suffix = opt(toks, "--suffix") or ""
        cfg = load_cfg(cfg_path)
        ckdir = Path(cfg["checkpoint"]["dir"] + suffix)
        ckdir = ckdir if ckdir.is_absolute() else cwd / ckdir
        runs.append({"name": cfg_path.stem + suffix, "cmd": toks, "cfg_path": cfg_path, "cfg": cfg,
                     "ckdir": ckdir, "final": ckdir / "final.pt"})
    if not runs:
        sys.exit(f"no train.py lines found in {runner}")
    return runs


def expected_controls_sha(runner: Path):
    mp = runner.parent / "matrix.json"
    if not mp.is_file():
        return None
    return json.load(open(mp, encoding="utf-8")).get("controls_sha")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--lr-scale", default=None, help="value substituted for ${LR} (stage 2)")
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--cwd", required=True, help="Machine_translation root (where train.py lives)")
    ap.add_argument("--logdir", default="logs/phase0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runner = Path(a.runner).expanduser()
    cwd = Path(a.cwd).expanduser().resolve()
    runs = parse_runs(runner, a.lr_scale, cwd, a.python)

    bad = check_controls(runner, runs, cwd)
    if bad:
        print(f"STALE CONTROLS — {bad}\nNothing launched.")
        return 2
    want = expected_controls_sha(runner)
    stale = []
    if want:
        for r in runs:
            if r["final"].exists():
                mk = r["ckdir"] / MARKER
                have = mk.read_text().strip() if mk.is_file() else None
                if have != want:
                    stale.append(f"{r['name']}: final.pt marker {have!r} != controls {want[:12]}...")
    else:
        print("  (runner has no matrix.json controls_sha: stale-run guard NOT active)")
    if stale:
        print("STALE RUNS — final.pt exists but was trained on different control sets:\n  "
              + "\n  ".join(stale)
              + "\nRegenerate the matrix (rental_setup.sh controls) or remove those checkpoint dirs. "
                "Nothing launched.")
        return 2

    todo = [r for r in runs if not r["final"].exists()]
    print(f"{len(runs)} runs in {runner.name}; {len(runs) - len(todo)} already have final.pt; "
          f"running {len(todo)} on {a.gpus} GPU(s)")
    if a.dry_run:
        for r in todo:
            print("  " + " ".join(shlex.quote(t) for t in r["cmd"]))
        return 0

    logdir = Path(a.logdir).expanduser()
    logdir = logdir if logdir.is_absolute() else cwd / logdir
    logdir.mkdir(parents=True, exist_ok=True)
    gpus: "queue.Queue[int]" = queue.Queue()
    for g in range(a.gpus):
        gpus.put(g)
    results, lock = {}, threading.Lock()

    def worker(r):
        g = gpus.get()
        try:
            if want:
                r["ckdir"].mkdir(parents=True, exist_ok=True)
                (r["ckdir"] / MARKER).write_text(want + "\n")
            log = logdir / f"{r['name']}.log"
            t0 = time.time()
            with open(log, "w") as fh:
                p = subprocess.run(r["cmd"], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
            ok = p.returncode == 0 and r["final"].exists()
            with lock:
                results[r["name"]] = ok
                print(f"  [{'done' if ok else 'FAILED'}] {r['name']}  gpu{g}  rc={p.returncode}  "
                      f"{time.time() - t0:.0f}s  -> {log}", flush=True)
        finally:
            gpus.put(g)

    with ThreadPoolExecutor(max_workers=a.gpus) as ex:
        list(ex.map(worker, todo))
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} run(s) FAILED: {', '.join(failed)}\n"
              "Read their logs; rerunning this command retries only runs without final.pt.")
        return 1
    print("\nall runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
