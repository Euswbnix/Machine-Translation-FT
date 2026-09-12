#!/usr/bin/env python3
"""E0.3d — evaluate every E0.3 run and assemble the TSV that e03_decide.py reads.

The pre-registered gate needs BLEU on newstest2014 AND both held-out in-domain
sets, for every stage-2 run and for the pre-fine-tuning baseline. Doing 30
evaluations by hand is exactly where a row gets mislabelled or a crashed run is
silently left out, so this is scripted and refuses to hide gaps:

  * Which runs exist is parsed from the runner e03_run_matrix.py generated
    (run_stage2.sh): condition, seed and --suffix come from what was launched,
    not from a list retyped here.
  * A run with no final.pt, or whose final.pt stopped short of max_steps, is
    INCOMPLETE. Any incompleteness writes <out>.partial.tsv and exits 1; the
    canonical TSV that e03_decide reads is only written when nothing is missing.
  * Finished evaluations are cached by (checkpoint path, size, mtime, test set),
    so an interrupted collection resumes instead of re-decoding.

USAGE (on the GPU box)
    python phase0/e03_collect.py --mt-root ~/mt/Machine_translation \
        --runner ~/mt/Machine-Translation-SFT/configs/phase0/run_stage2.sh --lr-scale 0.15 \
        --baseline-ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
        --baseline-config ~/mt/Machine-Translation-SFT/configs/sft_base_enfr.yaml \
        --controls-dir ~/mt/Machine-Translation-SFT/data/phase0 \
        --out ~/mt/Machine-Translation-SFT/results/phase0_bleu.tsv --jobs 4
    python phase0/e03_decide.py --results ~/mt/Machine-Translation-SFT/results/phase0_bleu.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLEU_RE = re.compile(r"BLEU \([a-z]+, sacrebleu 13a\): ([0-9.]+)")
RUN_RE = re.compile(r"python\s+train\.py\s.*?--config\s+(\S+).*?--seed\s+(\d+).*?--suffix\s+(\S+)")


def load_cfg(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        return json.load(open(path))       # JSON is a YAML subset; used by the offline tests


def parse_runner(runner: Path, lr: str):
    runs = []
    for line in runner.read_text().splitlines():
        m = RUN_RE.search(line)
        if not m:
            continue
        cfg = m.group(1).replace("${LR}", lr)
        cond = re.match(r"(ft_[a-z]+)_lr", Path(cfg).name)
        if not cond:
            sys.exit(f"cannot derive a condition from config name {cfg!r}")
        runs.append({"condition": cond.group(1), "seed": int(m.group(2)),
                     "suffix": m.group(3), "config": Path(cfg)})
    if not runs:
        sys.exit(f"no train.py lines parsed from {runner}")
    return runs


try:
    import torch                                            # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def final_step(ckpt: Path):
    """global_step recorded in the checkpoint, or None if torch is unavailable."""
    if not HAVE_TORCH:
        return None
    return torch.load(ckpt, map_location="cpu", weights_only=False).get("global_step")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt-root", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--lr-scale", required=True, help="the value passed to run_stage2.sh, e.g. 0.15")
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--baseline-config", required=True)
    ap.add_argument("--controls-dir", required=True)
    ap.add_argument("--news-src", default="data_enfr_v1/test.en")
    ap.add_argument("--news-ref", default="data_enfr_v1/test.fr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel evals, one GPU each")
    ap.add_argument("--eval-script", default="scripts/eval_bleu.py",
                    help="relative to --mt-root; replaceable for offline tests")
    a = ap.parse_args()

    mt = Path(a.mt_root).expanduser().resolve()
    ctrl = Path(a.controls_dir).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    tests = {
        "newstest2014": (mt / a.news_src, mt / a.news_ref),
        "heldout_un": (ctrl / "heldout_un.en", ctrl / "heldout_un.fr"),
        "heldout_europarl": (ctrl / "heldout_europarl.en", ctrl / "heldout_europarl.fr"),
    }
    missing_sets = [f"{k}: {p}" for k, (s, r) in tests.items() for p in (s, r) if not p.exists()]
    if missing_sets:
        sys.exit("test sets missing — the gate cannot be evaluated:\n  " + "\n  ".join(missing_sets))

    runs = parse_runner(Path(a.runner).expanduser(), a.lr_scale)
    jobs, incomplete = [], []
    base_ckpt = (mt / a.baseline_ckpt) if not Path(a.baseline_ckpt).is_absolute() else Path(a.baseline_ckpt)
    if not base_ckpt.exists():
        sys.exit(f"baseline checkpoint {base_ckpt} does not exist — run rental_setup.sh accept first")
    for ts in tests:
        jobs.append(("baseline", "-", ts, base_ckpt, Path(a.baseline_config).expanduser()))
    for r in runs:
        cfg_path = r["config"] if r["config"].is_absolute() else (mt / r["config"])
        cfg = load_cfg(cfg_path)
        ckpt = mt / (cfg["checkpoint"]["dir"] + r["suffix"]) / "final.pt"
        if not ckpt.exists():
            incomplete.append(f"{r['condition']} seed {r['seed']}: no {ckpt}")
            continue
        step, want = final_step(ckpt), cfg["training"]["max_steps"]
        if step is not None and step < want:
            incomplete.append(f"{r['condition']} seed {r['seed']}: stopped at step {step} < {want}")
            continue
        for ts in tests:
            jobs.append((r["condition"], str(r["seed"]), ts, ckpt, cfg_path))
    if not HAVE_TORCH:
        print("  (torch not importable: checkpoint step counts NOT verified)")

    cache_path = out.parent / ".e03_collect_cache.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    devices: "queue.Queue[int]" = queue.Queue()
    for i in range(max(1, a.jobs)):
        devices.put(i)

    def key(j):
        st = j[3].stat()
        return f"{j[3]}|{st.st_size}|{int(st.st_mtime)}|{j[2]}"

    def run_one(j):
        cond, seed, ts, ckpt, cfg_path = j
        k = key(j)
        if k in cache:
            return j, cache[k], None
        dev = devices.get()
        try:
            src, ref = tests[ts]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev)}
            p = subprocess.run([sys.executable, a.eval_script, "--ckpt", str(ckpt),
                                "--config", str(cfg_path), "--src", str(src), "--ref", str(ref),
                                "--beam", "5", "--length-penalty", "1.0"],
                               cwd=mt, capture_output=True, text=True, env=env)
        finally:
            devices.put(dev)
        m = BLEU_RE.search(p.stdout + p.stderr)
        if p.returncode != 0 or not m:
            return j, None, (p.stdout + p.stderr)[-400:]
        return j, float(m.group(1)), None

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for j, bleu, err in ex.map(run_one, jobs):
            if bleu is None:
                failures.append(f"{j[0]} seed {j[1]} {j[2]}: {err}")
                continue
            cache[key(j)] = bleu
            rows.append((j[0], j[1], j[2], bleu))
            print(f"  {j[0]:10s} {j[1]:>3} {j[2]:18s} {bleu:6.2f}")
    json.dump(cache, open(cache_path, "w"), indent=1)

    order = {"baseline": 0, "ft_topk": 1, "ft_random": 2, "ft_bottom": 3}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1], r[2]))
    body = "condition\tseed\ttestset\tbleu\n" + "".join(f"{c}\t{s}\t{t}\t{b:.2f}\n" for c, s, t, b in rows)

    if incomplete or failures:
        partial = out.with_suffix(".partial.tsv")
        partial.write_text(body)
        print(f"\nINCOMPLETE — wrote {partial}, NOT {out}")
        for x in incomplete + failures:
            print(f"  {x}")
        print("e03_decide.py must not be run on a partial collection.")
        return 1
    out.write_text(body)
    print(f"\nwrote {out}: {len(rows)} rows ({len(runs)} runs x {len(tests)} test sets + baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
