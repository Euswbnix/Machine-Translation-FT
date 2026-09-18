#!/usr/bin/env python3
"""E0.3d — evaluate every E0.3 run and assemble the TSV that e03_decide.py reads.

The pre-registered gate needs BLEU on newstest2014 AND the held-out in-domain
sets, for every stage-2 run and for the pre-fine-tuning baseline. Doing 30
evaluations by hand is exactly where a row gets mislabelled or a crashed run is
silently left out, so this is scripted and refuses to hide gaps:

  * Which runs exist is parsed from the runner e03_run_matrix.py generated
    (run_stage2.sh): condition, seed and --suffix come from what was launched,
    not from a list retyped here.
  * A run with no final.pt, or whose final.pt stopped short of its budget
    (max_steps, or max_target_tokens under --budget tokens), is INCOMPLETE. Any
    incompleteness writes <out>.partial.tsv and exits 1; the canonical TSV that
    e03_decide reads is only written when nothing is missing.
  * Stale-run guard: when the runner's matrix.json carries a controls_sha, the
    --controls-dir must still fingerprint to it and every run's <ckdir>/controls_sha
    marker (written by run_parallel.py) must equal it; otherwise exit 2.
  * Finished evaluations are cached under a key built from everything that affects
    the score (sha256 of the test src/ref files, checkpoint path/size/mtime, sha256
    of the config and the eval script, beam, length penalty). The cache is rewritten
    atomically after EVERY finished evaluation, so a killed collection resumes.

Checkpoint type (training-path finding; a pre-registration decision, default = the
current behaviour): --ft-ckpt final evaluates <run>/final.pt; --ft-ckpt avg-last5
evaluates the average of the last 4 step_*.pt plus final.pt (scripts/average_checkpoints.py
--ckpts, never --ckpt-dir/--n, whose regex skips final.pt). The baseline is the averaged
HF release. <out stem>.meta.json records both kinds and e03_decide prints a warning when
they differ: the paper puts checkpoint averaging at +0.2-0.4 BLEU. (The TSV itself keeps
its 4-column format.)

Budget accounting (training-path finding): applied_target_tokens, optimizer_steps,
dropped_tokens and total_train_tokens from each final.pt (trainer patch) go into
<out stem>.meta.json. A max/min applied-token ratio above 1 + --max-token-imbalance, or a dropped
fraction above --max-dropped-frac, prints a WARNING; with --strict-budget it makes the
collection incomplete instead. Without torch these are "not verified".

Canonical output: <out> and <out stem>.meta.json are deleted at start and rewritten only
when this collection completes, so they always belong to the most recent complete
collection; meta.json records lr_scale, the stage-1 selected lr and the decisions sha.

Exit: 0 canonical TSV written; 1 incomplete or failed evaluations (.partial.tsv only);
2 stale runs/controls or bad inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controls_fingerprint import MARKER, fingerprint  # noqa: E402

BLEU_RE = re.compile(r"BLEU \([a-z]+, sacrebleu 13a\): ([0-9.]+)")
RUN_RE = re.compile(r"python\s+train\.py\s.*?--config\s+(\S+).*?--seed\s+(\d+).*?--suffix\s+(\S+)")
BEAM, LENGTH_PENALTY = "5", "1.0"
BUDGET_KEYS = ("global_step", "applied_target_tokens", "optimizer_steps", "dropped_tokens", "total_train_tokens")


def load_cfg(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(open(path))
    except ImportError:
        return json.load(open(path))       # JSON is a YAML subset; used by the offline tests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


DEFAULT_HELDOUT = ("heldout_un", "heldout_europarl")


def held_out_sets(ctrl: Path) -> list[str]:
    """Held-out test sets to evaluate: the ones <controls-dir>/manifest.json lists
    under "heldout_sets" (written by e03_build_controls.py), in that order; the
    original two when there is no manifest or it predates that key. A listed set
    whose files are missing fails in main(), like any missing test set."""
    mp = ctrl / "manifest.json"
    if not mp.exists():
        print(f"  (no {mp}: using the default held-out sets {list(DEFAULT_HELDOUT)})")
        return list(DEFAULT_HELDOUT)
    try:
        man = json.load(open(mp, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read {mp}: {exc}")
    if "heldout_sets" not in man:
        print(f"  ({mp} lists no heldout_sets: using the default {list(DEFAULT_HELDOUT)})")
        return list(DEFAULT_HELDOUT)
    names = list(man["heldout_sets"])
    if not names:
        sys.exit(f"{mp} says NO held-out sets were built (--allow-no-heldout?) — "
                 "the gate cannot be evaluated")
    bad = [x for x in names if not re.fullmatch(r"heldout_[A-Za-z0-9_.\-]+", x)]
    if bad:
        sys.exit(f"{mp} lists malformed held-out set names: {bad}")
    print(f"  held-out sets from {mp}: {names}")
    return names


try:
    import torch                                            # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def ckpt_info(ckpt: Path):
    """Budget fields recorded in the checkpoint, or None if torch is unavailable."""
    if not HAVE_TORCH:
        return None
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    return {k: d.get(k) for k in BUDGET_KEYS}


def step_files(ckdir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in ckdir.glob("step_*.pt"):
        m = re.fullmatch(r"step_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def build_avg_last5(ckdir: Path, cfg: dict, mt: Path, avg_script: str, python: str):
    """-> (path, None) or (None, reason). Inputs: last 4 step_*.pt (contiguous at
    save_interval, all below max_steps) + final.pt."""
    steps = step_files(ckdir)
    if len(steps) < 4:
        return None, f"avg-last5 needs 4 step_*.pt in {ckdir}, found {len(steps)} (keep_last too low?)"
    last4 = steps[-4:]
    si = int(cfg["training"].get("save_interval", 0))
    gaps = {b[0] - a[0] for a, b in zip(last4, last4[1:])}
    if not si or gaps != {si}:
        return None, f"step checkpoints {[s for s, _ in last4]} are not contiguous at save_interval {si}"
    if last4[-1][0] >= int(cfg["training"]["max_steps"]):
        return None, f"last step checkpoint {last4[-1][0]} is not below max_steps"
    inputs = [p for _, p in last4] + [ckdir / "final.pt"]
    out = ckdir / "avg_last5.pt"
    rec = ckdir / "avg_last5.inputs.json"
    want = [str(p) for p in inputs]
    if out.exists() and rec.exists() and json.load(open(rec)) == want \
            and out.stat().st_mtime >= max(p.stat().st_mtime for p in inputs):
        return out, None
    tmp = ckdir / "avg_last5.pt.tmp"
    script = Path(avg_script) if Path(avg_script).is_absolute() else mt / avg_script
    p = subprocess.run([python, str(script), "--ckpts", *want, "--out", str(tmp)],
                       cwd=mt, capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return None, f"average_checkpoints failed: {(p.stdout + p.stderr)[-300:]}"
    os.replace(tmp, out)
    rec.write_text(json.dumps(want))
    return out, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt-root", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--lr-scale", required=True, help="the value passed to run_stage2.sh, e.g. 0.15")
    ap.add_argument("--selected-lr", default=None, help="lr_scale stage 1 selected (recorded in meta.json)")
    ap.add_argument("--decisions-sha", default=None, help="sha256 of the decisions file (recorded in meta.json)")
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--baseline-config", required=True)
    ap.add_argument("--controls-dir", required=True)
    ap.add_argument("--news-src", default="data_enfr_v1/test.en")
    ap.add_argument("--news-ref", default="data_enfr_v1/test.fr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="parallel evals, one GPU each")
    ap.add_argument("--eval-script", default="scripts/eval_bleu.py",
                    help="relative to --mt-root; replaceable for offline tests")
    ap.add_argument("--ft-ckpt", choices=("final", "avg-last5"), default="final",
                    help="pre-registration decision; default final = the behaviour the gate was written with")
    ap.add_argument("--average-script", default="scripts/average_checkpoints.py", help="relative to --mt-root")
    ap.add_argument("--max-token-imbalance", type=float, default=0.02)
    ap.add_argument("--max-dropped-frac", type=float, default=0.01)
    ap.add_argument("--strict-budget", action="store_true",
                    help="a budget WARNING makes the collection incomplete (exit 1)")
    a = ap.parse_args()

    mt = Path(a.mt_root).expanduser().resolve()
    ctrl = Path(a.controls_dir).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only a complete run of THIS invocation may leave a canonical TSV (audit F9): a failed or
    # stale collection must not leave an older collection (e.g. another lr_scale) in place.
    for old in (out, out.with_suffix(".meta.json")):
        if old.exists():
            print(f"  removing previous {old.name} (rewritten only if this collection completes)")
            old.unlink()
    runner = Path(a.runner).expanduser()

    tests = {"newstest2014": (mt / a.news_src, mt / a.news_ref)}
    heldout = held_out_sets(ctrl)
    for name in heldout:
        tests[name] = (ctrl / f"{name}.en", ctrl / f"{name}.fr")
    missing_sets = [f"{k}: {p}" for k, (s, r) in tests.items() for p in (s, r) if not p.exists()]
    if missing_sets:
        sys.exit("test sets missing — the gate cannot be evaluated:\n  " + "\n  ".join(missing_sets))

    want_sha = None
    mp = runner.parent / "matrix.json"
    if mp.is_file():
        want_sha = json.load(open(mp, encoding="utf-8")).get("controls_sha")
    if want_sha:
        have = fingerprint(ctrl)["controls_sha"]
        if have != want_sha:
            print(f"STALE: {ctrl} fingerprints to {have[:12]}... but the runs were generated for "
                  f"{want_sha[:12]}... — the control sets were rebuilt after the matrix. Not collecting.")
            return 2
    else:
        print("  (runner has no matrix.json controls_sha: stale-run guard NOT active)")

    runs = parse_runner(runner, a.lr_scale)
    jobs, incomplete, stale, budget, unverifiable = [], [], [], {}, []
    base_ckpt = (mt / a.baseline_ckpt) if not Path(a.baseline_ckpt).is_absolute() else Path(a.baseline_ckpt)
    if not base_ckpt.exists():
        sys.exit(f"baseline checkpoint {base_ckpt} does not exist — run rental_setup.sh accept first")
    for ts in tests:
        jobs.append(("baseline", "-", ts, base_ckpt, Path(a.baseline_config).expanduser()))
    for r in runs:
        cfg_path = r["config"] if r["config"].is_absolute() else (mt / r["config"])
        cfg = load_cfg(cfg_path)
        ckdir = mt / (cfg["checkpoint"]["dir"] + r["suffix"])
        final = ckdir / "final.pt"
        label = f"{r['condition']} seed {r['seed']}"
        if want_sha:
            mk = ckdir / MARKER
            got = mk.read_text().strip() if mk.is_file() else None
            if final.exists() and got != want_sha:
                stale.append(f"{label}: marker {got!r} != {want_sha[:12]}...")
                continue
        if not final.exists():
            incomplete.append(f"{label}: no {final}")
            continue
        info = ckpt_info(final)
        # A config without a training block cannot say what "finished" means: verify what
        # is there and say so, rather than raising KeyError halfway through a collection.
        tr = cfg.get("training") or {}
        if info is not None:
            budget[(r["condition"], r["seed"])] = info
            if tr.get("max_target_tokens"):
                if (info.get("applied_target_tokens") or 0) < int(tr["max_target_tokens"]):
                    incomplete.append(f"{label}: applied {info.get('applied_target_tokens')} "
                                      f"< max_target_tokens {tr['max_target_tokens']}")
                    continue
            elif tr.get("max_steps") is None:
                unverifiable.append(f"{label}: config has no training.max_steps; step count not verified")
            elif info.get("global_step") is not None and info["global_step"] < tr["max_steps"]:
                incomplete.append(f"{label}: stopped at step {info['global_step']} < {tr['max_steps']}")
                continue
        ckpt = final
        if a.ft_ckpt == "avg-last5":
            ckpt, why = build_avg_last5(ckdir, cfg, mt, a.average_script, sys.executable)
            if ckpt is None:
                incomplete.append(f"{label}: {why}")
                continue
        for ts in tests:
            jobs.append((r["condition"], str(r["seed"]), ts, ckpt, cfg_path))
    if unverifiable:
        print("NOT VERIFIED (budget):\n  " + "\n  ".join(unverifiable))
    if stale:
        print("STALE RUNS — trained on different control sets:\n  " + "\n  ".join(stale))
        return 2
    if not HAVE_TORCH:
        print("  (torch not importable: checkpoint step/token counts NOT verified)")

    # ---- budget accounting ------------------------------------------------
    warnings = []
    if budget:
        applied = {k: v.get("applied_target_tokens") for k, v in budget.items()}
        vals = [x for x in applied.values() if isinstance(x, (int, float)) and x > 0]
        if len(vals) == len(applied) and vals:
            ratio = max(vals) / min(vals)
            print(f"  applied target tokens: min {min(vals):,} max {max(vals):,} ratio {ratio:.4f}")
            if ratio > 1 + a.max_token_imbalance:
                warnings.append(f"applied-token imbalance across runs {ratio:.4f} > 1+{a.max_token_imbalance}")
        else:
            warnings.append("some final.pt lack applied_target_tokens (unpatched trainer?)")
        for k, v in budget.items():
            ap_, dr = v.get("applied_target_tokens") or 0, v.get("dropped_tokens") or 0
            if ap_ + dr and dr / (ap_ + dr) > a.max_dropped_frac:
                warnings.append(f"{k[0]} seed {k[1]}: spike guard dropped {dr / (ap_ + dr):.2%} of tokens")
    for w in warnings:
        print(f"  WARNING (budget): {w}")
    if warnings and a.strict_budget:
        incomplete += [f"budget: {w}" for w in warnings]

    # ---- evaluation with a content-keyed, incrementally saved cache -----------
    cache_path = out.parent / ".e03_collect_cache.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.load(open(cache_path))
            if not isinstance(cache, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as e:
            bad = cache_path.with_name(cache_path.name + ".corrupt")
            os.replace(cache_path, bad)
            print(f"  cache {cache_path} unreadable ({e}); moved to {bad}, starting empty")
            cache = {}
    ts_sha = {ts: (sha256_file(s), sha256_file(r)) for ts, (s, r) in tests.items()}
    eval_script = mt / a.eval_script
    eval_sha = sha256_file(eval_script) if eval_script.is_file() else "missing"
    cfg_sha: dict = {}

    def key(j):
        st = j[3].stat()
        c = str(j[4])
        if c not in cfg_sha:
            cfg_sha[c] = sha256_file(j[4])
        return json.dumps({"ckpt": str(j[3]), "size": st.st_size, "mtime": int(st.st_mtime),
                           "testset": j[2], "src_sha": ts_sha[j[2]][0], "ref_sha": ts_sha[j[2]][1],
                           "config_sha": cfg_sha[c], "eval_script": a.eval_script, "eval_sha": eval_sha,
                           "beam": BEAM, "lp": LENGTH_PENALTY}, sort_keys=True)

    lock = threading.Lock()

    def save_cache():
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache_path)

    devices: "queue.Queue[int]" = queue.Queue()
    for i in range(max(1, a.jobs)):
        devices.put(i)

    def run_one(j, k):
        cond, seed, ts, ckpt, cfg_path = j
        if k in cache:
            return j, cache[k], None
        dev = devices.get()
        try:
            src, ref = tests[ts]
            # scripts/eval_bleu.py imports `src.*`, and `python scripts/x.py` puts only
            # scripts/ on sys.path; a fresh clone is not pip-installed.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(dev),
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(mt), os.environ.get("PYTHONPATH")]))}
            p = subprocess.run([sys.executable, a.eval_script, "--ckpt", str(ckpt),
                                "--config", str(cfg_path), "--src", str(src), "--ref", str(ref),
                                "--beam", BEAM, "--length-penalty", LENGTH_PENALTY],
                               cwd=mt, capture_output=True, text=True, env=env)
        finally:
            devices.put(dev)
        m = BLEU_RE.search(p.stdout + p.stderr)
        if p.returncode != 0 or not m:
            return j, None, (p.stdout + p.stderr)[-400:]
        bleu = float(m.group(1))
        with lock:                               # saved by the worker before it takes the next job,
            cache[k] = bleu                      # so a kill during job n+1 never loses job n
            save_cache()
        return j, bleu, None

    rows, failures = [], []
    keys = [key(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = [ex.submit(run_one, j, k) for j, k in zip(jobs, keys)]
        for fu in as_completed(futs):
            j, bleu, err = fu.result()
            if bleu is None:
                failures.append(f"{j[0]} seed {j[1]} {j[2]}: {err}")
                continue
            rows.append((j[0], j[1], j[2], bleu))
            print(f"  {j[0]:10s} {j[1]:>3} {j[2]:18s} {bleu:6.2f}")

    order = {"baseline": 0, "ft_topk": 1, "ft_random": 2, "ft_bottom": 3}
    rows.sort(key=lambda r: (order.get(r[0], 9), r[1], r[2]))
    meta = {"ft_ckpt": a.ft_ckpt, "baseline_kind": "averaged", "baseline_ckpt": str(base_ckpt),
            "lr_scale": a.lr_scale, "selected_lr": a.selected_lr, "decisions_sha": a.decisions_sha,
            "lr_deviation": bool(a.selected_lr is not None and a.selected_lr != a.lr_scale),
            "controls_sha": want_sha, "torch_verified": HAVE_TORCH,
            "budget": {f"{c} {s}": v for (c, s), v in sorted(budget.items())},
            "budget_warnings": warnings, "complete": not (incomplete or failures)}
    body = "condition\tseed\ttestset\tbleu\n" + \
        "".join(f"{c}\t{s}\t{t}\t{b:.2f}\n" for c, s, t, b in rows)

    if incomplete or failures:
        partial = out.with_suffix(".partial.tsv")
        partial.write_text(body)
        partial.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(f"\nINCOMPLETE — wrote {partial}, NOT {out}")
        for x in incomplete + failures:
            print(f"  {x}")
        print("e03_decide.py must not be run on a partial collection.")
        return 1
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    out.write_text(body)
    print(f"\nwrote {out}: {len(rows)} rows ({len(runs)} runs x {len(tests)} test sets + baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
