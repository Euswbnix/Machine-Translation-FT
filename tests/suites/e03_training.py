"""Integration lane: the verified training-path findings.

e03_select_lr (LR rule), warm_cache (tokenisation-cache race), run_parallel /
e03_collect stale-run guards and content-keyed, incrementally saved BLEU cache,
e03_collect --ft-ckpt avg-last5 and budget accounting, e03_decide floored Welch df and
fail-closed completeness, hf_to_ckpt .part/sha download, hf_devtest_to_text transform,
e03_decisions schema. Fixtures are generated here; expectations are recomputed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


def _w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- select_lr
def select_lr(ctx):
    d = ctx.d / "sel"
    curves = {
        "1": [29.9, 29.6, 29.7, 29.4, 29.5, 29.3, 29.2, 29.25, 29.1, 29.0],   # falling, not strictly
        "0.5": [30.0, 29.9, 30.1, 29.8, 30.0, 29.9, 29.85, 29.9, 29.8, 29.75],
        "0.15": [30.0, 30.1, 30.0, 30.05, 30.0, 30.1, 30.0, 30.05, 30.0, 30.02],
        "0.05": [30.0, 30.0, 30.1, 30.0, 30.0, 30.1, 30.0, 30.0, 30.1, 30.02],
    }

    def write(cur, root=d):
        for s, ys in cur.items():
            _w(root / f"ft_topk_lr{s}_s42_st1.log",
               "noise line\n" + "".join(f"Step {106000 + 1000 * k:>7d} | Valid BLEU: {y:.2f} (next eval at 0)\n"
                                        for k, y in enumerate(ys)))

    write(curves)
    S = ctx.ROOT / "phase0/e03_select_lr.py"

    def sel(*args, root=d):
        out = root / "sel.json"
        out.unlink(missing_ok=True)
        rc, o = ctx.run([S, "--logdir", root, "--out", out, *args])
        return rc, o, (json.load(open(out)) if out.exists() else {})

    rc, o, j = sel()
    ctx.check("select_lr default rule (the runner's current wording) keeps a noisy non-strict decline: selects 1",
              rc == 0 and j.get("selected") == "1", o[-300:])
    rc, o, j = sel("--rule", "flat-endpoints", "--tol", "0.3")
    ctx.check("select_lr flat-endpoints tol 0.3 rejects the same curve (|29.0-29.9| = 0.9): selects 0.5",
              rc == 0 and j.get("selected") == "0.5", o[-300:])
    strict = dict(curves, **{"1": [29.9 - 0.1 * k for k in range(10)]})
    d2 = ctx.ctx_d2 = ctx.d / "sel_strict"
    write(strict, d2)
    rc, o, j = sel(root=d2)
    ctx.check("select_lr default rule fails a strictly monotone decline: selects 0.5", rc == 0 and j.get("selected") == "0.5", o[-300:])
    tie = dict(curves, **{"1": [29.9, 29.8, 29.8, 29.7, 29.6, 29.5, 29.4, 29.3, 29.2, 29.1]})
    d5 = ctx.d / "sel_tie"
    write(tie, d5)
    rc, o, j = sel(root=d5)
    ctx.check("select_lr default rule is STRICT: a never-rising curve with one tie is not a strict decline, so 1 is kept",
              rc == 0 and j.get("selected") == "1", o[-300:])
    ys = curves["0.5"]; xs = [106000 + 1000 * k for k in range(10)]
    mx, my = sum(xs) / 10, sum(ys) / 10
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    rc, o, j = sel("--rule", "flat-slope", "--tol", "0.3")
    st = {r["lr_scale"]: r["statistic"] for r in j.get("rungs", [])}
    ctx.check("select_lr flat-slope statistic equals an independent OLS |slope| x span",
              rc in (0, 1) and abs(st.get(0.5, -1) - abs(slope) * 9000) < 1e-9, f"{st.get(0.5)} vs {abs(slope) * 9000}")
    rc, o, j = sel("--rule", "flat-vs-baseline", "--tol", "0.3")
    ctx.check("select_lr flat-vs-baseline without --baseline-bleu -> exit 2", rc == 2, o[-200:])
    rc, o, j = sel("--rule", "flat-vs-baseline", "--tol", "0.3", "--baseline-bleu", "30.0")
    ctx.check("select_lr flat-vs-baseline B=30.0 tol 0.3 selects 0.5 (lr 1 ends 1.0 below)", rc == 0 and j.get("selected") == "0.5", o[-300:])
    rc, o, j = sel("--tol", "0.3")
    ctx.check("select_lr refuses a tolerance for the no-tolerance rule (exit 2)", rc == 2, o[-200:])
    rc, o, j = sel("--rule", "flat-endpoints", "--tol", "0.0")
    ctx.check("select_lr: no rung passes -> exit 1, nothing selected", rc == 1 and j.get("selected") is None, o[-200:])
    short = dict(curves, **{"0.15": curves["0.15"][:9]})
    d3 = ctx.d / "sel_short"; write(short, d3)
    rc, o, j = sel(root=d3)
    ctx.check("select_lr: a rung with 9 of 10 evals -> exit 2", rc == 2 and "9 evals" in o, o[-200:])
    (d3 / "ft_topk_lr0.15_s42_st1.log").unlink()
    rc, o, j = sel(root=d3)
    ctx.check("select_lr: a missing rung log -> exit 2", rc == 2 and "missing" in o, o[-200:])
    d4 = ctx.d / "sel_dup"; write(curves, d4)
    with open(d4 / "ft_topk_lr0.05_s42_st1.log", "a") as f:
        f.write("Step  106000 | Valid BLEU: 30.00\n")
    rc, o, j = sel(root=d4)
    ctx.check("select_lr: a repeated step (restarted run) -> exit 2", rc == 2 and "twice" in o, o[-200:])


# --------------------------------------------------------------------------- warm_cache
FAKE_DATASET = r'''
import os
from pathlib import Path
import numpy as np
class TranslationDataset:
    def __init__(self, src_path, tgt_path, tokenizer, max_tokens=256):
        open(os.environ["FAKE_BUILD_LOG"], "a").write(f"{src_path} {max_tokens}\n")
        n = sum(1 for _ in open(src_path))
        off = np.arange(0, 3 * n + 1, 3, dtype=np.int64)
        np.savez(Path(str(src_path) + f".cached_{max_tokens}.npz"), src_tokens=np.zeros(3 * n, np.uint16),
                 src_offsets=off, tgt_tokens=np.zeros(3 * n, np.uint16), tgt_offsets=off)
'''


def warm(ctx):
    mt = (ctx.d / "wc_mt")
    mt.mkdir(parents=True, exist_ok=True)
    mt = mt.resolve()
    _w(mt / "src/__init__.py", ""); _w(mt / "src/data/__init__.py", "")
    _w(mt / "src/data/dataset.py", FAKE_DATASET)
    _w(mt / "src/data/tokenizer.py", "class Tokenizer:\n    def __init__(self, p):\n        self.p = p\n")
    for n in ("data/a.en", "data/a.fr", "data/b.en", "data/b.fr", "v/valid.en", "v/valid.fr", "v/spm.model"):
        _w(mt / n, "one\ntwo\n")
    lines = ["#!/usr/bin/env bash", 'LR="${1:?}"', 'VALID="0.5"']
    for c, tr in (("ft_topk", "a"), ("ft_random", "a"), ("ft_bottom", "b")):
        _w(mt / f"cfg/{c}_lr0.5.yaml", json.dumps({
            "model": {"max_seq_len": 128}, "checkpoint": {"dir": f"ck/{c}"},
            "data": {"train_src": f"data/{tr}.en", "train_tgt": f"data/{tr}.fr", "valid_src": "v/valid.en",
                     "valid_tgt": "v/valid.fr", "spm_model": "v/spm.model"}}))
        for sd in (42, 1):
            lines.append(f"python train.py --config cfg/{c}_lr${{LR}}.yaml --resume b.pt --seed {sd} --suffix _s{sd}")
    runner = mt / "run.sh"; _w(runner, "\n".join(lines) + "\n")
    blog = ctx.d / "wc_build.log"; blog.write_text("")
    env = {"FAKE_BUILD_LOG": str(blog)}
    args = [ctx.ROOT / "phase0/warm_cache.py", "--runner", runner, "--lr-scale", "0.5", "--cwd", mt]
    rc, o = ctx.run(args, env=env)
    built = blog.read_text().splitlines()
    ctx.check("warm_cache builds one cache per distinct train file plus one valid cache (3 of 12 references)",
              rc == 0 and sorted(built) == sorted([f"{mt}/data/a.en 128", f"{mt}/data/b.en 128", f"{mt}/v/valid.en 128"]),
              f"rc={rc} {built} {o[-200:]}")
    rc, o = ctx.run(args, env=env)
    ctx.check("warm_cache rerun builds nothing", rc == 0 and len(blog.read_text().splitlines()) == 3, o[-200:])
    bad = mt / "data/b.en.cached_128.npz"
    bad.write_bytes(bad.read_bytes()[:40])
    rc, o = ctx.run(args, env=env)
    ctx.check("warm_cache detects a truncated cache, deletes it and exits 1", rc == 1 and not bad.exists() and "UNREADABLE" in o, o[-200:])
    rc, o = ctx.run(args, env=env)
    ctx.check("warm_cache then rebuilds the deleted cache", rc == 0 and bad.exists() and len(blog.read_text().splitlines()) == 4, o[-200:])


# --------------------------------------------------------------------------- run_parallel stale guard
def parallel_stale(ctx):
    cwd = ctx.d / "rp"
    _w(cwd / "train.py", "import argparse, json, os\nap = argparse.ArgumentParser()\n"
       "for f in ('--config','--resume','--seed','--suffix'): ap.add_argument(f)\n"
       "ap.add_argument('--reset-optimizer', action='store_true')\na = ap.parse_args()\n"
       "open(os.environ['RP_LOG'], 'a').write(a.suffix + '\\n')\n"
       "dd = json.load(open(a.config))['checkpoint']['dir'] + a.suffix\nos.makedirs(dd, exist_ok=True)\n"
       "import json as _j; _j.dump({'global_step': 115000, 'applied_target_tokens': 1234567,\n"
       "    'optimizer_steps': 3000, 'dropped_tokens': 0, 'total_train_tokens': 2345678},\n"
       "    open(dd + '/final.pt', 'w'))\n")
    _w(cwd / "cfg/ft_topk_lr1.yaml", json.dumps({"checkpoint": {"dir": "ck/ft_topk_lr1"}}))
    runner = cwd / "configs/run_stage1.sh"
    _w(runner, "#!/usr/bin/env bash\n" + "".join(
        f"python train.py --config cfg/ft_topk_lr1.yaml --resume b.pt --seed {s} --suffix _s{s}\n" for s in (1, 2)))
    mj = runner.parent / "matrix.json"
    mj.write_text(json.dumps({"controls_sha": "a" * 64}))
    log = ctx.d / "rp.log"; log.write_text("")
    args = [ctx.ROOT / "phase0/run_parallel.py", "--runner", runner, "--gpus", "2", "--cwd", cwd, "--logdir", ctx.d / "rp_logs"]
    env = {"RP_LOG": str(log)}
    rc, o = ctx.run(args, env=env)
    ctx.check("run_parallel writes the controls_sha marker next to each launched run",
              rc == 0 and (cwd / "ck/ft_topk_lr1_s1/controls_sha").read_text().strip() == "a" * 64, o[-200:])
    mj.write_text(json.dumps({"controls_sha": "b" * 64}))
    rc, o = ctx.run(args, env=env)
    ctx.check("run_parallel refuses final.pt trained on other controls (exit 2) and launches nothing",
              rc == 2 and "STALE" in o and len(log.read_text().splitlines()) == 2, o[-200:])
    mj.write_text(json.dumps({"controls_sha": "a" * 64}))
    (cwd / "ck/ft_topk_lr1_s2/controls_sha").unlink()
    rc, o = ctx.run(args, env=env)
    ctx.check("run_parallel refuses a final.pt with no marker", rc == 2 and "None" in o, o[-200:])


def parallel_controls(ctx):
    """audit F10: before any launch, the control-set dir must still fingerprint to matrix.json."""
    cwd = ctx.d / "rpc"
    ctrl = cwd / "data/phase0"
    written = ["ft_topk.en", "ft_topk.fr"]
    for x in written:
        _w(ctrl / x, f"{x}\n")
    json.dump({"written": written + ["manifest.json"]}, open(ctrl / "manifest.json", "w"))
    _w(cwd / "train.py", "import sys\nopen(__import__('os').environ['RPC_LOG'], 'a').write('x\\n')\n")
    _w(cwd / "cfg/ft_topk_lr1.yaml", json.dumps({"checkpoint": {"dir": "ck/ft_topk_lr1"},
                                                 "data": {"train_src": "data/phase0/ft_topk.en"}}))
    runner = cwd / "configs/run_stage1.sh"
    _w(runner, "#!/usr/bin/env bash\npython train.py --config cfg/ft_topk_lr1.yaml --resume b.pt --seed 1 --suffix _s1\n")
    files = {n: hashlib.sha256((ctrl / n).read_bytes()).hexdigest() for n in written + ["manifest.json"]}
    (runner.parent / "matrix.json").write_text(json.dumps({"controls_sha": fp_indep(ctrl, written), "data_dir": "data/phase0",
                                                           "controls_files": files}))
    log = ctx.d / "rpc.log"; log.write_text("")
    args = [ctx.ROOT / "phase0/run_parallel.py", "--runner", runner, "--gpus", "1", "--cwd", cwd,
            "--logdir", ctx.d / "rpc_logs", "--dry-run"]
    rc, o = ctx.run(args, env={"RPC_LOG": str(log)})
    ctx.check("run_parallel --dry-run passes when the control sets still match matrix.json", rc == 0, o[-200:])
    (ctrl / "ft_topk.fr").write_text("changed\n")
    rc, o = ctx.run(args, env={"RPC_LOG": str(log)})
    ctx.check("run_parallel refuses (exit 2, nothing launched) when a control file changed after the matrix",
              rc == 2 and "STALE CONTROLS" in o and "ft_topk.fr" in o and log.read_text() == "", o[-300:])
    rc, o = ctx.run([ctx.ROOT / "phase0/warm_cache.py", "--runner", runner, "--cwd", cwd, "--dry-run"])
    ctx.check("warm_cache refuses (exit 2) when a control file changed after the matrix", rc == 2 and "STALE CONTROLS" in o, o[-300:])


# --------------------------------------------------------------------------- e03_collect
FAKE_EVAL = r'''
import argparse, hashlib, os, signal
ap = argparse.ArgumentParser()
for f in ("--ckpt", "--config", "--src", "--ref", "--beam", "--length-penalty"):
    ap.add_argument(f)
a = ap.parse_args()
cnt = os.environ["EVAL_COUNTER"]
n = len(open(cnt).read().splitlines()) + 1 if os.path.exists(cnt) else 1
kill_at = os.environ.get("KILL_PARENT_AT")
if kill_at and n == int(kill_at):
    os.kill(os.getppid(), signal.SIGKILL)
    raise SystemExit(9)
open(cnt, "a").write(f"{a.ckpt} {a.src}\n")
h = int(hashlib.md5((a.ckpt + a.src + open(a.ref).read()).encode()).hexdigest(), 16)
print(f"BLEU (fr, sacrebleu 13a): {30 + h % 1000 / 100:.2f}")
'''

FAKE_AVG = r'''
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="+"); ap.add_argument("--out")
a = ap.parse_args()
json.dump({"averaged": a.ckpts, "global_step": 115000}, open(a.out, "w"))
'''


def fp_indep(ctrl: Path, written):
    names = sorted(set(written) | {"manifest.json"})
    blob = "".join(f"{n}\t{hashlib.sha256((ctrl / n).read_bytes()).hexdigest()}\n" for n in names)
    return hashlib.sha256(blob.encode()).hexdigest()


def collect(ctx):
    base = ctx.d / "col"
    mt, sf = base / "mt", base / "sft"
    for x in ("test.en", "test.fr"):
        _w(mt / "data_enfr_v1" / x, f"news {x}\n")
    ctrl = sf / "data/phase0"
    written = ["heldout_un.en", "heldout_un.fr", "ft_topk.en"]
    for x in written:
        _w(ctrl / x, f"{x} content\n")
    json.dump({"heldout_sets": {"heldout_un": {}}, "written": written + ["manifest.json"]}, open(ctrl / "manifest.json", "w"))
    _w(mt / "ckpt_hf/base.pt", "w")
    _w(sf / "configs/sft_base_enfr.yaml", '{"model": {}}')
    _w(mt / "scripts/fake_eval.py", FAKE_EVAL)
    _w(mt / "scripts/fake_avg.py", FAKE_AVG)
    sha = fp_indep(ctrl, written)
    lines = ["#!/usr/bin/env bash", 'LR="${1:?}"']
    for c in ("ft_topk", "ft_random", "ft_bottom"):
        _w(sf / f"configs/phase0/{c}_lr0.15.yaml", json.dumps(
            {"checkpoint": {"dir": f"checkpoints/phase0/{sha[:12]}/{c}_lr0.15"},
             "training": {"max_steps": 115000, "save_interval": 2000}, "model": {}}))
        for sd in (42, 1, 2):
            lines.append(f"python train.py --config {sf}/configs/phase0/{c}_lr${{LR}}.yaml "
                         f"--resume ckpt_hf/base.pt --reset-optimizer --seed {sd} --suffix _s{sd}_st2")
            ck = mt / f"checkpoints/phase0/{sha[:12]}/{c}_lr0.15_s{sd}_st2"
            ck.mkdir(parents=True, exist_ok=True)
            ctx.write_ckpt(ck / "final.pt", global_step=115000, applied_target_tokens=1_000_000,
                           optimizer_steps=2500, dropped_tokens=0, total_train_tokens=1_000_000)
            (ck / "controls_sha").write_text(sha + "\n")
            for st in range(106000, 116000, 2000):
                ctx.write_ckpt(ck / f"step_{st}.pt", global_step=st)
    runner = sf / "configs/phase0/run_stage2.sh"; _w(runner, "\n".join(lines) + "\n")
    (sf / "configs/phase0/matrix.json").write_text(json.dumps({"controls_sha": sha}))
    counter = base / "count"
    out = sf / "results/phase0_bleu.tsv"
    args = [ctx.ROOT / "phase0/e03_collect.py", "--mt-root", mt, "--runner", runner, "--lr-scale", "0.15",
            "--baseline-ckpt", "ckpt_hf/base.pt", "--baseline-config", sf / "configs/sft_base_enfr.yaml",
            "--controls-dir", ctrl, "--out", out, "--eval-script", "scripts/fake_eval.py", "--jobs", "2"]
    # Every collect run reads its checkpoints through the shared torch stub, so the JSON
    # fixtures mean the same thing with or without real torch installed (tests/regress.py).
    env = {"EVAL_COUNTER": str(counter), "PYTHONPATH": str(ctx.make_faketorch(ctx.d))}

    def n_evals():
        return len(counter.read_text().splitlines()) if counter.exists() else 0

    rc, o = ctx.run(args, env=env)
    meta = lambda p: json.load(open(p.with_suffix(".meta.json"))) if p.with_suffix(".meta.json").exists() else {}
    ctx.check("collect: complete run evaluates 9 runs x 2 sets + 2 baseline = 20; TSV keeps its 4-column format; "
              "meta.json records ft_ckpt final vs averaged",
              rc == 0 and n_evals() == 20 and out.read_text().startswith("condition\tseed\ttestset\tbleu\n")
              and meta(out).get("ft_ckpt") == "final" and meta(out).get("baseline_kind") == "averaged"
              and meta(out).get("controls_sha") == sha, o[-300:])
    rc, o = ctx.run(args, env=env)
    ctx.check("collect: rerun is served from the cache (0 new evaluations)", rc == 0 and n_evals() == 20, o[-200:])

    # stale controls: content change of a manifest-listed file
    orig = (ctrl / "heldout_un.fr").read_text()
    (ctrl / "heldout_un.fr").write_text("CHANGED reference\n")
    rc, o = ctx.run(args, env=env)
    ctx.check("collect: controls rebuilt after the matrix -> exit 2 STALE, nothing evaluated",
              rc == 2 and "STALE" in o and n_evals() == 20, o[-200:])
    # cache key on content (legacy runner without matrix.json -> guard inactive, cache must still invalidate)
    (sf / "configs/phase0/matrix.json").rename(sf / "configs/phase0/matrix.json.off")
    rc, o = ctx.run(args, env=env)
    ctx.check("collect: a changed heldout_un.fr re-evaluates exactly the baseline + 9 runs on heldout_un (10)",
              rc == 0 and n_evals() == 30, f"rc={rc} evals={n_evals()} {o[-200:]}")
    (ctrl / "heldout_un.fr").write_text(orig)
    (sf / "configs/phase0/matrix.json.off").rename(sf / "configs/phase0/matrix.json")
    rc, _ = ctx.run(args, env=env)                                   # back to the original content: cached
    mk = mt / f"checkpoints/phase0/{sha[:12]}/ft_random_lr0.15_s1_st2/controls_sha"
    mk.write_text("f" * 64 + "\n")
    rc, o = ctx.run(args, env=env)
    ctx.check("collect: a run whose marker differs from the matrix -> exit 2", rc == 2 and "ft_random seed 1" in o, o[-200:])
    mk.write_text(sha + "\n")

    # incremental cache: parent killed on the 5th evaluation
    out2 = sf / "results2/phase0_bleu.tsv"
    a2 = [x if x != out else out2 for x in args]
    a2[a2.index("--jobs") + 1] = "1"
    before = n_evals()
    rc, o = ctx.run(a2, env={**env, "KILL_PARENT_AT": str(before + 5)})
    cache2 = sf / "results2/.e03_collect_cache.json"
    n_cached = len(json.load(open(cache2))) if cache2.exists() else 0
    ctx.check("collect: SIGKILL during the 5th evaluation leaves exactly the 4 finished ones in the cache",
              rc != 0 and n_cached == 4, f"rc={rc} cached={n_cached}")
    mid = n_evals()
    rc, o = ctx.run(a2, env=env)
    ctx.check("collect: the resumed collection decodes only the remaining 16", rc == 0 and n_evals() - mid == 16,
              f"rc={rc} new={n_evals() - mid}")
    cache2.write_text("{not json")
    rc, o = ctx.run(a2, env=env)
    ctx.check("collect: a corrupt cache is moved aside, not a traceback", rc == 0 and cache2.with_name(cache2.name + ".corrupt").exists(), o[-200:])

    # avg-last5
    out3 = sf / "results3/phase0_bleu.tsv"
    a3 = [x if x != out else out3 for x in args] + ["--ft-ckpt", "avg-last5", "--average-script", "scripts/fake_avg.py"]
    before = n_evals()
    rc, o = ctx.run(a3, env=env)
    ck = mt / f"checkpoints/phase0/{sha[:12]}/ft_topk_lr0.15_s42_st2"
    inputs = json.load(open(ck / "avg_last5.inputs.json")) if (ck / "avg_last5.inputs.json").exists() else []
    evald = [l.split()[0] for l in counter.read_text().splitlines()[before:]]
    ctx.check("collect --ft-ckpt avg-last5 averages step_108000..114000 + final.pt and evaluates the average",
              rc == 0 and [Path(p).name for p in inputs] == ["step_108000.pt", "step_110000.pt", "step_112000.pt", "step_114000.pt", "final.pt"]
              and all(e.endswith("avg_last5.pt") or e.endswith("base.pt") for e in evald) and len(evald) == 20
              and meta(out3).get("ft_ckpt") == "avg-last5", f"rc={rc} {inputs} {o[-200:]}")
    (mt / f"checkpoints/phase0/{sha[:12]}/ft_bottom_lr0.15_s2_st2/step_110000.pt").unlink()
    out4 = sf / "results4/phase0_bleu.tsv"
    rc, o = ctx.run([x if x != out3 else out4 for x in a3], env=env)
    ctx.check("collect avg-last5: a missing step checkpoint makes that run incomplete (exit 1)",
              rc == 1 and "not contiguous" in o and not out4.exists(), o[-300:])
    ctx.write_ckpt(mt / f"checkpoints/phase0/{sha[:12]}/ft_bottom_lr0.15_s2_st2/step_110000.pt", global_step=110000)
    ckm = mt / f"checkpoints/phase0/{sha[:12]}/ft_topk_lr0.15_s1_st2"
    ctx.write_ckpt(ckm / "step_116000.pt", global_step=116000)
    for x in ("avg_last5.pt", "avg_last5.inputs.json"):
        (ckm / x).unlink(missing_ok=True)
    out4b = sf / "results4b/phase0_bleu.tsv"
    rc, o = ctx.run([x if x != out3 else out4b for x in a3], env=env)
    ctx.check("collect avg-last5: a step checkpoint at max_steps (116000 >= 115000) makes that run incomplete, no average",
              rc == 1 and "is not below max_steps" in o and not out4b.exists() and not (ckm / "avg_last5.pt").exists(), o[-300:])
    (ckm / "step_116000.pt").unlink()

    # a failed collection must not leave an earlier canonical TSV in place (audit F9)
    out6 = sf / "results6/phase0_bleu.tsv"
    a6 = [x if x != out else out6 for x in args] + ["--selected-lr", "0.15", "--decisions-sha", "d" * 64]
    rc, o = ctx.run(a6, env=env)
    m6 = meta(out6)
    ctx.check("collect meta records lr_scale, selected_lr, lr_deviation and decisions_sha",
              rc == 0 and m6.get("lr_scale") == "0.15" and m6.get("selected_lr") == "0.15"
              and m6.get("lr_deviation") is False and m6.get("decisions_sha") == "d" * 64, str(m6)[:300])
    fin = mt / f"checkpoints/phase0/{sha[:12]}/ft_random_lr0.15_s1_st2/final.pt"
    fin.rename(fin.with_name("final.off"))
    rc, o = ctx.run(a6, env=env)
    ctx.check("collect: an incomplete collection deletes the previous canonical TSV and meta.json",
              rc == 1 and not out6.exists() and not out6.with_suffix(".meta.json").exists(), o[-300:])
    fin.with_name("final.off").rename(fin)

    # budget accounting with a fake torch
    ft = ctx.make_faketorch(ctx.d)
    ckr = mt / f"checkpoints/phase0/{sha[:12]}/ft_random_lr0.15_s2_st2/final.pt"
    ckr.write_text(json.dumps({"global_step": 115000, "applied_target_tokens": 950_000, "optimizer_steps": 2500,
                               "dropped_tokens": 20_000, "total_train_tokens": 970_000}))
    out5 = sf / "results5/phase0_bleu.tsv"
    a5 = [x if x != out else out5 for x in args]
    rc, o = ctx.run(a5, env={**env, "PYTHONPATH": str(ft)})
    m5 = meta(out5)
    ctx.check("collect: applied-token imbalance (1.0526) and a 2.06% dropped fraction print WARNINGs; meta.json records them",
              rc == 0 and o.count("WARNING (budget)") == 2 and "ratio 1.0526" in o
              and m5.get("budget", {}).get("ft_random 2", {}).get("applied_target_tokens") == 950000
              and len(m5.get("budget_warnings", [])) == 2, o[-400:])
    rc_d, od = ctx.run([ctx.ROOT / "phase0/e03_decide.py", "--results", out5, "--indomain", "heldout_un"])
    ctx.check("e03_decide repeats the collector's budget warnings", "WARNING (budget): applied-token imbalance" in od, od[-300:])
    out5.unlink(); out5.with_suffix(".meta.json").unlink()
    rc, o = ctx.run(a5 + ["--strict-budget"], env={**env, "PYTHONPATH": str(ft)})
    ctx.check("collect --strict-budget: the same imbalance makes the collection incomplete (exit 1, no TSV)",
              rc == 1 and not out5.exists(), o[-200:])
    ckr.write_text(json.dumps({"global_step": 114000, "applied_target_tokens": 1_000_000, "dropped_tokens": 0}))
    rc, o = ctx.run(a5, env={**env, "PYTHONPATH": str(ft)})
    ctx.check("collect: a final.pt short of max_steps is incomplete when torch can read it", rc == 1 and "stopped at step 114000" in o, o[-200:])


# --------------------------------------------------------------------------- e03_decide
GO = """condition seed testset bleu
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
"""


def decide(ctx):
    d = ctx.d / "dec"; d.mkdir()
    DEC = ctx.ROOT / "phase0/e03_decide.py"
    rc, o = ctx.run(["-c", "import sys, json; sys.path.insert(0, sys.argv[1]); import e03_decide as m; "
                     "g=[1+0.1*i for i in range(400)]; "
                     "print(json.dumps([m.tcrit(2.6), m.tcrit(3.6), m.tcrit(11.5), m.tcrit(50), m.tcrit(500), "
                     "min(m.tcrit(x) for x in g), all(m.tcrit(a) >= m.tcrit(b) for a, b in zip(g, g[1:]))]))",
                     ctx.ROOT / "phase0"])
    v = json.loads(o.strip().splitlines()[-1]) if rc == 0 else []
    ctx.check("tcrit is conservative: t(2.6) >= 3.47, t(3.6) >= 2.90, t(11.5) >= 2.20, never < 1.96, non-increasing",
              rc == 0 and v[0] >= 3.47 and v[1] >= 2.90 and v[2] >= 2.20 and v[3] >= 2.04 and v[4] >= 1.96 and v[5] >= 1.96 and v[6],
              str(v))
    # df ~ 2.65, t ~ 3.3: significant under round(df)=3 (3.182), not under exact t(2.65) ~ 3.45
    sdr = math.sqrt(6)
    mr = 31 + 3.3 * math.sqrt(7 / 3)
    fx = ["condition seed testset bleu", "baseline - newstest2014 40.0", "baseline - heldout_un 35.0",
          "baseline - heldout_europarl 33.0"]
    fx += [f"ft_topk {s} newstest2014 {x}" for s, x in zip((42, 1, 2), (30.0, 31.0, 32.0))]
    fx += [f"ft_topk {s} heldout_un 36.0" for s in (42, 1, 2)] + [f"ft_topk {s} heldout_europarl 34.0" for s in (42, 1, 2)]
    fx += [f"ft_random {s} newstest2014 {x:.6f}" for s, x in zip((42, 1, 2), (mr - sdr, mr, mr + sdr))]
    va, vb = 1.0, 6.0
    df = (va / 3 + vb / 3) ** 2 / ((va / 3) ** 2 / 2 + (vb / 3) ** 2 / 2)
    p = d / "df.tsv"; p.write_text("\n".join(fx) + "\n")
    rc, o = ctx.run([DEC, "--results", p])
    ctx.check(f"decide: Welch df {df:.2f}, |t| 3.30 is NOT significant with floored df (criterion 1 FAIL, exit 1)",
              rc == 1 and "criterion 1: FAIL" in o and "t=3.30" in o and "NOT significant" in o, o[-500:])
    partial = "".join(l + "\n" for l in GO.splitlines() if not l.startswith("ft_topk") or "heldout_europarl" not in l)
    p = d / "partial.tsv"; p.write_text(partial)
    rc, o = ctx.run([DEC, "--results", p, "--indomain", "heldout_un,heldout_europarl"])
    ctx.check("decide: a listed set's ft_topk rows missing -> exit 2 (explicit two-set list; the frozen gate is UN-only)",
              rc == 2 and "NO DECISION" in o, o[-200:])
    short = GO.replace("ft_topk 2 newstest2014 36.50\n", "")
    p = d / "short.tsv"; p.write_text(short)
    rc, o = ctx.run([DEC, "--results", p])
    ctx.check("decide: one ft_topk newstest seed missing -> exit 2 (incomplete cells)", rc == 2 and "incomplete cells" in o, o[-200:])
    p = d / "go.tsv"; p.write_text(GO)
    rc, o = ctx.run([DEC, "--results", p])
    ctx.check("decide: a header-less TSV is assumed final vs averaged and warns; verdict unchanged",
              rc == 0 and "different kinds" in o and "legacy TSV" in o, o[-300:])
    p.with_suffix(".meta.json").write_text(json.dumps({"ft_ckpt": "avg-last5", "baseline_kind": "averaged"}))
    rc, o = ctx.run([DEC, "--results", p])
    ctx.check("decide: avg-last5 vs averaged meta.json -> no checkpoint-kind warning; verdict unchanged",
              rc == 0 and "different kinds" not in o and "fine-tuned = avg-last5" in o, o[-300:])
    p2 = d / "hand.tsv"; p2.write_text("# ft_ckpt=final baseline_kind=final\n" + GO)
    rc, o = ctx.run([DEC, "--results", p2])
    ctx.check("decide: a hand-built '#' header is still honoured (final vs final: no warning)",
              rc == 0 and "different kinds" not in o and "baseline = final" in o, o[-300:])


# --------------------------------------------------------------------------- hf_to_ckpt download
def hf_download(ctx):
    d = ctx.d / "hf"; d.mkdir()
    code = r'''
import hashlib, io, json, sys, urllib.request
sys.path.insert(0, sys.argv[1])
import hf_to_ckpt as m
from pathlib import Path
data = {"config.json": b"{}", "pytorch_model.bin": b"W" * 1000, "sentencepiece.model": b"SPM"}
served = dict(data); calls = []
def fake(url, timeout=None):
    name = url.rsplit("/", 1)[-1]; calls.append(name)
    return io.BytesIO(served[name])
m.urllib.request.urlopen = fake
dest = Path(sys.argv[2])
exp = {"pytorch_model.bin": hashlib.sha256(data["pytorch_model.bin"]).hexdigest()}
res = {}
dest.mkdir(parents=True, exist_ok=True)
(dest / "pytorch_model.bin.part").write_bytes(b"half")
m.download("r", "s", dest, exp)
res["fresh"] = [(dest / k).read_bytes() == v for k, v in data.items()] + [not any(dest.glob("*.part")), len(calls) == 3]
(dest / "pytorch_model.bin").write_bytes(b"W" * 10)          # truncated cached file
calls.clear()
m.download("r", "s", dest, exp)
res["recache"] = [calls == ["pytorch_model.bin"], (dest / "pytorch_model.bin").read_bytes() == data["pytorch_model.bin"]]
class Broken(io.BytesIO):
    def read(self, n=-1):
        if self.tell() >= 100:
            raise OSError("connection reset")
        return super().read(100)
def fake_broken(url, timeout=None):
    return Broken(served[url.rsplit("/", 1)[-1]])
m.urllib.request.urlopen = fake_broken
(dest / "pytorch_model.bin").unlink()
try:
    m.download("r", "s", dest, exp); res["interrupted"] = False
except OSError:
    res["interrupted"] = not (dest / "pytorch_model.bin").exists()
m.urllib.request.urlopen = fake
served["pytorch_model.bin"] = b"X" * 1000
(dest / "pytorch_model.bin").write_bytes(b"W" * 10)
try:
    m.download("r", "s", dest, exp); res["bad"] = False
except SystemExit as e:
    res["bad"] = "refusing" in str(e)
print(json.dumps(res))
'''
    rc, o = ctx.run(["-c", code, ctx.ROOT / "phase0", d / "dl"])
    r = json.loads(o.strip().splitlines()[-1]) if rc == 0 else {}
    ctx.check("hf_to_ckpt: downloads via .part (a stale .part is discarded), all files intact", all(r.get("fresh", [False])), o[-300:])
    ctx.check("hf_to_ckpt: a cached file failing --expect-sha is deleted and fetched again", all(r.get("recache", [False])), o[-300:])
    ctx.check("hf_to_ckpt: a fresh download failing --expect-sha is fatal", r.get("bad") is True, o[-300:])
    ctx.check("hf_to_ckpt: an interrupted download leaves no file under the final name", r.get("interrupted") is True, o[-300:])


# --------------------------------------------------------------------------- hf_devtest_to_text
def devtest(ctx):
    d = ctx.d / "dt"; d.mkdir()
    rows = [("  Hello world  ", "Bonjour le monde"), ("two\nlines", "deux\nlignes"), ("", "vide"),
            ("only src", "   "), ("sep \u2028 inside", "sep \u2028 dedans"), ("tab\there", "tab\tici")]
    exp_s, exp_t = [], []
    for s, t in rows:                                   # download_wmt_enfr.py _save_split, retyped
        s2, t2 = s.strip().replace("\n", " "), t.strip().replace("\n", " ")
        if s2 and t2:
            exp_s.append(s2 + "\n"); exp_t.append(t2 + "\n")
    es, et = "".join(exp_s).encode(), "".join(exp_t).encode()
    code = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import hf_devtest_to_text as m
rows = json.loads(sys.argv[3]); out = sys.argv[2]
q = lambda *a: None
r = {}
n = int(sys.argv[6])
r["ok"] = m.convert(rows, out + "/v.en", out + "/v.fr", sys.argv[4], sys.argv[5], n, log=q)
r["badsha"] = m.convert(rows, out + "/w.en", out + "/w.fr", "0" * 64, sys.argv[5], n, log=q)
r["badlines"] = m.convert(rows, out + "/x.en", out + "/x.fr", sys.argv[4], sys.argv[5], n + 1, log=q)
print(json.dumps(r))
'''
    rc, o = ctx.run(["-c", code, ctx.ROOT / "phase0", d, json.dumps(rows),
                     hashlib.sha256(es).hexdigest(), hashlib.sha256(et).hexdigest(), str(len(exp_s))])
    r = json.loads(o.strip().splitlines()[-1]) if rc == 0 else {}
    ctx.check("hf_devtest_to_text reproduces _save_split (strip, newline->space, skip empty) byte for byte",
              len(exp_s) == 4 and r.get("ok") == 0 and (d / "v.en").read_bytes() == es and (d / "v.fr").read_bytes() == et, o[-300:])
    ctx.check("hf_devtest_to_text: a sha or line-count mismatch writes only .rejected (exit 1)",
              r.get("badsha") == 1 and r.get("badlines") == 1 and not (d / "w.en").exists()
              and (d / "w.en.rejected").exists() and not (d / "x.fr").exists(), o[-300:])
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        ctx.skip("hf_devtest_to_text parquet read", "pyarrow not importable by this python")


# --------------------------------------------------------------------------- e03_decisions
def decisions(ctx):
    d = ctx.d / "decs"; d.mkdir()
    V = ctx.ROOT / "phase0/e03_decisions.py"
    rc, o = ctx.run([V, "check", ctx.ROOT / "phase0/e03_decisions.example.json"])
    keys = ("pool", "n_ft", "heldout_domains", "indomain", "exclude_pretrain_from_heldout", "lr_selection_rule",
            "lr_tolerance_bleu", "score_mode", "calibration", "ft_checkpoint", "budget", "target_tokens", "loss_spike_ratio")
    ctx.check("decisions: the shipped example is invalid and names all 13 decisions",
              rc == 2 and all(f'{k}: still "CHOOSE"' in o for k in keys), o[-500:])
    good = {"pool": "reused", "n_ft": 1000000, "heldout_domains": ["un", "europarl", "giga-fren"],
            "indomain": ["heldout_un"], "exclude_pretrain_from_heldout": True, "lr_selection_rule": "flat-endpoints",
            "lr_tolerance_bleu": 0.3, "score_mode": "full",
            "calibration": {"max_mean_diff": 2e-4, "max_p99_diff": 2e-3, "min_spearman": 0.999, "min_jaccard": 0.99, "min_rows": 1000},
            "ft_checkpoint": "avg-last5", "budget": "steps", "target_tokens": None, "loss_spike_ratio": 0, "_note": "x"}

    def chk(obj):
        p = d / "x.json"; p.write_text(json.dumps(obj))
        return ctx.run([V, "shell", p])

    rc, o = chk(good)
    ctx.check("decisions: a complete file validates and exports shell values",
              rc == 0 and "D_CAL_ARGS='--max-mean-diff 0.0002 --max-p99-diff 0.002 --min-spearman 0.999 --min-jaccard 0.99 --min-rows 1000'" in o
              and "D_HELDOUT_DOMAINS=un,europarl,giga-fren" in o and "D_EXCLUDE_PRETRAIN=1" in o, o[-400:])
    cases = {
        "unknown key": (dict(good, pools="full"), "unknown key 'pools'"),
        "indomain not built": (dict(good, indomain=["heldout_commoncrawl"]), "not built by heldout_domains"),
        "tolerance for the no-tolerance rule": (dict(good, lr_selection_rule="no-strict-monotone-decline-from-first-eval"), "takes no tolerance"),
        "full without calibration": (dict(good, calibration=None), "needs an object"),
        "tokens without target": (dict(good, budget="tokens"), "needs a positive integer"),
        "none with pool full": (dict(good, score_mode="none", calibration=None, pool="full"), 'requires pool "reused"'),
        "bool n_ft": (dict(good, n_ft=True), "not a positive integer"),
    }
    for name, (obj, want) in cases.items():
        rc, o = chk(obj)
        ctx.check(f"decisions: {name} is refused", rc == 2 and want in o, o[-300:])
    import re as _re
    sh = lambda o_: dict(_re.findall(r"^(D_[A-Z_0-9]+)=(.*)$", o_, _re.M))        # noqa: E731
    _, o = chk(good)
    rc1, o1 = chk(dict(good, calibration={**good["calibration"], "jaccard_min_k": 200}))
    ctx.check("decisions: optional calibration.jaccard_min_k is exported as --jaccard-min-k; absent -> not passed",
              rc1 == 0 and "--jaccard-min-k 200" in sh(o1).get("D_CAL_ARGS", "") and "--jaccard-min-k" not in o, o1[-300:])
    rc2, o2 = chk(dict(good, calibration={**good["calibration"], "jaccard_min_k": -1}))
    ctx.check("decisions: a negative jaccard_min_k is refused", rc2 == 2 and "jaccard_min_k" in o2, o2[-200:])
    base_sha = sh(o).get("D_SCORE_SHA")
    _, o3 = chk(dict(good, n_ft=5, _note="other", indomain=["heldout_un", "heldout_europarl"]))
    _, o4 = chk(dict(good, calibration={**good["calibration"], "min_rows": 999}))
    ctx.check("decisions: D_SCORE_SHA ignores n_ft/indomain/comments but changes with calibration; D_SHA256 changes with any byte",
              base_sha and sh(o3).get("D_SCORE_SHA") == base_sha and sh(o4).get("D_SCORE_SHA") != base_sha
              and sh(o3).get("D_SHA256") != sh(o).get("D_SHA256"), f"{base_sha} {sh(o3).get('D_SCORE_SHA')} {sh(o4).get('D_SCORE_SHA')}")
    pa, pb = d / "a.json", d / "b.json"
    pa.write_text(json.dumps(good)); pb.write_text(json.dumps(dict(good, indomain=["heldout_europarl"], _note="y")))
    rc5, o5 = ctx.run([V, "diff", pa, pb])
    ctx.check("decisions diff names the changed decision keys only (comments ignored)", rc5 == 0 and o5.strip() == "indomain", o5[-200:])
    obj = dict(good); del obj["budget"]
    rc, o = chk(obj)
    ctx.check("decisions: a missing key is refused by name", rc == 2 and "budget: MISSING" in o, o[-200:])


def suite(ctx):
    for part in (select_lr, warm, parallel_stale, parallel_controls, collect, decide, hf_download, devtest, decisions):
        try:
            part(ctx)
        except Exception:                                          # noqa: BLE001
            import traceback
            ctx.check(f"e03_training.{part.__name__} ran without crashing", False, traceback.format_exc()[-600:])
