"""Integration lane: phase0/rental_setup.sh end to end, offline.

A copy of phase0/, scripts/ and configs/ is placed where the script expects the SFT
clone (a real git repo with a local bare remote, so the committed/tagged/pushed checks on
the decisions run for real), with fakes only at the hardware/network edges: nvidia-smi,
hf / huggingface-cli, curl, rebuild_corpus.py (copies toy corpora), hf_to_ckpt.py and
hf_devtest_to_text.py, score_with_comet.py (deterministic hash scorer with the real CLI),
e03_build_controls.py (writes a manifest; optionally drops a held-out file), the trainer
(train.py + src/data) and eval_bleu.py. Everything between those edges is the production
code: the stage logic, e03_decisions.py, the bundle SHA256SUMS and pin checks,
rescore_plan.py extract/merge/calibrate, score_sharded.py, e03_run_matrix.py,
warm_cache.py, run_parallel.py, e03_select_lr.py, e03_collect.py and e03_decide.py.
Scored TSVs are checked against an expectation computed here from the old scored file
and the fake scoring function, not from the plan.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

SOURCES = ["europarl", "commoncrawl", "un", "news-commentary", "giga-fren"]
MEMBERS = ["plan.json", "missing_rows.npy", "reuse_scores.npy", "provenance_labels.npy",
           "provenance_report.json", "pool_mask_reused.npy"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shab(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fscore(s: str, t: str) -> float:
    return int.from_bytes(hashlib.blake2b((s + "\0" + t).encode("utf-8"), digest_size=8).digest(), "little") / 2**64


FAKE_SCORER = r'''
import argparse, hashlib, json, os, sys
ap = argparse.ArgumentParser()
for f in ("--src", "--tgt", "--out", "--model", "--meta-out", "--model-revision", "--encoder-revision"):
    ap.add_argument(f, default="")
for f in ("--gpus", "--batch-size", "--chunk-size"):
    ap.add_argument(f, type=int, default=1)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--allow-stack-change", action="store_true")
a = ap.parse_args()
if a.gpus > 1:
    sys.exit(2)
dev = os.environ.get("CUDA_VISIBLE_DEVICES")
if os.environ.get("FAKE_STACK_REFUSE") and not a.allow_stack_change:
    print("ERROR: the scoring stack changed since rows 0..9 were scored (differs: packages). Refusing.", file=sys.stderr)
    sys.exit(3)
def sc(s, t):
    return int.from_bytes(hashlib.blake2b((s + "\0" + t).encode("utf-8"), digest_size=8).digest(), "little") / 2**64
table = json.load(open(os.environ["FAKE_SCORE_TABLE"])) if os.environ.get("FAKE_SCORE_TABLE") else {}
done = sum(1 for _ in open(a.out, encoding="utf-8", newline="\n")) if a.resume and os.path.exists(a.out) else 0
with open(a.src, encoding="utf-8", newline="\n") as fs, open(a.tgt, encoding="utf-8", newline="\n") as ft, \
     open(a.out, "a" if done else "w", encoding="utf-8", newline="\n") as fo:
    for i, (s, t) in enumerate(zip(fs, ft)):
        if i < done:
            continue
        s = s[:-1].replace("\t", " "); t = t[:-1].replace("\t", " ")
        v = table.get(s + "\t" + t)
        fo.write(f"{v if v is not None else format(sc(s, t), '.6f')}\t{s}\t{t}\n")
if a.meta_out:
    torch = os.environ.get("FAKE_META_TORCH") if os.environ.get("FAKE_META_TORCH_DEV") == dev else "1.0"
    meta = {"fake": True, "cuda_visible_devices": dev, "packages": {"torch": torch}}
    if os.environ.get("FAKE_STACK_REFUSE") and a.allow_stack_change:
        meta["segments"] = [{"packages": {"torch": "0.9"}}, {"packages": {"torch": "1.0"}}]
    json.dump(meta, open(a.meta_out, "w"))
if os.environ.get("FAKE_CORRUPT_REPORT"):          # simulates a read error on an input calibrate needs
    open(os.environ["FAKE_CORRUPT_REPORT"], "w").write("{}")
open(os.environ["FAKE_SCORER_LOG"], "a").write(f"{os.environ.get('CUDA_VISIBLE_DEVICES', '-')} {a.src}\n")
'''

FAKE_REBUILD = r'''
import argparse, os, shutil, sys
ap = argparse.ArgumentParser()
for f in ("--parquet-dir", "--src", "--tgt", "--mode", "--out-dir", "--expect-sha-src", "--expect-sha-tgt"):
    ap.add_argument(f)
ap.add_argument("--max-rows", type=int, default=0)
a = ap.parse_args()
if os.environ.get("FAKE_REBUILD_FAIL") == "1":
    print("REJECTED: on-disk sha256 differs"); sys.exit(1)
src = os.environ["TOY_V1" if a.max_rows else "TOY_V2"]
for ext in ("en", "fr"):
    shutil.copyfile(f"{src}/train.clean.{ext}", f"{a.out_dir}/train.clean.{ext}")
'''

FAKE_BUILDER = r'''
import argparse, json, os
ap = argparse.ArgumentParser()
for f in ("--qe-scores", "--provenance", "--provenance-report", "--out-dir", "--n-ft", "--heldout-domains",
          "--pool-mask", "--pretrain-src", "--pretrain-tgt"):
    ap.add_argument(f)
a = ap.parse_args()
open(os.environ["FAKE_CTRL_ARGS"], "w").write(json.dumps(vars(a)))
os.makedirs(a.out_dir, exist_ok=True)
written = []
for c in ("ft_topk", "ft_random", "ft_bottom"):
    for e in ("en", "fr"):
        open(f"{a.out_dir}/{c}.{e}", "w").write(f"{c} {e}\n"); written.append(f"{c}.{e}")
hs = {}
for d in a.heldout_domains.split(","):
    n = f"heldout_{d}"; hs[n] = {"domain": d}
    for e in ("en", "fr"):
        if os.environ.get("FAKE_CTRL_DROP") == f"{n}.{e}":
            continue
        open(f"{a.out_dir}/{n}.{e}", "w").write(f"{n} {e} line\n"); written.append(f"{n}.{e}")
written.append("manifest.json")
json.dump({"heldout_sets": hs, "written": written}, open(f"{a.out_dir}/manifest.json", "w"))
print("fake builder wrote", len(written), "files")
'''

FAKE_TRAIN = r'''
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--resume"); ap.add_argument("--seed"); ap.add_argument("--suffix")
ap.add_argument("--reset-optimizer", action="store_true")
a = ap.parse_args()
cfg = json.load(open(a.config))
L = cfg["model"]["max_seq_len"]
for p in (cfg["data"]["train_src"], cfg["data"]["valid_src"]):
    if not os.path.exists(p + f".cached_{L}.npz"):
        print("tokenisation cache missing at launch:", p); sys.exit(7)
lr = cfg["training"]["lr_scale"]
for k in range(10):
    b = 30.0 - 0.1 * k if lr == 1.0 else 30.0 + (0.05 if k % 2 else 0.0)
    print(f"Step {106000 + 1000 * k:>7d} | Valid BLEU: {b:.2f} (next eval at {107000 + 1000 * k})")
d = cfg["checkpoint"]["dir"] + a.suffix
os.makedirs(d, exist_ok=True)
open(d + "/final.pt", "w").write("{}")
'''

FAKE_DATASET = r'''
import os
from pathlib import Path
import numpy as np
class TranslationDataset:
    def __init__(self, src_path, tgt_path, tokenizer, max_tokens=256):
        open(os.environ["FAKE_BUILD_LOG"], "a").write(f"{src_path}\n")
        n = sum(1 for _ in open(src_path))
        off = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
        np.savez(Path(str(src_path) + f".cached_{max_tokens}.npz"), src_tokens=np.zeros(2 * n, np.uint16),
                 src_offsets=off, tgt_tokens=np.zeros(2 * n, np.uint16), tgt_offsets=off)
'''

FAKE_EVAL = r'''
import argparse, hashlib, os
ap = argparse.ArgumentParser()
for f in ("--ckpt", "--config", "--src", "--ref", "--beam", "--length-penalty"):
    ap.add_argument(f)
a = ap.parse_args()
if os.environ.get("FAKE_EVAL_RC"):
    import sys
    print("Traceback (most recent call last):\nRuntimeError: CUDA out of memory (fake)", file=sys.stderr)
    sys.exit(int(os.environ["FAKE_EVAL_RC"]))
h = int(hashlib.md5((a.ckpt + a.src).encode()).hexdigest(), 16)
if not os.environ.get("FAKE_NO_BLEU"):
    print(f"BLEU (fr, sacrebleu 13a): {os.environ.get('FAKE_BLEU') or format(30 + h % 300 / 100, '.2f')}")
'''

FAKE_CURL = r'''
import os, shutil, sys
argv = sys.argv[1:]
out = argv[argv.index("-o") + 1]
url = argv[-1]
rel = url.split("/resolve/", 1)[1].split("/", 1)[1]
open(os.environ["FAKE_CURL_LOG"], "a").write(f"existed={int(os.path.exists(out))} {rel}\n")
src = os.path.join(os.environ["FAKE_CURL_DIR"], rel)
data = open(src, "rb").read()
if os.environ.get("FAKE_CURL_BAD") == "1":
    data = b"Z" * len(data)
open(out, "wb").write(data)
'''

FAKE_HF_TO_CKPT = r'''
import argparse, os, shutil
ap = argparse.ArgumentParser()
ap.add_argument("--out")
a, _ = ap.parse_known_args()
open(a.out, "w").write("w")
shutil.copyfile(os.environ["FAKE_SPM_FILE"], a.out[:-len(".pt")] + ".sentencepiece.model")
'''

FAKE_DEVTEST = r'''
import argparse, os, shutil
ap = argparse.ArgumentParser()
for f in ("--parquet", "--out-src", "--out-tgt"):
    ap.add_argument(f)
a, _ = ap.parse_known_args()
open(os.environ["FAKE_DEVTEST_LOG"], "a").write(os.path.basename(a.parquet) + "\n")
for p in (a.out_src, a.out_tgt):
    shutil.copyfile(os.path.join(os.environ["FAKE_DEVTEST_DIR"], os.path.basename(p)), p)
'''

BASE_CFG = {"model": {"d_model": 512, "max_seq_len": 256},
            "training": {"batch_size": 24576, "max_sentences": 192, "accumulate_steps": 4, "max_steps": 115000,
                         "lr_scale": 1.0, "min_lr": 1e-5, "early_stopping": True, "patience": 5,
                         "eval_interval": 2000, "eval_interval_min": 1000, "save_interval": 2000,
                         "loss_spike_ratio": 1.3},
            "data": {"train_src": "x.en", "train_tgt": "x.fr", "valid_src": "data_enfr_v1/valid.en",
                     "valid_tgt": "data_enfr_v1/valid.fr", "spm_model": "data_enfr_v1/spm.model"},
            "checkpoint": {"dir": "ck", "keep_last": 5},
            "logging": {"swanlab": {"enabled": True, "experiment": "sft"}}}


def decisions(**over):
    d = {"pool": "reused", "n_ft": 10, "heldout_domains": ["un", "europarl"], "indomain": ["heldout_un"],
         "exclude_pretrain_from_heldout": True, "lr_selection_rule": "no-strict-monotone-decline-from-first-eval",
         "lr_tolerance_bleu": None, "score_mode": "none", "calibration": None, "ft_checkpoint": "final",
         "budget": "steps", "target_tokens": None, "loss_spike_ratio": "inherit"}
    d.update(over)
    return d


def suite(ctx):
    np, d = ctx.np, ctx.d
    if shutil.which("bash") is None or shutil.which("sha256sum") is None:
        ctx.skip("rental_setup.sh offline", "bash or sha256sum not available")
        return
    # An x86_64 python on Apple silicon cannot exec the arm64-only xcrun git shim (nor can the
    # bash it starts), so find a git that runs and put a shim for it first on PATH.
    git_cmd = None
    for cand in (["git"], ["/Library/Developer/CommandLineTools/usr/bin/git"], ["arch", "-arm64", "git"]):
        try:
            if subprocess.run([*cand, "--version"], capture_output=True, text=True).returncode == 0:
                git_cmd = cand
                break
        except OSError:
            continue
    if git_cmd is None:
        ctx.skip("rental_setup.sh offline", "no runnable git")
        return
    check = ctx.check

    # ---- docs that must not drift from the script's real status (audit F16, F23)
    header = (ctx.ROOT / "phase0/rental_setup.sh").read_text()
    check("rental_setup.sh header does not claim the current env/accept ran on a box",
          "Only 'env' and 'accept' have run" not in header and "have NOT run on a box" in header)
    rb_lines = (ctx.ROOT / "phase0/RUNBOOK.md").read_text().splitlines()
    bad = [i for i, ln in enumerate(rb_lines) if "not a substitute" in ln
           and not any("Superseded" in x for x in rb_lines[max(0, i - 10):i])]
    check("RUNBOOK: every 'not a substitute' (HF releases) paragraph is marked Superseded", not bad, str(bad))
    check("README does not say nothing requires renting a machine",
          "Nothing here requires renting" not in (ctx.ROOT / "phase0/README.md").read_text())

    work = d / "w"
    sft, mt = work / "Machine-Translation-SFT", work / "Machine_translation"
    for sub in ("phase0", "scripts", "configs"):
        shutil.copytree(ctx.ROOT / sub, sft / sub, ignore=shutil.ignore_patterns("__pycache__", "e03_decisions.json"))
    (sft / "configs/sft_base_enfr.yaml").write_text(json.dumps(BASE_CFG))      # JSON is YAML; yaml shim below
    (sft / "scripts/score_with_comet.py").write_text(FAKE_SCORER)
    (sft / "phase0/rebuild_corpus.py").write_text(FAKE_REBUILD)
    (sft / "phase0/e03_build_controls.py").write_text(FAKE_BUILDER)
    (sft / "phase0/hf_to_ckpt.py").write_text(FAKE_HF_TO_CKPT)
    (sft / "phase0/hf_devtest_to_text.py").write_text(FAKE_DEVTEST)
    toy_parquet = b"PAR1x"
    (sft / "phase0/hf_wmt14_filelist.tsv").write_text(f"fr-en/train-00000-of-00001.parquet\t5\t{shab(toy_parquet)}\n")
    (work / "hf_wmt14/fr-en").mkdir(parents=True)
    (work / "hf_wmt14/fr-en/train-00000-of-00001.parquet").write_bytes(toy_parquet)
    curl_dir = d / "curl_src"; (curl_dir / "fr-en").mkdir(parents=True)
    (curl_dir / "fr-en/train-00000-of-00001.parquet").write_bytes(toy_parquet)
    dev_pq, test_pq = b"DEVPARQUET", b"TESTPARQUET!"
    (curl_dir / "fr-en/validation-00000-of-00001.parquet").write_bytes(dev_pq)
    (curl_dir / "fr-en/test-00000-of-00001.parquet").write_bytes(test_pq)
    shim = d / "shim"; shim.mkdir()
    (shim / "yaml.py").write_text("import json\ndef safe_load(f): return json.load(f)\n"
                                  "def safe_dump(o, f, **k): json.dump(o, f, indent=1)\n")
    bindir = d / "bin"; bindir.mkdir()
    (bindir / "python").symlink_to(ctx.PY)
    (bindir / "nvidia-smi").write_text('#!/bin/sh\ni=0\nwhile [ $i -lt "${FAKE_NGPU:-2}" ]; do echo "GPU $i: Fake (UUID: x)"; i=$((i+1)); done\n')
    for b in ("hf", "huggingface-cli"):
        (bindir / b).write_text("#!/bin/sh\nprintf '%s\\n' \"${FAKE_HF_OUT:-fake-account}\"\n")
    (bindir / "curl").write_text(f"#!{ctx.PY}\n" + FAKE_CURL)
    # Resolve to an ABSOLUTE git before bindir goes on PATH. "exec git" would re-find this
    # shim (bindir is first on PATH) and spin forever -- on Linux, where plain "git" is the
    # chosen candidate, that hung the whole suite for as long as it was allowed to run.
    git_abs = [shutil.which(x) or x if x == "git" else x for x in git_cmd]
    (bindir / "git").write_text("#!/bin/sh\nexec " + " ".join(git_abs) + ' "$@"\n')
    check("the fake git shim execs an absolute git outside the shim dir (no PATH self-recursion)",
          all(not x.startswith(str(bindir)) for x in git_abs)
          and any(os.path.isabs(x) for x in git_abs), str(git_abs))
    for b in ("nvidia-smi", "hf", "huggingface-cli", "curl", "git"):
        (bindir / b).chmod(0o755)

    # ---- the SFT clone is a git repo with a pushed branch
    genv = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def g(*args, cwd=sft):
        # timeout: a shim that cannot run must fail a check, never hang the suite
        try:
            r = subprocess.run([str(bindir / "git"), *[str(x) for x in args]], cwd=cwd,
                               capture_output=True, text=True, env=genv, timeout=120)
        except subprocess.TimeoutExpired:
            return 124, f"git {' '.join(str(x) for x in args)} timed out after 120 s"
        return r.returncode, (r.stdout + r.stderr).strip()

    remote = d / "remote.git"
    g("init", "-q", "--bare", remote, cwd=d)
    for args in (("init", "-q"), ("checkout", "-q", "-b", "wmt2027-phase0"), ("add", "-A"),
                 ("commit", "-q", "-m", "fixture"), ("remote", "add", "origin", remote),
                 ("push", "-q", "-u", "origin", "wmt2027-phase0")):
        rc_g, o_g = g(*args)
        if rc_g != 0:
            check(f"rental fixture: git {args[0]}", False, o_g[-300:])
            return
    ntag = [0]

    def commit_decisions():
        g("add", "phase0/e03_decisions.json")
        if g("diff", "--cached", "--quiet")[0] != 0:
            g("commit", "-q", "-m", "decisions")
        ntag[0] += 1
        g("tag", f"decisions-{ntag[0]}")
        return g("push", "-q", "origin", "HEAD", "--tags")

    # ---- toy corpora (one line carries U+2028, which must stay inside its row)
    N = 60
    en = [f"english sentence {i} words here" for i in range(N)]
    fr = [f"phrase francaise {i} mots ici" for i in range(N)]
    en[40] = "line with a separator   inside it"
    toy2, toy1 = d / "toy_v2", d / "toy_v1"
    for tdir, rows in ((toy2, range(N)), (toy1, range(25))):
        tdir.mkdir()
        (tdir / "train.clean.en").write_text("".join(en[i] + "\n" for i in rows), encoding="utf-8", newline="\n")
        (tdir / "train.clean.fr").write_text("".join(fr[i] + "\n" for i in rows), encoding="utf-8", newline="\n")
    mt.mkdir()

    # ---- bundle v2 built with the real rescore_plan.py plan over an "old" scored file
    rng = np.random.RandomState(5)
    old_lines, old_score = [], {}
    for i in range(35):
        s = f"{rng.uniform(0.2, 0.95):.6f}"
        old_lines.append(f"{s}\t{en[i]}\t{fr[i]}\n"); old_score[(en[i], fr[i])] = s
    old_lines.append(f"0.100000\t{en[50]}\tWRONG PARTNER\n")
    (d / "old.tsv").write_text("".join(old_lines), encoding="utf-8", newline="\n")
    plan = d / "plan"
    rc, o = ctx.run([ctx.ROOT / "phase0/rescore_plan.py", "plan", "--old-scored", d / "old.tsv",
                     "--new-src", toy2 / "train.clean.en", "--new-tgt", toy2 / "train.clean.fr", "--out-dir", plan])
    check("rental fixture: rescore plan over the toy corpus", rc == 0, o[-300:])
    if rc != 0:
        return
    R = np.load(plan / "reuse_scores.npy")
    np.save(plan / "pool_mask_reused.npy", ~np.isnan(R))
    labels = np.array([i % 5 if i % 13 else 255 for i in range(N)], dtype=np.uint8)
    np.save(plan / "provenance_labels.npy", labels)
    (plan / "provenance_report.json").write_text(json.dumps({"normalizer": "exact", "match_rate": 0.9, "sources": SOURCES,
                                                             "full_corpus": {}}))

    def make_bundle(tamper=False, plan_over=None, flip_label=False, keep_unpack=False):
        """-> sha256 of the bundle's SHA256SUMS. plan_over/flip_label build a self-consistent variant."""
        src = plan
        if plan_over or flip_label:
            src = d / "plan_variant"
            shutil.rmtree(src, ignore_errors=True)
            shutil.copytree(plan, src)
            if plan_over:
                pj = json.load(open(src / "plan.json")); pj.update(plan_over)
                (src / "plan.json").write_text(json.dumps(pj))
            if flip_label:
                lab = np.load(src / "provenance_labels.npy"); lab[0] = (int(lab[0]) + 1) % 5
                np.save(src / "provenance_labels.npy", lab)
        sums = "".join(f"{sha(src / m)}  {m}\n" for m in MEMBERS)
        if tamper:
            sums = sums.replace(sha(src / "reuse_scores.npy"), "0" * 64)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for m in MEMBERS:
                tf.add(src / m, arcname=m)
            info = tarfile.TarInfo("SHA256SUMS"); data = sums.encode(); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if not keep_unpack:
            shutil.rmtree(work / "rescore", ignore_errors=True)
        (work / "rental_bundle_v2.tar.gz").write_bytes(buf.getvalue())
        return shab(sums.encode())

    pinned_sums = make_bundle()
    meminfo = d / "meminfo"; meminfo.write_text("MemTotal: 99999999 kB\nMemAvailable: 67108864 kB\n")
    logs = {k: d / f"{k}.log" for k in ("scorer", "ctrl_args", "build", "curl", "devtest")}
    for p in logs.values():
        p.write_text("")
    x_sha = shab(b"x\n")
    spm_file = d / "spm.model"; spm_file.write_bytes(b"spm-bytes")
    devtest_dir = d / "devtest"; devtest_dir.mkdir()
    for n_ in ("valid.en", "valid.fr", "test.en", "test.fr"):
        (devtest_dir / n_).write_text("x\n")
    env = {**os.environ, "WORK": str(work), "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
           # the gate stage runs e03_collect, which reads checkpoints with torch: the fake
           # trainer writes JSON, so the stub must win over any real torch on the box
           "PYTHONPATH": f"{ctx.make_faketorch(d)}{os.pathsep}{shim}", "TOY_V2": str(toy2), "TOY_V1": str(toy1), "MEMINFO": str(meminfo),
           "DF_KB_OVERRIDE": str(10 ** 9), "FAKE_SCORER_LOG": str(logs["scorer"]),
           "FAKE_CTRL_ARGS": str(logs["ctrl_args"]), "FAKE_BUILD_LOG": str(logs["build"]),
           "FAKE_CURL_LOG": str(logs["curl"]), "FAKE_CURL_DIR": str(curl_dir),
           "FAKE_DEVTEST_LOG": str(logs["devtest"]), "FAKE_DEVTEST_DIR": str(devtest_dir), "FAKE_SPM_FILE": str(spm_file),
           "EXPECTED_V2_FIXED_SHA_EN": sha(toy2 / "train.clean.en"), "EXPECTED_V2_FIXED_SHA_FR": sha(toy2 / "train.clean.fr"),
           "EXPECTED_V1_SHA_EN": sha(toy1 / "train.clean.en"), "EXPECTED_V1_SHA_FR": sha(toy1 / "train.clean.fr"),
           "EXPECTED_BUNDLE_SUMS_SHA": pinned_sums,
           "DEV_PARQUET_SIZE": str(len(dev_pq)), "DEV_PARQUET_SHA": shab(dev_pq),
           "TEST_PARQUET_SIZE": str(len(test_pq)), "TEST_PARQUET_SHA": shab(test_pq),
           "VALID_EN_SHA": x_sha, "VALID_FR_SHA": x_sha, "TEST_EN_SHA": x_sha, "TEST_FR_SHA": x_sha,
           "HF_BASE_SPM_SHA": shab(b"spm-bytes"), "CGROUP_MEM_DIR": str(d / "no_cgroup")}
    for k in ("SCORE_SMOKE", "SCORE_REDO", "RECALIBRATE", "DECISIONS_AMEND", "DECISIONS_UNTAGGED", "LR_OVERRIDE",
              "PIN_SFT_REV", "MT_REV", "SCORE_ALLOW_STACK_CHANGE"):
        env.pop(k, None)

    def rent(*args, **extra):
        e = {**env, **{k: str(v) for k, v in extra.items()}}
        try:
            r = subprocess.run(["bash", str(sft / "phase0/rental_setup.sh"), *args], env=e,
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as t:
            out = (t.stdout or b"").decode("utf-8", "replace") if isinstance(t.stdout, bytes) else (t.stdout or "")
            return 124, out + f"\nSTAGE {' '.join(args)} TIMED OUT after 300 s"
        return r.returncode, r.stdout + r.stderr

    dec_path = sft / "phase0/e03_decisions.json"
    res = sft / "results/phase0"

    def set_decisions(commit=True, **over):
        dec_path.write_text(json.dumps(decisions(**over), indent=1))
        if commit:
            commit_decisions()

    def nlines(p):
        return len(p.read_text().splitlines()) if p.exists() else 0

    # ---- repos: both clones at pinned commits (audit F11, F12)
    mg = lambda *a: g(*a, cwd=mt)                                                   # noqa: E731
    mg("init", "-q"); (mt / "a.py").write_text("v1\n"); mg("add", "a.py"); mg("commit", "-q", "-m", "c1")
    c1 = mg("rev-parse", "HEAD")[1]
    (mt / "a.py").write_text("v2\n"); mg("commit", "-q", "-am", "c2")
    c2 = mg("rev-parse", "HEAD")[1]
    rc, o = rent("repos", MT_REV=c1)
    check("repos checks Machine_translation out at the pinned MT_REV (HEAD was a later commit) and records it",
          rc == 0 and mg("rev-parse", "HEAD")[1] == c1 != c2 and (work / "mt_rev.txt").read_text().strip() == c1, o[-300:])
    rc, o = rent("repos", MT_REV="f" * 40)
    check("repos stops when MT_REV cannot be checked out", rc != 0 and "Machine_translation" in o, o[-200:])
    (mt / "stray.py").write_text("x\n")
    rc, o = rent("repos", MT_REV=c1)
    check("repos stops when Machine_translation has changes outside the trainer patch", rc != 0 and "outside the trainer patch" in o, o[-200:])
    (mt / "stray.py").unlink()
    sft_head = g("rev-parse", "HEAD")[1]
    rc, o = rent("repos", MT_REV=c1, PIN_SFT_REV="e" * 40)
    check("repos stops when PIN_SFT_REV is not a commit of the SFT clone", rc != 0 and "PIN_SFT_REV" in o, o[-200:])
    rc, o = rent("repos", MT_REV=c1, PIN_SFT_REV=sft_head)
    check("repos accepts PIN_SFT_REV = the pushed HEAD and records it",
          rc == 0 and (work / "sft_rev.txt").read_text().strip() == sft_head, o[-300:])
    g("checkout", "-q", "wmt2027-phase0")

    # ---- env guards with a fake python (Python version, pins, CUDA after install)
    envbin = d / "envbin"; envbin.mkdir()
    torchstub = d / "torchstub" / "torch"; torchstub.mkdir(parents=True)
    (torchstub / "__init__.py").write_text(
        "import os\n__version__ = '2.8.0+fake'\nclass version:\n    cuda = '12.8'\n"
        "class _T:\n    def cuda(self):\n        if os.environ.get('FAKE_CUDA_BAD'):\n"
        "            raise RuntimeError('no CUDA device (fake)')\n        return self\n"
        "def zeros(*a, **k):\n    return _T()\n")
    pip_log = d / "pip.log"
    (envbin / "python").write_text(f"#!{ctx.PY}\n" + r'''
import json, os, subprocess, sys
real = os.environ["REAL_PY"]
args = sys.argv[1:]
if args[:2] == ["-m", "pip"]:
    open(os.environ["FAKE_PIP_LOG"], "a").write(" ".join(args[2:]) + "\n")
    sys.exit(0)
prelude = ("import os, sys, json\n"
           "if os.environ.get('FAKE_PYVER'):\n"
           "    _v = tuple(int(x) for x in os.environ['FAKE_PYVER'].split('.'))\n"
           "    sys.version_info = _v + (0, 'final', 0)\n"
           "    sys.version = os.environ['FAKE_PYVER'] + '.0 (fake)'\n"
           "if os.environ.get('FAKE_META'):\n"
           "    import importlib.metadata as _md\n"
           "    _t = json.loads(os.environ['FAKE_META'])\n"
           "    _md.version = lambda p, _o=_md.version: _t[p] if p in _t else _o(p)\n")
if args and args[0] == "-c":
    os.execv(real, [real, "-c", prelude + args[1]] + args[2:])
if args and args[0] == "-":
    code = sys.stdin.read()
    os.execv(real, [real, "-c", prelude + code] + args[1:])
os.execv(real, [real] + args)
''')
    (envbin / "python").chmod(0o755)
    good_meta = json.dumps({"unbabel-comet": "2.2.7", "pytorch-lightning": "2.5.5", "transformers": "4.57.1", "numpy": "1.26.4"})
    env_extra = dict(MT_REV=c1, PATH=f"{envbin}{os.pathsep}{env['PATH']}", REAL_PY=ctx.PY, FAKE_PIP_LOG=pip_log,
                     PYTHONPATH=f"{shim}{os.pathsep}{torchstub.parent}")
    (mt / "requirements.txt").write_text("")
    rc, o = rent("env", FAKE_PYVER="3.13", FAKE_META=good_meta, **env_extra)
    check("env stops on Python 3.13 ('outside 3.10-3.12') before installing anything",
          rc != 0 and "outside 3.10-3.12" in o and not pip_log.exists(), o[-300:])
    rc, o = rent("env", FAKE_PYVER="3.11", FAKE_META=json.dumps({**json.loads(good_meta), "unbabel-comet": "2.2.6"}), **env_extra)
    # Must stop AT the pin guard: a later step failing in the fake environment must not
    # count (a mutation turning this die into '|| true' once went uncaught).
    check("env stops when an installed version differs from its pin (unbabel-comet 2.2.6)",
          rc != 0 and "NOT PINNED: unbabel-comet==2.2.7 (installed 2.2.6)" in o
          and "installed versions differ from the pins above" in o and "torch before:" not in o, o[-300:])
    pins = pip_log.read_text() if pip_log.exists() else ""
    check("env's pip install carries all four pins",
          all(x in pins for x in ("unbabel-comet==2.2.7", "pytorch-lightning==2.5.5", "transformers==4.57.1", "numpy==1.26.4")),
          pins[-300:])
    rc, o = rent("env", FAKE_PYVER="3.11", FAKE_META=good_meta, FAKE_CUDA_BAD=1, **env_extra)
    check("env stops with 'CUDA is not usable' when torch cannot reach the GPU after install",
          rc != 0 and "pins OK" in o and "CUDA is not usable" in o, o[-300:])
    rc, o = rent("env", FAKE_PYVER="3.11", FAKE_META=good_meta, **env_extra)
    check("env with good Python, pins and CUDA gets past the guards to the trainer patch step",
          "pins OK" in o and "torch before:" in o and "CUDA is not usable" not in o and "trainer patch" in o, o[-300:])
    (mt / "requirements.txt").unlink()

    # ---- PIN_SFT_REV older than the decisions commit (round 2)
    (sft / "phase0/e03_decisions.json").write_text(json.dumps(decisions(), indent=1))
    commit_decisions()
    rent("repos", MT_REV=c1, PIN_SFT_REV=sft_head)          # first run detaches and asks for a rerun
    rc, o = rent("repos", MT_REV=c1, PIN_SFT_REV=sft_head)
    check("repos with a PIN_SFT_REV that predates the pushed decisions commit warns (and still succeeds)",
          rc == 0 and "PIN_SFT_REV predates the decisions commit" in o, o[-300:])
    g("checkout", "-q", "wmt2027-phase0")
    g("rm", "-q", "phase0/e03_decisions.json"); g("commit", "-q", "-m", "fixture: drop decisions again")
    g("push", "-q", "origin", "HEAD")
    shutil.rmtree(mt / ".git"); (mt / "a.py").unlink()

    # ---- markers and data
    rc, o = rent("score")
    check("score refuses a corpus without .sha_verified", rc != 0 and "not verified" in o, o[-200:])
    rc, o = rent("data", DF_KB_OVERRIDE=1024)
    check("data stops when free disk is below the measured budget", rc != 0 and "needs >= 30 GB" in o, o[-200:])
    rc, o = rent("data", FAKE_REBUILD_FAIL=1)
    check("a rejected rebuild writes no .sha_verified marker",
          rc != 0 and not (mt / "data_enfr_v2/.sha_verified").exists(), o[-200:])
    rc, o = rent("data")
    mk = mt / "data_enfr_v2/.sha_verified"
    check("a passing rebuild writes v2 and v1.1 markers with the verified row counts (no download: size+sha match)",
          rc == 0 and mk.exists() and f"rows {N}" in mk.read_text() and logs["curl"].read_text() == ""
          and "rows 25" in (mt / "data_enfr_v1_pretrain/.sha_verified").read_text(), o[-300:])
    pq = work / "hf_wmt14/fr-en/train-00000-of-00001.parquet"
    pq.write_bytes(b"PAR1y")
    rc, o = rent("data")
    check("fetch_pinned: a same-size parquet with wrong bytes is downloaded again and then accepted",
          rc == 0 and pq.read_bytes() == toy_parquet and "train-00000-of-00001.parquet" in logs["curl"].read_text(), o[-300:])
    pq.write_bytes(b"PAR1y")
    rc, o = rent("data", FAKE_CURL_BAD=1)
    check("fetch_pinned: a parquet that stays wrong after re-download stops 'data', naming the file",
          rc != 0 and "fr-en/train-00000-of-00001.parquet: sha256" in o and "after re-download" in o, o[-300:])
    (work / "hf_wmt14/fr-en/train-00000-of-00001.parquet.bad").unlink(missing_ok=True)
    pq.write_bytes(b"PAR1xEXTRA")
    logs["curl"].write_text("")
    rc, o = rent("data")
    check("fetch_pinned: an oversized parquet is deleted and fetched fresh, never resumed",
          rc == 0 and pq.read_bytes() == toy_parquet and logs["curl"].read_text().startswith("existed=0"), o[-300:])
    logs["curl"].write_text("")
    with open(mt / "data_enfr_v2/train.clean.fr", "a") as f:
        f.write("extra\n")
    rc, o = rent("score")
    check("a corpus that changed after verification is refused", rc != 0 and "changed size since verification" in o, o[-200:])
    shutil.copyfile(toy2 / "train.clean.fr", mt / "data_enfr_v2/train.clean.fr")

    # ---- decisions gate
    rc, o = rent("score")
    check("score refuses without phase0/e03_decisions.json", rc != 0 and "DECISIONS MISSING" in o, o[-200:])
    shutil.copyfile(sft / "phase0/e03_decisions.example.json", dec_path)
    for stage in ("score", "controls"):
        rc, o = rent(stage)
        check(f"{stage} with the all-CHOOSE template lists every one of the 13 decisions",
              rc != 0 and o.count('still "CHOOSE"') == 13, o[-400:])
    set_decisions(commit=False, score_mode="none", pool="full")
    rc, o = rent("score")
    check("an inconsistent decision (score_mode none with pool full) is refused",
          rc != 0 and 'requires pool "reused"' in o, o[-200:])

    # ---- decisions must be committed, tagged and pushed (audit F7, F11)
    set_decisions(commit=False, score_mode="none")
    rc, o = rent("score")
    check("an untracked decisions file is refused", rc != 0 and "not tracked by git" in o, o[-200:])
    g("add", "phase0/e03_decisions.json"); g("commit", "-q", "-m", "untagged decisions")
    rc, o = rent("score")
    check("a committed but untagged decisions file is refused", rc != 0 and "no git tag contains" in o, o[-200:])
    ntag[0] += 1; g("tag", f"decisions-{ntag[0]}")
    rc, o = rent("score")
    check("a tagged but unpushed decisions commit is refused", rc != 0 and "on no remote branch" in o, o[-200:])
    g("push", "-q", "origin", "HEAD", "--tags")
    dec_path.write_text(dec_path.read_text().replace('"n_ft": 10', '"n_ft": 11'))
    rc, o = rent("score")
    check("a tracked decisions file with an uncommitted edit is refused", rc != 0 and "differs from its committed version" in o, o[-200:])
    make_bundle(tamper=True)
    rc, o = rent("score", DECISIONS_UNTAGGED=1)
    dev_txt = (res / "deviations.txt").read_text() if (res / "deviations.txt").exists() else ""
    check("DECISIONS_UNTAGGED=1 proceeds past the git check and records a deviation",
          "differ from SHA256SUMS" in o and "DECISIONS_UNTAGGED=1" in dev_txt, o[-300:])
    shutil.rmtree(sft / "results", ignore_errors=True)
    set_decisions(score_mode="none")

    # ---- bundle integrity
    rc, o = rent("score")
    check("a bundle member that differs from SHA256SUMS stops score",
          rc != 0 and "differ from SHA256SUMS" in o and not (mt / "data_enfr_v2/v2_scored.tsv").exists(), o[-200:])
    for side in ("src", "tgt"):
        s_ = make_bundle(plan_over={f"new_{side}_sha256": "0" * 64})
        rc, o = rent("score", EXPECTED_BUNDLE_SUMS_SHA=s_)
        check(f"a self-consistent bundle whose plan.json new_{side}_sha256 names another corpus stops score",
              rc != 0 and "made for another corpus" in o and not (mt / "data_enfr_v2/v2_scored.tsv").exists(), o[-200:])
    make_bundle(flip_label=True)
    rc, o = rent("score")
    check("a self-consistent bundle that is not the pinned one (other labels) is refused",
          rc != 0 and "not the pinned v2 bundle" in o, o[-200:])
    make_bundle()
    rc, o = rent("provenance")
    make_bundle(flip_label=True, keep_unpack=True)
    rc, o = rent("score")
    check("a different tarball copied over an existing unpack is unpacked again (and then refused as unpinned)",
          rc != 0 and "unpacking" in o and "not the pinned v2 bundle" in o, o[-300:])
    make_bundle()

    # ---- score none
    rc, o = rent("score")
    out = mt / "data_enfr_v2/v2_scored.tsv"
    done = mt / "data_enfr_v2/v2_scored.done"
    exp_none = []
    for i in range(N):
        s = old_score.get((en[i], fr[i]))
        exp_none.append(f"{np.float32(float(s)):.6f}\t{en[i]}\t{fr[i]}\n" if s is not None else f"nan\t{en[i]}\t{fr[i]}\n")
    got = out.read_text(encoding="utf-8") if out.exists() else ""
    check("score_mode none: reused old scores where the pair was scored, literal nan elsewhere, corpus order",
          rc == 0 and got == "".join(exp_none) and done.exists(), o[-300:])
    check("score_mode none used no GPU scorer", logs["scorer"].read_text() == "")

    # ---- changed scoring decisions after a completed score (audit F15) and HF login (F13)
    set_decisions(score_mode="reuse")
    done_before, out_before = done.read_bytes(), out.read_bytes()
    rc, o = rent("score")
    check("a completed score under other scoring decisions stops score (no SCORE_REDO) and deletes nothing",
          rc != 0 and "other scoring decisions" in o and done.read_bytes() == done_before and out.read_bytes() == out_before, o[-300:])
    rc, o = rent("score", SCORE_REDO=1, FAKE_HF_OUT="Not logged in")
    check("hf whoami printing 'Not logged in' with exit 0 stops score_mode reuse before extract",
          rc != 0 and "not logged in to Hugging Face" in o and not (work / "rescore/to_score.en").exists()
          and logs["scorer"].read_text() == "", o[-300:])
    shutil.rmtree(sft / "results", ignore_errors=True)

    # ---- score full: calibration against the reused old scores gates the result (audit F1)
    cal = {"max_mean_diff": 2e-4, "max_p99_diff": 2e-3, "min_spearman": 0.999, "min_jaccard": 0.99, "min_rows": 1}
    failed = mt / "data_enfr_v2/v2_scored.calibration_failed.tsv"
    sidecar = mt / "data_enfr_v2/v2_scored.calibration_failed.json"
    shards = mt / "data_enfr_v2/v2_scored.partial.tsv.shards"
    set_decisions(score_mode="full", calibration=cal)
    rc, o = rent("score", SCORE_REDO=1)
    side = json.load(open(sidecar)) if sidecar.exists() else {}
    check("score_mode full: new scores that disagree with the reused ones fail calibration and STOP "
          "(no v2_scored.tsv, no done marker, scores kept aside with a sidecar, shards removed)",
          rc != 0 and "calibration did not pass" in o and not out.exists() and not done.exists()
          and failed.exists() and side.get("verdict") == "fail" and side.get("tsv_sha256") == sha(failed)
          and not shards.exists(), o[-400:])
    table = d / "score_table.json"
    table.write_text(json.dumps({f"{s}\t{t}": sc for (s, t), sc in old_score.items()}))
    report_in_bundle = work / "rescore/provenance_report.json"
    rc, o = rent("score", FAKE_SCORE_TABLE=table, FAKE_CORRUPT_REPORT=report_in_bundle)
    check("calibrate exit 3 (unreadable input) keeps the verified shards and the partial TSV, renames nothing",
          rc != 0 and "could not run" in o and shards.exists() and (mt / "data_enfr_v2/v2_scored.partial.tsv").exists()
          and not out.exists(), o[-400:])
    shutil.copyfile(plan / "provenance_report.json", report_in_bundle)
    n0 = nlines(logs["scorer"])
    rc, o = rent("score", FAKE_SCORE_TABLE=table)
    exp_full = "".join(f"{old_score.get((en[i], fr[i])) or format(fscore(en[i], fr[i]), '.6f')}\t{en[i]}\t{fr[i]}\n"
                       for i in range(N))
    got = out.read_text(encoding="utf-8") if out.exists() else ""
    rep_ = json.load(open(mt / "data_enfr_v2/calibration_report.json")) if rc == 0 else {}
    check("after fixing the input, score completes from the kept shards with 0 scorer invocations; calibration passes",
          rc == 0 and got == exp_full and rep_.get("verdict") == "pass" and nlines(logs["scorer"]) == n0
          and not shards.exists(), f"rc={rc} scorer calls {n0}->{nlines(logs['scorer'])} {o[-400:]}")

    # full mode: shards whose metas record different stacks stop before calibration (round 2)
    done.unlink(missing_ok=True)
    rc, o = rent("score", FAKE_SCORE_TABLE=table, FAKE_META_TORCH_DEV="1", FAKE_META_TORCH="9.9")
    check("score_mode full: shard 1 scored under another stack than shard 0 -> STOP before calibration, scores and shards kept",
          rc != 0 and "full-mode scores mix scoring stacks" in o and "calibration:" not in o
          and (mt / "data_enfr_v2/v2_scored.partial.tsv").exists() and shards.exists() and not done.exists(), o[-400:])
    shutil.rmtree(shards, ignore_errors=True)
    (mt / "data_enfr_v2/v2_scored.partial.tsv").unlink(missing_ok=True)

    set_decisions(score_mode="full", calibration={**cal, "min_rows": 100000})
    rc, o = rent("score", SCORE_REDO=1, FAKE_SCORE_TABLE=table)
    side = json.load(open(sidecar)) if sidecar.exists() else {}
    check("an 'insufficient' calibration (exit 2) also stops and keeps the scores with a sidecar",
          rc != 0 and "calibration did not pass (exit 2" in o and side.get("verdict") == "insufficient", o[-300:])
    n1 = nlines(logs["scorer"])
    rc, o = rent("score", RECALIBRATE=1)
    check("RECALIBRATE=1 with unchanged thresholds is refused", rc != 0 and "unchanged" in o and failed.exists(), o[-300:])
    set_decisions(score_mode="full", calibration=cal)
    kept = failed.read_bytes()
    failed.write_bytes(kept + b"0.5\tx\ty\n")
    rc, o = rent("score", RECALIBRATE=1)
    check("RECALIBRATE=1 refuses kept scores whose sha256 differs from the sidecar",
          rc != 0 and "the kept scores changed" in o and not out.exists(), o[-300:])
    failed.write_bytes(kept)
    rc, o = rent("score", RECALIBRATE=1)
    dev_txt = (res / "deviations.txt").read_text() if (res / "deviations.txt").exists() else ""
    got = out.read_text(encoding="utf-8") if out.exists() else ""
    check("RECALIBRATE=1 after a threshold change passes with 0 scorer invocations and records the deviation",
          rc == 0 and got == exp_full and nlines(logs["scorer"]) == n1 and "RECALIBRATE" in dev_txt
          and done.exists() and not failed.exists(), f"rc={rc} scorer calls {n1}->{nlines(logs['scorer'])} {o[-400:]}")
    shutil.rmtree(sft / "results", ignore_errors=True)

    # ---- score reuse (+ SCORE_SMOKE on 2 fake GPUs)
    set_decisions(score_mode="reuse")
    rc, o = rent("score", SCORE_SMOKE=1, SCORE_REDO=1)
    exp_reuse = []
    for i in range(N):
        s = old_score.get((en[i], fr[i]))
        v = float(s) if s is not None else float(f"{fscore(en[i], fr[i]):.6f}")
        exp_reuse.append(f"{np.float32(v):.6f}\t{en[i]}\t{fr[i]}\n")
    got = out.read_text(encoding="utf-8") if out.exists() else ""
    devs = {ln.split()[0] for ln in logs["scorer"].read_text().splitlines()}
    check("score_mode reuse: new pairs scored sharded, merged with reused scores in corpus order",
          rc == 0 and got == "".join(exp_reuse), o[-400:])
    shard_files = sorted((work / "rescore/new_scores.tsv.shards").glob("shard.*.tsv")) \
        if (work / "rescore/new_scores.tsv.shards").exists() else []
    check("SCORE_SMOKE ran the 2-GPU order check before the full run", rc == 0 and "smoke OK" in o and {"0", "1"} <= devs, o[-300:])
    smeta = json.load(open(mt / "data_enfr_v2/v2_scored.meta.json")) if (mt / "data_enfr_v2/v2_scored.meta.json").exists() else {}
    # the stage returns early when a completed scored file exists for these scoring
    # decisions, so clear it: what is under test here is the split, not the marker
    set_decisions(score_mode="reuse")
    done.unlink(missing_ok=True); out.unlink(missing_ok=True)
    shutil.rmtree(work / "rescore/new_scores.tsv.shards", ignore_errors=True)
    (work / "rescore/new_scores.tsv").unlink(missing_ok=True)
    rc, o = rent("score", SCORE_REDO=1, SCORE_SHARDS=3, FAKE_NGPU=1)
    got_s = out.read_text(encoding="utf-8") if out.exists() else ""
    check("SCORE_SHARDS=3 on one GPU splits into 3 shards and merges to the same scores",
          rc == 0 and "into 3 shards" in o and got_s == "".join(exp_reuse), f"rc={rc} {o[-300:]}")
    # (changing --shards on a work dir that still exists is refused by score_sharded itself;
    # tests/suites/scoring.py covers that. Here the reuse path deletes the shards after a
    # successful merge, so the next run legitimately starts from a fresh split.)
    shutil.rmtree(work / "rescore/new_scores.tsv.shards", ignore_errors=True)
    (work / "rescore/new_scores.tsv").unlink(missing_ok=True)
    done.unlink(missing_ok=True); out.unlink(missing_ok=True)
    set_decisions(score_mode="reuse")
    rc, o = rent("score", SCORE_REDO=1)
    check("reuse cleans up to_score.* and shards and writes scoring metadata",
          not (work / "rescore/to_score.en").exists() and not (work / "rescore/new_scores.tsv.shards").exists()
          and smeta.get("mode") == "reuse")
    check("SCORE_SMOKE's measured max and p99 |sharded - single| are recorded in v2_scored.meta.json",
          isinstance(smeta.get("smoke"), dict) and "p99_abs_diff" in smeta["smoke"] and "max_abs_diff" in smeta["smoke"],
          str(smeta.get("smoke"))[:200])
    n2 = nlines(logs["scorer"])
    set_decisions(score_mode="reuse", n_ft=11)
    rc, o = rent("score")
    check("editing a non-scoring decision (n_ft) after score reuses the completed scores (no scorer call)",
          rc == 0 and "already complete" in o and nlines(logs["scorer"]) == n2, o[-300:])
    set_decisions(score_mode="reuse")

    # ---- a scorer refusing a stack change (exit 3): distinct message, knob, deviation (round 2)
    done.unlink()
    rc, o = rent("score", FAKE_STACK_REFUSE=1)
    check("a scorer stack refusal stops 'score' naming the recovery options, without 'rerun to resume'",
          rc != 0 and "SCORE_ALLOW_STACK_CHANGE=1" in o and "REFUSED a scoring-stack change" in o
          and "rerun 'score' to resume" not in o and not done.exists(), o[-400:])
    rc, o = rent("score", FAKE_STACK_REFUSE=1, SCORE_ALLOW_STACK_CHANGE=1)
    dev_txt = (res / "deviations.txt").read_text() if (res / "deviations.txt").exists() else ""
    got = out.read_text(encoding="utf-8") if out.exists() else ""
    check("SCORE_ALLOW_STACK_CHANGE=1 passes --allow-stack-change, records a deviation, completes, and the reuse "
          "within-shard warning is reached", rc == 0 and got == "".join(exp_reuse) and "SCORE_ALLOW_STACK_CHANGE=1" in dev_txt
          and "resumed under a different scoring stack" in o, o[-400:])
    done.unlink()
    rc, o = rent("score", FAKE_META_TORCH_DEV="1", FAKE_META_TORCH="9.9")
    check("score_mode reuse (rescored) with a between-shard stack difference WARNs and completes",
          rc == 0 and "new scores do not come from one scoring stack" in o and done.exists(), o[-300:])

    # ---- provenance from the bundle, then controls
    rc, o = rent("provenance")
    check("provenance copies SHA256SUMS-verified bundle labels", rc == 0 and (sft / "phase0/provenance_labels.npy").exists(), o[-200:])
    done_bytes = done.read_bytes()
    for label, content in (("a marker for another score_mode/sha", "none " + "0" * 64 + "\n"), ("no marker", None)):
        if content is None:
            done.unlink()
        else:
            done.write_text(content)
        rc, o = rent("controls")
        check(f"controls refuses with {label} (run 'score' first)",
              rc != 0 and "run 'score' first" in o and not (sft / "data/phase0").exists(), o[-200:])
    done.write_bytes(done_bytes)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:-1]), encoding="utf-8")
    rc, o = rent("controls")
    check("controls refuses when scored rows != corpus rows != label rows",
          rc != 0 and "row counts disagree" in o and not (sft / "data/phase0").exists(), o[-300:])
    out.write_text("".join(lines), encoding="utf-8")
    low = d / "meminfo_low"; low.write_text("MemAvailable: 1048576 kB\n")
    rc, o = rent("controls", MEMINFO=low)
    check("controls refuses below the RAM guard", rc != 0 and "MemAvailable" in o, o[-200:])
    GiB = 1024 ** 3
    cg2 = d / "cg2"; cg2.mkdir()
    (cg2 / "memory.max").write_text(f"{8 * GiB}\n"); (cg2 / "memory.current").write_text(f"{GiB}\n")
    (cg2 / "memory.high").write_text("max\n"); (cg2 / "memory.stat").write_text("anon 1\ninactive_file 0\n")
    rc, o = rent("controls", CGROUP_MEM_DIR=cg2)
    check("controls refuses when MemAvailable is high (64 GiB) but the cgroup v2 memory.max leaves 7 GiB",
          rc != 0 and "cgroup limit" in o and "memory.max" in o, o[-300:])
    cg1 = d / "cg1"; (cg1 / "memory").mkdir(parents=True)
    (cg1 / "memory/memory.limit_in_bytes").write_text(f"{4 * GiB}\n"); (cg1 / "memory/memory.usage_in_bytes").write_text("0\n")
    rc, o = rent("controls", CGROUP_MEM_DIR=cg1)
    check("controls refuses under a low cgroup v1 memory.limit_in_bytes", rc != 0 and "cgroup limit" in o, o[-300:])
    cgc = d / "cgcache"; cgc.mkdir()
    (cgc / "memory.max").write_text(f"{40 * GiB}\n"); (cgc / "memory.current").write_text(f"{39 * GiB}\n")
    (cgc / "memory.stat").write_text(f"inactive_file {30 * GiB}\n")
    cgmax = d / "cgmax"; cgmax.mkdir()
    (cgmax / "memory.max").write_text("max\n"); (cgmax / "memory.current").write_text(f"{GiB}\n")
    rc, o = rent("controls", FAKE_CTRL_DROP="heldout_europarl.fr")
    check("controls stops when a held-out file the decisions require was not built",
          rc != 0 and "heldout_europarl.fr missing or empty" in o and not (sft / "configs/phase0/matrix.json").exists(), o[-300:])
    rc_cc, o_cc = rent("controls", CGROUP_MEM_DIR=cgc, FAKE_CTRL_DROP="heldout_europarl.fr")
    check("inactive file cache charged to the cgroup does not count against the guard (40 GiB max, 39 used, 30 cache)",
          rc_cc != 0 and "cgroup limit" not in o_cc and "heldout_europarl.fr missing" in o_cc, o_cc[-300:])
    rc, o = rent("controls", CGROUP_MEM_DIR=cgmax)
    check("controls passes the RAM guard when cgroup v2 memory.max is 'max'", rc == 0 and "needs >=" not in o, o[-300:])
    args = json.loads(logs["ctrl_args"].read_text() or "{}")
    check("controls passes the decisions to the builder (pool mask, domains, n_ft, pretraining corpus)",
          rc == 0 and args.get("pool_mask", "").endswith("pool_mask_reused.npy") and args.get("heldout_domains") == "un,europarl"
          and args.get("n_ft") == "10" and args.get("pretrain_src", "").endswith("data_enfr_v1_pretrain/train.clean.en"), o[-300:])
    rc, o = rent("controls", FAKE_CTRL_DROP="heldout_europarl.fr")
    check("a controls rebuild that fails part-way removes the old decisions record and matrix (stage1 would refuse)",
          rc != 0 and not (res / "decisions.sha256").exists() and not (sft / "configs/phase0/matrix.json").exists(), o[-300:])
    rc, o = rent("controls")

    # ---- fake trainer + accept (audit F17)
    (mt / "src/data").mkdir(parents=True)
    (mt / "src/__init__.py").write_text(""); (mt / "src/data/__init__.py").write_text("")
    (mt / "src/data/dataset.py").write_text(FAKE_DATASET)
    (mt / "src/data/tokenizer.py").write_text("class Tokenizer:\n    def __init__(self, p):\n        self.p = p\n")
    (mt / "train.py").write_text(FAKE_TRAIN)
    (mt / "scripts").mkdir(exist_ok=True); (mt / "scripts/eval_bleu.py").write_text(FAKE_EVAL)
    (mt / "data_enfr_v1").mkdir(exist_ok=True)
    for n_ in ("valid.en", "valid.fr", "test.en", "test.fr", "spm.model"):
        (mt / "data_enfr_v1" / n_).write_text("x\n")
    (mt / "ckpt_hf").mkdir(exist_ok=True); ctx.write_ckpt(mt / "ckpt_hf/enfr_base_v1.1_averaged.pt")
    avb = work / "accept_valid_bleu.txt"
    rc, o = rent("accept", FAKE_BLEU="35.31")
    check("accept with matching valid/test downloads nothing, reproduces BLEU and writes accept_valid_bleu.txt",
          rc == 0 and "ACCEPTED" in o and logs["curl"].read_text() == "" and avb.exists()
          and avb.read_text().strip() == "35.31", o[-300:])
    (mt / "data_enfr_v1/test.fr").write_text("corrupt\n")
    rc, o = rent("accept", FAKE_BLEU="35.31")
    check("accept re-fetches the pinned dev/test parquet and regenerates text when test.fr's sha256 is wrong",
          rc == 0 and "test-00000-of-00001.parquet" in logs["curl"].read_text()
          and "test-00000-of-00001.parquet" in logs["devtest"].read_text()
          and (mt / "data_enfr_v1/test.fr").read_text() == "x\n", o[-300:])
    rc, o = rent("accept", FAKE_BLEU="30.00")
    check("accept stops when test BLEU is outside the tolerance and writes no accept_valid_bleu.txt",
          rc != 0 and "Do NOT run E0.3" in o and not avb.exists(), o[-300:])
    rc, o = rent("accept", FAKE_NO_BLEU=1)
    check("accept stops when eval prints no BLEU line", rc != 0 and "no BLEU line" in o, o[-200:])
    avb.unlink(missing_ok=True)
    rc, o = rent("accept", FAKE_EVAL_RC=3)
    check("accept stops with a STOP line when eval_bleu.py itself fails (exit 3) and writes no accept_valid_bleu.txt",
          rc != 0 and "STOP" in o and "exited 3" in o and "Do NOT run E0.3" in o and not avb.exists(), o[-300:])
    rc, o = rent("accept", FAKE_BLEU="35.31", HF_BASE_SPM_SHA="0" * 64)
    check("accept stops when the copied spm differs from the release sha256", rc != 0 and "copied spm differs" in o, o[-200:])
    avb.unlink(missing_ok=True)

    # ---- flat-vs-baseline precondition before any GPU run (F17) and the avg-last5 stage2 disk guard (F22)
    set_decisions(score_mode="reuse", lr_selection_rule="flat-vs-baseline", lr_tolerance_bleu=0.1)
    rc_c, o_c = rent("controls")
    rc, o = rent("stage1")
    check("stage1 with flat-vs-baseline and no accept_valid_bleu.txt stops before any training run",
          rc_c == 0 and rc != 0 and "rerun 'accept'" in o and logs["build"].read_text() == ""
          and not list((mt / "checkpoints").rglob("final.pt")), o_c[-200:] + o[-300:])
    set_decisions(score_mode="reuse", ft_checkpoint="avg-last5")
    rc_c, o_c = rent("controls")
    rc, o = rent("stage2", "0.5", DF_KB_OVERRIDE=45 * 1024 * 1024)
    check("stage2 under avg-last5 requires 55 GB free (45 GB is refused)", rc_c == 0 and rc != 0 and "needs >= 55 GB" in o, o[-300:])
    set_decisions(score_mode="reuse")
    rc, o = rent("controls")
    check("controls records manifest, matrix, decisions sha and history before stage1, and tees its log",
          rc == 0 and (res / "controls_manifest.json").read_bytes() == (sft / "data/phase0/manifest.json").read_bytes()
          and (res / "decisions.sha256").read_text().strip() == sha(dec_path)
          and (res / f"decisions.{sha(dec_path)[:12]}.json").exists()
          and (res / "decisions_history.tsv").read_text().splitlines()[-1].split("\t")[1] == sha(dec_path)
          and json.load(open(res / "matrix.json")).get("controls_sha")
          and "fake builder wrote" in (sft / "logs/phase0/controls_build.log").read_text(), o[-300:])
    kl = json.load(open(sft / "configs/phase0/ft_topk_lr1.yaml"))["checkpoint"]["keep_last"] if rc == 0 else None
    check("ft_checkpoint final -> keep_last 1 in every generated config", kl == 1, str(kl))

    # ---- stage1/stage2/gate
    dec_bytes = dec_path.read_bytes()
    dec_path.write_text(json.dumps(decisions(score_mode="reuse", indomain=["heldout_un", "heldout_europarl"]), indent=1))
    rc, o = rent("stage1")
    check("stage1 refuses decisions changed after controls", rc != 0 and "changed after 'controls'" in o, o[-200:])
    dec_path.write_bytes(dec_bytes)
    rc, o = rent("stage1")
    sel = json.load(open(res / "lr_selection.json")) if (res / "lr_selection.json").exists() else {}
    built = logs["build"].read_text().splitlines()
    check("stage1 warms every tokenisation cache before launching (fake trainer exits 7 otherwise)",
          rc == 0 and len(built) == 2 and "tokenisation cache missing" not in o, o[-400:])
    check("stage1 selects mechanically: lr 1 declines strictly -> the frozen rule picks 0.5",
          sel.get("selected") == "0.5" and sel.get("rule") == "no-strict-monotone-decline-from-first-eval", str(sel)[:300])
    rc, o = rent("stage2", "0.15")
    check("stage2 refuses an lr_scale other than the selected one", rc != 0 and "refusing 0.15" in o, o[-200:])
    rc, o = rent("stage2", "0.5")
    finals = list((mt / "checkpoints/phase0").rglob("final.pt"))
    check("stage2 runs the 9 runs at the selected lr_scale", rc == 0 and len(finals) == 13, f"rc={rc} finals={len(finals)} {o[-300:]}")
    rc, o = rent("gate", "0.15")
    check("gate refuses an lr_scale that is not the stage-1 selection and has no recorded override",
          rc != 0 and "not the stage-1 selection" in o, o[-200:])
    rc, o = rent("gate", "0.5")
    tsv = res / "phase0_bleu.tsv"
    gmeta = json.load(open(tsv.with_suffix(".meta.json"))) if tsv.with_suffix(".meta.json").exists() else {}
    check("gate collects with the decided checkpoint type and applies e03_decide with the decided --indomain",
          rc in (0, 1) and tsv.exists() and gmeta.get("ft_ckpt") == "final"
          and "(2) in-domain heldout_un" in o and "heldout_europarl" in o and "[aux] heldout_europarl" in o, o[-500:])
    check("gate meta records lr_scale, the stage-1 selection and the decisions sha; gate prints the deviations record",
          gmeta.get("lr_scale") == "0.5" and gmeta.get("selected_lr") == "0.5" and gmeta.get("decisions_sha") == sha(dec_path)
          and "deviations" in o, str(gmeta)[:300])
    victim = sorted((mt / "checkpoints/phase0").rglob("ft_random_lr0.5_s1_st2/final.pt"))
    if victim:
        victim[0].rename(victim[0].with_name("final.pt.off"))
    rc, o = rent("gate", "0.5")
    check("a failed collection leaves no canonical phase0_bleu.tsv or meta from the earlier collection",
          rc != 0 and not tsv.exists() and not tsv.with_suffix(".meta.json").exists(), o[-300:])
    if victim:
        victim[0].with_name("final.pt.off").rename(victim[0])

    # ---- decisions changed after results exist (audit F6)
    hist_rows = nlines(res / "decisions_history.tsv")
    sha_before = (res / "decisions.sha256").read_text()
    set_decisions(score_mode="reuse", indomain=["heldout_un", "heldout_europarl"])
    for stage in ("score", "controls"):
        rc, o = rent(stage)
        check(f"{stage} refuses a decisions change after stage1 results exist (no DECISIONS_AMEND)",
              rc != 0 and "pre-registration deviation" in o and "indomain" in o
              and (res / "decisions.sha256").read_text() == sha_before and done.read_bytes() == done_bytes, o[-300:])
    rc1, o1 = rent("score", DECISIONS_AMEND=1)
    rc2, o2 = rent("controls", DECISIONS_AMEND=1)
    dev_txt = (res / "deviations.txt").read_text() if (res / "deviations.txt").exists() else ""
    check("DECISIONS_AMEND=1 lets score and controls proceed, records the deviation naming the changed key, adds a history row",
          rc1 == 0 and rc2 == 0 and "decisions amended" in dev_txt and "indomain" in dev_txt
          and nlines(res / "decisions_history.tsv") == hist_rows + 1, o1[-200:] + o2[-300:])
    rc, o = rent("gate", "0.5")
    check("gate prints the recorded deviation before the verdict", "decisions amended" in o and "VERDICT" in o, o[-400:])
