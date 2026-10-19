"""Re-decode E0.5's kept checkpoints and keep the hypotheses, reproducing the recorded BLEU.

E0.5 scored every run with Machine_translation's scripts/eval_bleu.py (beam 5, length penalty 1.0)
through e03_collect.py, but kept only the BLEU numbers (results/.e03_collect_cache.json), not the
outputs. Token-level analyses (WMT2027_PLAN.md §3.2 item 6) need the outputs. This script decodes
every (checkpoint, test set) pair of that cache again and refuses anything it cannot pin:

  - Machine_translation must be at E0.5's revision, and scripts/eval_bleu.py must have the cache's sha;
  - each source and reference file must have the cache's src_sha / ref_sha;
  - each fine-tune config must have the cache's config_sha (found by hash in --configs);
  - each fine-tune checkpoint must have the sha recorded in E0.5's ckpt.sha.

Every re-decoded BLEU is compared with the cached one; the report says which reproduce exactly and
the largest difference. Hypotheses are written one per source line, with their sha256.

  python phase0/e05_redecode.py --mt <Machine_translation at 200f6c0> --cache <.e03_collect_cache.json> \
      --ckpt-root <dir holding checkpoints/phase0/...> --ckpt-sha <ckpt.sha> --configs <configs/phase0_e05> \
      --filtered <results/filtered_tests> --baseline-ckpt <averaged v1.1 .pt> --out <dir>
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MT_REV = "200f6c0624c0c1f810e639dde3c53545f359849e"
BLEU_RE = re.compile(r"BLEU \([a-z]+, sacrebleu 13a\): ([0-9.]+)")   # as e03_collect.py


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def die(msg: str):
    sys.exit(f"REFUSED: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--ckpt-sha", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--filtered", required=True)
    ap.add_argument("--baseline-ckpt", required=True,
                    help="averaged v1.1 base; tensor-identical to E0.5's ckpt_hf copy, so it is checked by its BLEU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default=None, help="substring filter on run names (for a smoke test)")
    a = ap.parse_args()

    mt = Path(a.mt).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rev = subprocess.run(["git", "-C", str(mt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if rev != MT_REV:
        die(f"Machine_translation is at {rev or '?'}, E0.5 used {MT_REV}")

    cache = {json.loads(k)["ckpt"] + "|" + json.loads(k)["testset"]: (json.loads(k), v)
             for k, v in json.load(open(a.cache)).items()}
    evals = {k["eval_sha"] for k, _ in cache.values()}
    if len(evals) != 1 or sha(mt / "scripts/eval_bleu.py") != evals.pop():
        die("scripts/eval_bleu.py differs from the one E0.5 used")
    if {k["beam"] for k, _ in cache.values()} != {"5"} or {k["lp"] for k, _ in cache.values()} != {"1.0"}:
        die("the cache holds settings other than beam 5 / length penalty 1.0")

    cfg_by_sha = {sha(p): p.resolve() for p in list(Path(a.configs).glob("*.yaml")) + list((mt / "configs").glob("*.yaml"))}
    ckpt_sha = {}
    for line in open(a.ckpt_sha):
        h, rel = line.split(None, 1)
        ckpt_sha[rel.strip()] = h
    tests = {"newstest2014": (mt / "data_enfr_v1/test.en", mt / "data_enfr_v1/test.fr"),
             "heldout_un": (Path(a.filtered).resolve() / "heldout_un.en", Path(a.filtered).resolve() / "heldout_un.fr")}
    test_sha = {n: (sha(s), sha(r)) for n, (s, r) in tests.items()}

    jobs, checked = [], {}
    for key, (k, cached) in sorted(cache.items()):
        ts = k["testset"]
        if test_sha[ts] != (k["src_sha"], k["ref_sha"]):
            die(f"{ts}: source/reference sha differs from the cache")
        cfg = cfg_by_sha.get(k["config_sha"])
        if cfg is None:
            die(f"no config with sha {k['config_sha'][:12]} (for {k['ckpt']})")
        rel = None
        if "/ckpt_hf/" in k["ckpt"]:
            name, ckpt = "baseline_v1.1_averaged", Path(a.baseline_ckpt).resolve()
        else:
            rel = k["ckpt"].split("/Machine_translation/", 1)[1]          # checkpoints/phase0/<sha>/<run>/final.pt
            name, ckpt = Path(rel).parent.name, Path(a.ckpt_root).resolve() / rel
        if a.only and a.only not in name:
            continue
        if rel is not None:
            if rel not in checked:
                checked[rel] = ckpt.is_file() and ckpt_sha.get(rel) == sha(ckpt)
            if not checked[rel]:
                die(f"{rel}: missing, or its sha differs from E0.5's ckpt.sha")
        jobs.append((name, ts, ckpt, cfg, cached))

    rows = []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "PYTHONPATH": str(mt)}
    for i, (name, ts, ckpt, cfg, cached) in enumerate(jobs, 1):
        hyp = out / f"{name}.{ts}.hyp"
        src, ref = tests[ts]
        p = subprocess.run([sys.executable, "scripts/eval_bleu.py", "--ckpt", str(ckpt), "--config", str(cfg),
                            "--src", str(src), "--ref", str(ref), "--beam", "5", "--length-penalty", "1.0",
                            "--out", str(hyp)], cwd=mt, capture_output=True, text=True, env=env)
        m = BLEU_RE.search(p.stdout + p.stderr)
        if p.returncode != 0 or not m:
            die(f"{name} {ts}: eval failed\n{(p.stdout + p.stderr)[-600:]}")
        bleu = float(m.group(1))
        n_src = sum(1 for _ in open(src, encoding="utf-8", newline="\n"))
        n_hyp = sum(1 for _ in open(hyp, encoding="utf-8", newline="\n"))
        if n_src != n_hyp:
            die(f"{name} {ts}: {n_hyp} hypotheses for {n_src} source lines")
        rows.append({"run": name, "testset": ts, "cached_bleu": cached, "bleu": bleu,
                     "diff": round(bleu - cached, 2), "hyp": hyp.name, "hyp_sha256": sha(hyp)})
        print(f"[{i}/{len(jobs)}] {name:32s} {ts:13s} cached {cached:6.2f} now {bleu:6.2f}", flush=True)

    exact = sum(r["diff"] == 0 for r in rows)
    worst = max((abs(r["diff"]) for r in rows), default=0.0)
    report = {"mt_rev": MT_REV, "n": len(rows), "exact": exact, "max_abs_diff": worst, "rows": rows}
    tmp = out / "redecode_report.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out / "redecode_report.json")
    print(f"\n{exact} of {len(rows)} reproduce the recorded BLEU exactly; largest difference {worst:.2f}")
    sys.exit(0 if exact == len(rows) else 2)


if __name__ == "__main__":
    main()
