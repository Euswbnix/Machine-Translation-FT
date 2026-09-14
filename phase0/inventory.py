#!/usr/bin/env python3
"""Read-only inventory of the training machine, run BEFORE copying anything.

Answers three questions in one pass, so nothing is fetched on a guessed path:
  1. Where are the irreplaceable artifacts (training logs, checkpoints)?
  2. Where are the expensive-to-rebuild ones (QE scores = ~14 GPU-hours,
     tokenizer caches, the 30M cleaned corpus)?
  3. Is the code that ACTUALLY ran here the same as what is on GitHub?

It never modifies anything, never prints file contents except log lines that
match known trainer print statements (step counts and learning rates only), and
skips anything that looks like a credential.

USAGE (on the Linux box):
    python3 inventory.py                    # scans $HOME
    python3 inventory.py /data /mnt/ssd     # add more roots
    python3 inventory.py --inspect-ckpt     # also read keys from averaged/best .pt
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time

# (priority, label, glob patterns)
CLASSES = [
    ("P0", "training log / report", ["*.log", "nohup.out", "*.out", "training_report*.txt",
                                     "report.txt", "*train*.txt"]),
    ("P0", "checkpoint (averaged/best/final)", ["averaged*.pt", "best*.pt", "final*.pt",
                                                "interrupted_step_*.pt"]),
    ("P0", "dev/eval trace", ["dev_ce_trace.tsv", "*bleu*.tsv", "*bleu*.json",
                              "*comet*.json", "*results*.json"]),
    ("P1", "QE scores (≈14 GPU-h to rebuild)", ["*scored*.tsv", "*scores*.tsv", "*kiwi*.tsv"]),
    ("P1", "tokenizer cache", ["*.cached_*.npz"]),
    ("P1", "SentencePiece model", ["spm*.model", "*.spm.model", "sentencepiece.model"]),
    ("P1", "FT / cleaned corpus", ["sft_train.*", "train.en", "train.fr", "train.de",
                                   "*clean*.en", "*clean*.fr", "*clean*.de"]),
    ("P1", "provenance / phase0 output", ["provenance_*.npy", "provenance_*.json", "e0*_*.json"]),
    ("P2", "dev/test sets", ["valid.en", "valid.fr", "valid.de", "test.en", "test.fr",
                             "test.de", "newstest*"]),
    ("P2", "step checkpoint (rotating)", ["step_*.pt"]),
    ("P2", "config", ["*.yaml"]),
]
SKIP_DIRS = {".git", "__pycache__", "node_modules", "site-packages", ".vscode-server",
             "snap", ".npm", ".cargo", ".rustup", "pkgs", ".conda", "miniconda3",
             "anaconda3", ".local", ".mozilla", ".thunderbird", "Trash", ".Trash"}
SECRET = re.compile(r"(^\.env|id_rsa|id_ed25519|\.pem$|\.key$|credential|secret|token\.json|"
                    r"\.netrc|\.pgpass|authorized_keys|known_hosts)", re.I)
# Credentials hide in VALUES, not only in files: a git remote like
# https://user:github_pat_...@github.com/... puts a live token into
# `git remote get-url`. The first version printed that verbatim and wrote it to
# inventory.json. Redact userinfo and known token shapes everywhere a value is
# reported, and FLAG it, so the owner knows to rotate rather than never finding out.
CRED_URL = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)
TOKEN = re.compile(r"\b(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                   r"hf_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})")


def has_credential(text) -> bool:
    return bool(text) and bool(CRED_URL.search(text) or TOKEN.search(text))


def redact(text):
    if not text:
        return text
    return TOKEN.sub("<redacted-token>", CRED_URL.sub(r"\1<redacted>@", text))


LOG_LINES = re.compile(r"(Optimizer/scheduler RESET.*|Resumed from step.*|Starting training for.*|"
                       r"Stop reason:.*|Total steps:.*|Best BLEU:.*)")
MAX_LOG_BYTES = 512 * 1024 * 1024


def human(n: float) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}P"


def classify(name: str):
    for prio, label, pats in CLASSES:
        if any(fnmatch.fnmatch(name, p) for p in pats):
            return prio, label
    return None


def walk(root: str, max_depth: int):
    root = os.path.abspath(os.path.expanduser(root))
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        depth = dirpath.count(os.sep) - base
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not SECRET.search(d)]
        if depth >= max_depth:
            dirnames[:] = []
        for fn in filenames:
            if SECRET.search(fn):
                continue
            yield dirpath, fn


def git_state(path: str):
    """HEAD, remote, dirty count and unpushed count -- or an explicit error.

    A failed git call must never be reported as a clean repo. The first version
    computed dirty_files as len("".splitlines()) == 0 when git itself errored,
    i.e. it printed "0 uncommitted files" for a repo it could not read at all.
    """
    errors = []

    def g(*a):
        try:
            r = subprocess.run(["git", "-C", path, *a], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"git {a[0]}: {type(e).__name__}")
            return None
        if r.returncode != 0:
            errors.append(f"git {a[0]}: {(r.stderr.strip().splitlines() or ['rc=%d' % r.returncode])[0][:120]}")
            return None
        return r.stdout.strip()

    head = g("log", "--oneline", "-1")
    remote_raw = g("remote", "get-url", "origin")
    status = g("status", "--porcelain")
    unpushed = g("rev-list", "--count", "@{u}..HEAD")   # fails legitimately with no upstream
    return {
        "head": redact(head),
        "remote": redact(remote_raw),
        "remote_embeds_credential": has_credential(remote_raw),
        "dirty_files": None if status is None else len(status.splitlines()),
        "unpushed_commits": unpushed,
        "errors": [redact(e) for e in errors] or None,
    }


def inspect_ckpt(path: str):
    try:
        import torch
    except ImportError:
        return {"error": "torch not importable"}
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                        # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    if not isinstance(ck, dict):
        return {"type": type(ck).__name__}
    tr = (ck.get("config") or {}).get("training", {}) if isinstance(ck.get("config"), dict) else {}
    hist = ck.get("history") or {}
    return {
        "keys": sorted(k for k in ck.keys())[:20],
        "global_step": ck.get("global_step"),
        "best_bleu": ck.get("best_bleu"),
        "has_model_key": "model" in ck,
        "has_optimizer": "optimizer" in ck,
        "has_token_counters": "applied_target_tokens" in ck,
        "accumulate_steps": tr.get("accumulate_steps"),
        "batch_size": tr.get("batch_size"),
        "max_sentences": tr.get("max_sentences"),
        "lr_scale": tr.get("lr_scale"),
        "warmup_steps": tr.get("warmup_steps"),
        "history_points": {k: len(v) for k, v in hist.items()} if isinstance(hist, dict) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["~"])
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--inspect-ckpt", action="store_true")
    ap.add_argument("--json-out", default="inventory.json")
    args = ap.parse_args()

    found, repos, log_hits = [], {}, []
    t0 = time.time()
    for root in args.roots:
        for dirpath, fn in walk(root, args.max_depth):
            full = os.path.join(dirpath, fn)
            if fn == "HEAD" and dirpath.endswith(os.sep + ".git"):
                continue
            c = classify(fn)
            if not c:
                continue
            try:
                st = os.stat(full)
            except OSError:
                continue
            found.append({"prio": c[0], "kind": c[1], "path": full, "size": st.st_size,
                          "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
        for dirpath, dirnames, _ in os.walk(os.path.expanduser(root)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS - {".git"}]
            if ".git" in dirnames and os.path.basename(dirpath) in (
                    "Machine_translation", "Machine-Translation-SFT"):
                repos[dirpath] = git_state(dirpath)
            if dirpath.count(os.sep) - os.path.expanduser(root).count(os.sep) >= 4:
                dirnames[:] = []

    # grep logs for the trainer's own print statements (numbers only)
    for f in found:
        if f["kind"] != "training log / report" or f["size"] > MAX_LOG_BYTES:
            continue
        try:
            with open(f["path"], "r", errors="replace") as fh:
                hits = [redact(m.group(1).strip()) for line in fh for m in [LOG_LINES.search(line)] if m]
        except OSError:
            continue
        if hits:
            log_hits.append({"path": f["path"], "lines": hits[:40]})

    hit_paths = {h["path"] for h in log_hits}
    for f in found:
        if f["kind"] == "training log / report" and f["path"] not in hit_paths:
            f["prio"], f["kind"] = "P2", "other log (no trainer lines)"

    if args.inspect_ckpt:
        for f in found:
            if f["kind"].startswith("checkpoint"):
                f["inspect"] = inspect_ckpt(f["path"])

    # ---- report --------------------------------------------------------
    print(f"scanned {', '.join(args.roots)} in {time.time()-t0:.0f}s\n")
    print("=== code actually on this machine vs GitHub ===")
    for p, s in repos.items() or {"(none found)": {}}.items():
        print(f"  {p}")
        for k, v in s.items():
            print(f"      {k:17s} {v}")
        if s.get("remote_embeds_credential"):
            print("      ⚠️  the origin URL EMBEDS A CREDENTIAL (redacted above). Anyone who can read "
                  "this repo's .git/config — or any log that ran `git remote -v` — has it. "
                  "Revoke it, and switch the remote to a token-free URL with a credential helper or SSH.")
    for prio in ("P0", "P1", "P2"):
        items = sorted((f for f in found if f["prio"] == prio), key=lambda f: (f["kind"], f["path"]))
        total = sum(f["size"] for f in items)
        print(f"\n=== {prio} — {len(items)} files, {human(total)} ===")
        last = None
        for f in items:
            if f["kind"] != last:
                print(f"  [{f['kind']}]")
                last = f["kind"]
            print(f"    {human(f['size']):>7}  {f['mtime']}  {f['path']}")
            if "inspect" in f:
                print(f"             {json.dumps(f['inspect'], ensure_ascii=False)}")
    print("\n=== trainer log lines (LR at resume, steps, stop reasons) ===")
    if not log_hits:
        print("  none — the trainer's stdout was not captured to a file anywhere scanned")
    for h in log_hits:
        print(f"  {h['path']}")
        for line in h["lines"]:
            print(f"      {line}")

    json.dump({"roots": args.roots, "repos": repos, "files": found, "log_hits": log_hits},
              open(args.json_out, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
