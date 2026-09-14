#!/usr/bin/env python3
"""Build the trainer's tokenisation caches ONCE, serially, before parallel runs start.

Training-path finding: TranslationDataset (Machine_translation/src/data/dataset.py,
read-only) checks `<src>.cached_<max_seq_len>.npz`, and if absent tokenises and calls
np.savez directly on the final path (no temp file, no rename). run_parallel.py starts up
to one run per GPU at once, so runs sharing a train or valid file race on that write; a
run that loads inside the write window dies (EOFError/BadZipFile), and a run killed
mid-write leaves a truncated cache that every retry then loads and fails on.

This parses the same runner run_parallel.py does (shared parse_runs, so they cannot
drift), collects every distinct (train_src, train_tgt) and (valid_src, valid_tgt) with
the config's spm_model and model.max_seq_len (what create_dataloader passes as
max_tokens), builds each missing cache with the trainer's own TranslationDataset, then
verifies EVERY cache by loading all four arrays. An unreadable cache is deleted and the
command exits 1, so the next attempt rebuilds it.

    python phase0/warm_cache.py --runner configs/phase0/run_stage2.sh --lr-scale 0.15 --cwd ~/mt/Machine_translation

Exit: 0 all caches present and readable; 1 a cache was unreadable (deleted) or a build failed;
2 the control sets no longer match matrix.json's controls_sha (nothing built).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parallel import parse_runs  # noqa: E402
from controls_fingerprint import check_controls  # noqa: E402

KEYS = ("src_tokens", "src_offsets", "tgt_tokens", "tgt_offsets")


def cache_path(src: str, max_len: int) -> Path:
    return Path(str(src) + f".cached_{max_len}.npz")


def collect(runs, cwd: Path) -> list[dict]:
    seen, out = set(), []
    for r in runs:
        cfg = r["cfg"]
        d, L = cfg["data"], int(cfg["model"]["max_seq_len"])
        for s, t in ((d["train_src"], d["train_tgt"]), (d["valid_src"], d["valid_tgt"])):
            sp = Path(s) if Path(s).is_absolute() else cwd / s
            tp = Path(t) if Path(t).is_absolute() else cwd / t
            spm = Path(d["spm_model"]) if Path(d["spm_model"]).is_absolute() else cwd / d["spm_model"]
            k = (str(sp), str(tp), L)
            if k not in seen:
                seen.add(k)
                out.append({"src": sp, "tgt": tp, "len": L, "spm": spm, "cache": cache_path(sp, L)})
    return out


def verify(p: Path):
    import numpy as np
    try:
        with np.load(p) as z:
            arrs = {k: z[k] for k in KEYS}
        if len(arrs["src_offsets"]) != len(arrs["tgt_offsets"]) or len(arrs["src_offsets"]) < 1:
            return "offset arrays disagree"
        if int(arrs["src_offsets"][-1]) != len(arrs["src_tokens"]) or \
                int(arrs["tgt_offsets"][-1]) != len(arrs["tgt_tokens"]):
            return "offsets do not end at the token array length"
    except Exception as e:                                      # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--lr-scale", default=None)
    ap.add_argument("--cwd", required=True, help="Machine_translation root")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cwd = Path(a.cwd).expanduser().resolve()
    runs = parse_runs(Path(a.runner).expanduser(), a.lr_scale, cwd, sys.executable)
    bad = check_controls(Path(a.runner).expanduser(), runs, cwd)
    if bad:
        print(f"STALE CONTROLS — {bad}\nNo cache built.")
        return 2
    items = collect(runs, cwd)
    missing = [it for it in items if not it["cache"].exists()]
    print(f"{len(items)} distinct dataset file pair(s); {len(missing)} cache(s) to build")
    for it in items:
        print(f"  {'BUILD' if it in missing else 'have '} {it['cache']}")
    if a.dry_run:
        return 0
    if missing:
        sys.path.insert(0, str(cwd))
        os.chdir(cwd)                       # the trainer resolves relative data paths from here
        from src.data.dataset import TranslationDataset   # noqa: E402  (trainer's own code)
        from src.data.tokenizer import Tokenizer           # noqa: E402
        tok_cache = {}
        for it in missing:
            if not it["src"].is_file() or not it["tgt"].is_file():
                print(f"FAILED: {it['src']} or {it['tgt']} missing")
                return 1
            tok = tok_cache.setdefault(str(it["spm"]), Tokenizer(str(it["spm"])))
            TranslationDataset(str(it["src"]), str(it["tgt"]), tok, it["len"])
    bad = 0
    for it in items:
        err = verify(it["cache"]) if it["cache"].exists() else "not built"
        if err:
            bad += 1
            print(f"UNREADABLE cache {it['cache']}: {err} -> deleted; rerun to rebuild")
            it["cache"].unlink(missing_ok=True)
    if bad:
        return 1
    print("all caches present and readable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
