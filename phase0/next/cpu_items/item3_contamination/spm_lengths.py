#!/usr/bin/env python3
"""Item 3, stage 1: SentencePiece piece counts for item3_contamination.py's "spm" length unit.

Runs with the E0.6 reference interpreter (~/e06env/venv-arm64, which already has sentencepiece
0.2.2; nothing is installed into it). The py3langid venv that runs stage 2 has no sentencepiece,
so the counts are made here and handed over in one file.

For every line of the 18 text inputs (e05_src/{ft_topk,ft_random}.{en,fr} and the seven
phase0_e05 set files .{en,fr}) it stores len(sp.encode(line)) with the base model's
spm_enfr_v1_fixed.model (ids, no BOS/EOS), the same call item3_contamination.py made before the
split. Lines are split on '\\n' only, as in item3_contamination.py's read_lines(). Each file's
sha256 and the model's sha256 go into the output; stage 2 checks them against its pins.

Usage: ~/e06env/venv-arm64/bin/python -I spm_lengths.py --data ~/e06data --out spm_lengths.npz
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

SETS = ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div",
        "ft_random_small", "heldout_un"]
FILES = ([f"e05_src/{p}.{x}" for p in ("ft_topk", "ft_random") for x in ("en", "fr")]
         + [f"phase0_e05/{s}.{x}" for s in SETS for x in ("en", "fr")])
MODEL = "spm/spm_enfr_v1_fixed.model"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(p: Path) -> list:
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.data)
    t0 = time.time()
    sp = spm.SentencePieceProcessor(model_file=str(d / MODEL))
    cache: dict = {}
    arrays, shas = {}, {}
    for rel in FILES:
        shas[rel] = sha256_file(d / rel)
        lines = read_lines(d / rel)
        new = [x for x in dict.fromkeys(lines) if x not in cache]
        B = 100_000
        for i in range(0, len(new), B):
            part = new[i:i + B]
            for x, ids in zip(part, sp.encode(part, num_threads=1)):
                cache[x] = len(ids)
        arrays[rel] = np.fromiter((cache[x] for x in lines), np.int32, len(lines))
        print(f"{time.strftime('%H:%M:%S')} {rel}: {len(lines):,} lines, {len(new):,} new strings",
              flush=True)
    meta = {"sentencepiece": spm.__version__, "python": sys.version.split()[0],
            "interpreter": sys.executable, "model": MODEL, "model_sha256": sha256_file(d / MODEL),
            "files_sha256": shas, "rule": "len(SentencePieceProcessor.encode(line)), ids, no BOS/EOS"}
    tmp = Path(a.out + ".tmp.npz")
    np.savez(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(arrays)} files, {len(cache):,} unique strings, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
