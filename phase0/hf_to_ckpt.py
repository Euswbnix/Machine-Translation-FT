#!/usr/bin/env python3
"""Turn a public HF release into a checkpoint that trainer.load_checkpoint accepts.

WHY THIS REMOVES THE NEED FOR THE ORIGINAL TRAINING MACHINE
-----------------------------------------------------------
The paper fine-tuned "the averaged Base v1.1 and Big v1.1 checkpoints (seed 42)"
(paper_wmt_upload/sections/05_sft.tex). Those are exactly the public releases
euswbnix/transformer-wmt14-enfr-{base,big}: their config.json records
training.steps 105000 / 210000 and valid/test BLEU 30.52/35.31 and 31.14/35.87,
which match 05_sft.tex:26 and :29 digit for digit.

But scripts/prepare_hf_release.py:256-265 saves only the bare state_dict, while
trainer.load_checkpoint reads ckpt["model"] and ckpt["global_step"] -- and
global_step is what places the Noam scheduler on its decay curve at resume
(scheduler._step = global_step // accumulate_steps). This restores those keys.

WHAT IT REFUSES TO DO
---------------------
It writes nothing unless the weights load into the training repo's OWN
Transformer class with strict=True, and (if --train-config is given) the model
section the trainer will build matches the release architecture field by field.

It does NOT prove the weights are good. The acceptance test is reproducing the
released test BLEU (35.31 Base / 35.87 Big) on newstest2014 with the same
decoding settings; that needs a GPU and is a separate RUNBOOK step.

USAGE (on the GPU box, from anywhere)
-------------------------------------
    python phase0/hf_to_ckpt.py --repo euswbnix/transformer-wmt14-enfr-base \
        --mt-root ~/Machine_translation \
        --train-config configs/sft_base_enfr.yaml \
        --out ckpt/enfr_base_v1.1_averaged.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

FILES = ("config.json", "pytorch_model.bin", "sentencepiece.model")
ARCH_KEYS = ("vocab_size", "d_model", "n_heads", "n_encoder_layers", "n_decoder_layers",
             "d_ff", "dropout", "max_seq_len", "share_embeddings")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_revision(repo: str, revision: str) -> str:
    """Pin the exact HF commit, so the checkpoint's provenance is reproducible."""
    url = f"https://huggingface.co/api/models/{repo}"
    if revision != "main":
        url += f"/revision/{revision}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["sha"]


def download(repo: str, sha: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in FILES:
        target = dest / name
        if not target.exists():
            url = f"https://huggingface.co/{repo}/resolve/{sha}/{name}"
            print(f"  downloading {name} …")
            with urllib.request.urlopen(url, timeout=600) as r, open(target, "wb") as f:
                shutil.copyfileobj(r, f)
        out[name] = target
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--mt-root", required=True, help="clone of Machine_translation")
    ap.add_argument("--train-config", help="yaml the FT run will use; architecture is cross-checked")
    ap.add_argument("--download-dir", default="hf_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    mt_root = Path(a.mt_root).expanduser().resolve()
    sys.path.insert(0, str(mt_root))
    from src.model import Transformer                                   # noqa: E402

    sha = resolve_revision(a.repo, a.revision)
    print(f"{a.repo} @ {sha}")
    files = download(a.repo, sha, Path(a.download_dir) / a.repo.replace("/", "__") / sha)
    cfg = json.load(open(files["config.json"]))

    missing = [k for k in ARCH_KEYS if k not in cfg]
    if missing:
        sys.exit(f"config.json lacks architecture keys {missing}; refusing to guess")
    steps = (cfg.get("training") or {}).get("steps")
    if not isinstance(steps, int) or steps <= 0:
        sys.exit(f"config.json training.steps is {steps!r}; global_step cannot be restored, "
                 "and without it the scheduler would resume at the wrong point. Refusing.")

    if a.train_config:
        import yaml
        tcfg = yaml.safe_load(open(a.train_config))["model"]
        diffs = {k: (tcfg.get(k), cfg[k]) for k in ARCH_KEYS if tcfg.get(k) != cfg[k]}
        if diffs:
            sys.exit(f"--train-config model section differs from the release: {diffs}")
        print(f"  architecture matches {a.train_config}")

    model = Transformer(**{k: cfg[k] for k in ARCH_KEYS}, pad_idx=cfg.get("pad_idx", 0))
    sd = torch.load(files["pytorch_model.bin"], map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        sys.exit(f"strict load into {mt_root}/src/model FAILED — the release does not match "
                 f"this code:\n{e}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  strict load OK: {n_params/1e6:.1f}M parameters, global_step {steps:,}")

    ckpt = {
        "model": model.state_dict(),
        "global_step": steps,
        # best_bleu only gates best.pt saving; the E0.3 configs disable early
        # stopping, so it has no effect on training dynamics there.
        "best_bleu": 0.0,
        "provenance": {
            "source": f"https://huggingface.co/{a.repo}", "revision": sha,
            "weights_sha256": sha256(files["pytorch_model.bin"]),
            "spm_sha256": sha256(files["sentencepiece.model"]),
            "release_training": cfg.get("training"),
            "note": "rebuilt by phase0/hf_to_ckpt.py; optimizer/scheduler state were "
                    "never released and are not needed under --reset-optimizer",
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    spm_out = out.with_name(out.stem + ".sentencepiece.model")
    shutil.copyfile(files["sentencepiece.model"], spm_out)
    print(f"wrote {out}\n      {spm_out}")
    print(f"  weights sha256 {ckpt['provenance']['weights_sha256']}")
    print("NEXT: acceptance test — reproduce the release test BLEU "
          f"({(cfg.get('training') or {}).get('test_bleu_newstest2014')}) on newstest2014 "
          "with beam 5, length penalty 1.0, sacrebleu 13a, using THIS spm. "
          "Do not start E0.3 until it matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
