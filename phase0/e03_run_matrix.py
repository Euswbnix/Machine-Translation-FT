#!/usr/bin/env python3
"""E0.3b — Generate the fine-tuning control matrix and the runner scripts.

TWO-STAGE DESIGN (13 runs, not 36)
----------------------------------
A full 3 conditions x 4 LRs x 3 seeds cross is 36 runs. It is not needed: the LR
sweep exists only to LOCATE a rate at which fine-tuning is stable, and the
condition comparison only has to happen at that rate.

  Stage 1 (4 runs, 1 seed) : LR sweep on ft_topk alone, spanning > 1 decade.
                             The lr_scale is selected MECHANICALLY by
                             phase0/e03_select_lr.py under the rule frozen in
                             phase0/e03_decisions.json "lr_selection_rule"
                             (newstest2013 valid BLEU; the gate uses newstest2014).
  Stage 2 (9 runs, 3 seeds): all three conditions at that LR.

THE LR BUG THIS IS BUILT AROUND
-------------------------------
configs/sft_base_enfr.yaml says the scheduler "resumes at late-Noam (~1.4e-4
effective peak), which is in the standard SFT LR band (10-20% of pretraining
peak 7e-4)". That number is wrong, by the SAME off-by-accumulate_steps error
that inflated the paper's token budgets 4x -- here in the opposite direction.

trainer.py:576 (load_checkpoint, reset_optimizer=True) sets

    self.scheduler._step = self.global_step // self.accumulate_steps
                         = 105000 // 4 = 26250

and TransformerScheduler._get_lr returns, past warmup,

    lr_scale * d_model**-0.5 * step**-0.5
      = 1.0 * 512**-0.5 * 26250**-0.5 = 2.73e-4

not 1.36e-4 (which is what step=105000 would give). Pretraining peak is
lr_scale * 512**-0.5 * 4000**-0.5 = 6.99e-4, so fine-tuning actually ran at
**39% of the pretraining peak**, roughly double the 10-20% the config itself
targets, on an already-converged model.

So "restart LR too high -> catastrophic forgetting" is not merely a rival
hypothesis to the domain story: it is a CONFIRMED deviation from the intended
experimental condition. BLEU falling monotonically from the first eval is its
textbook signature. trainer.py:578 prints the real LR at load time, so the
training log settles it: look for "LR = 2.73e-04".

The good news is that the author DID guard against the catastrophic version --
the scheduler is not reset to warmup, it is placed at the right point on the
decay curve. The error is only in the arithmetic reported around it.

LR GRID
-------
Effective peak = lr_scale * 512**-0.5 * 26250**-0.5:

    lr_scale 1.00 -> 2.73e-4  (39% of pretrain peak)  <-- what actually ran
    lr_scale 0.50 -> 1.36e-4  (20%)                   <-- what the paper intended
    lr_scale 0.15 -> 4.09e-5  (5.9%)
    lr_scale 0.05 -> 1.36e-5  (2.0%)

Range 20x, and it brackets the intended setting. Over a 10K-micro-step run Noam
decays only ~4%, and `min_lr` is pinned to the same value, so each rung is flat.

USAGE
-----
    python e03_run_matrix.py --base-config configs/sft_base_enfr.yaml \
        --data-dir data/phase0 --out-dir configs/phase0
    bash configs/phase0/run_stage1.sh       # ~2 GPU-hours
    python phase0/e03_select_lr.py --logdir logs/phase0/stage1 --rule <frozen rule>   # then:
    bash configs/phase0/run_stage2.sh 0.15  # ~5 GPU-hours
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controls_fingerprint import fingerprint  # noqa: E402

try:
    import yaml
except ImportError:
    print("需要 pyyaml:  pip install pyyaml", file=sys.stderr)
    raise

LR_SCALES = [1.0, 0.50, 0.15, 0.05]
PRETRAIN_PEAK = 6.99e-4   # lr_scale=1 at warmup_steps=4000
CONDITIONS = ["ft_topk", "ft_random", "ft_bottom"]   # --conditions overrides (explanatory probes)
SEEDS = [42, 1, 2]
D_MODEL = 512
RESUME_GLOBAL_STEP = 105_000
RESUME_OPT_STEP = RESUME_GLOBAL_STEP // 4   # trainer.py:577 divides by accumulate_steps


def effective_peak(lr_scale: float) -> float:
    return lr_scale * D_MODEL ** -0.5 * RESUME_OPT_STEP ** -0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--data-dir", default="data/phase0")
    ap.add_argument("--out-dir", default="configs/phase0")
    ap.add_argument("--ckpt", default="checkpoints/base_enfr_v1/averaged.pt",
                    help="the converged Base v1.1 checkpoint to fine-tune from")
    ap.add_argument("--src-ext", default="en")
    ap.add_argument("--tgt-ext", default="fr")
    ap.add_argument("--swanlab-cloud", action="store_true",
                    help="keep SwanLab cloud logging. OFF by default: the base SFT configs "
                         "set enabled: true / mode: cloud, which on a freshly rented GPU box "
                         "with no SwanLab login blocks every run at startup")
    ap.add_argument("--no-controls-fingerprint", action="store_true",
                    help="do not hash --data-dir/manifest.json (offline tests only). By default the "
                         "controls content sha goes into matrix.json and every checkpoint dir name, "
                         "so a rebuilt control set can never reuse runs trained on the old one")
    ap.add_argument("--conditions", default=None,
                    help="comma list of condition names instead of the E0.3 three "
                         "(ft_topk,ft_random,ft_bottom). Each needs <name>.<src-ext> and "
                         "<name>.<tgt-ext> in --data-dir. For explanatory probes that fine-tune other "
                         "splits (E0.4); e03_decide still reads the gate's own condition names, so a "
                         "probe writes its own results file and cannot be mistaken for the gate.")
    ap.add_argument("--keep-last", type=int, default=None,
                    help="checkpoint.keep_last (trainer prunes step_*.pt only; final.pt is kept). "
                         "e03_collect --ft-ckpt final reads only final.pt -> 1 suffices; "
                         "avg-last5 needs the last 4 step_*.pt -> >= 4. Default: inherit")
    ap.add_argument("--budget", choices=("steps", "tokens"), default="steps",
                    help="steps (pre-registered): every arm stops at max_steps micro-batches. "
                         "tokens: every arm stops at --target-tokens applied target tokens "
                         "(trainer patch token gate), evals every target/10 tokens")
    ap.add_argument("--target-tokens", type=int, default=0)
    ap.add_argument("--spike-ratio", default="inherit",
                    help="training.loss_spike_ratio: 'inherit' (1.3 from the base config) or a number; "
                         "0 disables the spike guard (PROTOCOL.md 1.3)")
    args = ap.parse_args()
    if args.conditions:
        global CONDITIONS
        CONDITIONS = [c.strip() for c in args.conditions.split(",") if c.strip()]
        if not CONDITIONS:
            sys.exit("--conditions is empty")
        missing = [f"{c}.{e}" for c in CONDITIONS for e in (args.src_ext, args.tgt_ext)
                   if not (Path(args.data_dir) / f"{c}.{e}").is_file()]
        if missing:
            sys.exit(f"--conditions names sets that are not in {args.data_dir}: " + ", ".join(missing))
    if args.keep_last is not None and args.keep_last < 1:
        sys.exit("--keep-last must be >= 1 (the trainer treats 0 as 'keep every step checkpoint')")
    if args.budget == "tokens" and args.target_tokens <= 0:
        sys.exit("--budget tokens needs --target-tokens > 0")
    if args.budget == "steps" and args.target_tokens:
        sys.exit("--target-tokens is only meaningful with --budget tokens")
    spike = None
    if args.spike_ratio != "inherit":
        try:
            spike = float(args.spike_ratio)
        except ValueError:
            sys.exit(f"--spike-ratio {args.spike_ratio!r} is neither 'inherit' nor a number")
        if spike < 0:
            sys.exit("--spike-ratio must be >= 0")
    fp = None if args.no_controls_fingerprint else fingerprint(args.data_dir)
    ck_root = "checkpoints/phase0" + (f"/{fp['controls_sha'][:12]}" if fp else "")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(args.base_config, encoding="utf-8"))

    print(f"effective peak LR by lr_scale (scheduler resumes at optimizer step "
          f"{RESUME_OPT_STEP:,}; pretraining peak {PRETRAIN_PEAK:.2e}):")
    for sc in LR_SCALES:
        lr = effective_peak(sc)
        note = ""
        if sc == 1.0:
            note = "   <-- what the SFT run ACTUALLY used"
        elif abs(lr - 1.36e-4) / 1.36e-4 < 0.05:
            note = "   <-- what the config comment INTENDED"
        print(f"  lr_scale {sc:<5} -> {lr:.2e}  ({100*lr/PRETRAIN_PEAK:4.1f}% of pretrain peak){note}")

    manifest = {"lr_scales": LR_SCALES, "conditions": CONDITIONS, "seeds": SEEDS,
                "effective_peak": {str(s): effective_peak(s) for s in LR_SCALES},
                "controls_sha": fp["controls_sha"] if fp else None,
                "controls_files": fp["files"] if fp else None,
                "data_dir": str(args.data_dir),
                "checkpoint_root": ck_root, "keep_last": args.keep_last,
                "budget": args.budget, "target_tokens": args.target_tokens or None,
                "loss_spike_ratio": "inherit" if spike is None else spike,
                "configs": []}

    for cond in CONDITIONS:
        for s in LR_SCALES:
            cfg = copy.deepcopy(base)
            tag = f"{cond}_lr{s:g}"
            cfg["training"]["lr_scale"] = s
            # pin the floor to the same value so the curve is genuinely flat
            cfg["training"]["min_lr"] = float(effective_peak(s))

            # Every condition must run for EXACTLY the same number of steps on
            # EXACTLY the same eval grid. Inherited from sft_base_enfr.yaml,
            # neither holds:
            #   - early_stopping fires on patience counted in EVALUATIONS
            #     (trainer.py:376), so a condition that degrades faster stops
            #     sooner and is compared over a shorter run.
            #   - _update_eval_interval (trainer.py:237-269) interpolates the
            #     eval interval from an EMA of TRAINING LOSS, which differs by
            #     condition — so the eval grid itself would be condition-
            #     dependent, and so would the number of chances each condition
            #     gets to post its best BLEU.
            # Both make run length a function of the variable under study.
            cfg["training"]["early_stopping"] = False
            cfg["training"]["patience"] = 10 ** 9
            grid = int(cfg["training"].get("eval_interval_min", 1000))
            cfg["training"]["eval_interval"] = grid
            cfg["training"]["eval_interval_min"] = grid
            cfg["data"]["train_src"] = f"{args.data_dir}/{cond}.{args.src_ext}"
            cfg["data"]["train_tgt"] = f"{args.data_dir}/{cond}.{args.tgt_ext}"
            cfg["checkpoint"]["dir"] = f"{ck_root}/{tag}"
            if args.keep_last is not None:
                cfg["checkpoint"]["keep_last"] = args.keep_last
            if args.budget == "tokens":
                # patch semantics: max_steps is NOT consulted; the backstop compares the
                # absolute global_step, which starts at the resumed 105,000.
                span = int(cfg["training"]["max_steps"]) - RESUME_GLOBAL_STEP
                cfg["training"]["max_target_tokens"] = int(args.target_tokens)
                cfg["training"]["eval_every_tokens"] = int(args.target_tokens) // 10
                cfg["training"]["max_micro_steps_backstop"] = RESUME_GLOBAL_STEP + 2 * span
            if spike is not None:
                cfg["training"]["loss_spike_ratio"] = spike
            sw = cfg.setdefault("logging", {}).setdefault("swanlab", {})
            sw["experiment"] = f"phase0_{tag}"
            # Inherited from sft_base_enfr.yaml as enabled: true, mode: cloud. On a
            # rented box with no login that stalls every run in swanlab.init, so it
            # is disabled unless explicitly requested. TensorBoard logs and the
            # dev-CE trace are unaffected.
            sw["enabled"] = bool(args.swanlab_cloud)
            p = out / f"{tag}.yaml"
            yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"),
                           sort_keys=False, allow_unicode=True)
            manifest["configs"].append({"condition": cond, "lr_scale": s,
                                        "path": str(p)})

    n_cfg = len(manifest["configs"])

    stage1 = ["#!/usr/bin/env bash",
              "# E0.3 Stage 1 — LR sweep on ft_topk only, seed 42. ~2 GPU-hours.",
              "# The lr_scale for stage 2 is selected by phase0/e03_select_lr.py under the",
              "# rule frozen in phase0/e03_decisions.json (lr_selection_rule), not by eye.",
              "# --suffix carries a stage tag: train.py appends it to BOTH",
              "# checkpoint.dir and swanlab.experiment (train.py:41,43), and the",
              "# tensorboard dir is ckpt_dir/logs (trainer.py:133), so without the",
              "# tag stage 2 would overwrite this stage's run at the chosen LR.",
              "set -euo pipefail", ""]
    for s in LR_SCALES:
        stage1.append(
            f"python train.py --config {args.out_dir}/ft_topk_lr{s:g}.yaml "
            f"--resume {args.ckpt} --reset-optimizer --seed 42 --suffix _s42_st1")
    stage1 += ["", 'echo "Stage 1 done. Select the lr_scale with phase0/e03_select_lr.py'
                   ' (frozen rule), then: bash run_stage2.sh <lr_scale>"']

    stage2 = ["#!/usr/bin/env bash",
              "# E0.3 Stage 2 — all three conditions at the chosen LR, 3 seeds. ~5 GPU-hours.",
              "set -euo pipefail",
              'LR="${1:?usage: run_stage2.sh <lr_scale>}"',
              '# filenames use %g formatting, so 1.0 -> lr1, 0.50 -> lr0.5',
              f'VALID="{" ".join(f"{s:g}" for s in LR_SCALES)}"',
              'case " $VALID " in *" $LR "*) ;; *)',
              '  echo "bad lr_scale \'$LR\'; valid: $VALID" >&2; exit 1;; esac', ""]
    for cond in CONDITIONS:
        for seed in SEEDS:
            stage2.append(
                f'python train.py --config {args.out_dir}/{cond}_lr${{LR}}.yaml '
                f'--resume {args.ckpt} --reset-optimizer --seed {seed} --suffix _s{seed}_st2')
    stage2 += ["",
               'echo "Stage 2 done. Note ft_topk/seed42 was run in BOTH stages with',
               'identical config and seed — comparing the two is a free determinism',
               'check. If they differ, the seed does not control everything and the',
               'seed-noise floor in e03_decide.py is understated."',
               'echo "Now run: python phase0/e03_decide.py --results <tsv>"']

    for name, lines in (("run_stage1.sh", stage1), ("run_stage2.sh", stage2)):
        p = out / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        p.chmod(0o755)

    json.dump(manifest, open(out / "matrix.json", "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {n_cfg} configs + run_stage1.sh + run_stage2.sh to {out}/")
    print(f"  stage 1: {len(LR_SCALES)} runs   stage 2: "
          f"{len(CONDITIONS)*len(SEEDS)} runs   total {len(LR_SCALES)+len(CONDITIONS)*len(SEEDS)}")
    if fp:
        print(f"  controls_sha {fp['controls_sha']} -> checkpoints under {ck_root}/")
    print("\n⚠️  Evaluate EVERY run on newstest2014 AND every held-out set in manifest.json.")
    print("    The domain claim predicts the in-domain sets IMPROVE while newstest degrades;")
    print("    without them a decline is indistinguishable from plain forgetting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
