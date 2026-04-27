"""Plot SFT BLEU trajectory for Base v1.1 and Big v1.1.

Hardcodes the eval points from the SFT runs (see Section 7 / training
reports). Outputs a single PNG suitable as paper Figure 7.

Usage:
    python scripts/plot_sft_curves.py \\
        --out assets/sft_curves.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# Single-ckpt valid BLEU (newstest2013) at each SFT eval step.
# Steps are *relative to start of SFT* (0 = baseline/pretrain end).
BASE = {
    "name": "Base v1.1 (60M)",
    "color": "#1f77b4",
    "baseline_avg": 30.52,        # averaged-ckpt baseline
    "sft_steps_K": [2, 3, 6],     # SFT steps elapsed (×1000)
    "sft_bleu":    [29.44, 29.46, 28.97],
    "final_test":  33.77,         # newstest2014 BLEU after SFT
    "baseline_test": 35.31,
}

BIG = {
    "name": "Big v1.1 (209M)",
    "color": "#d62728",
    "baseline_avg": 31.14,
    "sft_steps_K": [2, 3, 4, 5, 6],
    "sft_bleu":    [29.39, 29.43, 29.19, 29.40, 29.05],
    "final_test":  33.54,
    "baseline_test": 35.87,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for run in (BASE, BIG):
        # SFT trajectory
        ax.plot(
            run["sft_steps_K"], run["sft_bleu"],
            marker="o", color=run["color"],
            label=f"{run['name']} + SFT",
            linewidth=2, markersize=7,
        )
        # Baseline (averaged-ckpt valid BLEU) as horizontal dashed line
        ax.axhline(
            run["baseline_avg"],
            color=run["color"], linestyle="--", alpha=0.55,
            label=f"{run['name']} baseline (avg) = {run['baseline_avg']:.2f}",
        )

    # Annotate final drop
    for run in (BASE, BIG):
        last_x = run["sft_steps_K"][-1]
        last_y = run["sft_bleu"][-1]
        delta = last_y - run["baseline_avg"]
        ax.annotate(
            f"Δ={delta:+.2f}",
            xy=(last_x, last_y),
            xytext=(last_x + 0.12, last_y - 0.10),
            color=run["color"], fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("SFT steps elapsed (×1000)")
    ax.set_ylabel("Valid BLEU (newstest2013, sacrebleu 13a)")
    ax.set_title("Quality-filtered SFT degrades both Base and Big v1.1\n"
                 "(top-1M of v2 by CometKiwi-22; same SPM, same eval, "
                 "fresh optimizer)")
    ax.set_xticks(range(0, 8))
    ax.set_xlim(-0.3, 7.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
