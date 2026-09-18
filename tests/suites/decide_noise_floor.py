"""Criterion 2 needs a noise floor, frozen 2026-09-18 (decision X1).

Before the amendment a +0.01 BLEU in-domain gain passed criterion 2 while the
seed sd is 0.05-0.2 BLEU, so the weaker half of the gate decided GO. Criterion 1
has had a noise floor from the start; these checks hold criterion 2 to the same
bar, and pin the frozen UN-only default (decision D5).
"""
HEAD = "condition seed testset bleu\n"
NEWS = ("baseline - newstest2014 38.21\n"
        "ft_topk 42 newstest2014 36.60\nft_topk 1 newstest2014 36.70\nft_topk 2 newstest2014 36.50\n"
        "ft_random 42 newstest2014 37.90\nft_random 1 newstest2014 38.00\nft_random 2 newstest2014 37.80\n")


def un(base, vals):
    rows = f"baseline - heldout_un {base:.2f}\n"
    for seed, v in zip((42, 1, 2), vals):
        rows += f"ft_topk {seed} heldout_un {v:.2f}\n"
    return rows


def suite(ctx):
    # mean 35.05, sd 0.10 -> gain +0.05 is inside the seed noise
    p = ctx.d / "within_noise.tsv"
    p.write_text(HEAD + NEWS + un(35.00, (35.05, 35.15, 34.95)), encoding="utf-8")
    rc, out = ctx.run([ctx.ROOT / "phase0/e03_decide.py", "--results", p])
    ctx.check("criterion 2 fails when the in-domain gain is inside the seed-noise floor (was GO before X1)",
              rc == 1 and "criterion 2: FAIL" in out and "within noise" in out and "VERDICT: NO-GO" in out,
              out[-300:])
    # same sd, gain +1.50 -> clears the floor, and criterion 1 passes -> GO
    p = ctx.d / "clears.tsv"
    p.write_text(HEAD + NEWS + un(35.00, (36.50, 36.60, 36.40)), encoding="utf-8")
    rc, out = ctx.run([ctx.ROOT / "phase0/e03_decide.py", "--results", p])
    ctx.check("a gain beyond the seed-noise floor still passes criterion 2",
              rc == 0 and "criterion 2: PASS" in out and "clears it" in out, out[-300:])
    # the floor is printed, so the verdict can be audited from the log alone
    ctx.check("the in-domain noise floor is reported next to the gain", "seed-noise floor (sd) = 0.10" in out,
              out[-200:])
    # frozen D5: UN-only gate, so a TSV without any Europarl rows still decides
    ctx.check("the frozen gate is UN-only: no heldout_europarl rows needed to decide",
              "heldout_europarl" not in (ctx.d / "clears.tsv").read_text(encoding="utf-8") and rc == 0,
              f"rc={rc}")
