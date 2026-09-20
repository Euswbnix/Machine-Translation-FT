"""e04_report must reach the outcome PROTOCOL.md fixed in advance, and refuse otherwise.

The point of the script is that nobody gets to look at the six BLEU numbers and then pick
which of the three pre-stated outcomes to tell. These checks feed it results engineered to
sit on each outcome, plus the shapes it must refuse to decide on at all.
"""
import subprocess


def tsv(p, rows, baseline=(("newstest2014", 30.0), ("heldout_un", 40.0))):
    lines = ["condition\tseed\ttestset\tbleu"]
    for ts, b in baseline:
        lines.append(f"baseline\t0\t{ts}\t{b:.2f}")
    for cond, ts, vals in rows:
        for i, v in enumerate(vals):
            lines.append(f"{cond}\t{i}\t{ts}\t{v:.2f}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def run(ctx, path, *extra):
    r = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e04_report.py"), "--results", str(path),
                        *extra], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def suite(ctx):
    d = ctx.d

    # 1. the repetitive half is clearly worse: H holds
    p = tsv(d / "h.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                          ("ft_topk_rep", "newstest2014", [27.0, 27.1, 26.9])])
    rc, out = run(ctx, p)
    ctx.check("outcome 1 when the repetitive half degrades more, beyond the noise floor",
              rc == 0 and "OUTCOME 1" in out, f"rc={rc} {out[-300:]}")

    # 2. the halves sit on top of each other: H refuted
    p = tsv(d / "flat.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                             ("ft_topk_rep", "newstest2014", [29.05, 28.95, 29.0])])
    rc, out = run(ctx, p)
    ctx.check("outcome 2 when the two halves are indistinguishable",
              rc == 0 and "OUTCOME 2" in out, f"rc={rc} {out[-300:]}")

    # 3. the DIVERSE half is worse: H refuted the other way
    p = tsv(d / "rev.tsv", [("ft_topk_div", "newstest2014", [27.0, 27.1, 26.9]),
                            ("ft_topk_rep", "newstest2014", [29.0, 29.1, 28.9])])
    rc, out = run(ctx, p)
    ctx.check("outcome 3 when the diverse half is the worse one",
              rc == 0 and "OUTCOME 3" in out, f"rc={rc} {out[-300:]}")

    # a big gap that seed noise can explain is NOT outcome 1
    p = tsv(d / "noisy.tsv", [("ft_topk_div", "newstest2014", [30.5, 27.0, 29.5]),
                              ("ft_topk_rep", "newstest2014", [28.0, 26.0, 30.0])])
    rc, out = run(ctx, p)
    ctx.check("a gap inside the seed-noise floor cannot reach outcome 1",
              rc == 0 and "OUTCOME 2" in out, f"rc={rc} {out[-300:]}")

    # fail-closed: a missing half
    p = tsv(d / "half.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9])])
    rc, out = run(ctx, p)
    ctx.check("one half missing -> NO DECISION (exit 2), never a verdict",
              rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")

    # fail-closed: wrong seed count
    p = tsv(d / "seeds.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1]),
                              ("ft_topk_rep", "newstest2014", [27.0, 27.1, 26.9])])
    rc, out = run(ctx, p)
    ctx.check("a half with 2 of 3 seeds -> NO DECISION, not a verdict on what survived",
              rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")

    # fail-closed: no baseline row on the news set
    p = tsv(d / "nobase.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                               ("ft_topk_rep", "newstest2014", [27.0, 27.1, 26.9])],
            baseline=(("heldout_un", 40.0),))
    rc, out = run(ctx, p)
    ctx.check("no baseline on the news set -> NO DECISION",
              rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")

    # fail-closed: zero variance on both halves leaves Welch undefined
    p = tsv(d / "zero.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.0, 29.0]),
                             ("ft_topk_rep", "newstest2014", [27.0, 27.0, 27.0])])
    rc, out = run(ctx, p)
    ctx.check("zero seed variance on both halves -> NO DECISION (Welch undefined)",
              rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")

    # the secondary in-domain set is reported but never decides
    p = tsv(d / "sec.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                            ("ft_topk_rep", "newstest2014", [29.05, 28.95, 29.0]),
                            ("ft_topk_div", "heldout_un", [35.0, 35.1, 34.9]),
                            ("ft_topk_rep", "heldout_un", [45.0, 45.1, 44.9])])
    rc, out = run(ctx, p)
    ctx.check("a large in-domain split does not override the news-set outcome",
              rc == 0 and "OUTCOME 2" in out and "[secondary]" in out, f"rc={rc} {out[-400:]}")

    # a missing --reference is noted, not fatal
    p = tsv(d / "ref.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                            ("ft_topk_rep", "newstest2014", [27.0, 27.1, 26.9])])
    rc, out = run(ctx, p, "--reference", str(d / "nope.tsv"))
    ctx.check("an absent reference file degrades to a note, not a crash",
              rc == 0 and "not found" in out, f"rc={rc} {out[-300:]}")

    # the applied-token imbalance the protocol says to report
    ld = d / "logs"
    ld.mkdir(parents=True, exist_ok=True)
    for cond, tok in (("ft_topk_div", 0.0800), ("ft_topk_rep", 0.0801)):
        for seed in (42, 1, 2):
            (ld / f"{cond}_lr1_s{seed}_st2.log").write_text(
                f"Step 114900 | Loss 1.8 | Tok(applied) {tok - 0.0005:.4f}B\n"
                f"Step 115000 | Loss 1.8 | Tok(applied) {tok:.4f}B\n", encoding="utf-8")
    p2 = tsv(d / "tok.tsv", [("ft_topk_div", "newstest2014", [29.0, 29.1, 28.9]),
                             ("ft_topk_rep", "newstest2014", [27.0, 27.1, 26.9])])
    rc, out = run(ctx, p2, "--logdir", str(ld))
    ctx.check("applied-token imbalance is computed from the LAST Tok(applied) of each run",
              "imbalance 0.12%" in out and "within the 2%" in out, out[-400:])

    (ld / "ft_topk_rep_lr1_s2_st2.log").write_text("Step 115000 | Loss 1.8\n", encoding="utf-8")
    rc, out = run(ctx, p2, "--logdir", str(ld))
    ctx.check("a run log with no Tok(applied) line is named, not counted as zero",
              "no Tok(applied) line in: ft_topk_rep_lr1_s2_st2.log" in out, out[-400:])

    rc, out = run(ctx, p2, "--logdir", str(d / "absent"))
    ctx.check("an absent logdir is a note, not a crash",
              rc == 0 and "not found" in out, f"rc={rc} {out[-300:]}")
