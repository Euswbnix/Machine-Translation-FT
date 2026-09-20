"""e03_collect must derive the condition from config names that contain more than one word.

2026-09-20: E0.4 introduced ft_topk_div / ft_topk_rep. The name regex was (ft_[a-z]+)_lr,
which cannot match an underscore, so e03_collect exited with "cannot derive a condition"
the moment it parsed the E0.4 runner -- after the six fine-tunes had already run. The
greedy form still has to resolve to the LAST _lr, or ft_topk_lr1 would parse as
ft_topk_lr1 instead of ft_topk.
"""
import importlib.util


def load(ctx):
    spec = importlib.util.spec_from_file_location("collect_cond_under_test",
                                                  ctx.ROOT / "phase0/e03_collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def runner_line(cfg, seed, suffix):
    return (f"python train.py --config /x/configs/{cfg} --resume ckpt.pt "
            f"--reset-optimizer --seed {seed} --suffix {suffix}")


def suite(ctx):
    mod = load(ctx)
    d = ctx.d

    r = d / "run_stage2.sh"
    r.write_text("\n".join([
        "#!/bin/bash",
        runner_line("ft_topk_div_lr${LR}.yaml", 42, "_s42_st2"),
        runner_line("ft_topk_rep_lr${LR}.yaml", 1, "_s1_st2"),
    ]) + "\n", encoding="utf-8")
    runs = mod.parse_runner(r, "1")
    ctx.check("a two-word condition parses (E0.4)",
              [x["condition"] for x in runs] == ["ft_topk_div", "ft_topk_rep"],
              str([x["condition"] for x in runs]))
    ctx.check("seed and suffix survive the same parse",
              (runs[0]["seed"], runs[0]["suffix"]) == (42, "_s42_st2"),
              str((runs[0]["seed"], runs[0]["suffix"])))
    ctx.check("${LR} is substituted in the config path",
              runs[0]["config"].name == "ft_topk_div_lr1.yaml", runs[0]["config"].name)

    r2 = d / "run_stage2_single.sh"
    r2.write_text("\n".join([
        runner_line("ft_topk_lr1.yaml", 42, "_s42_st2"),
        runner_line("ft_random_lr0.15.yaml", 1, "_s1_st2"),
        runner_line("ft_bottom_lr0.05.yaml", 2, "_s2_st2"),
    ]) + "\n", encoding="utf-8")
    runs2 = mod.parse_runner(r2, "1")
    ctx.check("one-word conditions are unchanged -- the greedy match stops at the last _lr",
              [x["condition"] for x in runs2] == ["ft_topk", "ft_random", "ft_bottom"],
              str([x["condition"] for x in runs2]))

    r3 = d / "run_bad.sh"
    r3.write_text(runner_line("baseline_v2.yaml", 42, "_s42_st2") + "\n", encoding="utf-8")
    try:
        mod.parse_runner(r3, "1")
        bad = "no exit"
    except SystemExit as e:
        bad = str(e)
    ctx.check("a name with no ft_<condition>_lr still refuses to guess",
              "cannot derive a condition" in bad, bad)

    r4 = d / "run_empty.sh"
    r4.write_text("#!/bin/bash\necho nothing here\n", encoding="utf-8")
    try:
        mod.parse_runner(r4, "1")
        empty = "no exit"
    except SystemExit as e:
        empty = str(e)
    ctx.check("a runner with no train.py lines is an error, not an empty result",
              "no train.py lines parsed" in empty, empty)
