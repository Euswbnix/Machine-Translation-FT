"""A test sentence longer than the model's positional encoding must not take down the gate.

2026-09-19, on the real run: one 649-token UN sentence in a 2,000-row held-out set made
eval_bleu raise inside the embedding (max_seq_len 256), and with it every one of the ten
evaluations of that set -- the baseline's included. e03_collect now drops such rows for the
baseline and every condition alike, and records how many, so the drop is visible in the
result instead of silently deciding the gate.
"""
import importlib.util


def load(ctx):
    spec = importlib.util.spec_from_file_location("collect_under_test", ctx.ROOT / "phase0/e03_collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")


def suite(ctx):
    mod = load(ctx)
    d = ctx.d
    words = lambda s: s.split()          # stand-in tokenizer: the real one is SentencePiece
    short, long_ = "a b c", " ".join(["w"] * 40)
    write(d / "ctrl/heldout_un.en", [short, long_, short])
    write(d / "ctrl/heldout_un.fr", [short, short, short])
    write(d / "ctrl/news.en", [short, short])
    write(d / "ctrl/news.fr", [short, short])
    tests = {"heldout_un": (d / "ctrl/heldout_un.en", d / "ctrl/heldout_un.fr"),
             "newstest2014": (d / "ctrl/news.en", d / "ctrl/news.fr")}

    out, dropped, how = mod.filter_long_rows(tests, 10, {}, d, d / "filtered", encode=words)
    ctx.check("an over-long row is dropped from the set that holds it",
              dropped == {"heldout_un": 1, "newstest2014": 0}, str(dropped))
    kept_src = (out["heldout_un"][0]).read_text(encoding="utf-8").splitlines()
    kept_ref = (out["heldout_un"][1]).read_text(encoding="utf-8").splitlines()
    ctx.check("both sides stay aligned after the drop (the row goes from source AND reference)",
              kept_src == [short, short] and kept_ref == [short, short], f"{kept_src} / {kept_ref}")
    ctx.check("a set with nothing over the limit keeps its original files, not a copy",
              out["newstest2014"] == tests["newstest2014"], str(out["newstest2014"]))
    ctx.check("the length measure used is reported, so a fallback cannot pass for the real tokenizer",
              how == "sentencepiece", how)

    # the reference side can be the long one too
    write(d / "ctrl2/heldout_un.en", [short, short])
    write(d / "ctrl2/heldout_un.fr", [short, long_])
    out2, dropped2, _ = mod.filter_long_rows(
        {"heldout_un": (d / "ctrl2/heldout_un.en", d / "ctrl2/heldout_un.fr")}, 10, {}, d, d / "filtered2",
        encode=words)
    ctx.check("a row whose REFERENCE is too long is dropped as well", dropped2 == {"heldout_un": 1}, str(dropped2))

    # every row over the limit is a stop, not an empty test set
    write(d / "ctrl3/heldout_un.en", [long_, long_])
    write(d / "ctrl3/heldout_un.fr", [short, short])
    rc, out3 = ctx.run(["-c",
                        "import importlib.util, sys;"
                        f"spec=importlib.util.spec_from_file_location('m', r'{ctx.ROOT}/phase0/e03_collect.py');"
                        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
                        f"m.filter_long_rows({{'heldout_un': (r'{d}/ctrl3/heldout_un.en', r'{d}/ctrl3/heldout_un.fr')}},"
                        f" 10, {{}}, r'{d}', __import__('pathlib').Path(r'{d}/filtered3'), encode=lambda s: s.split())"])
    ctx.check("a set where every row is too long stops the collection instead of scoring nothing",
              rc != 0 and "nothing to evaluate" in out3, f"rc={rc} {out3[-160:]}")
