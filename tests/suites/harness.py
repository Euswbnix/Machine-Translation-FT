"""Proves tests/regress.py discovers and runs tests/suites/*.py with a usable ctx."""


def suite(ctx):
    probe = ctx.d / "probe.txt"
    probe.write_text("ok\n", encoding="utf-8")
    rc, out = ctx.run(["-c", "import sys; print(open(sys.argv[1]).read().strip())", probe])
    ctx.check("external suite loader passes a fresh fixture dir and a working run()",
              rc == 0 and out.strip() == "ok" and ctx.ROOT.joinpath("tests/regress.py").exists(), out[-200:])
