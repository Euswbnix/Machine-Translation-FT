"""Proves tests/regress.py discovers and runs tests/suites/*.py with a usable ctx."""


def suite(ctx):
    probe = ctx.d / "probe.txt"
    probe.write_text("ok\n", encoding="utf-8")
    rc, out = ctx.run(["-c", "import sys; print(open(sys.argv[1]).read().strip())", probe])
    ctx.check("external suite loader passes a fresh fixture dir and a working run()",
              rc == 0 and out.strip() == "ok" and ctx.ROOT.joinpath("tests/regress.py").exists(), out[-200:])

    # rental_setup.sh switches must never reach a fixture (see LEAKY_ENV in tests/regress.py)
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("regress_under_test", ctx.ROOT / "tests/regress.py")
    reg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reg)
    fake = {"PY": "/opt/venv/bin/python", "WORK": "/home/u/mt", "PIN_SFT_REV": "deadbeef",
            "SCORE_REDO": "1", "PATH": "/usr/bin", "HOME": "/home/u"}
    dropped = reg.scrub_env(fake)
    ctx.check("regress scrubs operator env (PY/WORK/PIN_SFT_REV/switches) but keeps PATH and HOME",
              set(dropped) == {"PY", "WORK", "PIN_SFT_REV", "SCORE_REDO"} and fake == {"PATH": "/usr/bin", "HOME": "/home/u"},
              f"dropped={dropped} left={fake}")
    ctx.check("no operator env variable survives into a running suite",
              not [k for k in reg.LEAKY_ENV if k in os.environ], str([k for k in reg.LEAKY_ENV if k in os.environ]))
