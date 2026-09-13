# tests/suites

`python3 tests/regress.py` runs every `*.py` here after its built-in suites, in name order.

Contract: each module defines `suite(ctx)`. `ctx` has
- `d`: a fresh fixture directory for this suite only,
- `check(name, cond, detail="")`, `skip(name, why)`,
- `run(args, env=None) -> (rc, stdout+stderr)`: runs `python <args>` with cwd = repo root,
- `ROOT`, `PY`, `MT_REPO`, `np`.

Generate every fixture from scratch inside `ctx.d`; never read session state. Drive
scripts through their real CLI, and check results by independent recomputation.
An exception inside `suite` is reported as a FAIL, not a crash of the whole run.
