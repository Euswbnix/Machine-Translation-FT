"""rental_setup.sh must run on a box whose PATH has python3 but no python,
and must let the operator point PY at a venv or conda env so the box's own
environment is left alone. Ubuntu 24.04 ships python3 only; the previous
hard-coded `PY=python` died at the first stage there.
"""
import subprocess


def _py_line(root):
    for ln in (root / "phase0/rental_setup.sh").read_text(encoding="utf-8").splitlines():
        if ln.startswith("PY=") or ln.startswith('PY="'):
            return ln
    return ""


def _resolve(line, path, preset=None):
    env = {"PATH": str(path)}
    if preset:
        env["PY"] = preset
    # /bin/bash by absolute path: PATH here is deliberately restricted to `path`,
    # which is what makes "no python on this box" a real test.
    r = subprocess.run(["/bin/bash", "-c", f'{line}\nprintf "%s" "$PY"'],
                       capture_output=True, text=True, env=env)
    return r.stdout.strip()


def suite(ctx):
    line = _py_line(ctx.ROOT)
    ctx.check("rental_setup.sh sets PY on one line", bool(line), line)
    binp = ctx.d / "bin"
    binp.mkdir()
    py3 = binp / "python3"
    py3.write_text("#!/bin/sh\nexec /usr/bin/env true\n", encoding="utf-8")
    py3.chmod(0o755)
    got = _resolve(line, binp)
    ctx.check("PY falls back to python3 when the box has no 'python' (Ubuntu 24.04)",
              got == str(py3), f"resolved to {got!r}")
    got = _resolve(line, binp, preset="/opt/env/bin/python")
    ctx.check("PY is overridable, so deps go into a venv instead of the box's own environment",
              got == "/opt/env/bin/python", f"resolved to {got!r}")
    pyx = binp / "python"
    pyx.write_text("#!/bin/sh\nexec /usr/bin/env true\n", encoding="utf-8")
    pyx.chmod(0o755)
    got = _resolve(line, binp)
    ctx.check("PY still prefers 'python' when it exists (unchanged on the rental images)",
              got == str(pyx), f"resolved to {got!r}")
