#!/usr/bin/env python3
"""Content fingerprint of an E0.3 controls directory (training-path finding: stale
runs and stale BLEU cache after a controls rebuild).

controls_sha = sha256 over "<name>\\t<sha256>\\n" for manifest.json and every file the
manifest lists under "written", in sorted name order. e03_run_matrix.py puts it into
matrix.json and into every checkpoint directory name; run_parallel.py writes it next to
each run and refuses a final.pt whose marker differs; e03_collect.py refuses runs whose
marker differs from the controls it is evaluating on.

    python phase0/controls_fingerprint.py data/phase0      # prints the sha
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MARKER = "controls_sha"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(ctrl_dir) -> dict:
    """-> {"controls_sha": hex, "files": {name: sha256}}; SystemExit if the manifest or a
    listed file is missing."""
    ctrl = Path(ctrl_dir)
    mp = ctrl / "manifest.json"
    if not mp.is_file():
        raise SystemExit(f"{mp} missing: build the controls first (e03_build_controls.py)")
    man = json.load(open(mp, encoding="utf-8"))
    names = sorted({"manifest.json", *[str(x) for x in man.get("written", [])]})
    files = {}
    for n in names:
        p = ctrl / n
        if not p.is_file():
            raise SystemExit(f"{p} is listed in manifest.json 'written' but missing")
        files[n] = _sha(p)
    blob = "".join(f"{n}\t{files[n]}\n" for n in names).encode()
    return {"controls_sha": hashlib.sha256(blob).hexdigest(), "files": files}


def check_controls(runner, runs, cwd) -> str | None:
    """Pre-launch guard (audit F10): the control-set directory the runs train on must still
    fingerprint to matrix.json's controls_sha. -> None if OK or the guard is inactive (no
    controls_sha), else a message. The directory is matrix.json "data_dir" (relative to
    cwd), else the unique parent of the runs' data.train_src."""
    mp = Path(runner).parent / "matrix.json"
    if not mp.is_file():
        return None
    m = json.load(open(mp, encoding="utf-8"))
    want = m.get("controls_sha")
    if not want:
        return None
    if m.get("data_dir"):
        dirs = {Path(m["data_dir"])}
    else:
        dirs = {Path(r["cfg"].get("data", {}).get("train_src", "")).parent for r in runs
                if isinstance(r.get("cfg"), dict) and r["cfg"].get("data", {}).get("train_src")}
    if not dirs:
        print(f"  ({mp} has controls_sha but no data_dir and the configs name no train_src: "
              "control-set content NOT verified before launch)")
        return None
    if len(dirs) != 1:
        return f"runs train on several data dirs {sorted(map(str, dirs))}; expected one controls dir"
    d = next(iter(dirs))
    d = d if d.is_absolute() else Path(cwd) / d
    try:
        fp = fingerprint(d)
    except SystemExit as e:
        return f"cannot fingerprint {d}: {e}"
    if fp["controls_sha"] != want:
        files = m.get("controls_files") or {}
        diff = sorted(n for n in set(files) | set(fp["files"]) if files.get(n) != fp["files"].get(n))
        return (f"{d} fingerprints to {fp['controls_sha'][:12]}... but matrix.json was generated for "
                f"{want[:12]}...; differing files: {diff if files else '(matrix.json lists none)'}. "
                "The control sets changed after the matrix (e.g. a failed 'controls' rebuild): rerun 'controls'.")
    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: controls_fingerprint.py <controls-dir>")
    print(fingerprint(sys.argv[1])["controls_sha"])
