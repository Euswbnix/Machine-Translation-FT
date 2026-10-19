"""Check the staged April numpy copy against the conda package's own sha256 list, then report its version/dispatch."""
import hashlib, json, os, sys
B = os.path.expanduser("~/replay_check/build")
paths = json.load(open(f"{B}/np201/paths.json"))["paths"]
bad = n = 0
for p in paths:
    rel = p["_path"]
    pre = "lib/python3.10/site-packages/"
    if not rel.startswith(pre + "numpy/") or "sha256" not in p:
        continue
    f = os.path.join(B, "np201", rel[len(pre):])
    if not os.path.exists(f):
        continue  # __pycache__ etc.
    n += 1
    if hashlib.sha256(open(f, "rb").read()).hexdigest() != p["sha256"]:
        bad += 1
print(f"np201 files checked against conda paths.json: {n}, mismatched: {bad}")
