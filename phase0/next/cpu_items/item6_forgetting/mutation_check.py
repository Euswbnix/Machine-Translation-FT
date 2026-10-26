"""Mutation check for forgetgenuse.py: each mutant must make test_forgetgenuse.py fail.
Run:  python mutation_check.py   (writes nothing in this folder; the log goes to stdout)
Each mutant is a copy of forgetgenuse.py with one exact substring replaced, run against an
unmodified copy of the tests in a temporary directory."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTANTS = [
    ("unclipped F", "F = min(fc.get(v, 0), n)", "F = fc.get(v, 0)"),
    ("unclipped B", "B = min(bc.get(v, 0), n)", "B = bc.get(v, 0)"),
    ("ID and OOD swapped", 'side = "ID" if v in id_types else "OOD"', 'side = "OOD" if v in id_types else "ID"'),
    ("denominator B instead of reference occurrences (the 2026-10-19 reading)",
     'd["forgetgenuse"] = c["lost"] / c["ref_occ"] if c["ref_occ"] else None',
     'd["forgetgenuse"] = c["lost"] / c["base_correct"] if c["base_correct"] else None'),
    ("|B - F| in place of max(0, B - F)", "L = B - F if B > F else 0", "L = abs(B - F)"),
    ("copy split inverted", 'copy = f"{side}_in_src" if v in ss else f"{side}_not_in_src"',
     'copy = f"{side}_not_in_src" if v in ss else f"{side}_in_src"'),
    ("> in place of >= for the min count", "c >= a.id_min_count}", "c > a.id_min_count}"),
    ("splitlines()", 'return [ln[:-1] if ln.endswith("\\n") else ln for ln in f]', "return f.read().splitlines()"),
]
src = (HERE / "forgetgenuse.py").read_text(encoding="utf-8")
ok = True
for name, old, new in MUTANTS:
    if src.count(old) != 1:
        print(f"SETUP ERROR  {name}: pattern found {src.count(old)} times"); ok = False; continue
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "forgetgenuse.py").write_text(src.replace(old, new), encoding="utf-8")
        shutil.copy(HERE / "test_forgetgenuse.py", d)
        p = subprocess.run([sys.executable, "-I", "test_forgetgenuse.py"], cwd=d, capture_output=True, text=True)
    last = (p.stderr.strip().splitlines() or ["?"])[-1]
    killed = p.returncode != 0
    ok &= killed
    print(f"{'killed' if killed else 'SURVIVED':8s}  {name}  ({last})")
with tempfile.TemporaryDirectory() as d:
    shutil.copy(HERE / "forgetgenuse.py", d); shutil.copy(HERE / "test_forgetgenuse.py", d)
    p = subprocess.run([sys.executable, "-I", "test_forgetgenuse.py"], cwd=d, capture_output=True, text=True)
print(f"unmutated: exit {p.returncode} ({(p.stderr.strip().splitlines() or ['?'])[-1]})")
print("MUTATION_CHECK", "OK" if ok and p.returncode == 0 else "FAILED")
sys.exit(0 if ok and p.returncode == 0 else 1)
