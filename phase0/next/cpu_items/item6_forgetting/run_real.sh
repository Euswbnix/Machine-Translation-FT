#!/usr/bin/env bash
# The real run of item 6 (NOT YET RUN: newstest2014's test.en / test.fr are on no reachable machine).
# 30 E0.5 runs = 6 sets x 5 seeds; one forgetgenuse.py call per set and tokeniser (12 calls).
#
# Needs, on the Studio:
#   $REF_DIR/test.en, $REF_DIR/test.fr   newstest2014 en-fr exactly as E0.5 scored it
#        (Machine_translation data_enfr_v1/test.{en,fr}; sha256 967dca93... / 82526d36..., checked)
#   $HYP_DIR/   the 31 *.newstest2014.hyp files of ~/mt_local/phase0_final/e05_redecode/ on the Mac
#        plus that folder's redecode_report.json (every hypothesis's sha256 is checked against it)
#   ~/e06data/phase0_e05/<set>.{en,fr} and ~/e06data/spm/spm_enfr_v1_fixed.model (already there)
# Usage:  ./run_real.sh <REF_DIR> <HYP_DIR> [OUT_DIR=real]
set -euo pipefail
cd ~/cpu_items/item6_forgetting
REF_DIR=${1:?REF_DIR}; HYP_DIR=${2:?HYP_DIR}; OUT=${3:-real}
PY=~/e06env/venv-arm64/bin/python
D=~/e06data/phase0_e05
SPM=~/e06data/spm/spm_enfr_v1_fixed.model
SETS="ft_random_small ft_topk_div ft_topk_rep ft_un_div ft_un_random ft_un_rep"
SEEDS="42 1 2 3 4"
for f in "$REF_DIR/test.en" "$REF_DIR/test.fr" "$HYP_DIR/redecode_report.json"; do
  [ -f "$f" ] || { echo "REFUSED: missing $f"; exit 1; }
done
# every hypothesis must be the re-decode's file, byte for byte
$PY -I - "$HYP_DIR" $SETS <<'PY'
import hashlib, json, sys
from pathlib import Path
d, sets = Path(sys.argv[1]), sys.argv[2:]
want = {r["hyp"]: r["hyp_sha256"] for r in json.load(open(d / "redecode_report.json"))["rows"]}
names = ["baseline_v1.1_averaged.newstest2014.hyp"] + [f"{s}_lr1_s{k}_st2.newstest2014.hyp" for s in sets for k in (42, 1, 2, 3, 4)]
for n in names:
    p = d / n
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != want.get(n):
        sys.exit(f"REFUSED: {p} missing or not the re-decode's file")
print(f"{len(names)} hypothesis files match redecode_report.json")
PY
mkdir -p "$OUT"
for TOK in words spm; do
  X=(--tok $TOK); [ $TOK = spm ] && X+=(--spm $SPM)
  for SET in $SETS; do
    HYPS=(); for K in $SEEDS; do HYPS+=("$HYP_DIR/${SET}_lr1_s${K}_st2.newstest2014.hyp"); done
    nice -n 5 $PY -I forgetgenuse.py --src "$REF_DIR/test.en" --ref "$REF_DIR/test.fr" \
      --expect-src-sha newstest2014 --expect-ref-sha newstest2014 \
      --base-hyp "$HYP_DIR/baseline_v1.1_averaged.newstest2014.hyp" --ft-hyp "${HYPS[@]}" \
      --ft-train "$D/$SET" --label "$SET" "${X[@]}" --out "$OUT/${SET}_$TOK.json"
  done
done
# one table: mean (sd) over the 5 seeds
$PY -I - "$OUT" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = {}
for p in sorted(d.glob("*_words.json")) + sorted(d.glob("*_spm.json")):
    o = json.loads(p.read_text(encoding="utf-8")); s = o["summary_over_runs"]
    rows[p.stem] = {k: {"mean": s[f"{k}.forgetgenuse"]["mean"], "sd": s[f"{k}.forgetgenuse"]["sd"]}
                    for k in ("All", "ID", "OOD", "OOD_in_src", "OOD_not_in_src")}
    rows[p.stem]["ref_occ_share_ID"] = o["runs"][0]["results"]["ref_occ_share_ID"]
    f = lambda k: f"{rows[p.stem][k]['mean']:.4f} ({rows[p.stem][k]['sd']:.4f})"
    print(f"{p.stem:24s} All {f('All')}  ID {f('ID')}  OOD {f('OOD')}  ref share ID {rows[p.stem]['ref_occ_share_ID']:.4f}")
(d / "summary.json").write_text(json.dumps(rows, indent=1))
PY
echo REAL_EXIT=0
