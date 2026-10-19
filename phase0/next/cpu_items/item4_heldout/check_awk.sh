#!/usr/bin/env bash
# Independent check of heldout_hygiene.py's EXACT counts, with awk and no Python.
# heldout_un (e05_src) against: the scored pool (fields 2 and 3 of v2_scored.tsv), the base's
# pretraining corpus (train.clean.en/.fr read line by line in step), and the two parents
# ft_topk / ft_random (every E0.5 and E0.6 set is a subset of one of them, so zero here is
# zero for every set). Byte comparison (LC_ALL=C); awk splits records on LF only.
#   HOME_DIR (default $HOME)   OUT (default $HOME/cpu_items/item4_heldout)
set -euo pipefail
H=${HOME_DIR:-$HOME}
O=${OUT:-$HOME/cpu_items/item4_heldout}
D=$H/e06data
mkdir -p "$O/check"
export LC_ALL=C
S=$D/e05_src/heldout_un.en
T=$D/e05_src/heldout_un.fr

LOAD='
function load(   l, n, m, i) {
  n = 0; while ((getline l < S) > 0) { s[l] = 1; src[++n] = l }
  m = 0; while ((getline l < T) > 0) { t[l] = 1; tgt[++m] = l }
  if (n != m) { print "heldout_un sides differ" > "/dev/stderr"; exit 2 }
  for (i = 1; i <= n; i++) p[src[i] SUBSEP tgt[i]] = 1
  N = n
}
function report(name,   i, hs, ht, hp) {
  hs = ht = hp = 0
  for (i = 1; i <= N; i++) { if (src[i] in fs) hs++; if (tgt[i] in ft) ht++; if ((src[i] SUBSEP tgt[i]) in fp) hp++ }
  printf "%s rows %d | rows with a heldout source %d, target %d, pair %d | heldout rows (of %d) whose source %d, target %d, pair %d occurs\n", \
    name, rows, cs + 0, ct + 0, cp + 0, N, hs, ht, hp
}'

# the scored pool, one pass
pool() {
  awk -F'\t' -v S="$S" -v T="$T" "$LOAD"'
  BEGIN { load() }
  { rows++; if (NF != 3) odd++
    if ($2 in s) { cs++; fs[$2] = 1 }
    if ($3 in t) { ct++; ft[$3] = 1 }
    if (($2 SUBSEP $3) in p) { cp++; fp[$2 SUBSEP $3] = 1 } }
  END { report("v2_scored.tsv"); printf "rows with other than 3 fields %d\n", odd + 0 }' "$D/v2_scored.tsv" > "$O/check/pool_awk.txt"
}

# two aligned files, source side as the main input and target side by getline
paired() {   # name en fr outfile
  awk -v S="$S" -v T="$T" -v FR="$3" "$LOAD"'
  BEGIN { load() }
  { rows++; e = $0; if ((getline f < FR) <= 0) { short++; f = "" }
    if (e in s) { cs++; fs[e] = 1 }
    if (f in t) { ct++; ft[f] = 1 }
    if ((e SUBSEP f) in p) { cp++; fp[e SUBSEP f] = 1 } }
  END { if ((getline f < FR) > 0) extra = 1
        report(NAME); printf "target side shorter by %d / longer: %d\n", short + 0, extra + 0 }' NAME="$1" "$2" > "$4"
}

pool & p1=$!
paired train.clean "$D/v1_fixed/train.clean.en" "$D/v1_fixed/train.clean.fr" "$O/check/pretrain_awk.txt" & p2=$!
( paired ft_topk "$D/e05_src/ft_topk.en" "$D/e05_src/ft_topk.fr" "$O/check/ft_topk_awk.txt"
  paired ft_random "$D/e05_src/ft_random.en" "$D/e05_src/ft_random.fr" "$O/check/ft_random_awk.txt" ) & p3=$!
rc=0
for p in $p1 $p2 $p3; do wait "$p" || rc=1; done   # a bare `wait` would hide a failed job
cat "$O"/check/*_awk.txt
exit $rc
