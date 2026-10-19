#!/usr/bin/env bash
# End-to-end smoke run of forgetgenuse.py on the Mac Studio. NOT a measurement: newstest2014's
# references are not on the Studio, so a real E0.5 hypothesis file stands in for the reference.
#   hyps/baseline_v1.1_averaged.newstest2014.hyp   (sha256 21f41ffb...; E0.5 re-decode, BLEU 35.31)
#   hyps/ft_un_rep_lr1_s42_st2.newstest2014.hyp    (sha256 7455e7ba...; E0.5 re-decode, BLEU 28.09)
# --src is a stand-in too (heldout_un.en's first lines), used only for the line-count check and the
# copy split, which is therefore meaningless here.
# Checks (smoke_check.py, an independent recount): ref = base  -> base_correct = ref_occ, gained = 0,
# lost = ref_occ - clipped matches of ft against base; ref = ft -> lost = 0 in every split.
set -euo pipefail
cd ~/cpu_items/item6_forgetting
PY=~/e06env/venv-arm64/bin/python
D=~/e06data/phase0_e05
SPM=~/e06data/spm/spm_enfr_v1_fixed.model
S=smoke
mkdir -p $S
head -n 1000 hyps/baseline_v1.1_averaged.newstest2014.hyp > $S/base1000.hyp
head -n 1000 hyps/ft_un_rep_lr1_s42_st2.newstest2014.hyp > $S/ft1000.hyp
head -n 1000 ~/e06data/e05_src/heldout_un.en > $S/src1000_standin.en
NOTE="SMOKE: fake reference (an E0.5 hypothesis file); stand-in source; numbers check the code, not forgetting"

for TOK in words spm; do
  X=(--tok $TOK); [ $TOK = spm ] && X+=(--spm $SPM)
  # A: reference := base hypothesis; fine-tuning set ft_un_rep (the set the ft hypothesis was trained on)
  nice -n 5 $PY -I forgetgenuse.py --src $S/src1000_standin.en --ref $S/base1000.hyp --base-hyp $S/base1000.hyp \
    --ft-hyp $S/ft1000.hyp --ft-train $D/ft_un_rep --label ft_un_rep "${X[@]}" --fake-ref-note "$NOTE (ref = base)" \
    --out $S/A_refbase_$TOK.json
  # B: reference := fine-tuned hypothesis
  nice -n 5 $PY -I forgetgenuse.py --src $S/src1000_standin.en --ref $S/ft1000.hyp --base-hyp $S/base1000.hyp \
    --ft-hyp $S/ft1000.hyp --ft-train $D/ft_un_rep --label ft_un_rep "${X[@]}" --fake-ref-note "$NOTE (ref = ft)" \
    --out $S/B_refft_$TOK.json
  # V: vocabulary build time on every E0.5 training set (the cost a real run pays per set)
  for SET in ft_random_small ft_topk_div ft_topk_rep ft_un_div ft_un_random ft_un_rep; do
    nice -n 5 $PY -I forgetgenuse.py --src $S/src1000_standin.en --ref $S/base1000.hyp --base-hyp $S/base1000.hyp \
      --ft-hyp $S/ft1000.hyp --ft-train $D/$SET --label $SET "${X[@]}" --fake-ref-note "$NOTE (ref = base; vocab timing)" \
      --out $S/V_${SET}_$TOK.json > /dev/null
  done
done

# T: timing proxy at full size: the full 3,003-line files, the ft hypothesis passed 5 times (one set's 5 seeds).
# The stand-in source needs 3,003 lines: the base hypothesis file serves (French, so the copy split is meaningless).
for TOK in words spm; do
  X=(--tok $TOK); [ $TOK = spm ] && X+=(--spm $SPM)
  F=hyps/ft_un_rep_lr1_s42_st2.newstest2014.hyp
  /usr/bin/time -p nice -n 5 $PY -I forgetgenuse.py --src hyps/baseline_v1.1_averaged.newstest2014.hyp \
    --ref hyps/baseline_v1.1_averaged.newstest2014.hyp --base-hyp hyps/baseline_v1.1_averaged.newstest2014.hyp \
    --ft-hyp $F $F $F $F $F --ft-train $D/ft_un_rep --label ft_un_rep "${X[@]}" \
    --fake-ref-note "$NOTE (timing proxy: full files, ft hyp x5)" --out $S/T_full_x5_$TOK.json
done
nice -n 5 $PY -I smoke_check.py $S
echo SMOKE_EXIT=0
