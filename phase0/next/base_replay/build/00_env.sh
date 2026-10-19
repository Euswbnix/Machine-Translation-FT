#!/bin/bash
# Record environment, verify inputs, stage a read-only copy of the April numpy build and the April sampler code.
set -euo pipefail
B=~/replay_check/build
W=~/workspace/Machine_translation
D=$W/data_enfr_v1
PY=~/miniconda/envs/DL/bin/python
cd $B
start=$(date +%s)
{
echo "date: $(date -Is)"; uname -a; lscpu | grep -E "Model name"; nproc
$PY -c "import sys,numpy,sentencepiece; print('python',sys.version.split()[0]); print('numpy',numpy.__version__); print('sentencepiece',sentencepiece.__version__)"
echo "--- sha256 of inputs"
nice -n 10 sha256sum $D/train.clean.en $D/train.clean.fr $D/spm_enfr_v1_fixed.model $D/train.clean.en.cached_256.npz
stat -c "%n %s %y" $D/train.clean.en $D/train.clean.fr $D/spm_enfr_v1_fixed.model $D/train.clean.en.cached_256.npz
echo "--- April code on box vs bde629a (sha256 prefixes recorded on the Mac: dataset 593d4550c323c71c, trainer 7cf423eec974462c)"
sha256sum $W/src/data/dataset.py $W/src/training/trainer.py
echo "--- line counts and CR counts"
for f in $D/train.clean.en $D/train.clean.fr; do echo "$f lines=$(wc -l < $f) CR=$(tr -cd '\r' < $f | wc -c)"; done
} > 00_env.log 2>&1
# April numpy (conda numpy-base 2.0.1 py310hb5e798b_1, recorded in the DL env's conda-meta/history and in swanlab's
# requirements.txt of both April-24 runs). Copy it (read only source) so nothing is written into ~/.conda.
P=~/.conda/pkgs/numpy-base-2.0.1-py310hb5e798b_1
mkdir -p np201
cp -r $P/lib/python3.10/site-packages/numpy np201/
cp $P/info/paths.json np201/paths.json
cp $W/src/data/dataset.py april_dataset.py
cp $W/src/training/trainer.py april_trainer.py
echo "staged in $(( $(date +%s) - start ))s" >> 00_env.log
