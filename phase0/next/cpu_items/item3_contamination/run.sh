#!/usr/bin/env bash
# Item 3 on the Mac Studio (reached over Tailscale as alexs-mac-studio), in the order it is meant
# to run. Copy item3_contamination.py, spm_lengths.py and check.py to
# ~/cpu_items/item3_contamination/ first. At most 6 processes (5 workers + the parent), nice 5.
# Nothing is installed into the E0.6 reference venv (~/e06env/venv-arm64); it is only RUN, for
# stage 1, because it already has sentencepiece 0.2.2.
set -euo pipefail
D=~/cpu_items/item3_contamination
cd "$D"
mkdir -p logs uvcache
REF=~/e06env/venv-arm64/bin/python
PY="$D/venv/bin/python"
export UV_CACHE_DIR="$D/uvcache" UV_NO_CONFIG=1 UV_PYTHON_DOWNLOADS=never

# 0. A SEPARATE venv from the reference CPython 3.14.3, with numpy 2.4.5 (the reference venv's
#    numpy) and the one language-ID wheel the author approved: py3langid 0.4.0, py3-none-any,
#    hash-pinned (PyPI sha256 below), binary only, no dependency resolution.
if [ ! -x "$PY" ]; then
  ~/e06env/bin/uv venv --python ~/e06env/python/cpython-3.14.3-macos-aarch64-none/bin/python3.14 "$D/venv" 2>&1 | tee logs/install.log
  ~/e06env/bin/uv pip install --python "$PY" --only-binary :all: --no-deps numpy==2.4.5 2>&1 | tee -a logs/install.log
  printf 'py3langid==0.4.0 --hash=sha256:c25bd038147951ba3a89efe9ec12cc7d2116cb6f311ff8af84ade398a8bc0616\n' > req_lid.txt
  ~/e06env/bin/uv pip install --python "$PY" --only-binary :all: --no-deps --require-hashes -r req_lid.txt 2>&1 | tee -a logs/install.log
  ~/e06env/bin/uv pip check --python "$PY" 2>&1 | tee -a logs/install.log
fi

# 1. SentencePiece piece counts with the reference interpreter (single process).
nice -n 5 "$REF" -I -u spm_lengths.py --data ~/e06data --out "$D/spm_lengths.npz" > logs/spm.log 2>&1

# 2. The analysis (5 workers, nice 5), then the independent recount (5 workers).
nice -n 5 "$PY" -I -u item3_contamination.py --data ~/e06data --feas ~/e06runs/feasibility \
    --out "$D" --spm-lengths "$D/spm_lengths.npz" --workers 5 > logs/run.log 2>&1
nice -n 5 "$PY" -I -u check.py --data ~/e06data --res "$D/results.json" \
    --workers 5 > logs/check.log 2>&1
echo done > logs/DONE
