#!/usr/bin/env bash
# Phase 0 on a rented GPU box, from nothing.
# USAGE-BEGIN
#   bash rental_setup.sh repos        clone/check out BOTH repos at their pinned commits and verify (no install)
#   bash rental_setup.sh env          repos + install PINNED deps, apply trainer patch, self-test   (no GPU)
#   bash rental_setup.sh accept       pinned newstest2013/2014 + Base v1.1 from HF, reproduce test BLEU (1 GPU)
#   bash rental_setup.sh data         pinned WMT14 parquet -> CR-safe v2 corpus AND v1.1 pretraining
#                                     corpus, both sha-checked; writes .sha_verified markers  (CPU, ~1 h)
#   bash rental_setup.sh score        per phase0/e03_decisions.json score_mode:
#                                       none  - reused scores only, nan elsewhere (no GPU, no HF login)
#                                       reuse - score the 21,628,292 new pairs on all GPUs, merge
#                                       full  - score all 38,275,284 pairs, calibrate vs reused scores
#                                     SCORE_SMOKE=1: 3,000-row 2-GPU order/consistency check first
#   bash rental_setup.sh provenance   bundle labels (SHA256SUMS-verified) or statmt + e01   (CPU)
#   bash rental_setup.sh controls     E0.3 control sets + run matrix, from the frozen decisions (CPU)
#   bash rental_setup.sh stage1       cache warm-up, 4-run LR sweep, mechanical LR selection  (GPUs)
#   bash rental_setup.sh stage2 <lr>  cache warm-up, 3 conditions x 3 seeds at the SELECTED lr_scale (GPUs)
#   bash rental_setup.sh gate <lr>    evaluate every run, apply the pre-registered rule      (GPUs)
#
# Explicit switches (all default off; each one that bends the protocol writes a line to
# results/phase0/deviations.txt):
#   PIN_SFT_REV=<40-hex>   repos/env: check out this commit of Machine-Translation-SFT (detached)
#   MT_REV=<40-hex>        repos/env: Machine_translation commit (default: the pinned one below)
#   SCORE_REDO=1           score: rescore although a completed v2_scored.tsv exists for other
#                          scoring decisions (score_mode / calibration)
#   RECALIBRATE=1          score (full): re-run only 'calibrate' on the kept
#                          v2_scored.calibration_failed.tsv with changed thresholds (no GPU)
#   DECISIONS_AMEND=1      score/controls: accept a decisions change after E0.3 results exist
#   DECISIONS_UNTAGGED=1   any stage: skip the committed/tagged/pushed check on the decisions
#   LR_OVERRIDE=1          stage2: run an lr_scale other than the stage-1 selection
#   SCORE_ALLOW_STACK_CHANGE=1  score (reuse/full): let a resumed shard continue under a changed
#                          scoring stack (passes --allow-stack-change; recorded in segments).
#                          reuse then warns; full still refuses to calibrate mixed-stack scores
# Run 'score', 'stage1', 'stage2' and 'gate' inside tmux. Never kill only a parent process.
# USAGE-END
#
# Inputs you place on the box before 'score':
#   $WORK/rental_bundle_v2.tar.gz  (or its members already unpacked in $WORK/rescore/)
#   $SFT/phase0/e03_decisions.json committed, tagged and pushed in the $SFT clone (not copied
#                                  in by hand); template: e03_decisions.example.json
#
# Status: the PRE-pinning 'env' and 'accept' ran on a real box (2026-09-13; test BLEU 35.31).
# The current pinned 'env' (repo pins, pip pins, Python guard) and pinned 'accept'
# (fetch_pinned + hf_devtest_to_text.py, hf_to_ckpt --expect-sha) have NOT run on a box, nor
# has anything after them. 'accept' and 'data' onward are exercised offline by
# tests/suites/rental.py with fake GPUs, downloads, scorer and trainer; 'env's Python-version,
# pin and CUDA guards are exercised offline with a fake python (its real install is not).
#
# Disk (measured on the Mac unless marked derived): fr-en train parquet 7,776,517,237 B;
# fixed v2 corpus 13,777,607,691 B; v1.1 corpus 3,103,997,455 B; to_score.* 7,890,306,262 B;
# merged scored TSV ~14.2 GB (derived); checkpoints ~0.7 GB each (derived, not measured).
# Each stage checks free space against those numbers plus margin (see need_free_gb calls).

set -euo pipefail
WORK="${WORK:-$HOME/mt}"
MT="$WORK/Machine_translation"
# The SFT configs hard-code ../Machine-Translation-SFT/... paths, so the clone
# directory must keep this name.
SFT="$WORK/Machine-Translation-SFT"
BRANCH="${BRANCH:-wmt2027-phase0}"
BUNDLE="$WORK/rescore"
DECISIONS_REL=phase0/e03_decisions.json
DECISIONS="$SFT/$DECISIONS_REL"
RESULTS="$SFT/results/phase0"
PY=python

# ---- pinned inputs. `pin VAR value` keeps an environment value (tests/suites/rental.py
# ---- substitutes toy data that way) and warns at every stage when one differs from the pin.
# ---- Never override these on a real box.
PIN_OVERRIDES=""
pin() {
  local cur="${!1:-}"
  if [ -z "$cur" ]; then
    printf -v "$1" '%s' "$2"
  elif [ "$cur" != "$2" ]; then
    PIN_OVERRIDES="${PIN_OVERRIDES:+$PIN_OVERRIDES }$1"
  fi
}
pin HF_WMT14_REV b199e406369ec1b7634206d3ded5ba45de2fe696   # wmt/wmt14 main, lastModified 2024-04-03
# Machine_translation (training repo, read-only here): the commit the trainer patch, the paper
# and the local review were checked against (2026-09-13). Changing it is a deliberate, logged change.
pin MT_REV 200f6c0624c0c1f810e639dde3c53545f359849e
# Mac rebuild 2026-09-13 (~/mt_local/rebuild/v2_fixed): 38,275,284 rows
pin EXPECTED_V2_FIXED_SHA_EN c8cc665c8fc9971bf689bb02d7409bc76bbb489cb7b7dab6b35d13778863e655
pin EXPECTED_V2_FIXED_SHA_FR e4f5301af4c49b8cd91ff81807db08116a7da94f952ba0ee3962e82c0618421a
# v1.1 pretraining corpus = first 10,000,000 HF rows, fixed mode (byte-identical to the published
# data_enfr_v1, 9,312,233 rows). Used only to exclude pretraining pairs from held-out sets.
pin EXPECTED_V1_SHA_EN 7230adc6c823d4df51407e445b1181ada450363dfe43507885c9f529d49b6b6e
pin EXPECTED_V1_SHA_FR b0a6ba91eeba2b4bdf28e8a73816a2bdd7ccbddd5f889fcc7d3f3f0a52dad7e2
# sha256 of the SHA256SUMS inside rental_bundle_v2.tar.gz (measured on the Mac 2026-09-13).
# A future bundle must update this pin.
pin EXPECTED_BUNDLE_SUMS_SHA 6bbd472458a31dcce9bbda6a0ac170cc022c985c6b9abe58e9ac68887ba501d4
V1_MAX_ROWS=10000000
# newstest2013 (valid) / newstest2014 (test): parquet at the pinned revision (HF tree API: size, LFS
# sha256) and the text the paper used (sha256 of ~/mt_fetch/.../data_enfr_v1/{valid,test}.{en,fr}).
DEV_PARQUET=fr-en/validation-00000-of-00001.parquet; TEST_PARQUET=fr-en/test-00000-of-00001.parquet
pin DEV_PARQUET_SIZE 474541
pin DEV_PARQUET_SHA 113e01f028c35bc5ac4286ce95ac9dcfb70badc3e3c71c91ab7d74fe6357fb5e
pin TEST_PARQUET_SIZE 535966
pin TEST_PARQUET_SHA 8b9f62543c776b5e992d15524f76f9fe589d838c1dcb04a8738c21e8c42269f1
pin VALID_EN_SHA ace268a1d9724ca76f6d8e5c720ffdc08f01bdce45630ed8b269dc7918e1c16e
pin VALID_FR_SHA 909b63be5df3eb8c13e1d5967bf3f0adfdb9f6ed5e638c5bffff563ee2ff7f41
pin TEST_EN_SHA 967dca935c7cde4ffc2fc8c9424c682335e1690440dbda104948eb1ffa537fc2
pin TEST_FR_SHA 82526d36ae1cd4d5f20919fbf8130e519cc68a0e57fe47bb205d199626457de9
# euswbnix/transformer-wmt14-enfr-base, commit and LFS sha256 from the HF API (2026-09-13)
HF_BASE_REPO=euswbnix/transformer-wmt14-enfr-base
pin HF_BASE_REV 8a58dcc4205991a582c7868b99d1def473dcb46f
pin HF_BASE_WEIGHTS_SHA 4d7eb2bbfecd31cca6c85143bf41a865c41022db803fb95d8a57ef718f4a2536
pin HF_BASE_SPM_SHA d1c10ea80fb95984f3f240b3579c87a16313b77aa9bc103ca0d6a0e06672a561
pin EXPECTED_BASE_TEST_BLEU 35.31      # HF config.json and 05_sft.tex:26
pin BLEU_TOL 0.15                      # decode batch size can move BLEU by a few hundredths
# Scoring stack (PyPI JSON read 2026-09-13). unbabel-comet 2.2.7 is the newest release
# (2025-09-01) and requires pytorch-lightning>=2,<3, transformers>=4.17,<5, numpy<2.
# pytorch-lightning 2.5.5 (2025-09-05) is the release of comet 2.2.7's era; 2.6.6 appeared
# 2026-09-10 and has never been used with comet. transformers 4.57.1 satisfies <5 (not
# verified to be the newest 4.x). numpy 1.26.4 is the newest <2 and has wheels for cp39-cp312
# only, hence the Python 3.10-3.12 guard in env.
PIN_COMET="unbabel-comet==2.2.7"
PIN_LIGHTNING="pytorch-lightning==2.5.5"
PIN_TRANSFORMERS="transformers==4.57.1"
PIN_NUMPY="numpy==1.26.4"
# RAM for 'controls': the controls lane measured 613,883,904 B peak RSS for 2,014,489 rows with
# --pool-mask and --pretrain-* (304.7 B/row). Linear extrapolation to 38,275,284 rows: ~11.7 GB
# (NOT measured at full scale). Guard at 16 GiB MemAvailable.
CONTROLS_MIN_MEM_GB="${CONTROLS_MIN_MEM_GB:-16}"
MEMINFO="${MEMINFO:-/proc/meminfo}"      # test hook
# Inside a container (vast.ai is Docker) /proc/meminfo shows the HOST; the container's limit
# is in its cgroup. need_mem_gb takes the smaller of the two.
CGROUP_MEM_DIR="${CGROUP_MEM_DIR:-/sys/fs/cgroup}"   # test hook

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }
utc() { date -u +%FT%TZ; }
fsize() { wc -c < "$1" | tr -d ' '; }
nrows() { wc -l < "$1" | tr -d ' '; }
sha_of() { sha256sum "$1" | awk '{print $1}'; }
is_sha40() { printf '%s' "$1" | grep -qE '^[0-9a-f]{40}$'; }

note_deviation() {   # note_deviation <dedupe key> <line>: append once to results/phase0/deviations.txt
  mkdir -p "$RESULTS"
  grep -qF -- "$1" "$RESULTS/deviations.txt" 2>/dev/null || printf '%s\n' "$2" >> "$RESULTS/deviations.txt"
  echo "DEVIATION recorded in $RESULTS/deviations.txt: $2"
}

need_free_gb() {   # need_free_gb <GB> <why>
  local avail
  avail="${DF_KB_OVERRIDE:-$(df -Pk "$WORK" | awk 'NR==2 {print $4}')}"    # DF_KB_OVERRIDE: test hook
  [ "$avail" -ge $(( $1 * 1024 * 1024 )) ] || die "$2 needs >= $1 GB free under $WORK; $(( avail / 1024 / 1024 )) GB available"
}

cgroup_headroom_kb() {   # prints "<kB> <source>" for the tightest cgroup memory limit, or nothing
  local c="$CGROUP_MEM_DIR" cur inact lim f best="" src=""
  if [ -r "$c/memory.current" ]; then                                  # cgroup v2
    cur=$(cat "$c/memory.current")
    inact=$(awk '$1=="inactive_file" {print $2}' "$c/memory.stat" 2>/dev/null || true)
    for f in memory.max memory.high; do
      [ -r "$c/$f" ] || continue
      lim=$(cat "$c/$f")
      [ "$lim" = max ] && continue
      lim=$(( (lim - cur + ${inact:-0}) / 1024 ))
      if [ -z "$best" ] || [ "$lim" -lt "$best" ]; then best=$lim; src="$c/$f"; fi
    done
  elif [ -r "$c/memory/memory.limit_in_bytes" ] && [ -r "$c/memory/memory.usage_in_bytes" ]; then   # cgroup v1
    lim=$(cat "$c/memory/memory.limit_in_bytes")
    if [ "${#lim}" -lt 19 ] || [ "$lim" -lt 4611686018427387904 ]; then   # >= 2^62 means unlimited
      cur=$(cat "$c/memory/memory.usage_in_bytes")
      inact=$(awk '$1=="total_inactive_file" {print $2}' "$c/memory/memory.stat" 2>/dev/null || true)
      best=$(( (lim - cur + ${inact:-0}) / 1024 )); src="$c/memory/memory.limit_in_bytes"
    fi
  fi
  [ -n "$best" ] && echo "$best $src"
  return 0
}

need_mem_gb() {    # need_mem_gb <GiB> <why>
  local kb cg cg_kb src="MemAvailable in $MEMINFO"
  [ -r "$MEMINFO" ] || die "cannot read $MEMINFO to check RAM for $2"
  kb=$(awk '/^MemAvailable:/ {print $2}' "$MEMINFO")
  [ -n "$kb" ] || die "no MemAvailable in $MEMINFO"
  cg=$(cgroup_headroom_kb)
  if [ -n "$cg" ]; then
    cg_kb=${cg%% *}
    if [ "$cg_kb" -lt "$kb" ]; then kb=$cg_kb; src="cgroup limit ${cg#* } minus usage (inactive file cache excluded)"; fi
  else
    echo "note: no cgroup memory limit found under $CGROUP_MEM_DIR; using $MEMINFO (inside a container that shows the host; rent >= 32 GB RAM)"
  fi
  [ "$kb" -ge $(( $1 * 1024 * 1024 )) ] || die "$2 needs >= $1 GiB of memory headroom (extrapolated peak ~11.7 GB); $src gives $(( kb / 1024 / 1024 )) GiB"
}

ngpus() { nvidia-smi -L | grep -c '^GPU ' || true; }

# ---- markers: written only after a verified rebuild; required by every consumer
write_marker() {   # write_marker <dir> <sha_src> <sha_tgt> <src> <tgt>
  printf 'sha256_src %s\nsha256_tgt %s\nbytes_src %s\nbytes_tgt %s\nrows %s\n' \
    "$2" "$3" "$(fsize "$4")" "$(fsize "$5")" "$(nrows "$4")" > "$1/.sha_verified.tmp"
  mv "$1/.sha_verified.tmp" "$1/.sha_verified"
}

require_verified() {   # require_verified <dir> <sha_src> <sha_tgt> <src> <tgt>
  local m="$1/.sha_verified"
  [ -f "$m" ] || die "$1 is not verified (no .sha_verified): run 'data' and let it finish"
  [ "$(awk '$1=="sha256_src"{print $2}' "$m")" = "$2" ] && [ "$(awk '$1=="sha256_tgt"{print $2}' "$m")" = "$3" ] \
    || die "$m records other sha256 values than this script pins; rerun 'data'"
  [ "$(awk '$1=="bytes_src"{print $2}' "$m")" = "$(fsize "$4")" ] && [ "$(awk '$1=="bytes_tgt"{print $2}' "$m")" = "$(fsize "$5")" ] \
    || die "$4 / $5 changed size since verification; rerun 'data'"
}

require_v2() { require_verified "$MT/data_enfr_v2" "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" \
  "$MT/data_enfr_v2/train.clean.en" "$MT/data_enfr_v2/train.clean.fr"; }
require_v1() { require_verified "$MT/data_enfr_v1_pretrain" "$EXPECTED_V1_SHA_EN" "$EXPECTED_V1_SHA_FR" \
  "$MT/data_enfr_v1_pretrain/train.clean.en" "$MT/data_enfr_v1_pretrain/train.clean.fr"; }

# ---- bundle v2 from the Mac: (re)unpack when the tarball changed, verify every member against
# ---- SHA256SUMS, and SHA256SUMS itself against the pinned sha
BUNDLE_MEMBERS="plan.json missing_rows.npy reuse_scores.npy provenance_labels.npy provenance_report.json pool_mask_reused.npy"
bundle_ready() {
  mkdir -p "$BUNDLE"
  local tar="$WORK/rental_bundle_v2.tar.gz" tsha="" f
  if [ -f "$tar" ]; then
    tsha=$(sha_of "$tar")
    if [ ! -f "$BUNDLE/SHA256SUMS" ] || [ "$(cat "$BUNDLE/.tar_sha256" 2>/dev/null)" != "$tsha" ]; then
      log "unpacking $tar (sha256 ${tsha:0:12}) into $BUNDLE"
      for f in $BUNDLE_MEMBERS SHA256SUMS .tar_sha256; do rm -f "$BUNDLE/$f"; done
      tar -xzf "$tar" -C "$BUNDLE"
      echo "$tsha" > "$BUNDLE/.tar_sha256"
    fi
  fi
  [ -f "$BUNDLE/SHA256SUMS" ] || die "no bundle: copy rental_bundle_v2.tar.gz to $WORK/ (see RUNBOOK)"
  for f in $BUNDLE_MEMBERS; do
    grep -q "  $f\$" "$BUNDLE/SHA256SUMS" || die "bundle SHA256SUMS does not list $f (old bundle?)"
  done
  (cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) || die "bundle member(s) differ from SHA256SUMS; re-copy the bundle"
  [ "$(sha_of "$BUNDLE/SHA256SUMS")" = "$EXPECTED_BUNDLE_SUMS_SHA" ] \
    || die "bundle is not the pinned v2 bundle: SHA256SUMS sha256 $(sha_of "$BUNDLE/SHA256SUMS") != pinned $EXPECTED_BUNDLE_SUMS_SHA"
  "$PY" - "$BUNDLE/plan.json" "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" <<'PY' || die "bundle plan.json was made for another corpus"
import json, sys
p = json.load(open(sys.argv[1]))
sys.exit(0 if (p["new_src_sha256"], p["new_tgt_sha256"]) == (sys.argv[2], sys.argv[3]) else 1)
PY
}

# ---- decisions
D_COMMIT=""; D_TAG=""
parse_decisions() {
  local vals
  vals=$("$PY" "$SFT/phase0/e03_decisions.py" shell "$DECISIONS") \
    || die "E0.3 decisions missing or incomplete (listed above). Nothing chosen for you: see PROTOCOL.md, 'E0.3 decisions required before controls'. (If you already committed $DECISIONS_REL, PIN_SFT_REV may predate that commit: re-pin to it and rerun 'repos'.)"
  eval "$vals"
}

require_committed_decisions() {
  # Pre-registration evidence: the decisions must be a tracked, unmodified, tagged commit that is
  # on a remote branch (pushed before the results exist), not a file copied onto the box.
  if [ "${DECISIONS_UNTAGGED:-0}" = 1 ]; then
    D_COMMIT=unverified; D_TAG=unverified
    note_deviation "DECISIONS_UNTAGGED $D_SHA256" \
      "$(utc) DECISIONS_UNTAGGED=1: $DECISIONS_REL (sha256 $D_SHA256) used without the committed/tagged/pushed check"
    return 0
  fi
  local hint="(DECISIONS_UNTAGGED=1 proceeds and records a deviation)" c
  git -C "$SFT" rev-parse --git-dir >/dev/null 2>&1 || die "$SFT is not a git clone; the decisions must be committed there $hint"
  git -C "$SFT" ls-files --error-unmatch "$DECISIONS_REL" >/dev/null 2>&1 \
    || die "$DECISIONS_REL is not tracked by git: commit, tag and push it before 'score' $hint"
  git -C "$SFT" diff --quiet HEAD -- "$DECISIONS_REL" \
    || die "$DECISIONS_REL differs from its committed version (HEAD): commit, tag and push the change, or revert it $hint"
  c=$(git -C "$SFT" log -1 --format=%H -- "$DECISIONS_REL")
  D_TAG=$(git -C "$SFT" tag --contains "$c" | head -n 1)
  [ -n "$D_TAG" ] || die "no git tag contains commit ${c:0:12}, which last changed $DECISIONS_REL: tag it and push the tag $hint"
  [ -n "$(git -C "$SFT" branch -r --contains "$c" 2>/dev/null)" ] \
    || die "commit ${c:0:12} ($DECISIONS_REL) is on no remote branch in this clone: push it, then 'git -C $SFT fetch' $hint"
  D_COMMIT=$c
}

load_decisions() { parse_decisions; require_committed_decisions; }

require_frozen_decisions() {
  parse_decisions
  [ -f "$RESULTS/decisions.sha256" ] || die "run 'controls' first (it records the decisions the matrix was built from)"
  [ "$(cat "$RESULTS/decisions.sha256")" = "$D_SHA256" ] \
    || die "$DECISIONS changed after 'controls' (sha256 $D_SHA256 vs recorded $(cat "$RESULTS/decisions.sha256")). Revert it, or rerun 'controls' and every stage after it."
  require_committed_decisions
}

downstream_output() {   # prints the first existing E0.3 result, or nothing
  local p
  for p in "$RESULTS/lr_selection.json" "$SFT/logs/phase0/stage1" "$SFT/logs/phase0/stage2" \
           "$RESULTS/.e03_collect_cache.json" "$RESULTS"/phase0_bleu*.tsv; do
    [ -e "$p" ] && { echo "$p"; return 0; }
  done
  if [ -d "$MT/checkpoints/phase0" ]; then
    p=$(find "$MT/checkpoints/phase0" -name final.pt -print 2>/dev/null | head -n 1)
    [ -n "$p" ] && echo "$p"
  fi
  return 0
}

guard_decisions_amend() {   # guard_decisions_amend <stage>
  # decisions_history.tsv is append-only and never deleted, so a half-finished 'controls' cannot
  # open a gap through which a changed rule slips in unrecorded.
  local hist="$RESULTS/decisions_history.tsv" last down changed
  [ -s "$hist" ] || return 0
  last=$(tail -n 1 "$hist" | cut -f2)
  [ "$last" = "$D_SHA256" ] && return 0
  down=$(downstream_output)
  [ -n "$down" ] || return 0
  changed=$("$PY" "$SFT/phase0/e03_decisions.py" diff "$RESULTS/decisions.${last:0:12}.json" "$DECISIONS" 2>/dev/null || echo unknown)
  [ "${DECISIONS_AMEND:-0}" = 1 ] || die "decisions changed after E0.3 results exist ($down): sha256 ${last:0:12} -> ${D_SHA256:0:12}, keys: $changed.
Changing the rule after results exist is a pre-registration deviation. Revert $DECISIONS_REL, or set
DECISIONS_AMEND=1 to record the deviation in $RESULTS/deviations.txt and rebuild."
  note_deviation "decisions amended $last -> $D_SHA256" \
    "$(utc) decisions amended after E0.3 results existed ($down): sha256 $last -> $D_SHA256, changed keys: $changed (first seen at '$1')"
}

record_decisions() {   # after a successful 'controls'
  mkdir -p "$RESULTS"
  local hist="$RESULTS/decisions_history.tsv"
  [ -s "$hist" ] || printf 'utc\tdecisions_sha256\tsft_head\tdecisions_commit\tdecisions_tag\n' > "$hist"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(utc)" "$D_SHA256" "$(git -C "$SFT" rev-parse HEAD 2>/dev/null || echo none)" \
    "$D_COMMIT" "$D_TAG" >> "$hist"
  cp "$DECISIONS" "$RESULTS/decisions.${D_SHA256:0:12}.json"
  cp "$DECISIONS" "$RESULTS/decisions.json"
  printf 'commit %s\ntag %s\n' "$D_COMMIT" "$D_TAG" > "$RESULTS/decisions.git"
  echo "$D_SHA256" > "$RESULTS/decisions.sha256"
}

warn_pin_overrides() {
  [ -z "$PIN_OVERRIDES" ] || echo "WARNING: pinned values overridden from the environment (tests only): $PIN_OVERRIDES" >&2
}

stage_repos() {
  mkdir -p "$WORK"
  [ -d "$MT/.git" ]  || git clone https://github.com/Euswbnix/Machine_translation.git "$MT"
  [ -d "$SFT/.git" ] || git clone -b "$BRANCH" https://github.com/Euswbnix/Machine-Translation-SFT.git "$SFT"
  local before after f dirty untracked patch_files
  before=$(git -C "$SFT" rev-parse HEAD)
  if [ -n "${PIN_SFT_REV:-}" ]; then
    is_sha40 "$PIN_SFT_REV" || die "PIN_SFT_REV must be a full 40-hex commit sha, got '$PIN_SFT_REV'"
    git -C "$SFT" cat-file -e "$PIN_SFT_REV^{commit}" 2>/dev/null || git -C "$SFT" fetch -q --tags origin \
      || die "cannot fetch $SFT"
    git -C "$SFT" checkout -q --detach "$PIN_SFT_REV" 2>/dev/null || die "cannot check out PIN_SFT_REV $PIN_SFT_REV in $SFT (not pushed?)"
    [ "$(git -C "$SFT" rev-parse HEAD)" = "$PIN_SFT_REV" ] || die "$SFT HEAD $(git -C "$SFT" rev-parse HEAD) != PIN_SFT_REV $PIN_SFT_REV"
  else
    { git -C "$SFT" fetch -q --tags && git -C "$SFT" checkout -q "$BRANCH" && git -C "$SFT" pull -q --ff-only; } \
      || die "cannot update $SFT to origin/$BRANCH"
    echo "WARNING: PIN_SFT_REV not set; $SFT follows $BRANCH at $(git -C "$SFT" rev-parse HEAD). Set PIN_SFT_REV to the pushed, reviewed commit."
  fi
  after=$(git -C "$SFT" rev-parse HEAD)
  for f in phase0/score_sharded.py phase0/e03_decisions.py phase0/hf_devtest_to_text.py phase0/controls_fingerprint.py; do
    [ -f "$SFT/$f" ] || die "$SFT/$f is missing at ${after:0:12}: the reviewed branch was not pushed? (RUNBOOK, 'Things only you can do', step 0)"
  done
  echo "$after" > "$WORK/sft_rev.txt"
  if [ ! -f "$SFT/$DECISIONS_REL" ]; then
    # warning only: repos/env/accept/data do not need the decisions; 'score' does
    git -C "$SFT" fetch -q origin "$BRANCH" 2>/dev/null || true
    if git -C "$SFT" cat-file -e "origin/$BRANCH:$DECISIONS_REL" 2>/dev/null; then
      echo "WARNING: $DECISIONS_REL is absent at ${after:0:12} but present on origin/$BRANCH: PIN_SFT_REV predates the decisions commit; re-pin PIN_SFT_REV to a commit containing it and rerun 'repos' before 'score'."
    else
      echo "WARNING: $DECISIONS_REL is not committed at ${after:0:12} (nor on origin/$BRANCH); commit, tag and push it, then pin PIN_SFT_REV to that commit before 'score'."
    fi
  fi
  [ "$before" = "$after" ] || die "$SFT moved from ${before:0:12} to ${after:0:12}; this process is running the old rental_setup.sh. Rerun the same command."
  is_sha40 "$MT_REV" || die "MT_REV must be a full 40-hex commit sha, got '$MT_REV'"
  # Only the trainer patch may have modified Machine_translation.
  patch_files=$(sed -n 's#^+++ b/##p' "$SFT/phase0/trainer_token_accounting.patch" | sort -u)
  dirty=$(git -C "$MT" diff --name-only HEAD | grep -vxF -f <(printf '%s\n' "$patch_files") || true)
  untracked=$(git -C "$MT" ls-files --others --exclude-standard -- '*.py' '*.sh' || true)
  [ -z "$dirty$untracked" ] || die "Machine_translation has changes outside the trainer patch's files:
$dirty
$untracked"
  git -C "$MT" cat-file -e "$MT_REV^{commit}" 2>/dev/null || git -C "$MT" fetch -q origin \
    || die "cannot fetch Machine_translation to find $MT_REV"
  git -C "$MT" checkout -q --detach "$MT_REV" 2>/dev/null || die "cannot check out Machine_translation $MT_REV"
  [ "$(git -C "$MT" rev-parse HEAD)" = "$MT_REV" ] \
    || die "Machine_translation HEAD $(git -C "$MT" rev-parse HEAD) != pinned MT_REV $MT_REV"
  echo "$MT_REV" > "$WORK/mt_rev.txt"
  echo "repos pinned: Machine-Translation-SFT ${after:0:12}, Machine_translation ${MT_REV:0:12}"
}

stage_env() {
  stage_repos
  nvidia-smi -L || die "no GPU visible"
  "$PY" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' \
    || die "python $("$PY" -c 'import sys; print(sys.version.split()[0])') is outside 3.10-3.12: $PIN_NUMPY (required by unbabel-comet <2) has no wheel for it. Use an image with Python 3.10-3.12."
  torch_before=$("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo "none")
  log "installing python deps (image torch: $torch_before)"
  "$PY" -m pip install -q -r "$MT/requirements.txt" "$PIN_COMET" "$PIN_LIGHTNING" "$PIN_TRANSFORMERS" "$PIN_NUMPY" \
      pyarrow huggingface_hub pyyaml
  "$PY" - "$PIN_COMET" "$PIN_LIGHTNING" "$PIN_TRANSFORMERS" "$PIN_NUMPY" <<'PY' || die "installed versions differ from the pins above"
import sys
from importlib import metadata
bad = [f"{p} (installed {metadata.version(p.split('==')[0])})" for p in sys.argv[1:]
       if metadata.version(p.split("==")[0]) != p.split("==")[1]]
print("pins OK" if not bad else "NOT PINNED: " + ", ".join(bad)); sys.exit(1 if bad else 0)
PY
  "$PY" -m pip freeze > "$WORK/pip_freeze_env.txt"
  torch_after=$("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')
  echo "torch before: $torch_before | after: $torch_after"
  # unbabel-comet / lightning can pull in a different torch. An RTX 5090 needs a
  # recent CUDA build; a silently replaced torch fails later, mid-experiment.
  "$PY" -c 'import torch, sys; torch.zeros(1).cuda(); sys.exit(0)' || die \
"CUDA is not usable after installing deps (torch now: $torch_after, was: $torch_before).
pip likely replaced the image's CUDA torch. Reinstall the CUDA build matching this driver, then rerun 'env'."
  log "trainer patch"
  if git -C "$MT" apply --check -p1 "$SFT/phase0/trainer_token_accounting.patch" 2>/dev/null; then
    git -C "$MT" apply -p1 "$SFT/phase0/trainer_token_accounting.patch"
    echo "applied"
  elif git -C "$MT" apply --check --reverse -p1 "$SFT/phase0/trainer_token_accounting.patch" 2>/dev/null; then
    echo "already applied"
  else
    die "trainer patch neither applies nor is already applied — Machine_translation has diverged"
  fi
  log "self-test"
  (cd "$SFT" && MT_REPO="$MT" "$PY" tests/regress.py) || die "regression failed on this box"
}

fetch_pinned() {   # fetch_pinned <repo path> <size> <sha256> -> $WORK/hf_wmt14/<repo path>
  local P="$WORK/hf_wmt14" fp="$1" fs="$2" fsha="${3:-}" f try got
  [ -n "$fsha" ] || die "$fp: no pinned sha256 (hf_wmt14_filelist.tsv needs path<TAB>size<TAB>sha256)"
  f="$P/$fp"
  mkdir -p "$(dirname "$f")"
  if [ -f "$f" ] && [ "$(fsize "$f")" = "$fs" ] && [ "$(sha_of "$f")" = "$fsha" ]; then
    return 0
  fi
  for try in resume fresh; do
    # a full-size or oversized file with the wrong bytes is never resumed; the retry starts fresh
    if [ -f "$f" ] && { [ "$try" = fresh ] || [ "$(fsize "$f")" -ge "$fs" ]; }; then
      rm -f "$f"
    fi
    if ! curl -fL --retry 5 -C - -o "$f" "https://huggingface.co/datasets/wmt/wmt14/resolve/$HF_WMT14_REV/$fp"; then
      [ "$try" = fresh ] && die "$fp: download failed"
      continue
    fi
    if [ "$(fsize "$f")" = "$fs" ] && [ "$(sha_of "$f")" = "$fsha" ]; then
      return 0
    fi
    echo "  $fp: size or sha256 wrong after the $try download" >&2
  done
  got=$(sha_of "$f")
  mv "$f" "$f.bad"
  die "$fp: sha256 $got != pinned $fsha after re-download (kept as $f.bad)"
}

stage_accept() {
  warn_pin_overrides
  cd "$MT"
  mkdir -p data_enfr_v1 ckpt_hf
  local ok=1 f want
  for f in valid.en:$VALID_EN_SHA valid.fr:$VALID_FR_SHA test.en:$TEST_EN_SHA test.fr:$TEST_FR_SHA; do
    want=${f#*:}; f=${f%%:*}
    [ -f "data_enfr_v1/$f" ] && [ "$(sha_of "data_enfr_v1/$f")" = "$want" ] || ok=0
  done
  if [ "$ok" = 0 ]; then
    log "newstest2013/2014 from wmt/wmt14@${HF_WMT14_REV:0:8} (dev/test parquet only; no train dump)"
    fetch_pinned "$DEV_PARQUET" "$DEV_PARQUET_SIZE" "$DEV_PARQUET_SHA"
    fetch_pinned "$TEST_PARQUET" "$TEST_PARQUET_SIZE" "$TEST_PARQUET_SHA"
    "$PY" "$SFT/phase0/hf_devtest_to_text.py" --parquet "$WORK/hf_wmt14/$DEV_PARQUET" \
        --out-src data_enfr_v1/valid.en --out-tgt data_enfr_v1/valid.fr --expect-parquet-sha "$DEV_PARQUET_SHA" \
        --expect-sha-src "$VALID_EN_SHA" --expect-sha-tgt "$VALID_FR_SHA" --expect-lines 3000 \
        || die "newstest2013 does not reproduce the paper's valid.* byte for byte"
    "$PY" "$SFT/phase0/hf_devtest_to_text.py" --parquet "$WORK/hf_wmt14/$TEST_PARQUET" \
        --out-src data_enfr_v1/test.en --out-tgt data_enfr_v1/test.fr --expect-parquet-sha "$TEST_PARQUET_SHA" \
        --expect-sha-src "$TEST_EN_SHA" --expect-sha-tgt "$TEST_FR_SHA" --expect-lines 3003 \
        || die "newstest2014 does not reproduce the paper's test.* byte for byte"
  fi
  echo "valid/test sha256 match the paper's files"
  log "rebuild Base v1.1 averaged checkpoint from HF @ ${HF_BASE_REV:0:8}"
  "$PY" "$SFT/phase0/hf_to_ckpt.py" --repo "$HF_BASE_REPO" --revision "$HF_BASE_REV" \
      --expect-sha "pytorch_model.bin=$HF_BASE_WEIGHTS_SHA" --expect-sha "sentencepiece.model=$HF_BASE_SPM_SHA" \
      --mt-root "$MT" --train-config "$SFT/configs/sft_base_enfr.yaml" \
      --download-dir "$WORK/hf_cache" --out ckpt_hf/enfr_base_v1.1_averaged.pt
  # the SFT config points spm_model at data_enfr_v1/spm_enfr_v1_fixed.model
  cp ckpt_hf/enfr_base_v1.1_averaged.sentencepiece.model data_enfr_v1/spm_enfr_v1_fixed.model
  [ "$(sha_of data_enfr_v1/spm_enfr_v1_fixed.model)" = "$HF_BASE_SPM_SHA" ] || die "copied spm differs from the release"
  log "acceptance: reproduce the released test BLEU before any fine-tuning"
  rm -f "$WORK/accept_valid_bleu.txt"
  # scripts/*.py import src.*; a fresh clone is not pip-installed (the original box was)
  export PYTHONPATH="$MT${PYTHONPATH:+:$PYTHONPATH}"
  # set +e / PIPESTATUS: under pipefail a failing eval_bleu.py would otherwise end the script
  # silently at the pipeline, before any STOP line
  local rc
  set +e
  "$PY" scripts/eval_bleu.py --ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
      --config "$SFT/configs/sft_base_enfr.yaml" \
      --src data_enfr_v1/test.en --ref data_enfr_v1/test.fr \
      --beam 5 --length-penalty 1.0 2>&1 | tee "$WORK/accept_base.log"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || die "eval_bleu.py exited $rc on newstest2014 (see $WORK/accept_base.log); acceptance NOT established. Do NOT run E0.3."
  got=$(grep -oE "BLEU \([a-z]+, sacrebleu 13a\): [0-9.]+" "$WORK/accept_base.log" | awk '{print $NF}' || true)
  # eval_bleu.py exited 0 but printed no BLEU line
  [ -n "$got" ] || die "no BLEU line in $WORK/accept_base.log — eval_bleu.py failed; read the log"
  "$PY" - "$got" <<PY || die "test BLEU $got is not within $BLEU_TOL of $EXPECTED_BASE_TEST_BLEU — weights, tokenizer or test set do not match the paper. Do NOT run E0.3."
import sys; g=float(sys.argv[1]); sys.exit(0 if abs(g-$EXPECTED_BASE_TEST_BLEU) <= $BLEU_TOL else 1)
PY
  echo "ACCEPTED: test BLEU $got (released $EXPECTED_BASE_TEST_BLEU)"
  log "pre-fine-tuning newstest2013 BLEU (reference for lr_selection_rule flat-vs-baseline; not gated)"
  set +e
  "$PY" scripts/eval_bleu.py --ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
      --config "$SFT/configs/sft_base_enfr.yaml" \
      --src data_enfr_v1/valid.en --ref data_enfr_v1/valid.fr \
      --beam 5 --length-penalty 1.0 2>&1 | tee "$WORK/accept_valid.log"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || die "eval_bleu.py exited $rc on newstest2013 (see $WORK/accept_valid.log); no pre-FT baseline recorded. Do NOT run E0.3."
  got=$(grep -oE "BLEU \([a-z]+, sacrebleu 13a\): [0-9.]+" "$WORK/accept_valid.log" | awk '{print $NF}' || true)
  [ -n "$got" ] || die "no BLEU line in $WORK/accept_valid.log"
  echo "$got" > "$WORK/accept_valid_bleu.txt"
  echo "newstest2013 BLEU $got -> $WORK/accept_valid_bleu.txt (in-training eval of the release: 30.52)"
}

stage_data() {
  warn_pin_overrides
  cd "$MT"
  need_free_gb 30 "'data' (parquet 7.8 GB + v2 13.8 GB + v1.1 3.1 GB, measured)"
  local fp fs fsha
  log "WMT14 fr-en train parquet at pinned revision ${HF_WMT14_REV:0:8} (size + sha256 per file)"
  while IFS=$'\t' read -r fp fs fsha; do
    fetch_pinned "$fp" "$fs" "$fsha"
  done < <(grep '^fr-en/train-' "$SFT/phase0/hf_wmt14_filelist.tsv")
  log "CR-safe clean of the full stream (v2)"
  mkdir -p data_enfr_v2
  if [ -f data_enfr_v2/.sha_verified ] && ( require_v2 ) 2>/dev/null; then
    echo "v2 already verified"
  else
    rm -f data_enfr_v2/.sha_verified
    "$PY" "$SFT/phase0/rebuild_corpus.py" --parquet-dir "$WORK/hf_wmt14/fr-en" --src en --tgt fr --mode fixed \
        --out-dir data_enfr_v2 --expect-sha-src "$EXPECTED_V2_FIXED_SHA_EN" --expect-sha-tgt "$EXPECTED_V2_FIXED_SHA_FR" \
        || die "rebuilt v2 differs from the Mac rebuild (or was rejected); do not score or train on it"
    write_marker data_enfr_v2 "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" data_enfr_v2/train.clean.en data_enfr_v2/train.clean.fr
  fi
  echo "v2 rows = $(awk '$1=="rows"{print $2}' data_enfr_v2/.sha_verified) (sha256 matches the Mac rebuild)"
  log "v1.1 pretraining corpus (first $V1_MAX_ROWS HF rows), for held-out exclusion"
  mkdir -p data_enfr_v1_pretrain
  if [ -f data_enfr_v1_pretrain/.sha_verified ] && ( require_v1 ) 2>/dev/null; then
    echo "v1.1 already verified"
  else
    rm -f data_enfr_v1_pretrain/.sha_verified
    "$PY" "$SFT/phase0/rebuild_corpus.py" --parquet-dir "$WORK/hf_wmt14/fr-en" --src en --tgt fr --mode fixed \
        --max-rows "$V1_MAX_ROWS" --out-dir data_enfr_v1_pretrain \
        --expect-sha-src "$EXPECTED_V1_SHA_EN" --expect-sha-tgt "$EXPECTED_V1_SHA_FR" \
        || die "rebuilt v1.1 differs from the published pretraining corpus"
    write_marker data_enfr_v1_pretrain "$EXPECTED_V1_SHA_EN" "$EXPECTED_V1_SHA_FR" \
        data_enfr_v1_pretrain/train.clean.en data_enfr_v1_pretrain/train.clean.fr
  fi
  echo "v1.1 rows = $(awk '$1=="rows"{print $2}' data_enfr_v1_pretrain/.sha_verified)"
}

hf_logged_in() {   # hf_logged_in <whoami command...>; never prints the output (it names the account)
  local out
  out=$("$@" 2>&1) || return 1
  ! printf '%s\n' "$out" | grep -qi 'not logged in'
}

require_hf_login() {
  # huggingface_hub < 1.0 (forced by the transformers/comet pins) prints "Not logged in" and exits 0
  hf_logged_in hf auth whoami || hf_logged_in huggingface-cli whoami || die \
"not logged in to Hugging Face. Unbabel/wmt22-cometkiwi-da is gated (auto-approve, CC-BY-NC-SA-4.0):
  1. accept the terms at https://huggingface.co/Unbabel/wmt22-cometkiwi-da
  2. run 'hf auth login' YOURSELF on this box
This script never handles your token."
}

score_smoke() {   # score_smoke <src> <tgt>: 3,000 rows, 2 GPUs sharded vs 1 GPU, same text order
  local S="$WORK/score_smoke"
  [ "$(ngpus)" -ge 2 ] || die "SCORE_SMOKE=1 needs >= 2 GPUs"
  rm -rf "$S"; mkdir -p "$S"
  head -n 3000 "$1" > "$S/in.en"; head -n 3000 "$2" > "$S/in.fr"
  log "SCORE_SMOKE: 3,000 rows sharded over GPUs 0,1 (chunk 1000) and single-process on GPU 0"
  "$PY" "$SFT/phase0/score_sharded.py" --src "$S/in.en" --tgt "$S/in.fr" --out "$S/sharded.tsv" \
      --devices 0,1 --chunk-size 1000 --meta-out "$S/sharded.meta.json" || die "smoke: sharded scoring failed"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$SFT/scripts/score_with_comet.py" --src "$S/in.en" --tgt "$S/in.fr" \
      --out "$S/single.tsv" --gpus 1 --chunk-size 1000 --meta-out "$S/single.meta.json" || die "smoke: single-GPU scoring failed"
  "$PY" - "$S" "${SCORE_SMOKE_TOL:-1e-3}" <<'PY' || die "smoke: sharded and single-GPU outputs disagree (see above)"
import json, sys
S, tol = sys.argv[1], float(sys.argv[2])
def rows(p): return [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8", newline="\n")]
a, b = rows(f"{S}/sharded.tsv"), rows(f"{S}/single.tsv")
src = [l.rstrip("\n").replace("\t", " ") for l in open(f"{S}/in.en", encoding="utf-8", newline="\n")]
tgt = [l.rstrip("\n").replace("\t", " ") for l in open(f"{S}/in.fr", encoding="utf-8", newline="\n")]
ok = len(a) == len(b) == len(src) and all(x[1] == s and x[2] == t for x, s, t in zip(a, src, tgt)) \
     and all(y[1] == s and y[2] == t for y, s, t in zip(b, src, tgt))
diffs = sorted(abs(float(x[0]) - float(y[0])) for x, y in zip(a, b)) if ok else []
d = diffs[-1] if diffs else float("nan")
p99 = diffs[min(len(diffs) - 1, int(0.99 * len(diffs)))] if diffs else float("nan")
print(f"smoke: {len(a)} rows, text order {'identical' if ok else 'DIFFERS'}, max |sharded - single| = {d:.2e}, "
      f"p99 = {p99:.2e} (tol {tol})")
# the same-stack noise floor, recorded in v2_scored.meta.json (PROTOCOL.md D7)
json.dump({"rows": len(a), "order_identical": ok, "max_abs_diff": d, "p99_abs_diff": p99, "tol": tol},
          open(f"{S}/smoke_stats.json", "w"), indent=1)
sys.exit(0 if ok and d <= tol else 1)
PY
  echo "smoke OK"
}

write_score_meta() {   # write_score_meta <mode> <scorer meta json or ''> <smoke stats json or ''>
  "$PY" - "$MT/data_enfr_v2/v2_scored.meta.json" "$1" "$D_SHA256" "$BUNDLE/SHA256SUMS" "${2:-}" \
      "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" "$MT/data_enfr_v2/v2_scored.tsv" \
      "$D_SCORE_SHA" "$D_COMMIT" "$D_TAG" "$WORK/mt_rev.txt" "$WORK/sft_rev.txt" "${3:-}" <<'PY'
import hashlib, json, os, sys, datetime
out, mode, dsha, sums, extra, sen, sfr, tsv, ssha, dcommit, dtag, mtrev, sftrev, smoke = sys.argv[1:15]
def load(p): return json.load(open(p)) if p and os.path.exists(p) else None
def text(p): return open(p).read().strip() if os.path.exists(p) else None
m = {"mode": mode, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "decisions_sha256": dsha, "scoring_decisions_sha256": ssha, "decisions_commit": dcommit, "decisions_tag": dtag,
     "bundle_SHA256SUMS_sha256": hashlib.sha256(open(sums, "rb").read()).hexdigest(),
     "corpus_sha256": {"en": sen, "fr": sfr}, "scored_tsv_bytes": os.path.getsize(tsv),
     "mt_rev": text(mtrev), "sft_rev": text(sftrev),
     "scorer_meta": load(extra), "smoke": load(smoke)}
json.dump(m, open(out + ".tmp", "w"), indent=1); os.replace(out + ".tmp", out)
PY
}

stack_changed_shards() {   # stack_changed_shards <merged score_sharded meta>: shards resumed under another stack
  "$PY" -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1])).get("stack_changed_within_shard") or [])))' "$1" 2>/dev/null || true
}

stack_mixed_shards() {   # stack_mixed_shards <merged meta>: non-empty when shards differ from each other or are unverifiable
  "$PY" - "$1" <<'PY' 2>/dev/null || echo "merged meta $1 unreadable"
import json, sys
m = json.load(open(sys.argv[1]))
parts = []
if m.get("stack_mixed"):
    parts.append("stacks differ between shards: " + "; ".join(
        f"shards {x['shards']}" for x in m.get("stack_identities") or []))
if m.get("stack_unverified"):
    parts.append(f"shards {m['stack_unverified']} have no readable meta")
print(" / ".join(parts))
PY
}

run_score_sharded() {   # run_score_sharded <score_sharded args...>; dies with a message fitting the exit code
  local rc=0 allow=()
  if [ "${SCORE_ALLOW_STACK_CHANGE:-0}" = 1 ]; then
    allow=(--allow-stack-change)
    note_deviation "SCORE_ALLOW_STACK_CHANGE $D_SCORE_MODE $D_SCORE_SHA" \
      "$(utc) SCORE_ALLOW_STACK_CHANGE=1: score_mode $D_SCORE_MODE; resumed shards may continue under a changed scoring stack (recorded per shard in segments)"
  fi
  "$PY" "$SFT/phase0/score_sharded.py" "$@" ${allow[@]+"${allow[@]}"} || rc=$?
  case "$rc" in
  0) ;;
  3) die "a scorer refused to continue under a changed scoring stack (shards and differing keys above); nothing was mixed, and rerunning 'score' as is fails the same way. Options: (1) restore the stack the shard meta records; (2) delete those shards' shard.NNN.tsv and shard.NNN.meta.json to rescore them under the new stack (full mode then refuses mixed shards: delete all shards to rescore everything); (3) SCORE_ALLOW_STACK_CHANGE=1 bash $0 score (records a deviation). See RUNBOOK step 4c." ;;
  *) die "sharded scoring failed (score_sharded exit $rc, above); rerun 'score' to resume" ;;
  esac
}

recalibrate_kept() {   # recalibrate_kept <plan dir> <out>
  local R="$1" out="$2" failed=data_enfr_v2/v2_scored.calibration_failed.tsv side=data_enfr_v2/v2_scored.calibration_failed.json
  local old_args old_dsha rc=0
  [ -f "$failed" ] && [ -f "$side" ] || die "RECALIBRATE=1: no $failed with its sidecar $side; nothing to recalibrate"
  log "RECALIBRATE=1: checking $failed against its sidecar"
  "$PY" - "$side" "$failed" "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" "$D_CAL_ARGS" "$side.scorer_meta.json" <<'PY' \
    || die "RECALIBRATE=1 refused (above); nothing changed"
import hashlib, json, sys
side, tsv, en, fr, cal, meta_out = sys.argv[1:7]
s = json.load(open(side))
h = hashlib.sha256()
with open(tsv, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
bad = []
if s.get("corpus_sha256") != {"en": en, "fr": fr}:
    bad.append(f"sidecar corpus sha256 {s.get('corpus_sha256')} is not the pinned corpus")
if s.get("tsv_sha256") != h.hexdigest():
    bad.append(f"{tsv} sha256 {h.hexdigest()[:12]} != sidecar {str(s.get('tsv_sha256'))[:12]}: the kept scores changed")
if s.get("cal_args") == cal:
    bad.append("the calibration thresholds are unchanged since the failed calibration; it would fail again")
for b in bad:
    print("  " + b)
json.dump(s.get("scorer_meta"), open(meta_out, "w"))
sys.exit(1 if bad else 0)
PY
  old_args=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cal_args"])' "$side")
  old_dsha=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decisions_sha256"])' "$side")
  log "RECALIBRATE=1: calibrate the kept scores with the current thresholds (no scoring)"
  # shellcheck disable=SC2086
  "$PY" "$SFT/phase0/rescore_plan.py" calibrate --plan-dir "$R" --new-scored "$failed" \
      --provenance "$R/provenance_labels.npy" --provenance-report "$R/provenance_report.json" \
      --new-src data_enfr_v2/train.clean.en --new-tgt data_enfr_v2/train.clean.fr \
      --out data_enfr_v2/calibration_report.json $D_CAL_ARGS || rc=$?
  [ "$rc" = 0 ] || die "recalibration did not pass either (calibrate exit $rc, data_enfr_v2/calibration_report.json); $failed kept. Report to the user."
  mv "$failed" "$out"
  mv "$side.scorer_meta.json" data_enfr_v2/v2_scored.scorer_meta.json
  mv "$side" data_enfr_v2/v2_scored.recalibrated_from.json
  note_deviation "RECALIBRATE $old_dsha -> $D_SHA256" \
    "$(utc) RECALIBRATE: calibration thresholds changed after a failed calibration and the kept scores were accepted without rescoring; old '$old_args' (decisions $old_dsha) -> new '$D_CAL_ARGS' (decisions $D_SHA256)"
  write_score_meta full data_enfr_v2/v2_scored.scorer_meta.json ""
}

stage_score() {
  warn_pin_overrides
  cd "$MT"
  require_v2
  load_decisions
  guard_decisions_amend score
  bundle_ready
  local R="$BUNDLE" out=data_enfr_v2/v2_scored.tsv ngpu have smoke=""
  local want="$D_SCORE_MODE $D_SCORE_SHA"
  local corpus_rows; corpus_rows=$(awk '$1=="rows"{print $2}' data_enfr_v2/.sha_verified)
  [ "${SCORE_SMOKE:-0}" = 1 ] && smoke="$WORK/score_smoke/smoke_stats.json"
  if [ -f data_enfr_v2/v2_scored.done ] && [ -s "$out" ]; then
    have=$(cat data_enfr_v2/v2_scored.done)
    # the marker is keyed on the scoring decisions only (score_mode, calibration); a marker from
    # before that change holds the whole-file sha and is accepted when it equals today's
    if [ "$have" = "$want" ] || [ "$have" = "$D_SCORE_MODE $D_SHA256" ]; then
      echo "$want" > data_enfr_v2/v2_scored.done
      echo "v2_scored.tsv already complete for score_mode $D_SCORE_MODE and these scoring decisions"; return 0
    fi
    [ "${SCORE_REDO:-0}" = 1 ] || die "a completed v2_scored.tsv exists for other scoring decisions (marker '$have', now '$want').
Changing score_mode or calibration after 'score' is a pre-registration deviation. Revert the decisions,
or set SCORE_REDO=1 to rescore (reuse ~10 GPU-h, full ~17.5 GPU-h, derived). Nothing was deleted."
    note_deviation "SCORE_REDO $have -> $want" "$(utc) SCORE_REDO=1: rescoring; score marker '$have' -> '$want'"
  fi
  if [ "${RECALIBRATE:-0}" = 1 ] && [ "$D_SCORE_MODE" != full ]; then
    die "RECALIBRATE=1 applies only to score_mode full"
  fi
  rm -f data_enfr_v2/v2_scored.done
  case "$D_SCORE_MODE" in
  none)
    need_free_gb 20 "score_mode none (merged TSV ~14 GB, derived)"
    log "score_mode none: reused scores only; 21,628,292 rows get the literal score nan (no GPU)"
    "$PY" "$SFT/phase0/rescore_plan.py" merge --plan-dir "$R" --allow-missing \
        --new-src data_enfr_v2/train.clean.en --new-tgt data_enfr_v2/train.clean.fr --out "$out" || die "merge refused"
    write_score_meta none "" ""
    ;;
  reuse)
    require_hf_login
    ngpu=$(ngpus); [ "$ngpu" -ge 1 ] || die "no GPU visible"
    need_free_gb 50 "score_mode reuse (to_score 7.9 GB + shard copies 7.9 + shard outputs ~8.1 + merged ~14.2)"
    log "score_mode reuse: extract the pairs the old scored file lacks"
    "$PY" "$SFT/phase0/rescore_plan.py" extract --plan-dir "$R" \
        --new-src data_enfr_v2/train.clean.en --new-tgt data_enfr_v2/train.clean.fr || die "extract refused"
    [ "${SCORE_SMOKE:-0}" = 1 ] && score_smoke "$R/to_score.en" "$R/to_score.fr"
    log "scoring to_score.* sharded over $ngpu GPU(s) (single 5090: ~10 GPU-h derived; multi-GPU throughput not measured)"
    run_score_sharded --src "$R/to_score.en" --tgt "$R/to_score.fr" --out "$R/new_scores.tsv" \
        --gpus "$ngpu" --meta-out "$R/new_scores.meta.json"
    [ -z "$(stack_changed_shards "$R/new_scores.meta.json")" ] \
      || echo "WARNING: shard(s) $(stack_changed_shards "$R/new_scores.meta.json") were resumed under a different scoring stack" >&2
    [ -z "$(stack_mixed_shards "$R/new_scores.meta.json")" ] \
      || echo "WARNING: new scores do not come from one scoring stack: $(stack_mixed_shards "$R/new_scores.meta.json") (see $R/new_scores.meta.json)" >&2
    "$PY" "$SFT/phase0/rescore_plan.py" merge --plan-dir "$R" --new-scores "$R/new_scores.tsv" \
        --new-src data_enfr_v2/train.clean.en --new-tgt data_enfr_v2/train.clean.fr --out "$out" || die "merge refused"
    write_score_meta reuse "$R/new_scores.meta.json" "$smoke"
    rm -rf "$R/to_score.en" "$R/to_score.fr" "$R/new_scores.tsv" "$R/new_scores.tsv.shards"
    ;;
  full)
    local part=data_enfr_v2/v2_scored.partial.tsv failed=data_enfr_v2/v2_scored.calibration_failed.tsv
    local side=data_enfr_v2/v2_scored.calibration_failed.json rc=0
    if [ "${RECALIBRATE:-0}" = 1 ]; then
      recalibrate_kept "$R" "$out"
    else
      require_hf_login
      ngpu=$(ngpus); [ "$ngpu" -ge 1 ] || die "no GPU visible"
      need_free_gb 50 "score_mode full (shard copies 13.8 GB + shard outputs ~14.2 + concatenated ~14.2)"
      [ -f "$failed" ] && echo "note: $failed from an earlier failed calibration exists (RECALIBRATE=1 re-checks it without scoring); scoring afresh"
      [ "${SCORE_SMOKE:-0}" = 1 ] && score_smoke data_enfr_v2/train.clean.en data_enfr_v2/train.clean.fr
      log "score_mode full: all $corpus_rows pairs sharded over $ngpu GPU(s) (single 5090: ~17.5 GPU-h derived; multi-GPU not measured)"
      run_score_sharded --src data_enfr_v2/train.clean.en --tgt data_enfr_v2/train.clean.fr \
          --out "$part" --gpus "$ngpu" --meta-out data_enfr_v2/v2_scored.scorer_meta.json
      [ "$(nrows "$part")" = "$corpus_rows" ] || die "$part has $(nrows "$part") rows, corpus has $corpus_rows"
      [ -z "$(stack_changed_shards data_enfr_v2/v2_scored.scorer_meta.json)" ] \
        || die "shard(s) $(stack_changed_shards data_enfr_v2/v2_scored.scorer_meta.json) mix scoring stacks; not calibrating. Scores and shards kept."
      # full mode exists to give every row one stack (PROTOCOL.md D7): a stack change BETWEEN shards also stops
      [ -z "$(stack_mixed_shards data_enfr_v2/v2_scored.scorer_meta.json)" ] \
        || die "full-mode scores mix scoring stacks: $(stack_mixed_shards data_enfr_v2/v2_scored.scorer_meta.json) (data_enfr_v2/v2_scored.scorer_meta.json); not calibrating. Scores ($part) and shards ($part.shards) kept; delete the shards of the unwanted stack and rerun 'score'."
      mv "$part" "$out"
      log "calibration: new scores vs the 16,646,992 reused old scores, per source (thresholds from decisions)"
      # shellcheck disable=SC2086  # D_CAL_ARGS is a flag list built by e03_decisions.py
      "$PY" "$SFT/phase0/rescore_plan.py" calibrate --plan-dir "$R" --new-scored "$out" \
          --provenance "$R/provenance_labels.npy" --provenance-report "$R/provenance_report.json" \
          --new-src data_enfr_v2/train.clean.en --new-tgt data_enfr_v2/train.clean.fr \
          --out data_enfr_v2/calibration_report.json $D_CAL_ARGS || rc=$?
      case "$rc" in
      0) ;;
      1|2)
        mv "$out" "$failed"
        "$PY" - "$side" "$failed" "$rc" "$EXPECTED_V2_FIXED_SHA_EN" "$EXPECTED_V2_FIXED_SHA_FR" "$D_SHA256" \
            "$D_SCORE_SHA" "$D_CAL_ARGS" data_enfr_v2/v2_scored.scorer_meta.json data_enfr_v2/calibration_report.json <<'PY'
import datetime, hashlib, json, os, sys
side, tsv, rc, en, fr, dsha, ssha, cal, meta, rep = sys.argv[1:11]
h, n = hashlib.sha256(), 0
with open(tsv, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b); n += b.count(b"\n")
def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
json.dump({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "tsv_sha256": h.hexdigest(), "rows": n,
           "corpus_sha256": {"en": en, "fr": fr}, "decisions_sha256": dsha, "scoring_decisions_sha256": ssha,
           "cal_args": cal, "calibrate_exit": int(rc), "verdict": {1: "fail", 2: "insufficient"}[int(rc)],
           "scorer_meta": load(meta), "calibration_report": load(rep)}, open(side + ".tmp", "w"), indent=1)
os.replace(side + ".tmp", side)
PY
        # Without this a rerun (e.g. after fixing the stack) would resume from these verified
        # shards and silently reuse the rejected scores. RECALIBRATE=1 re-checks the kept TSV.
        rm -rf "$part.shards"
        die "calibration did not pass (exit $rc; data_enfr_v2/calibration_report.json); scores kept as $failed with $side. Report to the user; do not proceed. (If the user changes the thresholds, RECALIBRATE=1 re-checks these scores without rescoring and records a deviation.)"
        ;;
      *)
        mv "$out" "$part"
        die "calibrate could not run (exit $rc: invalid or unreadable input, above). Scores and verified shards are kept ($part, $part.shards); fix the input and rerun 'score' (no rescoring needed)."
        ;;
      esac
      write_score_meta full data_enfr_v2/v2_scored.scorer_meta.json "$smoke"
      rm -rf "$part.shards"
    fi
    ;;
  esac
  [ "$(nrows "$out")" = "$corpus_rows" ] || die "$out has $(nrows "$out") rows, corpus has $corpus_rows"
  echo "$want" > data_enfr_v2/v2_scored.done
  echo "v2_scored.tsv complete ($corpus_rows rows, score_mode $D_SCORE_MODE)"
}

statmt_url() {
  # URLs checked (HTTP 200) on 2026-09-12.
  case "$1" in
    europarl)        echo https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz ;;
    commoncrawl)     echo https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz ;;
    un)              echo https://www.statmt.org/wmt13/training-parallel-un.tgz ;;
    news-commentary) echo https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz ;;
    giga-fren)       echo https://www.statmt.org/wmt10/training-giga-fren.tar ;;
    *) die "unknown corpus $1" ;;
  esac
}

stage_provenance() {
  # Plain bash 3.2 on purpose (no associative arrays, no bare empty-array
  # expansion under set -u), so this stage can be exercised offline on a Mac.
  cd "$MT"
  require_v2
  if [ -f "$BUNDLE/SHA256SUMS" ] || [ -f "$WORK/rental_bundle_v2.tar.gz" ]; then
    bundle_ready
    # Row order is fixed by the corpus, and require_v2 ties it to the Mac rebuild's sha256,
    # so labels computed there index these exact rows.
    [ "$("$PY" -c 'import numpy as n,sys; print(len(n.load(sys.argv[1], mmap_mode="r")))' "$BUNDLE/provenance_labels.npy")" \
      = "$(nrows data_enfr_v2/train.clean.en)" ] || die "bundle provenance labels do not match the corpus row count"
    cp "$BUNDLE/provenance_labels.npy" "$SFT/phase0/provenance_labels.npy"
    cp "$BUNDLE/provenance_report.json" "$SFT/phase0/provenance_report.json"
    echo "using bundle provenance labels (match rate $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["match_rate"])' "$BUNDLE/provenance_report.json"))"
    return 0
  fi
  local R="$WORK/statmt" lab d f pat en fr n_en n_fr
  mkdir -p "$R"
  local args=()
  # This ORDER defines the integer codes in provenance_labels.npy;
  # e03_build_controls reads it back from the report rather than trusting a retype.
  for lab in europarl commoncrawl un news-commentary giga-fren; do
    d="$R/$lab"; mkdir -p "$d"
    f="$d/$(basename "$(statmt_url "$lab")")"
    if [ ! -s "$f" ]; then
      [ "${STATMT_OFFLINE:-0}" = 1 ] && die "$lab: $f missing and STATMT_OFFLINE=1"
      curl -fL --retry 3 -o "$f" "$(statmt_url "$lab")"
    fi
    if [ ! -f "$d/.extracted" ]; then
      tar -xf "$f" -C "$d"
      # giga-fren ships gzipped text inside the tar
      find "$d" -type f -name '*.gz' -exec gunzip -f {} \;
      touch "$d/.extracted"
    fi
    if [ "$lab" = giga-fren ]; then pat='*giga-fren*'; else pat='*fr-en*'; fi
    # Inner file names are DISCOVERED, not assumed: exactly one .en and one .fr
    # with equal line counts, or stop.
    en=$(find "$d" -type f -name "$pat" -name '*.en' | sort)
    fr=$(find "$d" -type f -name "$pat" -name '*.fr' | sort)
    n_en=$(printf '%s' "$en" | grep -c . || true)
    n_fr=$(printf '%s' "$fr" | grep -c . || true)
    if [ "$n_en" -ne 1 ] || [ "$n_fr" -ne 1 ]; then
      die "$lab: expected exactly one .en and one .fr under $d, found $n_en/$n_fr:
$en
$fr"
    fi
    [ "$(wc -l < "$en")" -eq "$(wc -l < "$fr")" ] || die "$lab: $en and $fr differ in line count"
    echo "  $lab: $(wc -l < "$en" | tr -d ' ') pairs  ($en)"
    args+=(--corpus "$lab:$en:$fr")
  done
  local qe=()
  [ -s data_enfr_v2/v2_scored.tsv ] && qe=(--qe-scores data_enfr_v2/v2_scored.tsv)
  "$PY" "$SFT/phase0/e01_provenance.py" \
      --clean-src data_enfr_v2/train.clean.en --clean-tgt data_enfr_v2/train.clean.fr \
      "${args[@]}" ${qe[@]+"${qe[@]}"} --norm exact --out "$SFT/phase0/provenance"
}

stage_controls() {
  cd "$MT"
  load_decisions
  guard_decisions_amend controls
  require_v2
  local labels="$SFT/phase0/provenance_labels.npy" report="$SFT/phase0/provenance_report.json"
  local scored=data_enfr_v2/v2_scored.tsv out="$SFT/data/phase0" have
  [ -s "$labels" ] && [ -s "$report" ] || die "run 'provenance' first. \
Without it there are no held-out domain sets and e03_decide cannot return GO."
  have=$(cat data_enfr_v2/v2_scored.done 2>/dev/null || true)
  if [ "$have" != "$D_SCORE_MODE $D_SCORE_SHA" ] && [ "$have" != "$D_SCORE_MODE $D_SHA256" ]; then
    [ -n "$have" ] && die "v2_scored.tsv was scored under other scoring decisions (marker '$have', now '$D_SCORE_MODE $D_SCORE_SHA'): revert score_mode/calibration, or run 'score' first with SCORE_REDO=1"
    die "run 'score' first (no completed v2_scored.tsv for score_mode $D_SCORE_MODE and these decisions)"
  fi
  local extra=()
  if [ "$D_POOL" = reused ]; then
    bundle_ready
    extra+=(--pool-mask "$BUNDLE/pool_mask_reused.npy")
  fi
  if [ "$D_EXCLUDE_PRETRAIN" = 1 ]; then
    require_v1
    extra+=(--pretrain-src "$MT/data_enfr_v1_pretrain/train.clean.en" --pretrain-tgt "$MT/data_enfr_v1_pretrain/train.clean.fr")
  fi
  log "row counts: corpus, scored TSV, provenance labels${extra[0]:+, pool mask}"
  "$PY" - "$(awk '$1=="rows"{print $2}' data_enfr_v2/.sha_verified)" "$scored" "$labels" \
      "${extra[1]:-}" <<'PY' || die "row counts disagree (above); nothing built"
import sys
import numpy as np
want, scored, labels, mask = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
n = {"corpus (.sha_verified)": want}
with open(scored, "rb") as f:
    n["scored TSV"] = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))
n["provenance labels"] = len(np.load(labels, mmap_mode="r"))
if mask.endswith(".npy"):
    m = np.load(mask, mmap_mode="r")
    n["pool mask"] = len(m) if m.dtype == np.bool_ and m.ndim == 1 else -1
for k, v in n.items():
    print(f"  {k}: {v:,}")
sys.exit(0 if len(set(n.values())) == 1 else 1)
PY
  need_mem_gb "$CONTROLS_MIN_MEM_GB" "e03_build_controls on 38.3M rows"
  need_free_gb 5 "'controls' (FT sets ~1.1 GB per 3M rows, derived)"
  mkdir -p "$SFT/logs/phase0" "$RESULTS"
  # Invalidate the previous records BEFORE the old sets are deleted: a rebuild that fails
  # part-way must leave nothing that stage1/stage2/gate accept (they then say "run 'controls' first").
  rm -f "$RESULTS/decisions.sha256" "$RESULTS/matrix.json" "$RESULTS/controls_manifest.json" \
        "$SFT/configs/phase0/matrix.json" "$SFT/configs/phase0/run_stage1.sh" "$SFT/configs/phase0/run_stage2.sh"
  rm -rf "$out"
  log "building control sets (log: $SFT/logs/phase0/controls_build.log)"
  set +e
  "$PY" "$SFT/phase0/e03_build_controls.py" --qe-scores "$scored" \
      --provenance "$labels" --provenance-report "$report" --out-dir "$out" \
      --n-ft "$D_N_FT" --heldout-domains "$D_HELDOUT_DOMAINS" ${extra[@]+"${extra[@]}"} 2>&1 \
      | tee "$SFT/logs/phase0/controls_build.log"
  local rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || die "e03_build_controls.py exited $rc (see the log); no matrix generated"
  "$PY" - "$out" "$D_HELDOUT_DOMAINS" "$D_INDOMAIN" <<'PY' || die "held-out sets required by the decisions are missing (above)"
import json, os, sys
out, doms, indomain = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
man = json.load(open(os.path.join(out, "manifest.json")))
listed = list(man.get("heldout_sets") or {})
bad = [f"heldout_{d} not in manifest heldout_sets {listed}" for d in doms if f"heldout_{d}" not in listed]
bad += [f"--indomain {s} not built" for s in indomain if s not in listed]
for s in listed:
    for ext in ("en", "fr"):
        p = os.path.join(out, f"{s}.{ext}")
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            bad.append(f"{p} missing or empty")
for b in bad:
    print("  " + b)
print("held-out sets OK: " + ", ".join(listed) if not bad else "")
sys.exit(1 if bad else 0)
PY
  local keep=1; [ "$D_FT_CKPT" = avg-last5 ] && keep=4
  local budget=(--budget "$D_BUDGET"); [ "$D_BUDGET" = tokens ] && budget+=(--target-tokens "$D_TARGET_TOKENS")
  set +e
  "$PY" "$SFT/phase0/e03_run_matrix.py" --base-config "$SFT/configs/sft_base_enfr.yaml" \
      --data-dir "$out" --out-dir "$SFT/configs/phase0" \
      --ckpt ckpt_hf/enfr_base_v1.1_averaged.pt --keep-last "$keep" "${budget[@]}" --spike-ratio "$D_SPIKE" 2>&1 \
      | tee "$SFT/logs/phase0/run_matrix.log"
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || die "e03_run_matrix.py exited $rc"
  cp "$out/manifest.json" "$RESULTS/controls_manifest.json"
  cp "$SFT/configs/phase0/matrix.json" "$RESULTS/matrix.json"
  record_decisions
  echo "next: bash $0 stage1"
}

stage_runs() {
  cd "$MT"
  local which="$1" lr="${2:-}" ngpu
  require_frozen_decisions
  ngpu=$(ngpus); [ "$ngpu" -ge 1 ] || die "no GPU visible"
  if [ "$which" = stage2 ]; then
    [ -n "$lr" ] || die "usage: $0 stage2 <lr_scale selected by stage1>"
    if [ "$D_FT_CKPT" = avg-last5 ]; then
      need_free_gb 55 "stage2 avg-last5 (9 runs x 8 checkpoints incl. avg_last5.pt written at gate, ~0.7 GB each, derived)"
    else
      need_free_gb 40 "stage2 (9 runs; ~0.7 GB per checkpoint, derived)"
    fi
    [ -f "$RESULTS/lr_selection.json" ] || die "no $RESULTS/lr_selection.json: run stage1 first"
    local sel; sel=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$RESULTS/lr_selection.json")
    if [ "$sel" != "$lr" ]; then
      [ "${LR_OVERRIDE:-0}" = 1 ] || die "stage1 selected lr_scale $sel under rule $D_LR_RULE; refusing $lr (LR_OVERRIDE=1 records a deviation)"
      note_deviation "stage2 lr_scale $lr overrides selected $sel" "$(utc) stage2 lr_scale $lr overrides selected $sel (rule $D_LR_RULE)"
    fi
    "$PY" "$SFT/phase0/warm_cache.py" --runner "$SFT/configs/phase0/run_stage2.sh" --lr-scale "$lr" --cwd "$MT" \
        || die "tokenisation cache warm-up failed or the control sets changed (above); rerun or rerun 'controls'"
    "$PY" "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0/run_stage2.sh" \
        --lr-scale "$lr" --gpus "$ngpu" --cwd "$MT" --logdir "$SFT/logs/phase0/stage2" || die "stage2 incomplete (above)"
    echo "next: bash $0 gate $lr"
  else
    # checked before any GPU run: the selection rule needs the pre-FT baseline written by 'accept'
    if [ "$D_LR_RULE" = flat-vs-baseline ]; then
      [ -s "$WORK/accept_valid_bleu.txt" ] || die "flat-vs-baseline needs $WORK/accept_valid_bleu.txt: rerun 'accept'"
    fi
    need_free_gb 20 "stage1 (4 runs; ~0.7 GB per checkpoint, derived)"
    "$PY" "$SFT/phase0/warm_cache.py" --runner "$SFT/configs/phase0/run_stage1.sh" --cwd "$MT" \
        || die "tokenisation cache warm-up failed or the control sets changed (above); rerun or rerun 'controls'"
    "$PY" "$SFT/phase0/run_parallel.py" --runner "$SFT/configs/phase0/run_stage1.sh" \
        --gpus "$ngpu" --cwd "$MT" --logdir "$SFT/logs/phase0/stage1" || die "stage1 incomplete (above)"
    local sel_args=(--logdir "$SFT/logs/phase0/stage1" --rule "$D_LR_RULE" --out "$RESULTS/lr_selection.json")
    [ -n "$D_LR_TOL" ] && sel_args+=(--tol "$D_LR_TOL")
    if [ "$D_LR_RULE" = flat-vs-baseline ]; then
      [ -s "$WORK/accept_valid_bleu.txt" ] || die "flat-vs-baseline needs $WORK/accept_valid_bleu.txt: rerun 'accept'"
      sel_args+=(--baseline-bleu "$(cat "$WORK/accept_valid_bleu.txt")")
    fi
    set +e
    "$PY" "$SFT/phase0/e03_select_lr.py" "${sel_args[@]}"
    local rc=$?
    set -e
    [ "$rc" = 0 ] || die "no lr_scale selected (e03_select_lr exit $rc). Report to the user; do not pick one by eye."
    echo "then: bash $0 stage2 $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$RESULTS/lr_selection.json")"
  fi
}

stage_gate() {
  cd "$MT"
  local lr="${1:-}" sel
  [ -n "$lr" ] || die "usage: $0 gate <lr_scale used for stage2>"
  require_frozen_decisions
  [ -f "$RESULTS/lr_selection.json" ] || die "no $RESULTS/lr_selection.json: run stage1 first"
  sel=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"])' "$RESULTS/lr_selection.json")
  if [ "$sel" != "$lr" ]; then
    grep -qF "stage2 lr_scale $lr overrides selected $sel" "$RESULTS/deviations.txt" 2>/dev/null \
      || die "gate lr_scale $lr is not the stage-1 selection $sel, and no stage2 LR_OVERRIDE deviation is recorded for it"
  fi
  if [ "$D_FT_CKPT" = avg-last5 ]; then
    need_free_gb 8 "gate avg-last5 (9 x avg_last5.pt + 1 transient .tmp, ~0.7 GB each, derived)"
  fi
  log "decisions record and deviations (read these with the verdict)"
  cat "$RESULTS/decisions_history.tsv" 2>/dev/null || echo "  (no decisions_history.tsv)"
  if [ -s "$RESULTS/deviations.txt" ]; then
    echo "DEVIATIONS:"; sed 's/^/  /' "$RESULTS/deviations.txt"
  else
    echo "deviations: none recorded"
  fi
  "$PY" "$SFT/phase0/e03_collect.py" --mt-root "$MT" \
      --runner "$SFT/configs/phase0/run_stage2.sh" --lr-scale "$lr" --selected-lr "$sel" --decisions-sha "$D_SHA256" \
      --baseline-ckpt ckpt_hf/enfr_base_v1.1_averaged.pt \
      --baseline-config "$SFT/configs/sft_base_enfr.yaml" \
      --controls-dir "$SFT/data/phase0" --out "$RESULTS/phase0_bleu.tsv" --ft-ckpt "$D_FT_CKPT" \
      --jobs "$(ngpus)" || die "collection incomplete or stale — see above; do not apply the gate to partial results"
  # exit 0 = GO, 1 = NO-GO (rule applied to complete data), 2 = cannot decide. The verdict is printed either way.
  "$PY" "$SFT/phase0/e03_decide.py" --results "$RESULTS/phase0_bleu.tsv" --indomain "$D_INDOMAIN"
}

# The whole dispatch and the exit are one parsed command, so bash never reads this file again
# after the stage starts (a 'git checkout' in 'repos' may rewrite it underneath).
case "${1:-}" in
  repos) stage_repos ;; env) stage_env ;; accept) stage_accept ;; data) stage_data ;;
  score) stage_score ;; provenance) stage_provenance ;; controls) stage_controls ;;
  stage1) stage_runs stage1 ;; stage2) stage_runs stage2 "${2:-}" ;; gate) stage_gate "${2:-}" ;;
  *) sed -n '/^# USAGE-BEGIN/,/^# USAGE-END/p' "$0" | sed '1d;$d'; exit 2 ;;
esac; exit $?
