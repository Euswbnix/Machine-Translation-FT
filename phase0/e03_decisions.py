#!/usr/bin/env python3
"""Validate phase0/e03_decisions.json -- the E0.3 design choices that must be frozen
(and git-tagged) BEFORE `rental_setup.sh score` / `controls` run.

Nothing here chooses anything. Every key is required; the literal value "CHOOSE"
counts as missing; unknown keys are refused (a typo must not silently fall back to a
default). Keys starting with "_" are free-form comments. Options and measured
consequences for every decision: PROTOCOL.md, "E0.3 decisions required before controls".

    python phase0/e03_decisions.py check phase0/e03_decisions.json
    python phase0/e03_decisions.py shell phase0/e03_decisions.json   # KEY='value' lines for eval
    python phase0/e03_decisions.py diff OLD.json NEW.json            # changed decision keys, one line

Exit codes: 0 valid; 2 missing file, invalid JSON, or any missing/invalid decision
(every problem is listed, not just the first).

Schema (value -> meaning):
  pool                           "reused" | "full"   FT/held-out pool: the 16,646,992 rows with
                                                     reused old scores, or all 38,275,284 rows
  n_ft                           int > 0             e03_build_controls --n-ft
  heldout_domains                [names]             e03_build_controls --heldout-domains (order =
                                                     draw order); names from the e01 report
  indomain                       ["heldout_<d>",...] e03_decide --indomain; each <d> in heldout_domains
  exclude_pretrain_from_heldout  true | false        pass --pretrain-src/--pretrain-tgt (v1.1 corpus)
  lr_selection_rule              see LR_RULES        e03_select_lr --rule
  lr_tolerance_bleu              number >= 0 | null  --tol; null iff the rule takes no tolerance
  score_mode                     "none" | "reuse" | "full"
  calibration                    object | null       rescore_plan calibrate thresholds; object iff
                                                     score_mode == "full". Optional key jaccard_min_k
                                                     (int >= 0, absent = 0 = every top-k Jaccard gated)
  ft_checkpoint                  "final" | "avg-last5"   e03_collect --ft-ckpt
  budget                         "steps" | "tokens"  e03_run_matrix --budget
  target_tokens                  int > 0 | null      --target-tokens; int iff budget == "tokens"
  loss_spike_ratio               "inherit" | number >= 0   e03_run_matrix --spike-ratio
"""
from __future__ import annotations

import hashlib
import json
import shlex
import sys

SOURCES = ("europarl", "commoncrawl", "un", "news-commentary", "giga-fren")
LR_RULES = {
    # rule name -> needs a tolerance?
    "no-strict-monotone-decline-from-first-eval": False,   # the wording run_stage1.sh prints today
    "flat-endpoints": True,
    "flat-slope": True,
    "flat-vs-baseline": True,
}
CAL_KEYS = {"max_mean_diff": float, "max_p99_diff": float, "min_spearman": float,
            "min_jaccard": float, "min_rows": int}
# optional calibration keys: absent -> the flag is not passed (rescore_plan default)
CAL_OPTIONAL = {"jaccard_min_k": int}
KEYS = ("pool", "n_ft", "heldout_domains", "indomain", "exclude_pretrain_from_heldout",
        "lr_selection_rule", "lr_tolerance_bleu", "score_mode", "calibration",
        "ft_checkpoint", "budget", "target_tokens", "loss_spike_ratio")
CHOOSE = "CHOOSE"


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate(d) -> list[str]:
    if not isinstance(d, dict):
        return ["top level must be a JSON object"]
    p: list[str] = []
    for k in d:
        if not k.startswith("_") and k not in KEYS:
            p.append(f"unknown key {k!r} (typo?)")

    def missing(k):
        if k not in d:
            p.append(f"{k}: MISSING")
            return True
        if d[k] == CHOOSE:
            p.append(f"{k}: still \"CHOOSE\"")
            return True
        return False

    def enum(k, opts):
        if not missing(k) and d[k] not in opts:
            p.append(f"{k}: {d[k]!r} is not one of {list(opts)}")

    enum("pool", ("reused", "full"))
    if not missing("n_ft") and not (_int(d["n_ft"]) and d["n_ft"] > 0):
        p.append(f"n_ft: {d['n_ft']!r} is not a positive integer")
    doms: list = []
    if not missing("heldout_domains"):
        v = d["heldout_domains"]
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            p.append("heldout_domains: must be a non-empty list of source names")
        else:
            bad = [x for x in v if x not in SOURCES]
            if bad:
                p.append(f"heldout_domains: unknown source(s) {bad}; valid: {list(SOURCES)}")
            if len(set(v)) != len(v):
                p.append("heldout_domains: duplicate names")
            doms = v
    if not missing("indomain"):
        v = d["indomain"]
        if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v)):
            p.append("indomain: must be a non-empty list like [\"heldout_un\"]")
        else:
            bad = [x for x in v if not x.startswith("heldout_") or x[len("heldout_"):] not in doms]
            if bad:
                p.append(f"indomain: {bad} not built by heldout_domains {doms}")
    if not missing("exclude_pretrain_from_heldout") and not isinstance(d["exclude_pretrain_from_heldout"], bool):
        p.append("exclude_pretrain_from_heldout: must be true or false")
    enum("lr_selection_rule", tuple(LR_RULES))
    if "lr_tolerance_bleu" not in d:
        p.append("lr_tolerance_bleu: MISSING (null for rules without a tolerance)")
    elif d["lr_tolerance_bleu"] == CHOOSE:
        p.append("lr_tolerance_bleu: still \"CHOOSE\"")
    elif d.get("lr_selection_rule") in LR_RULES:
        tol = d["lr_tolerance_bleu"]
        if LR_RULES[d["lr_selection_rule"]]:
            if not (_num(tol) and tol >= 0):
                p.append(f"lr_tolerance_bleu: rule {d['lr_selection_rule']} needs a number >= 0, got {tol!r}")
        elif tol is not None:
            p.append(f"lr_tolerance_bleu: rule {d['lr_selection_rule']} takes no tolerance; set null")
    enum("score_mode", ("none", "reuse", "full"))
    if "calibration" not in d:
        p.append("calibration: MISSING (null unless score_mode is \"full\")")
    elif d["calibration"] == CHOOSE:
        p.append("calibration: still \"CHOOSE\"")
    elif d.get("score_mode") == "full":
        c = d["calibration"]
        if not isinstance(c, dict):
            p.append("calibration: score_mode \"full\" needs an object with " + ", ".join(CAL_KEYS))
        else:
            for ck, typ in CAL_KEYS.items():
                cv = c.get(ck)
                if cv is None or cv == CHOOSE:
                    p.append(f"calibration.{ck}: MISSING")
                elif not ((_int(cv) if typ is int else _num(cv)) and cv >= 0):
                    p.append(f"calibration.{ck}: {cv!r} is not a non-negative {typ.__name__}")
            for ck, typ in CAL_OPTIONAL.items():
                if ck in c and not (_int(c[ck]) and c[ck] >= 0):
                    p.append(f"calibration.{ck}: {c[ck]!r} is not a non-negative int")
            extra = [ck for ck in c if ck not in CAL_KEYS and ck not in CAL_OPTIONAL and not ck.startswith("_")]
            if extra:
                p.append(f"calibration: unknown key(s) {extra}")
    elif d.get("score_mode") in ("none", "reuse") and d["calibration"] is not None:
        p.append(f"calibration: must be null when score_mode is {d['score_mode']!r}")
    enum("ft_checkpoint", ("final", "avg-last5"))
    enum("budget", ("steps", "tokens"))
    if "target_tokens" not in d:
        p.append("target_tokens: MISSING (null unless budget is \"tokens\")")
    elif d["target_tokens"] == CHOOSE:
        p.append("target_tokens: still \"CHOOSE\"")
    elif d.get("budget") == "tokens":
        if not (_int(d["target_tokens"]) and d["target_tokens"] > 0):
            p.append(f"target_tokens: budget \"tokens\" needs a positive integer, got {d['target_tokens']!r}")
    elif d.get("budget") == "steps" and d["target_tokens"] is not None:
        p.append("target_tokens: must be null when budget is \"steps\"")
    if not missing("loss_spike_ratio"):
        v = d["loss_spike_ratio"]
        if not (v == "inherit" or (_num(v) and v >= 0)):
            p.append(f"loss_spike_ratio: {v!r} is neither \"inherit\" nor a number >= 0")
    # cross-field
    if d.get("score_mode") == "none" and d.get("pool") == "full":
        p.append("score_mode \"none\" leaves 21,628,292 rows unscored (nan); it requires pool \"reused\"")
    return p


def load(path: str):
    raw = open(path, "rb").read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def score_sha(d: dict) -> str:
    """sha256 over the decisions that change v2_scored.tsv: score_mode, and the calibration
    thresholds when score_mode is "full" ("_" comment keys dropped). The 'score' done marker
    is keyed on this, so editing n_ft, held-out domains, the LR rule or a comment after
    'score' does not throw away finished scores (audit F15/F18). Every later stage stays
    keyed on the whole-file D_SHA256."""
    cal = d["calibration"] if d["score_mode"] == "full" else None
    if isinstance(cal, dict):
        cal = {k: v for k, v in cal.items() if not k.startswith("_")}
    blob = json.dumps({"score_mode": d["score_mode"], "calibration": cal}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def shell_lines(d: dict, sha: str) -> list[str]:
    cal = ""
    if d["score_mode"] == "full":
        c = d["calibration"]
        cal = " ".join(f"--{k.replace('_', '-')} {c[k]}" for k in CAL_KEYS)
        cal += "".join(f" --{k.replace('_', '-')} {c[k]}" for k in CAL_OPTIONAL if k in c)
    vals = {
        "D_POOL": d["pool"], "D_N_FT": d["n_ft"],
        "D_HELDOUT_DOMAINS": ",".join(d["heldout_domains"]), "D_INDOMAIN": ",".join(d["indomain"]),
        "D_EXCLUDE_PRETRAIN": 1 if d["exclude_pretrain_from_heldout"] else 0,
        "D_LR_RULE": d["lr_selection_rule"],
        "D_LR_TOL": "" if d["lr_tolerance_bleu"] is None else d["lr_tolerance_bleu"],
        "D_SCORE_MODE": d["score_mode"], "D_CAL_ARGS": cal,
        "D_FT_CKPT": d["ft_checkpoint"], "D_BUDGET": d["budget"],
        "D_TARGET_TOKENS": "" if d["target_tokens"] is None else d["target_tokens"],
        "D_SPIKE": d["loss_spike_ratio"], "D_SHA256": sha, "D_SCORE_SHA": score_sha(d),
    }
    return [f"{k}={shlex.quote(str(v))}" for k, v in vals.items()]


def diff(old: dict, new: dict) -> list[str]:
    """Decision keys whose values differ ("_" comment keys ignored)."""
    keys = sorted({k for k in (*old, *new) if not k.startswith("_")})
    return [k for k in keys if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "diff":
        try:
            old, new = load(argv[1])[0], load(argv[2])[0]
        except (OSError, ValueError) as e:
            print(f"unreadable: {e}", file=sys.stderr)
            return 2
        ch = diff(old, new) if isinstance(old, dict) and isinstance(new, dict) else ["<not an object>"]
        print(",".join(ch) if ch else "(only comments/whitespace)")
        return 0
    if len(argv) != 2 or argv[0] not in ("check", "shell"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd, path = argv
    try:
        d, sha = load(path)
    except FileNotFoundError:
        print(f"DECISIONS MISSING: {path} does not exist. Copy phase0/e03_decisions.example.json, "
              "replace every \"CHOOSE\" (options: PROTOCOL.md, 'E0.3 decisions required before "
              "controls'), commit and tag it.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"DECISIONS INVALID: cannot read {path}: {e}", file=sys.stderr)
        return 2
    probs = validate(d)
    if probs:
        print(f"DECISIONS INCOMPLETE/INVALID in {path} ({len(probs)} problem(s)):", file=sys.stderr)
        for x in probs:
            print(f"  - {x}", file=sys.stderr)
        return 2
    if cmd == "check":
        print(f"decisions OK: {path} sha256 {sha}")
    else:
        print("\n".join(shell_lines(d, sha)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
