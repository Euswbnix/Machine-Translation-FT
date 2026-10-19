"""item6_forgetting: token-level forgetting (ForgetGenUse), split by the fine-tuning set's vocabulary.

WMT2027_PLAN.md §3.2 item 6. The measure is Saunders & DeNeefe (2024, EMNLP, Sec 2.1, Eq 1), as the
saunders2024 note in phase0/next/groundwork.json describes it: "For each reference token,
ForgetGenUse counts how often the original model produced it correctly and the adapted model no
longer does. It can be split into tokens that appear in the adaptation set (ID) and tokens that do
not (OOD)." The note gives this sentence, not Eq 1's typography, so the counting below is this
script's operationalisation of it (README.md, "Definition"):

  for test segment s and token type v, with r, b, f the counts of v in the reference, the base
  hypothesis and the fine-tuned hypothesis of s:
      B(s,v) = min(b, r)          base model's correct uses of v (clipped, as BLEU's unigram matches)
      F(s,v) = min(f, r)          fine-tuned model's correct uses
      L(s,v) = max(0, B - F)      correct uses the fine-tuned model no longer produces  (forgotten)
      G(s,v) = max(0, F - B)      correct uses only the fine-tuned model produces      (reported only)
  ForgetGenUse_X = sum_{s, v in X} L / sum_{s, v in X} B
      X = All; ID = types that occur >= --id-min-count times in the fine-tuning set's TARGET side
      (tokenised the same way); OOD = the other reference types.
  ForgetGenUse_X_macro (alternative aggregation, reported beside it): the mean over the types of X
      that the base got right at least once of  sum_s L(s,v) / sum_s B(s,v).
  Secondary split (not in the paper): (s,v) pairs whose type v also occurs in the source segment s
  (copyable tokens: names, numbers) versus those whose type does not.

Every count behind the ratios is written, so another normalisation (e.g. by reference occurrences)
can be recomputed from the JSON without re-running.

Tokenisers (--tok): words = lowercased \\w+ (the project's token rule, e04_split.py / e05_build.py;
the default); words_cased = \\w+ with case kept; spm = the base model's SentencePiece pieces
(spm_enfr_v1_fixed, sha256 checked), on raw text.

Fail-closed: source, reference and every hypothesis must have the same number of lines (lines split
on "\\n" only, so a stray \\r, \\x85 or \\u2028 cannot shift them); each fine-tuning prefix must have
equal-length .<src> and .<tgt> files; --expect-src-sha / --expect-ref-sha refuse other files.

  python forgetgenuse.py --src test.en --ref test.fr --base-hyp base.hyp \\
      --ft-hyp run_s42.hyp run_s1.hyp ... --ft-train <dir>/ft_un_rep --out out.json [--tok words|words_cased|spm]

Stdlib only, except sentencepiece for --tok spm. Descriptive analysis; not a registered result.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)
SPM_SHA = "d1c10ea80fb95984f3f240b3579c87a16313b77aa9bc103ca0d6a0e06672a561"   # spm_enfr_v1_fixed.model
# newstest2014 en-fr as E0.5 scored it (Machine_translation data_enfr_v1/test.{en,fr}; the
# src_sha / ref_sha of results/.e03_collect_cache.json in E0.5's results archive)
NEWSTEST2014_SRC_SHA = "967dca935c7cde4ffc2fc8c9424c682335e1690440dbda104948eb1ffa537fc2"
NEWSTEST2014_REF_SHA = "82526d36ae1cd4d5f20919fbf8130e519cc68a0e57fe47bb205d199626457de9"

SPLITS = ("All", "ID", "OOD")
COPY_SPLITS = ("ID_in_src", "ID_not_in_src", "OOD_in_src", "OOD_not_in_src")
DEFINITION = (
    "Per test segment s and reference token type v: B=min(count in base hyp, count in ref), "
    "F=min(count in fine-tuned hyp, count in ref), lost L=max(0,B-F), gained G=max(0,F-B). "
    "ForgetGenUse_X = sum L / sum B over (s,v) with v in X. ID = types occurring >= id_min_count "
    "times in the fine-tuning set's target side under the same tokeniser; OOD = the rest. "
    "_macro = mean over types of X with sum_s B>0 of sum_s L / sum_s B. Copy splits: v also in the "
    "source segment (same tokeniser) or not. Operationalisation of Saunders & DeNeefe (2024) Sec 2.1 "
    "as described in groundwork.json's saunders2024 note; Eq 1's exact normalisation is to be checked "
    "against the paper.")


def die(msg):
    sys.exit(f"REFUSED: {msg}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_lines(p):
    """Lines split on "\\n" only; strict UTF-8."""
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]


def make_tokenizer(mode, spm_path=None, expect_spm_sha=SPM_SHA):
    """Returns (tokenise_one, tokenise_many, info)."""
    if mode == "words":
        one = lambda s: [w.lower() for w in WORD.findall(s)]
        return one, (lambda xs: [one(x) for x in xs]), {"tok": "words", "rule": "lowercased \\w+ (re.UNICODE)"}
    if mode == "words_cased":
        return WORD.findall, (lambda xs: [WORD.findall(x) for x in xs]), {"tok": "words_cased", "rule": "\\w+ (re.UNICODE), case kept"}
    if mode == "spm":
        if spm_path is None:
            die("--tok spm needs --spm")
        got = sha256(spm_path)
        if expect_spm_sha not in ("any", got):
            die(f"{spm_path}: sha256 {got[:12]}..., expected {expect_spm_sha[:12]}...")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(spm_path))

        def many(xs):
            try:
                return sp.encode(list(xs), out_type=str, num_threads=1)
            except TypeError:
                return [sp.encode(x, out_type=str) for x in xs]
        return (lambda s: sp.encode(s, out_type=str)), many, {
            "tok": "spm", "model": str(spm_path), "model_sha256": got,
            "pieces": sp.get_piece_size(), "sentencepiece": spm.__version__, "rule": "SentencePiece pieces of raw text"}
    die(f"unknown --tok {mode}")


def ft_target_vocab(prefixes, src_lang, tgt_lang, tok_many):
    """Counter of target-side token types over the fine-tuning set(s); file records for the JSON."""
    vocab, files = collections.Counter(), []
    for pre in prefixes:
        tgt = Path(f"{pre}.{tgt_lang}")
        src = Path(f"{pre}.{src_lang}")
        if not tgt.is_file():
            die(f"{tgt}: no target-side training file")
        lines = read_lines(tgt)
        rec = {"tgt": str(tgt), "tgt_sha256": sha256(tgt), "rows": len(lines)}
        if src.is_file():
            n_src = len(read_lines(src))
            if n_src != len(lines):
                die(f"{pre}: {n_src} .{src_lang} lines but {len(lines)} .{tgt_lang} lines")
            rec.update({"src": str(src), "src_sha256": sha256(src)})
        else:
            die(f"{src}: no source-side training file (pass the set's prefix, with both sides)")
        CH = 20000
        for i in range(0, len(lines), CH):
            for toks in tok_many(lines[i:i + CH]):
                vocab.update(toks)
        files.append(rec)
    return vocab, files


def score(ref_t, base_t, ft_t, src_t, id_types):
    """Sums per split and per type. All inputs are lists (one per segment) of token lists."""
    sums = {k: collections.Counter() for k in SPLITS + COPY_SPLITS}
    per_type = collections.defaultdict(lambda: [0, 0, 0, 0, 0])      # ref, B, F, L, G
    for r, b, f, s in zip(ref_t, base_t, ft_t, src_t):
        rc, bc, fc, ss = collections.Counter(r), collections.Counter(b), collections.Counter(f), set(s)
        for v, n in rc.items():
            B = min(bc.get(v, 0), n)
            F = min(fc.get(v, 0), n)
            L = B - F if B > F else 0
            G = F - B if F > B else 0
            side = "ID" if v in id_types else "OOD"
            copy = f"{side}_in_src" if v in ss else f"{side}_not_in_src"
            for k in ("All", side, copy):
                c = sums[k]
                c["ref_occ"] += n
                c["base_correct"] += B
                c["ft_correct"] += F
                c["lost"] += L
                c["gained"] += G
            t = per_type[v]
            t[0] += n; t[1] += B; t[2] += F; t[3] += L; t[4] += G
    out = {}
    for k, c in sums.items():
        d = {x: c[x] for x in ("ref_occ", "base_correct", "ft_correct", "lost", "gained")}
        d["forgetgenuse"] = c["lost"] / c["base_correct"] if c["base_correct"] else None
        out[k] = d
    for k in SPLITS:
        types = [v for v in per_type if k == "All" or (k == "ID") == (v in id_types)]
        ratios = [per_type[v][3] / per_type[v][1] for v in types if per_type[v][1] > 0]
        out[k]["ref_types"] = len(types)
        out[k]["types_base_correct"] = len(ratios)
        out[k]["forgetgenuse_macro"] = sum(ratios) / len(ratios) if ratios else None
    out["ref_occ_share_ID"] = out["ID"]["ref_occ"] / out["All"]["ref_occ"] if out["All"]["ref_occ"] else None
    return out, per_type


def top_types(per_type, id_types, vocab, side, key_ix, k=25):
    rows = [(v, t) for v, t in per_type.items() if (side == "All" or (side == "ID") == (v in id_types)) and t[key_ix] > 0]
    rows.sort(key=lambda x: (-x[1][key_ix], x[0]))
    return [{"type": v, "ref_occ": t[0], "base_correct": t[1], "ft_correct": t[2], "lost": t[3],
             "gained": t[4], "ft_train_count": vocab.get(v, 0)} for v, t in rows[:k]]


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="test source (newstest2014 .en)")
    ap.add_argument("--ref", required=True, help="test reference (newstest2014 .fr)")
    ap.add_argument("--base-hyp", required=True, help="the base model's hypotheses on the same test set")
    ap.add_argument("--ft-hyp", required=True, nargs="+", help="fine-tuned hypotheses (one per run, e.g. 5 seeds of one set)")
    ap.add_argument("--ft-train", required=True, nargs="+",
                    help="the fine-tuning set's file prefix(es): <prefix>.<src-lang> and <prefix>.<tgt-lang> (target side defines ID)")
    ap.add_argument("--out", required=True, help="results JSON")
    ap.add_argument("--label", default=None, help="name of the fine-tuning set, for the JSON")
    ap.add_argument("--tok", default="words", choices=("words", "words_cased", "spm"))
    ap.add_argument("--spm", default=None, help="SentencePiece model for --tok spm")
    ap.add_argument("--expect-spm-sha", default=SPM_SHA, help="'any' to accept another model")
    ap.add_argument("--id-min-count", type=int, default=1, help="occurrences in the training target side to count as ID")
    ap.add_argument("--src-lang", default="en")
    ap.add_argument("--tgt-lang", default="fr")
    ap.add_argument("--expect-src-sha", default=None, help="sha256 the --src file must have ('newstest2014' = E0.5's)")
    ap.add_argument("--expect-ref-sha", default=None, help="sha256 the --ref file must have ('newstest2014' = E0.5's)")
    ap.add_argument("--top", type=int, default=25, help="how many most-lost / most-gained types to list")
    ap.add_argument("--fake-ref-note", default=None, help="free text recorded in the JSON (e.g. for a smoke run)")
    a = ap.parse_args(argv)
    if a.id_min_count < 1:
        die("--id-min-count must be >= 1")
    t0 = time.time()

    inputs = {}
    for name, p, expect, named in (("src", a.src, a.expect_src_sha, NEWSTEST2014_SRC_SHA),
                                   ("ref", a.ref, a.expect_ref_sha, NEWSTEST2014_REF_SHA)):
        h = sha256(p)
        if expect is not None:
            want = named if expect == "newstest2014" else expect
            if h != want:
                die(f"--{name} {p}: sha256 {h[:12]}..., expected {want[:12]}...")
        inputs[name] = {"path": str(p), "sha256": h}
    src, ref, base = read_lines(a.src), read_lines(a.ref), read_lines(a.base_hyp)
    n = len(ref)
    if n == 0:
        die("empty reference")
    for name, lines in (("src", src), ("base-hyp", base)):
        if len(lines) != n:
            die(f"--{name} has {len(lines)} lines, --ref has {n}")
    inputs["src"]["lines"] = len(src)
    inputs["ref"]["lines"] = n
    inputs["base_hyp"] = {"path": str(a.base_hyp), "sha256": sha256(a.base_hyp), "lines": len(base)}
    fts = []
    for p in a.ft_hyp:
        lines = read_lines(p)
        if len(lines) != n:
            die(f"--ft-hyp {p} has {len(lines)} lines, --ref has {n}")
        fts.append((p, lines))

    tok_one, tok_many, tok_info = make_tokenizer(a.tok, a.spm, a.expect_spm_sha)
    t1 = time.time()
    vocab, ft_files = ft_target_vocab(a.ft_train, a.src_lang, a.tgt_lang, tok_many)
    id_types = {v for v, c in vocab.items() if c >= a.id_min_count}
    t2 = time.time()
    ref_t, base_t, src_t = tok_many(ref), tok_many(base), tok_many(src)

    runs = []
    for p, lines in fts:
        res, per_type = score(ref_t, base_t, tok_many(lines), src_t, id_types)
        runs.append({"ft_hyp": str(p), "sha256": sha256(p), "lines": len(lines), "results": res,
                     "top_lost_OOD": top_types(per_type, id_types, vocab, "OOD", 3, a.top),
                     "top_lost_ID": top_types(per_type, id_types, vocab, "ID", 3, a.top),
                     "top_gained": top_types(per_type, id_types, vocab, "All", 4, a.top)})
    t3 = time.time()

    summary = {}
    for k in SPLITS + COPY_SPLITS:
        for m in ("forgetgenuse", "forgetgenuse_macro"):
            vals = [r["results"][k].get(m) for r in runs]
            vals = [v for v in vals if v is not None]
            if m == "forgetgenuse_macro" and k in COPY_SPLITS:
                continue
            summary[f"{k}.{m}"] = {"n": len(vals), "mean": statistics.fmean(vals) if vals else None,
                                   "sd": statistics.stdev(vals) if len(vals) > 1 else None,
                                   "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    out = {"item": "item6_forgetting", "label": a.label, "definition": DEFINITION,
           "options": {**tok_info, "id_min_count": a.id_min_count, "src_lang": a.src_lang, "tgt_lang": a.tgt_lang},
           "note": a.fake_ref_note, "inputs": inputs,
           "ft_train": {"files": ft_files, "target_types": len(vocab), "id_types": len(id_types),
                        "target_tokens": sum(vocab.values())},
           "runs": runs, "summary_over_runs": summary,
           "seconds": {"read_check": round(t1 - t0, 3), "ft_vocab": round(t2 - t1, 3),
                       "score_all_runs": round(t3 - t2, 3), "total": round(time.time() - t0, 3)},
           "python": sys.version.split()[0]}
    tmp = Path(f"{a.out}.tmp")
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, a.out)
    return out


def main():
    out = run()
    for r in out["runs"]:
        x = r["results"]
        f = lambda k: "n/a" if x[k]["forgetgenuse"] is None else f"{x[k]['forgetgenuse']:.4f}"
        print(f"{Path(r['ft_hyp']).name}: ForgetGenUse All {f('All')}  ID {f('ID')}  OOD {f('OOD')}  "
              f"(OOD base-correct {x['OOD']['base_correct']:,} of {x['All']['base_correct']:,})")
    print(f"total {out['seconds']['total']} s")


if __name__ == "__main__":
    main()
