"""Independent recount for smoke_run.sh's outputs (does not import forgetgenuse.py).

  A_refbase_<tok>.json  reference = base hypothesis: base_correct = ref_occ and gained = 0 in every
                        split; ft_correct = clipped matches of ft against the reference
                        (Counter intersection); lost = ref_occ - matches; ID/OOD recounted from a
                        fresh read of the training target side.
  B_refft_<tok>.json    reference = ft hypothesis: lost = 0 in every split, ft_correct = ref_occ.
Writes <dir>/smoke_summary.json; exits 1 on any failed check.
Usage: python smoke_check.py <smoke dir>
"""
import collections
import json
import re
import sys
from pathlib import Path

W = re.compile(r"\w+")


def lines(p):
    t = Path(p).read_bytes().decode("utf-8")
    parts = t.split("\n")
    return parts[:-1] if t.endswith("\n") else parts


def tokenizer(opts):
    if opts["tok"] == "words":
        return lambda s: [w.lower() for w in W.findall(s)]
    if opts["tok"] == "words_cased":
        return W.findall
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=opts["model"])
    return lambda s: sp.encode(s, out_type=str)


d = Path(sys.argv[1])
fails, summary = [], {"checks": {}, "fake_numbers": {}, "vocab": {}, "timing_s": {}}
for js in sorted(d.glob("*.json")):
    if js.name == "smoke_summary.json":
        continue
    o = json.loads(js.read_text(encoding="utf-8"))
    tag, res = js.stem, o["runs"][0]["results"]
    summary["timing_s"][tag] = o["seconds"]
    summary["vocab"][tag] = {"set": o["label"], "tok": o["options"]["tok"], "target_types": o["ft_train"]["target_types"],
                             "id_types": o["ft_train"]["id_types"], "rows": o["ft_train"]["files"][0]["rows"]}
    summary["fake_numbers"][tag] = {k: res[k]["forgetgenuse"] for k in ("All", "ID", "OOD")} | {
        "OOD_base_correct": res["OOD"]["base_correct"], "All_base_correct": res["All"]["base_correct"],
        "ref_occ_share_ID": res["ref_occ_share_ID"]}
    if not (tag.startswith("A_") or tag.startswith("B_")):
        continue
    tok = tokenizer(o["options"])
    ref = [tok(x) for x in lines(o["inputs"]["ref"]["path"])]
    ft = [tok(x) for x in lines(o["runs"][0]["ft_hyp"])]
    vocab = set()
    for x in lines(o["ft_train"]["files"][0]["tgt"]):
        vocab.update(tok(x))
    ref_occ = sum(len(r) for r in ref)
    match = sum(sum((collections.Counter(f) & collections.Counter(r)).values()) for f, r in zip(ft, ref))
    id_occ = sum(1 for r in ref for v in r if v in vocab)
    id_match = sum(sum(c for v, c in (collections.Counter(f) & collections.Counter(r)).items() if v in vocab)
                   for f, r in zip(ft, ref))
    got = {k: res[k] for k in ("All", "ID", "OOD")}
    if tag.startswith("A_"):
        want = {"All.ref_occ": ref_occ, "All.base_correct": ref_occ, "All.gained": 0, "All.ft_correct": match,
                "All.lost": ref_occ - match, "ID.ref_occ": id_occ, "ID.lost": id_occ - id_match,
                "OOD.lost": (ref_occ - id_occ) - (match - id_match), "ID.gained": 0, "OOD.gained": 0}
    else:
        want = {"All.ref_occ": ref_occ, "All.ft_correct": ref_occ, "All.lost": 0, "ID.lost": 0, "OOD.lost": 0,
                "ID.ref_occ": id_occ}
    for k, v in want.items():
        s, f = k.split(".")
        ok = got[s][f] == v
        summary["checks"][f"{tag}:{k}"] = {"want": v, "got": got[s][f], "ok": ok}
        if not ok:
            fails.append(f"{tag}:{k} want {v} got {got[s][f]}")
    if tag.startswith("A_"):
        fgu = 1 - match / ref_occ
        ok = abs(res["All"]["forgetgenuse"] - fgu) < 1e-12
        summary["checks"][f"{tag}:All.forgetgenuse=1-matches/ref_occ"] = {"want": fgu, "got": res["All"]["forgetgenuse"], "ok": ok}
        if not ok:
            fails.append(f"{tag}: forgetgenuse")
summary["n_checks"] = len(summary["checks"])
summary["failed"] = fails
(d / "smoke_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{summary['n_checks']} checks, {len(fails)} failed")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
