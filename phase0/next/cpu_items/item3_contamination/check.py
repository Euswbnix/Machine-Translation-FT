#!/usr/bin/env python3
"""Independent recount for item 3. Shares no code with item3_contamination.py.

Reads the set files / index files directly, classifies every line with py3langid 0.4.0's PUBLIC
API (LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True).classify, all labels; its
confidence is the 0.9 rule's probability; and a second identifier restricted with
set_languages(['en', 'fr'])), recomputes the length-ratio flags with integer arithmetic, and
compares every count with results.json.

Usage: ~/cpu_items/item3_contamination/venv/bin/python -I check.py --data ~/e06data --res ~/cpu_items/item3_contamination/results.json --workers 5
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PIN = {
    "phase0_e05/ft_un_rep.en": "940cde0f051d6b473c55a6771af64900a20c9c52d66e1a4772e18a5bf9cafe68",
    "phase0_e05/ft_un_rep.fr": "9373be1fc7afb86a1b5ebf76d7cb567bde0ef349443ca9c3c6a87c175185edbd",
    "phase0_e05/ft_un_div.en": "51d9ec1157c58591a7b25b2a1bde52e9dc93342a34c55e55547f0e3857ba94da",
    "phase0_e05/ft_un_div.fr": "1957872adf3f96b8e7fe551eeeae42bbd38f03811511e013f9017b65b954a7e1",
    "phase0_e05/ft_un_random.en": "80fa6aea32f342cc43adec8e66063f8f08c20b6e3b66a60b0bfd7baf3e35c59c",
    "phase0_e05/ft_un_random.fr": "1b54a882d989153697417b1f4b2300b81500acac258e29c542776d39a32094c0",
    "e05_src/ft_random.en": "937d4d4ab2683b585942107d4d260badef54a5a136e3caf97ef6da48d93552a0",
    "e05_src/ft_random.fr": "382e0de574feaaa323939d73a24736b762c76a8b55a0ffd258d64b15e5303e29",
    "vocab/R0.npy": "22eeb53dd3cb621831939cb41a5b9280045dbc5f6b1ce48686234479ac354dc9",
    "vocab/A_narrow_42_58.npy": "e9fac76975bb83e2a4b12ed913c333212cc4a70c440346278c12ad7c0bd7cfa8",
}
FULL = TWO = None


def init():
    global FULL, TWO
    from py3langid.langid import LanguageIdentifier, MODEL_FILE
    FULL = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    TWO = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    TWO.set_languages(["en", "fr"])


def work(batch):
    out = []
    for s in batch:
        lab, p = FULL.classify(s)
        lab2, _ = TWO.classify(s)
        out.append((lab, float(p), lab2))
    return out


def lines(p):
    out = []
    with open(p, "rb") as f:
        data = f.read()
    parts = data.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return [x.decode("utf-8") for x in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.workers > 5:
        sys.exit("at most 5 workers (6 processes with the parent)")
    d = Path(a.data)
    for rel, h in PIN.items():
        g = hashlib.sha256((d / rel).read_bytes()).hexdigest()
        if g != h:
            sys.exit(f"pin mismatch {rel}")
    res = json.loads(Path(a.res).read_text())
    S = {}
    for s in ("ft_un_rep", "ft_un_div", "ft_un_random"):
        S[s] = (lines(d / f"phase0_e05/{s}.en"), lines(d / f"phase0_e05/{s}.fr"))
    ren, rfr = lines(d / "e05_src/ft_random.en"), lines(d / "e05_src/ft_random.fr")
    for name, rel in (("R0", "vocab/R0.npy"), ("A_narrow", "vocab/A_narrow_42_58.npy")):
        idx = np.load(d / rel).tolist()
        S[name] = ([ren[i] for i in idx], [rfr[i] for i in idx])
    del ren, rfr
    uniq = sorted({x for en, fr in S.values() for x in en} | {x for en, fr in S.values() for x in fr})
    print(f"{len(uniq):,} unique strings (both sides pooled)", flush=True)
    t0 = time.time()
    B = 3000
    lab = {}
    with get_context("spawn").Pool(a.workers, initializer=init) as pool:
        for batch, r in zip((uniq[i:i + B] for i in range(0, len(uniq), B)),
                            pool.imap(work, (uniq[i:i + B] for i in range(0, len(uniq), B)))):
            for s, v in zip(batch, r):
                lab[s] = v
    print(f"classified in {time.time() - t0:.0f}s", flush=True)
    report = {}
    allok = True
    for name, (en, fr) in S.items():
        c = dict.fromkeys(["en_wrong", "fr_wrong", "lid_any", "lid_any_conf", "swap_or_copy_2way",
                           "identical_sides", "ratio_ws_1.3", "ratio_ws_1.5", "ratio_ws_2.0",
                           "either_headline"], 0)
        for e, f in zip(en, fr):
            le, pe, le2 = lab[e]
            lf, pf, lf2 = lab[f]
            ew, fw = le != "en", lf != "fr"
            c["en_wrong"] += ew
            c["fr_wrong"] += fw
            c["lid_any"] += ew or fw
            c["lid_any_conf"] += (ew and pe >= 0.9) or (fw and pf >= 0.9)
            c["swap_or_copy_2way"] += le2 == "fr" or lf2 == "en"
            c["identical_sides"] += e.strip() == f.strip()
            ne, nf = len(e.split()), len(f.split())
            out = {}
            for t10 in (13, 15, 20):          # threshold * 10, integer arithmetic
                o = ne == 0 or nf == 0 or 10 * nf > t10 * ne or 10 * ne > t10 * nf
                c[f"ratio_ws_{t10 / 10}"] += o
                out[t10] = o
            c["either_headline"] += (ew or fw) or out[15]
        mine = res["sets"][name]["counts"]
        diff = {k: (v, mine[k]) for k, v in c.items() if v != mine[k]}
        report[name] = {"rows": len(en), "counts": c, "differences_vs_results": diff}
        allok &= not diff
        print(name, "OK" if not diff else f"DIFF {diff}", flush=True)
    # the 20 headline examples
    exdiff = []
    for ex in res["examples_headline_20"]:
        en, fr = S[ex["set"]]
        e, f = en[ex["line"] - 1], fr[ex["line"] - 1]
        if e != ex["en"] or f != ex["fr"] or lab[e][0] != ex["en_top1"][0] or lab[f][0] != ex["fr_top1"][0]:
            exdiff.append(ex["set"] + ":" + str(ex["line"]))
    allok &= not exdiff
    report["examples_headline_20_mismatches"] = exdiff
    report["all_match"] = bool(allok)
    out = Path(a.res).parent / "check_results.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("ALL MATCH" if allok else "MISMATCHES", out, flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
