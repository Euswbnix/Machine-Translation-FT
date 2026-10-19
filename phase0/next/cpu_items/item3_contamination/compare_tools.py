#!/usr/bin/env python3
"""Item 3: compare this run (py3langid 0.4.0) with the independent check (Apple NaturalLanguage,
~/cpu_items/item3_contamination_check/). Reads only; writes compare_tools.json.

1. Length ratio (same rule, so it must agree EXACTLY): every ratio_* count (3 units x 3
   thresholds, long/short, Tukey fences), the fences themselves, L0 and identical_sides, in every
   set both runs report (the 13 named sets and the 46 feasibility draws).
2. Language ID (different tool): per-set rates for both tools, and the requested contrasts
   (ft_un_rep - ft_un_div, ft_un_div - ft_un_random) plus the extra ones: sign and size of the
   difference, a/b ratio, binomial sd.
3. Row overlap in the three UN sets, R0 and A_narrow: how many rows each tool flags, how many
   both flag. Per-string labels: this run's per_string.npz (its unique-string order is rebuilt here
   exactly as item3_contamination.py builds it, and spot-checked against a fresh classify()) and
   the checker's work/lid_in_k.txt + lid_out_k.tsv.

Usage (Studio): ~/cpu_items/item3_contamination/venv/bin/python -I compare_tools.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

HOME = Path.home()
DATA = HOME / "e06data"
MINE = HOME / "cpu_items" / "item3_contamination"
CHK = HOME / "cpu_items" / "item3_contamination_check"
E05 = ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div", "ft_random_small",
       "heldout_un"]
TUKEY = {"ftrandom_all": "ft_random", "ftrandom_un": "ft_random[un]"}
LID_KEYS = ["lid_any", "en_wrong", "fr_wrong", "lid_any_conf", "swap_or_copy_2way",
            "same_top1_both_sides", "either_headline"]


def read_lines(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]


def main():
    A = json.loads((MINE / "results.json").read_text())
    B = json.loads((CHK / "results.json").read_text())
    out = {"py3langid_results": str(MINE / "results.json"), "apple_nl_results": str(CHK / "results.json")}

    # ---- 1. ratio: exact ----
    names = sorted(set(A["sets"]) & set(B["sets"]))
    only = sorted(set(A["sets"]) ^ set(B["sets"]))
    diffs, compared = [], 0
    for s in names:
        ca, cb = A["sets"][s]["counts"], B["sets"][s]["counts"]
        if A["sets"][s]["rows"] != B["sets"][s]["rows"]:
            diffs.append([s, "rows", A["sets"][s]["rows"], B["sets"][s]["rows"]])
        for k, v in ca.items():
            if not (k.startswith("ratio_") or k in ("L0_violation", "identical_sides")):
                continue
            kb = k
            for mine, theirs in TUKEY.items():
                kb = kb.replace(f"tukey_{mine}", f"tukey_{theirs}")
            compared += 1
            if cb.get(kb) != v:
                diffs.append([s, k, v, cb.get(kb)])
    fen = []
    for u, d in A["rules"]["tukey_fences_log_r"].items():
        for mine, theirs in TUKEY.items():
            if list(d[mine]) != list(B["tukey_fences_log_r"][f"{u}|{theirs}"]):
                fen.append([u, mine, d[mine], B["tukey_fences_log_r"][f"{u}|{theirs}"]])
    out["ratio_exact"] = {"sets_compared": len(names), "sets_in_one_run_only": only,
                          "counts_compared": compared, "count_differences": diffs,
                          "fence_differences": fen, "all_equal": not diffs and not fen}
    print("ratio:", out["ratio_exact"]["sets_compared"], "sets,", compared, "counts,",
          len(diffs), "differences,", len(fen), "fence differences", flush=True)

    # ---- 2. LID rates and contrasts ----
    rates = {}
    for s in ["ft_un_rep", "ft_un_div", "ft_un_random", "R0", "A_narrow", "ft_topk_rep", "ft_topk_div",
              "ft_random_small", "heldout_un", "ft_topk", "ft_topk[un]", "ft_random", "ft_random[un]",
              "feas:ft_un_div_narrow__od2-c2__od1-b", "feas:ft_un_rep_dedup__od4-a",
              "feas:ft_un_div_dd__od4-a"]:
        rates[s] = {"rows": A["sets"][s]["rows"]}
        for k in LID_KEYS:
            rates[s][k] = {"py3langid": [A["sets"][s]["counts"][k], 100 * A["sets"][s]["shares"][k]],
                           "apple_nl": [B["sets"][s]["counts"][k], 100 * B["sets"][s]["shares"][k]]}
    out["lid_rates_count_pct"] = rates
    con = []
    for ca in A["contrasts"]:
        cb = next(x for x in B["contrasts"] if x["a"] == ca["a"] and x["b"] == ca["b"])
        row = {"a": ca["a"], "b": ca["b"], "what": ca["what"], "measures": {}}
        for k in LID_KEYS:
            ma, mb = ca["measures"][k], cb["measures"][k]
            da, db = ma["diff_pp"], mb["diff_pp"]
            ra = ma["a_share"] / ma["b_share"] if ma["b_share"] > 0 else None
            rb = mb["a_pct"] / mb["b_pct"] if mb["b_pct"] > 0 else None
            row["measures"][k] = {
                "py3langid": {"a_pct": 100 * ma["a_share"], "b_pct": 100 * ma["b_share"], "diff_pp": da,
                              "sd_pp": ma["binomial_sd_of_diff_pp"], "a_over_b": ra},
                "apple_nl": {"a_pct": mb["a_pct"], "b_pct": mb["b_pct"], "diff_pp": db,
                             "sd_pp": mb["binomial_sd_diff_pp"], "a_over_b": rb},
                "same_sign": (da > 0) == (db > 0) and da != 0 and db != 0,
                "diff_ratio_py3langid_over_nl": (da / db) if db else None}
        con.append(row)
    out["lid_contrasts"] = con

    # ---- 3. row overlap ----
    P = {}
    for par in ("ft_topk", "ft_random"):
        P[par] = (read_lines(DATA / "e05_src" / f"{par}.en"), read_lines(DATA / "e05_src" / f"{par}.fr"))
    F = {s: (read_lines(DATA / "phase0_e05" / f"{s}.en"), read_lines(DATA / "phase0_e05" / f"{s}.fr"))
         for s in E05}

    def table(side):          # = item3_contamination.py's table()
        idx = {}
        for par in ("ft_topk", "ft_random"):
            for x in P[par][side]:
                if x not in idx:
                    idx[x] = len(idx)
        for s in E05:
            for x in F[s][side]:
                if x not in idx:
                    idx[x] = len(idx)
        return idx
    en_idx, fr_idx = table(0), table(1)
    if (len(en_idx), len(fr_idx)) != (A["unique_strings"]["en"], A["unique_strings"]["fr"]):
        raise SystemExit("unique tables differ from the run's")
    labels = A["environment"]["labels"]
    with np.load(MINE / "per_string.npz") as z:
        en_top, fr_top = z["en_top"], z["fr_top"]
        en_p, fr_p = z["en_topp"], z["fr_topp"]
    # spot check the rebuilt order against a fresh public classify()
    from py3langid.langid import LanguageIdentifier, MODEL_FILE
    ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    rng = random.Random(7)
    inv_en = list(en_idx)
    inv_fr = list(fr_idx)
    bad = 0
    for inv, top, pp in ((inv_en, en_top, en_p), (inv_fr, fr_top, fr_p)):
        for i in rng.sample(range(len(inv)), 500):
            lab, p = ident.classify(inv[i])
            bad += lab != labels[top[i]] or float(p) != pp[i]
    if bad:
        raise SystemExit(f"rebuilt order does not match per_string.npz ({bad} of 1,000)")
    del inv_en, inv_fr
    sets = {s: F[s] for s in ("ft_un_rep", "ft_un_div", "ft_un_random")}
    for name, rel in (("R0", "vocab/R0.npy"), ("A_narrow", "vocab/A_narrow_42_58.npy")):
        ix = np.load(DATA / rel).tolist()
        sets[name] = ([P["ft_random"][0][i] for i in ix], [P["ft_random"][1][i] for i in ix])
    need = set()
    for en, fr in sets.values():
        need.update(en)
        need.update(fr)
    nl = {}
    for k in range(3):
        ins = (CHK / "work" / f"lid_in_{k}.txt").read_bytes()[:-1].decode("utf-8").split("\n")
        with open(CHK / "work" / f"lid_out_{k}.tsv", encoding="utf-8") as f:
            for s, ln in zip(ins, f):
                if s in need:
                    c = ln.rstrip("\n").split("\t")
                    nl[s] = (c[0], float(c[2]))
    missing = len(need - set(nl))
    if missing:
        raise SystemExit(f"{missing} strings have no Apple NL label")
    ov = {}
    for name, (en, fr) in sets.items():
        pe = np.array([labels[en_top[en_idx[x]]] != "en" for x in en])
        pf = np.array([labels[fr_top[fr_idx[x]]] != "fr" for x in fr])
        ne = np.array([nl[x][0] != "en" for x in en])
        nf = np.array([nl[x][0] != "fr" for x in fr])
        r = {"rows": len(en)}
        for key, a_, b_ in (("en_wrong", pe, ne), ("fr_wrong", pf, nf), ("lid_any", pe | pf, ne | nf)):
            both = int((a_ & b_).sum())
            union = int((a_ | b_).sum())
            r[key] = {"py3langid": int(a_.sum()), "apple_nl": int(b_.sum()), "both": both,
                      "jaccard": both / union if union else None,
                      "share_of_nl_flags_also_py3langid": both / int(b_.sum()) if b_.sum() else None,
                      "share_of_py3langid_flags_also_nl": both / int(a_.sum()) if a_.sum() else None}
        if r["lid_any"]["py3langid"] != A["sets"][name]["counts"]["lid_any"] or \
                r["lid_any"]["apple_nl"] != B["sets"][name]["counts"]["lid_any"]:
            raise SystemExit(f"{name}: overlap recount differs from the results files")
        # anatomy of the English-side flags (descriptive): labels, repeated lines, and how much
        # of the flag is the 'pcm' label (Nigerian Pidgin, English-based) that py3langid gives
        # to English name / date / country-list lines
        cnt = Counter(en)
        la = [labels[en_top[en_idx[x]]] for x in en]
        rows = [i for i in range(len(en)) if pe[i]]
        rows_nl = [i for i in range(len(en)) if ne[i]]
        r["en_wrong_anatomy"] = {
            "py3langid": {"rows": len(rows), "distinct_lines": len({en[i] for i in rows}),
                          "rows_whose_line_repeats_in_set": sum(1 for i in rows if cnt[en[i]] > 1),
                          "top_labels": Counter(la[i] for i in rows).most_common(6),
                          "rows_label_pcm": sum(1 for i in rows if la[i] == "pcm"),
                          "conf_ge_0.5": int(sum(1 for i in rows if en_p[en_idx[en[i]]] >= 0.5))},
            "apple_nl": {"rows": len(rows_nl), "distinct_lines": len({en[i] for i in rows_nl}),
                         "rows_whose_line_repeats_in_set": sum(1 for i in rows_nl if cnt[en[i]] > 1),
                         "top_labels": Counter(nl[en[i]][0] for i in rows_nl).most_common(6),
                         "conf_ge_0.5": sum(1 for i in rows_nl if nl[en[i]][1] >= 0.5)}}
        r["lid_any_without_pcm_only_rows_py3langid"] = int(
            sum(1 for i in range(len(en)) if (pe[i] and la[i] != "pcm") or pf[i]))
        ov[name] = r
        print(name, json.dumps(r), flush=True)
    out["row_overlap"] = ov
    out["spot_check_rebuilt_order"] = "1,000 strings, label and confidence identical"
    (MINE / "compare_tools.json").write_text(json.dumps(out, indent=1))
    print("wrote", MINE / "compare_tools.json", flush=True)


if __name__ == "__main__":
    main()
