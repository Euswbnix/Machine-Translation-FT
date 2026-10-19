#!/usr/bin/env python3
"""Compare every number of this item's results.json (heldout_hygiene.py + check_awk.sh, run on the
Mac Studio) with the independent checker's measurements (../item4_heldout_check/results/*.json,
hcheck.py, its own code). stdlib only; reads JSON, writes compare_with_check.json next to itself.

  compare_with_check.py [--per pool_per_heldout_row.json]

--per: optional local copy of the Studio-side pool_per_heldout_row.json (written by `pool`), for
the per-row copy counts the checker reports (duplicated pair, source_max, target_max).
A row is "derived" when the checker's figure is computed here from its own fields (stated in the row).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHK = HERE.parent / "item4_heldout_check" / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=Path)
    a = ap.parse_args()
    B = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    C = {n: json.loads((CHK / f"{n}.json").read_text(encoding="utf-8")) for n in ("inputs", "sets", "pre", "evalcopy", "reduce")}
    rows: list[dict] = []

    def cmp(section, claim, b, c, how="direct"):
        rows.append({"section": section, "claim": claim, "builder": b, "checker": c, "how": how, "agree": b == c})

    # ---- inputs -----------------------------------------------------------------------------
    bi, ci = B["inputs"], C["inputs"]
    for p, v in bi["sha256"].items():
        cmp("inputs", f"sha256 {p}", v["sha256"], ci["sha256_vs_e06data_SHA256SUMS"][p]["got"])
    for k, v in bi["e03_fingerprints"].items():
        cmp("inputs", f"E0.3 fingerprint matches: {k}", v["match"], ci["e03_fingerprints"][k]["ok"])
    for f, v in bi["bundle_sha256"].items():
        cmp("inputs", f"sha256 inputs/{f}", v["sha256"], ci["bundle_sha256"][f]["got"])

    # ---- (a) sets ---------------------------------------------------------------------------
    bs, cs = B["sets"], C["sets"]
    for k, v in bs["heldout_un"].items():
        cmp("heldout_un", k, v, cs["heldout_un"][k])
    cmp("sets", "feasibility_report.json sha256", bs["feasibility_report_sha256"], cs["feasibility_report_sha256"]["got"])
    for f, v in bs["report_inputs_match"].items():
        cmp("sets", f"report input sha256 {f}", v["sha256"], cs["report_inputs"][f]["got"])
    cmp("sets", "failed options without a file", sorted(bs["failed_options_no_file"]),
        sorted(cs["feasibility_listing"]["report_results_without_file"]))
    cmp("sets", "file names (46)", sorted(bs["feasibility_sets"]), sorted(cs["feasibility_sets"]))
    SET = {"source": "exact_source", "target": "exact_target", "pair": "exact_pair", "norm_source": "norm_source",
           "norm_target": "norm_target"}
    per_file = {"compared": 0, "agree": 0, "differ": []}
    for fn, b in bs["feasibility_sets"].items():
        c = cs["feasibility_sets"][fn]
        pairs = [("rows", b["rows"], c["rows"]), ("tag", b["tag"], c["tag"]), ("parent", b["parent"], c["parent"]),
                 ("sha256", b["sha256"], c["sha256"]), ("un_rows", b["un_rows"], c["un_rows"]),
                 ("report_field", b["report_field"], c["report_field"]),
                 ("agrees_with_report", b["agrees_with_report"], c["agrees_with_report_field"]),
                 ("all_zero", b["all_zero"], c["all_zero"]),
                 ("shared_lines", b["shared_lines_as_feasibility_report"], c["distinct_shared_lines_as_report"])]
        pairs += [(f"set rows {k}", b["set_rows_touching_heldout"][k], c["set_rows_with_heldout"][ck]) for k, ck in SET.items()]
        h = b["heldout_rows_in_set"]
        pairs += [(f"heldout rows exact {k}", h["exact"][k], c["heldout_rows_found_in_set"][f"exact_{k}"]) for k in ("source", "target", "pair")]
        pairs += [(f"heldout rows norm {k}", h["norm"][k], c["heldout_rows_found_in_set"][f"norm_{k}"]) for k in ("source", "target")]
        for name, x, y in pairs:
            per_file["compared"] += 1
            if x == y:
                per_file["agree"] += 1
            else:
                per_file["differ"].append([fn, name, x, y])
    cmp("sets", f"per-file values ({per_file['compared']} = 46 files x 19 values) that agree", per_file["compared"], per_file["agree"])
    s, t = bs["feasibility_summary"], cs["feasibility_summary"]
    cmp("sets", "files", s["files"], t["files"])
    cmp("sets", "files sharing nothing with heldout_un", s["all_zero"], t["all_zero"])
    cmp("sets", "files whose report field equals the recount", s["agree_with_report"], t["agree_with_report_field"])
    cmp("sets", "files by parent", s["by_parent"], t["by_parent"])
    cmp("sets", "distinct index contents", s["distinct_index_contents"], t["distinct_index_contents"])
    cmp("sets", "repo SHA256SUMS: entries matching (46 .npy + report)", 47 if bs["repo_feasibility_SHA256SUMS"]["match"] else -1,
        t["sha256_ok_repo"] + (cs["feasibility_report_sha256"]["got"] == cs["feasibility_report_sha256"]["repo_SHA256SUMS"]),
        "checker: sha256_ok_repo + report sha256 equal to the repo's")
    SUP = {"vocab/R0.npy": "vocab/R0.npy", "vocab/A_narrow_42_58.npy": "vocab/A_narrow_42_58.npy",
           "ft_topk (all rows)": "ft_topk (all 926,999 rows)", "ft_random (all rows)": "ft_random (all 926,999 rows)"}
    SUP |= {k: k for k in bs["supplementary_sets"] if k.startswith("phase0_e05/")}
    for bk, ck in SUP.items():
        b, c = bs["supplementary_sets"][bk], cs["supplementary_sets"][ck]
        cmp("sets", f"supplementary {bk}: rows", b["rows"], c["rows"])
        cmp("sets", f"supplementary {bk}: all zero", b["all_zero"], c["all_zero"])
        cmp("sets", f"supplementary {bk}: set rows (5 counts)", b["set_rows_touching_heldout"],
            {k: c["set_rows_with_heldout"][ck2] for k, ck2 in SET.items()})
        if bk.startswith("vocab/"):
            cmp("sets", f"supplementary {bk}: identical to", b["identical_to"], c["same_bytes_as"])
    cmp("sets", "parent rows ft_topk / ft_random", [bs["supplementary_sets"]["ft_topk (all rows)"]["rows"],
        bs["supplementary_sets"]["ft_random (all rows)"]["rows"]], [cs["parents_rows"]["ft_topk"], cs["parents_rows"]["ft_random"]])

    # ---- (b) pool ---------------------------------------------------------------------------
    bp, cr = B["pool"], C["reduce"]
    cmp("pool", "pool rows", bp["pool_rows"], cr["pool_rows"])
    cmp("pool", "rows with other than 3 fields", bp["rows_not_3_fields"], cr["rows_not_3_fields"])
    POOL = {"exact_pair": "exact_pair", "exact_source": "exact_source", "exact_target": "exact_target",
            "exact_source_not_pair": "exact_source_not_pair", "exact_target_not_pair": "exact_target_not_pair",
            "exact_any": "exact_source_or_target", "norm_source": "norm_source", "norm_target": "norm_target",
            "norm_source_not_exact_source": "norm_source_not_exact", "norm_target_not_exact_target": "norm_target_not_exact"}
    for k, ck in POOL.items():
        cmp("pool", f"pool rows: {k}", bp["pool_rows_touching_heldout"][k], cr["pool_rows_with_heldout"][ck],
            "direct" if k != "exact_any" else "checker exact_source_or_target (a pair hit is a source hit, so the same set)")
    for k, v in bp["e03_reservation_categories"].items():
        cmp("pool", f"E0.3 category {k}", v, cr["e03_reservation_categories"]["measured"][k])
        cmp("pool", f"E0.3 category {k} vs E0.3's manifest", v, cr["e03_reservation_categories"]["e03_manifest"][k], "E0.3 manifest (via checker)")
    hp = bp["heldout_rows_in_pool"]
    for k in ("source", "target", "pair"):
        cmp("pool", f"heldout rows with exact {k} in pool", hp["exact"][k], cr["heldout_rows_found_in_pool"][f"exact_{k}"])
    for k in ("source", "target"):
        cmp("pool", f"heldout rows with norm {k} in pool", hp["norm"][k], cr["heldout_rows_found_in_pool"][f"norm_{k}"])
    for k, ck in (("source", "distinct_sources"), ("target", "distinct_targets"), ("pair", "distinct_pairs")):
        cmp("pool", f"distinct heldout {k}s found in pool", hp["distinct_exact"][k],
            cs["heldout_un"][ck] if cr["heldout_rows_found_in_pool"][f"exact_{k}"] == 2000 else None,
            "derived: checker found all 2,000 rows, so every distinct string")
    cmp("pool", "heldout rows by pool copies of their pair", bp["heldout_rows_by_pool_copies_of_their_pair"],
        cr["pool_copies_per_heldout_row"]["pair"])
    dist = {int(k) for k in cr["pool_copies_per_heldout_row"]["pair"]}
    cmp("pool", "pool copies per distinct heldout pair, max", bp["pool_rows_exact_pair_per_distinct_heldout_pair"]["max"],
        cr["pool_copies_per_heldout_row"]["pair_max"])
    cmp("pool", "pool copies per distinct heldout pair, min", bp["pool_rows_exact_pair_per_distinct_heldout_pair"]["min"], min(dist),
        "derived: smallest key of the checker's distribution")
    for k in ("pair", "source", "target"):
        cmp("pool", f"heldout rows with no pool row ({k})", bp["heldout_rows_with_no_pool_row"][k],
            2000 - cr["heldout_rows_found_in_pool"][f"exact_{k}"], "derived: 2,000 - checker's rows found")
    ev, ce = bp["evaluated_copy"], C["evalcopy"]
    cmp("evaluated copy", "rows", ev["rows"], ce["rows"][0])
    cmp("evaluated copy", "sha256", ev["sha256"], ce["sha256"])
    cmp("evaluated copy", "dropped rows (0-based)", ev["dropped_rows_0based"], ce["dropped_rows_0based"])
    cmp("evaluated copy", "kept rows with exact pair in pool", ev["kept_rows_with_exact_pair_in_pool"],
        ce["rows"][0] if cr["heldout_rows_found_in_pool"]["exact_pair"] == 2000 else None, "derived: all rows found")
    bm, cm = bp["by_source_and_mask"], cr["touching_rows_by_source_and_mask"]
    for k in ("exact_pair", "extra_by_source", "extra_by_target", "rows_reserved_total"):
        cmp("pool", f"{k}: rows / inside reused mask / sources", [bm[k]["rows"], bm[k]["inside_reused_mask"], bm[k]["sources"]],
            [cm[k]["rows"], cm[k]["inside_reused_mask"], cm[k]["sources"]])
    for k, ck in (("exact_source_not_pair", "exact_source"), ("exact_target_not_pair", "exact_target")):
        src = {n: cm[ck]["sources"].get(n, 0) - cm["exact_pair"]["sources"].get(n, 0) for n in cm[ck]["sources"]}
        cmp("pool", f"{k}: rows / inside reused mask / sources", [bm[k]["rows"], bm[k]["inside_reused_mask"], bm[k]["sources"]],
            [cm[ck]["rows"] - cm["exact_pair"]["rows"], cm[ck]["inside_reused_mask"] - cm["exact_pair"]["inside_reused_mask"],
             {n: x for n, x in src.items() if x}], f"derived: checker {ck} minus exact_pair (pair rows are source and target rows)")
    if a.per:
        per = json.loads(a.per.read_text(encoding="utf-8"))
        for d in cr["duplicated_heldout_pair"]:
            cmp("pool", f"pool copies of the duplicated heldout pair (heldout rows {d['heldout_rows']})",
                [per[r]["pair"] for r in d["heldout_rows"]], [d["pool_copies"]] * len(d["heldout_rows"]), "builder: pool_per_heldout_row.json")
        cmp("pool", "max pool rows sharing one heldout source", max(p["source"] for p in per),
            cr["pool_copies_per_heldout_row"]["source_max"], "builder: pool_per_heldout_row.json")
        cmp("pool", "max pool rows sharing one heldout target", max(p["target"] for p in per),
            cr["pool_copies_per_heldout_row"]["target_max"], "builder: pool_per_heldout_row.json")
        cmp("pool", "heldout rows whose pair has more than one pool copy", sum(1 for p in per if p["pair"] > 1),
            cr["pool_copies_per_heldout_row"]["pair_rows_with_more_than_1"], "builder: pool_per_heldout_row.json")

    # ---- (b) pretraining --------------------------------------------------------------------
    bt, cp = B["pretrain"], C["pre"]
    cmp("pretrain", "pretraining rows", bt["pretrain_rows"], cp["rows"])
    for k, v in bt["pretrain_rows_touching_heldout"].items():
        cmp("pretrain", f"pretraining rows: {k}", v, cp["pretrain_rows_with_heldout"][k])
    h = bt["heldout_rows_in_pretrain"]
    for k in ("source", "target", "pair"):
        cmp("pretrain", f"heldout rows with exact {k} in pretraining", h["exact"][k], cp["heldout_rows_found_in_pretrain"][f"exact_{k}"])
    for k in ("source", "target"):
        cmp("pretrain", f"heldout rows with norm {k} in pretraining", h["norm"][k], cp["heldout_rows_found_in_pretrain"][f"norm_{k}"])

    # ---- the builder's awk recount (check_awk.sh) against the checker -------------------------
    pat = re.compile(r"(\S+) rows (\d+) \| rows with a heldout source (\d+), target (\d+), pair (\d+) \| "
                     r"heldout rows \(of (\d+)\) whose source (\d+), target (\d+), pair (\d+) occurs")
    aw = {}
    for f, txt in B.get("independent_check_awk", {}).items():
        m = pat.search(txt)
        if m:
            aw[m.group(1)] = [int(x) for x in m.groups()[1:]]
    pw = cr["pool_rows_with_heldout"]
    hf = cr["heldout_rows_found_in_pool"]
    cmp("awk", "v2_scored.tsv: rows, src, tgt, pair rows | heldout of N, src, tgt, pair", aw.get("v2_scored.tsv"),
        [cr["pool_rows"], pw["exact_source"], pw["exact_target"], pw["exact_pair"], 2000, hf["exact_source"], hf["exact_target"], hf["exact_pair"]])
    pr, ph = cp["pretrain_rows_with_heldout"], cp["heldout_rows_found_in_pretrain"]
    cmp("awk", "train.clean: rows, src, tgt, pair rows | heldout of N, src, tgt, pair", aw.get("train.clean"),
        [cp["rows"], pr["exact_source"], pr["exact_target"], pr["exact_pair"], 2000, ph["exact_source"], ph["exact_target"], ph["exact_pair"]])
    for n in ("ft_topk", "ft_random"):
        c = cs["supplementary_sets"][f"{n} (all 926,999 rows)"]
        cmp("awk", f"{n}: rows, src, tgt, pair rows | heldout of N, src, tgt, pair", aw.get(n),
            [c["rows"], c["set_rows_with_heldout"]["exact_source"], c["set_rows_with_heldout"]["exact_target"],
             c["set_rows_with_heldout"]["exact_pair"], 2000, c["heldout_rows_found_in_set"]["exact_source"],
             c["heldout_rows_found_in_set"]["exact_target"], c["heldout_rows_found_in_set"]["exact_pair"]])
    odd = re.search(r"rows with other than 3 fields (\d+)", B.get("independent_check_awk", {}).get("pool_awk.txt", ""))
    cmp("awk", "v2_scored.tsv rows with other than 3 fields", int(odd.group(1)) if odd else None, cr["rows_not_3_fields"])

    out = {"compared": len(rows), "agree": sum(r["agree"] for r in rows),
           "disagreements": [r for r in rows if not r["agree"]], "per_file_differences": per_file["differ"], "rows": rows}
    (HERE / "compare_with_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        b = json.dumps(r["builder"], ensure_ascii=False)
        c = json.dumps(r["checker"], ensure_ascii=False)
        print(f"{'  ' if r['agree'] else '!!'} [{r['section']}] {r['claim']}: builder {b[:90]} | checker {c[:90]}")
    print(f"\n{out['agree']} of {out['compared']} comparisons agree; per-file differences: {len(per_file['differ'])}")
    return 0 if out["agree"] == out["compared"] and not per_file["differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
