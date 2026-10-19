#!/usr/bin/env python3
"""Compare this check's measured numbers (results/*.json, from hcheck.py on the Studio) with the
figures the builder's item4_heldout/README.md states (quoted from E0.3's manifest, the feasibility
report and PROTOCOL.md; the builder ran nothing). Writes compare_results.json. stdlib only."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
J = {n: json.loads((R / f"{n}.json").read_text(encoding="utf-8")) for n in ("inputs", "sets", "pre", "evalcopy", "reduce")}
s, p, e, r = J["sets"], J["pre"], J["evalcopy"], J["reduce"]
fs = s["feasibility_sets"]

rows = [
    # (claim in the builder's README, builder's figure, measured)
    ("(a) feasibility files", 46, s["feasibility_summary"]["files"]),
    ("(a) files sharing nothing with heldout_un (exact and norm, both sides)", 46, s["feasibility_summary"]["all_zero"]),
    ("(a) report field heldout_un_shared_lines = 0/0 and equal to the recount", 46, s["feasibility_summary"]["agree_with_report_field"]),
    ("(a) failed options without a file", 5, len(s["feasibility_listing"]["report_results_without_file"])),
    ("(a) supplementary sets all zero (R0, A_narrow, six E0.5, two parents)", 10,
     sum(v["all_zero"] for v in s["supplementary_sets"].values())),
    ("(a) E0.5 rows whose pair is in the stated parent = all rows",
     sum(v["rows"] for k, v in s["supplementary_sets"].items() if k.startswith("phase0_e05/")),
     sum(v["rows_whose_pair_is_in_parent"]["rows"] for k, v in s["supplementary_sets"].items() if k.startswith("phase0_e05/"))),
    ("(b) heldout rows with exact pair in pool", 2000, r["heldout_rows_found_in_pool"]["exact_pair"]),
    ("(b) heldout rows with exact source in pool", 2000, r["heldout_rows_found_in_pool"]["exact_source"]),
    ("(b) heldout rows with exact target in pool", 2000, r["heldout_rows_found_in_pool"]["exact_target"]),
    ("(b) pool rows, E0.3 category exact pair", 3290, r["e03_reservation_categories"]["measured"]["exact_pair"]),
    ("(b) pool rows, E0.3 category + normalised source", 1078, r["e03_reservation_categories"]["measured"]["extra_by_source"]),
    ("(b) pool rows, E0.3 category + normalised target", 284, r["e03_reservation_categories"]["measured"]["extra_by_target"]),
    ("(b) pool rows touching heldout_un, total", 4652, r["e03_reservation_categories"]["measured"]["rows_reserved_total"]),
    ("(b) of those inside the reused mask (item 5)", 4645, r["touching_rows_by_source_and_mask"]["rows_reserved_total"]["inside_reused_mask"]),
    ("(b) pool rows with an exact heldout source (no prior figure)", None, r["pool_rows_with_heldout"]["exact_source"]),
    ("(b) pool rows with an exact heldout target (no prior figure)", None, r["pool_rows_with_heldout"]["exact_target"]),
    ("(b) pretraining rows touching heldout_un (exact or norm, any side)", 0, p["pretrain_rows_with_heldout"]["any"]),
    ("(b) heldout rows found in pretraining, exact pair", 0, p["heldout_rows_found_in_pretrain"]["exact_pair"]),
    ("(b) heldout rows found in pretraining, norm source", 0, p["heldout_rows_found_in_pretrain"]["norm_source"]),
    ("(b) heldout rows found in pretraining, norm target", 0, p["heldout_rows_found_in_pretrain"]["norm_target"]),
    ("build: pool rows (v2_scored.tsv)", 38275284, r["pool_rows"]),
    ("build: eligible (reused mask)", 16646992, r["heldout_candidates"]["measured"]["n_pool_eligible"]),
    ("build: UN-labelled rows", 11598490, r["heldout_candidates"]["measured"]["label_rows"]),
    ("build: candidates after the pretraining exclusion", 6767225, r["heldout_candidates"]["measured"]["pool_rows"]),
    ("build: excluded as touching pretraining", 4831265, r["heldout_candidates"]["measured"]["pool_rows_excluded_pretrain"]),
    ("build: pretraining rows", 9312233, p["rows"]),
    ("build: RandomState(42) draw equals heldout_un row by row", 2000, r["draw_rederived"]["rows_equal_to_heldout_un_row_by_row"]),
    ("build: distinct pairs in the draw", 1999, r["draw_rederived"]["distinct_pairs_in_draw"]),
    ("build: pool copies of the duplicated heldout pair ('occurs twice in the pool')", 2,
     r["duplicated_heldout_pair"][0]["pool_copies"]),
    ("build: PROTOCOL D4 default draw, pairs verbatim in pretraining", 793, r["protocol_d4_default_draw"]["pairs_verbatim_in_pretrain"]),
    ("evaluated copy: rows", 1999, e["rows"][0]),
    ("evaluated copy: dropped 0-based row", 1764, e["dropped_rows_0based"][0] if len(e["dropped_rows_0based"]) == 1 else e["dropped_rows_0based"]),
]
out = {"inputs_all_ok": J["inputs"]["all_ok"],
       "positive_controls": {"sets_planted": s["positive_controls"]["planted"]["ok"], "sets_self": s["positive_controls"]["self_ok"],
                             "pretrain_e05_table": p["positive_control_e05_pairs_in_pretraining_pct"]["measured"]
                             == p["positive_control_e05_pairs_in_pretraining_pct"]["protocol_table"],
                             "pool_key_path_same_rows": r["key_path"]["same_rows_as_string_path"]},
       "rows": [{"claim": c, "builder": b, "measured": m, "agree": (None if b is None else b == m)} for c, b, m in rows]}
out["disagreements"] = [x for x in out["rows"] if x["agree"] is False]
(Path(__file__).resolve().parent / "compare_results.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for x in out["rows"]:
    print(f"{'  ' if x['agree'] else '!!' if x['agree'] is False else '--'} {x['claim']}: builder {x['builder']}, measured {x['measured']}")
print("positive controls:", out["positive_controls"], "inputs ok:", out["inputs_all_ok"])
