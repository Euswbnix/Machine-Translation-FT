#!/usr/bin/env python3
"""Compare check_results.json (this checker's own run) with the builder's results.json, number by number.
usage: compare.py <builder results.json> <check_results.json> <out.json>"""
import json
import sys

B = json.load(open(sys.argv[1]))
M = json.load(open(sys.argv[2]))
rows, bad = [], []


def cmp(what, b, m, tol=0.0):
    ok = (b is None and m is None) or (b is not None and m is not None and abs(b - m) <= tol)
    rows.append({"what": what, "builder": b, "checker": m, "ok": ok})
    if not ok:
        bad.append(rows[-1])


ORDER = ("ft_un_rep", "ft_un_div", "ft_un_random", "A_narrow", "R0", "ft_topk_rep", "ft_topk_div",
         "ft_random_small", "ft_topk", "ft_random")
cov, wt, mb = M["coverage"], M["word_types"], M["coverage_at_matched_pieces"]
tw = M.get("token_weighted_and_side_vocab", {})
for n in ORDER:
    bs = B["sets"][n]
    cmp(f"{n} rows", bs["rows"], cov[n]["rows"])
    cmp(f"{n} rows dropped by trainer", bs["rows_dropped_by_trainer"], cov[n]["rows"] - cov[n]["rows_kept_by_trainer"])
    for side in ("source", "target"):
        b, m = bs[side], cov[n][side]
        cmp(f"{n} {side} piece types", b["piece_types"], m["types_k1"])
        cmp(f"{n} {side} coverage %", 100 * b["coverage"], m["coverage_pct"], 1e-9)
        cmp(f"{n} {side} piece tokens", b["piece_tokens"], m["piece_tokens"])
        cmp(f"{n} {side} unk tokens", b["unknown_tokens"], m["unk_tokens"])
        for k in (2, 5, 20):
            cmp(f"{n} {side} types k>={k}", b["piece_types_min_count"][f"k{k}"], m[f"types_k{k}"])
        cmp(f"{n} {side} matched-piece types", b.get("piece_types_at_matched_pieces"), mb[n][f"{side}_types"])
        if tw:
            t = tw[n][f"{side}_all"]
            cmp(f"{n} {side} uncovered pretraining share (ppm, all pretraining rows)",
                b["pretraining_token_share_uncovered_ppm"], 1e4 * t["uncovered_share_pct_excl_special"], 1e-6)
            cmp(f"{n} {side} coverage of side vocab", b["coverage_of_side_vocab"], t["coverage_of_side_vocab_pct"] / 100, 1e-12)
            cmp(f"{n} {side} set types outside side vocab", b["types_outside_side_vocab"], t["set_types_outside_side_vocab"])
    bw, mw = bs["words"], wt[n]
    cmp(f"{n} source words", bw["source_words"], mw["source_words"])
    cmp(f"{n} french words", bw["french_words"], mw["french_words"])
    cmp(f"{n} source word types D11", bw["source_word_types_D11"], mw["source_word_types_D11"])
    cmp(f"{n} french types @6.5M D12", bw["french_word_types_at_6500000_D12"], mw["french_types_at_6500000_D12"])
    cmp(f"{n} source types @5.6M", bw["source_word_types_at_5600000"], mw["source_types_at_5600000"])
cmp("budget source pieces", B["budgets"]["pieces_matched"]["source"], mb["budget_source"])
cmp("budget target pieces", B["budgets"]["pieces_matched"]["target"], mb["budget_target"])
if "pretraining" in M:
    for side, key in (("source", "en_all"), ("target", "fr_all")):
        cmp(f"pretraining {side} piece tokens", B["pretraining"][side]["piece_tokens"], M["pretraining"][key]["piece_tokens"])
        cmp(f"pretraining {side} unk tokens", B["pretraining"][side]["unknown_tokens"], M["pretraining"][key]["unk_tokens"])
        cmp(f"pretraining {side} types seen", B["pretraining"][side]["piece_types_seen"], M["pretraining"][key]["types_seen_countable"])
for rule in ("R0_rule", "ft_un_random_rule"):
    s = B["resampling"]["summary"][rule]
    for j, side in enumerate(("source", "target")):
        cmp(f"redraw {rule} {side} sd types (ddof=1)", s[f"{side}_piece_types"]["sd"], M["redraws"][rule]["sd_sample_types"][j], 1e-9)
for c in B["contrasts"]:
    key = f"{c['a']} - {c['b']}"
    if key not in M["gaps"]:
        continue
    g = M["gaps"][key]
    mm = c["measures"]
    cmp(f"{key} src cov diff (pp)", 100 * mm["source_coverage"]["a_minus_b"], g["src_cov_pp"]["diff"], 1e-9)
    cmp(f"{key} tgt cov diff (pp)", 100 * mm["target_coverage"]["a_minus_b"], g["tgt_cov_pp"]["diff"], 1e-9)
    cmp(f"{key} src cov rel (%)", 100 * mm["source_coverage"]["relative_(a-b)/b"], g["src_cov_pp"]["rel_pct"], 1e-9)
    cmp(f"{key} tgt cov rel (%)", 100 * mm["target_coverage"]["relative_(a-b)/b"], g["tgt_cov_pp"]["rel_pct"], 1e-9)
    cmp(f"{key} src k20 rel (%)", 100 * mm["source_coverage_k20"]["relative_(a-b)/b"], g["src_types_k20"]["rel_pct"], 1e-9)
    cmp(f"{key} tgt k20 rel (%)", 100 * mm["target_coverage_k20"]["relative_(a-b)/b"], g["tgt_types_k20"]["rel_pct"], 1e-9)
    cmp(f"{key} src word types rel (%)", 100 * mm["source_word_types_D11"]["relative_(a-b)/b"], g["src_word_types_D11"]["rel_pct"], 1e-9)
    cmp(f"{key} fr types D12 rel (%)", 100 * mm["french_word_types_at_6500000_D12"]["relative_(a-b)/b"], g["fr_types_D12"]["rel_pct"], 1e-9)
out = {"compared": len(rows), "disagree": len(bad), "disagreements": bad, "all": rows}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"compared {len(rows)} numbers, {len(bad)} disagree")
for r in bad:
    print("  DIFF", r)
