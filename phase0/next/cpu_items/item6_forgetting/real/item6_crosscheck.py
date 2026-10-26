"""Cross-check item 6's real run with the independent checker's fgu_check.py (words tokeniser).
Run on the Studio from ~/cpu_items/item6_forgetting_check.

2026-10-25: the headline is now Eq 1 (lost uses / reference occurrences; the checker's
alt_per_ref_occ). Besides the ratios, every per-run count (r, B, F, L, G per split) must equal the
builder's (ref_occ, base_correct, ft_correct, lost, gained) exactly."""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import fgu_check as fc
I = Path.home() / "cpu_items" / "item6_forgetting"; H = I / "real_hyps"; R = I / "real_refs"; E = Path.home() / "e06data" / "phase0_e05"
OUT = I / (sys.argv[1] if len(sys.argv) > 1 else "real")
want = {r["hyp"]: r["hyp_sha256"] for r in json.load(open(H / "redecode_report.json"))["rows"]}
assert fc.sha(R / "test.en").startswith("967dca935c7c") and fc.sha(R / "test.fr").startswith("82526d36ae1c")
tok = fc.Tok("words")
ref = [tok(s) for s in fc.read(R / "test.fr")]; src = [tok(s) for s in fc.read(R / "test.en")]
bn = "baseline_v1.1_averaged.newstest2014.hyp"; assert fc.sha(H / bn) == want[bn]
base = [tok(s) for s in fc.read(H / bn)]
builder = json.load(open(OUT / "summary.json"))
COUNTS = {"r": "ref_occ", "B": "base_correct", "F": "ft_correct", "L": "lost", "G": "gained"}
out, worst = {}, 0.0
for st in ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div", "ft_random_small"]:
    vocab, _ = fc.train_vocab(str(E / st), tok)
    runs = json.load(open(OUT / f"{st}_words.json"))["runs"]
    per = {"All": [], "ID": [], "OOD": []}
    for seed, b in zip((42, 1, 2, 3, 4), runs):
        n = f"{st}_lr1_s{seed}_st2.newstest2014.hyp"; assert fc.sha(H / n) == want[n], n
        assert Path(b["ft_hyp"]).name == n, (b["ft_hyp"], n)
        r = fc.fgu(ref, base, [tok(s) for s in fc.read(H / n)], src, vocab)
        for g in ("All", "ID", "OOD", "OOD_in_src", "OOD_not_in_src"):
            for k, bk in COUNTS.items():
                assert r[g][k] == b["results"][g][bk], (st, seed, g, k, r[g][k], b["results"][g][bk])
        for g in per: per[g].append(r[g]["alt_per_ref_occ"])   # Eq 1: L / reference occurrences
    out[st] = {g: (statistics.mean(v), statistics.stdev(v)) for g, v in per.items()}
    for g, (m, s) in out[st].items():
        worst = max(worst, abs(m - builder[f"{st}_words"][g]["mean"]), abs(s - builder[f"{st}_words"][g]["sd"]))
    print(f"{st:16s} " + "  ".join(f"{g} {m:.4f} ({s:.4f})" for g, (m, s) in out[st].items()))
json.dump({"checker_words_eq1": out, "max_abs_diff_vs_builder": worst,
           "per_run_counts": "r, B, F, L, G equal the builder's for All/ID/OOD/OOD_in_src/OOD_not_in_src, 30 runs"},
          open("crosscheck_real_words_eq1.json", "w"), indent=1)
print(f"all 30 runs: per-split counts identical; max |mean or sd difference| vs builder = {worst:.2e}")
