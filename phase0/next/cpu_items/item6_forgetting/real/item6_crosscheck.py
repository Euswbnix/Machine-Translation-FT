"""Cross-check item 6's real run with the independent checker's fgu_check.py (words tokeniser).
Run on the Studio from ~/cpu_items/item6_forgetting_check."""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import fgu_check as fc
I = Path.home() / "cpu_items" / "item6_forgetting"; H = I / "real_hyps"; R = I / "real_refs"; E = Path.home() / "e06data" / "phase0_e05"
want = {r["hyp"]: r["hyp_sha256"] for r in json.load(open(H / "redecode_report.json"))["rows"]}
assert fc.sha(R / "test.en").startswith("967dca935c7c") and fc.sha(R / "test.fr").startswith("82526d36ae1c")
tok = fc.Tok("words")
ref = [tok(s) for s in fc.read(R / "test.fr")]; src = [tok(s) for s in fc.read(R / "test.en")]
bn = "baseline_v1.1_averaged.newstest2014.hyp"; assert fc.sha(H / bn) == want[bn]
base = [tok(s) for s in fc.read(H / bn)]
builder = json.load(open(I / "real" / "summary.json"))
out = {}
for st in ["ft_un_rep", "ft_un_div", "ft_un_random", "ft_topk_rep", "ft_topk_div", "ft_random_small"]:
    vocab, _ = fc.train_vocab(str(E / st), tok)
    per = {"All": [], "ID": [], "OOD": []}
    for seed in (42, 1, 2, 3, 4):
        n = f"{st}_lr1_s{seed}_st2.newstest2014.hyp"; assert fc.sha(H / n) == want[n], n
        r = fc.fgu(ref, base, [tok(s) for s in fc.read(H / n)], src, vocab)
        for g in per: per[g].append(r[g]["fgu"])
    out[st] = {g: (statistics.mean(v), statistics.stdev(v)) for g, v in per.items()}
    print(f"{st:16s} " + "  ".join(f"{g} {m:.4f} ({s:.4f})" for g, (m, s) in out[st].items()))
json.dump({"checker_words": out}, open("crosscheck_real_words.json", "w"), indent=1)
print("builder summary keys:", list(builder)[:6])
