from common import *
import math
pre = pickle.load(open("pre_en.pkl","rb"))["counts"]
rank = {w: r for r, (w, _) in enumerate(pre.most_common())}
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy"); rnd = np.where(Q["in_ft_un_random"])[0]
def counter(name, idx):
    t = tok(name, "en"); c = collections.Counter()
    for i in idx: c.update(t[i])
    return c
cd, cr, cq = counter("ft_topk", div), counter("ft_random", R0), counter("ft_random", rnd)
for nm, c in (("div", cd), ("R0", cr), ("rand", cq)):
    v = list(c.values())
    print(nm, "types", len(c), "hapax", sum(1 for x in v if x == 1), "count<=5", sum(1 for x in v if x <= 5), "count>=100", sum(1 for x in v if x >= 100))
extra = set(cr) - set(cd); missing = set(cd) - set(cr)
print("types in R0 not in div:", len(extra), " in div not in R0:", len(missing))
def anat(S, c, label):
    has_digit = sum(1 for w in S if any(ch.isdigit() for ch in w))
    bands = collections.Counter()
    for w in S:
        r = rank.get(w)
        b = "OOV-pretrain" if r is None else ("<10k" if r < 10000 else "<30k" if r < 30000 else "<100k" if r < 100000 else ">=100k")
        bands[b] += 1
    tok_mass = sum(c[w] for w in S)
    print(f"{label}: {len(S)} types, {tok_mass} tokens ({100*tok_mass/sum(c.values()):.2f}% of set tokens); contains digit {has_digit}; pretrain-rank bands {dict(bands)}")
anat(extra, cr, "R0-only types")
anat(missing, cd, "div-only types")
# how many rows carry the extra types
t = tok("ft_random", "en")
rows_with_extra = sum(1 for i in R0 if any(w in extra for w in t[i]))
print("R0 rows containing >=1 type absent from div:", rows_with_extra, f"({100*rows_with_extra/len(R0):.1f}%)")
td = tok("ft_topk", "en")
rows_with_missing = sum(1 for i in div if any(w in missing for w in td[i]))
print("div rows containing >=1 type absent from R0:", rows_with_missing, f"({100*rows_with_missing/len(div):.1f}%)")
# samples of extra types by frequency in R0
ex = sorted(extra, key=lambda w: -cr[w])
print("most frequent R0-only types:", [(w, cr[w]) for w in ex[:40]])
rng = random.Random(1); print("random R0-only types:", rng.sample(ex, 40))
ms = sorted(missing, key=lambda w: -cd[w]); print("most frequent div-only types:", [(w, cd[w]) for w in ms[:40]])
# top distribution difference in frequent words: log-odds between div and R0
N1, N2 = sum(cd.values()), sum(cr.values())
lo = {w: math.log((cd[w]+5)/N1) - math.log((cr[w]+5)/N2) for w in set(cd) | set(cr) if cd[w] + cr[w] >= 500}
s = sorted(lo, key=lo.get)
print("words over-represented in R0 vs div:", [(w, round(lo[w],2), cd[w], cr[w]) for w in s[:40]])
print("words over-represented in div vs R0:", [(w, round(lo[w],2), cd[w], cr[w]) for w in s[-40:]])
