from arms import *
from a8_lib import nn_match
div = np.where(T["in_ft_un_div"])[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
hist = hist_of(T["len"], div)
LT = np.load("lexhit_top.npy")
# within-length-bucket median split of the top band (UN, rep=0) by lexical literalness
b = collections.defaultdict(list)
for i in TD: b[int(T["len"][i])].append(int(i))
rng = random.Random(42); lit, free = [], []
for L, idx in b.items():
    v = np.array([LT[i] if np.isfinite(LT[i]) else -1 for i in idx]); jit = np.array([rng.random() for _ in idx])
    o = np.lexsort((jit, v)); h = len(idx)//2
    free += [idx[k] for k in o[:h]]; lit += [idx[k] for k in o[len(idx)-h:]]
lit, free = np.array(lit), np.array(free)
for frac in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4):
    h = {L: int(np.floor(frac*n)) for L, n in hist.items()}
    r = [match_exact(h, T["len"], p) for p in (lit, free)]
    print(f"x{frac}: short literal {r[0][2]:,} free {r[1][2]:,}")
    if all(x[0] is not None for x in r):
        print("feasible at", round(frac,4), sum(h.values()), "rows each"); break
for nm, (a, s, _) in zip(("C1 literal", "C1 free"), r):
    a = np.array(a)
    report(f"{nm} x{frac:.3f}", "ft_topk", a, f"lexhit={np.nanmean(LT[a]):.3f} frEn_sd={np.std(T['len_fr'][a]/T['len'][a]):.3f} frEn_mean={np.mean(T['len_fr'][a]/T['len'][a]):.3f} overlap_div={len(np.intersect1d(a,div)):,}")
    np.save(f"{nm.replace(' ','_')}.npy", a)
la, fa = np.load("C1_literal.npy"), np.load("C1_free.npy")
print("C1 maxrank AUC literal>free", round(auc(T["maxrank"][la], T["maxrank"][fa]),3), " qe AUC", round(auc(T["qe"][la], T["qe"][fa]),3))
