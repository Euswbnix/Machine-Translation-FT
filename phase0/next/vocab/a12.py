from arms import *
from a8_lib import nn_match
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy"); B_nn = np.load("B_nn.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
hist = hist_of(T["len"], div)
tr, tf = tok("ft_random", "en"), tok("ft_random", "fr")
pe, pf = collections.Counter(), collections.Counter()
for i in RU0: pe.update(tr[i]); pf.update(tf[i])
oe = [w for w, _ in pe.most_common()]; of = [w for w, _ in pf.most_common()]
def capped(Ke, Kf):
    Ve = set(oe[:Ke]) if Ke else None; Vf = set(of[:Kf]) if Kf else None
    return RU0[np.array([(Ve is None or all(w in Ve for w in tr[i])) and (Vf is None or all(w in Vf for w in tf[i])) for i in RU0])]
def dup(par, idx):
    t = text(par, "en"); return 1 - len({t[i] for i in idx})/len(idx)
def maxrep(par, idx):
    tt = tok(par, "en"); g = collections.Counter()
    for i in idx:
        t = tt[i]; g.update({" ".join(t[j:j+8]) for j in range(len(t)-7)})
    return sum(1 for v in g.values() if v > 1)
print("dup/rep8 check: div", dup("ft_topk", div), maxrep("ft_topk", div), " R0", dup("ft_random", R0), maxrep("ft_random", R0))
# narrow random arm: tune caps to div (33,874 src types, 43,021 fr types@6.5M); seeds for robustness
for Ke, Kf in ((41000, 60000), (42000, 60000), (42000, 58000)):
    pool = capped(Ke, Kf)
    for seed in (42, 1, 2):
        a = nn_match(T, div, Q, pool, seed=seed)
        report(f"A_narrow V{Ke//1000}k/F{Kf//1000}k s{seed}", "ft_random", a, f"pool={len(pool):,} dup={dup('ft_random',a):.4f} rep8={maxrep('ft_random',a)} overlapR0={len(np.intersect1d(a,R0)):,}")
        if seed == 42: np.save(f"A_narrow_{Ke//1000}_{Kf//1000}.npy", a)
# random-wide partner for B_nn (41,112 src / 49,277 fr types@6.5M), rarity matched to B_nn
for Ke, Kf in ((60000, 60000), (70000, 62000), (None, 62000), (None, 65000)):
    pool = capped(Ke, Kf)
    try:
        a = nn_match(T, B_nn, Q, pool, seed=42)
    except AssertionError as e:
        print("infeasible", Ke, Kf, e); continue
    report(f"A_wide V{Ke}/F{Kf} ~B_nn rarity", "ft_random", a, f"maxrankAUC(vs B_nn)={auc(Q['maxrank'][a], T['maxrank'][B_nn]):.3f} dup={dup('ft_random',a):.4f}")
    np.save(f"A_wide_{Ke}_{Kf}.npy", a)
report("B_nn (top, wide)", "ft_topk", B_nn, f"dup={dup('ft_topk',B_nn):.4f} rep8={maxrep('ft_topk',B_nn)}")
# C1 at div's exact histogram
LT = np.load("lexhit_top.npy"); med = np.nanmedian(LT[TD])
lit = TD[np.nan_to_num(LT[TD], nan=med) > med]; free = TD[np.nan_to_num(LT[TD], nan=med) <= med]
for frac in (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5):
    h = {L: int(round(frac*n)) for L, n in hist.items()}
    r = [match_exact(h, T["len"], p) for p in (lit, free)]
    if all(x[0] is not None for x in r): break
    print(f"C1 at div hist x{frac}: short literal {r[0][2]:,} free {r[1][2]:,}")
print("C1 feasible at x", frac, sum(h.values()), "rows each")
for nm, (a, s, _) in zip(("C1 literal", "C1 free"), r):
    report(nm + f" x{frac}", "ft_topk", a, f"lexhit={np.nanmean(LT[a]):.3f} overlap_div={len(np.intersect1d(a,div)):,}")
