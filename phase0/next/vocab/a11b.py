from arms import *
def strat_pair(A, a_pool, b_pool, seed=42, cap=None):
    """E0.4's stratified matching: per length bucket k = min(|a_L|, |b_L|) rows from each"""
    rng = random.Random(seed); ba, bb = collections.defaultdict(list), collections.defaultdict(list)
    for i in a_pool: ba[int(A["len"][i])].append(int(i))
    for i in b_pool: bb[int(A["len"][i])].append(int(i))
    ka, kb = [], []
    for L in sorted(set(ba) | set(bb)):
        k = min(len(ba[L]), len(bb[L]))
        if k: ka += rng.sample(ba[L], k); kb += rng.sample(bb[L], k)
    return np.array(sorted(ka)), np.array(sorted(kb))
div = np.where(T["in_ft_un_div"])[0]
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
LT = np.load("lexhit_top.npy"); LQ = np.load("lexhit_rnd.npy")
med = np.nanmedian(LT[TD])
lit_pool = TD[np.nan_to_num(LT[TD], nan=med) > med]; free_pool = TD[np.nan_to_num(LT[TD], nan=med) <= med]
a, b = strat_pair(T, lit_pool, free_pool)
for nm, x in (("C1 top literal", a), ("C1 top free", b)):
    report(nm, "ft_topk", x, f"lexhit={np.nanmean(LT[x]):.3f} lenmean={T['len'][x].mean():.2f}")
print("C1 maxrank AUC literal>free:", round(auc(T["maxrank"][a], T["maxrank"][b]), 3), " fr/en ratio sd lit/free:",
      round(np.std(T["len_fr"][a]/T["len"][a]),3), round(np.std(T["len_fr"][b]/T["len"][b]),3))
# C2: QE bands in the narrow random pool
tr, tf = tok("ft_random", "en"), tok("ft_random", "fr")
pe, pf = collections.Counter(), collections.Counter()
for i in RU0: pe.update(tr[i]); pf.update(tf[i])
Ve = {w for w, _ in pe.most_common(40000)}; Vf = {w for w, _ in pf.most_common(60000)}
NP = RU0[np.array([all(w in Ve for w in tr[i]) and all(w in Vf for w in tf[i]) for i in RU0])]
print("narrow random pool", len(NP), "QE quantiles .1 .25 .5 .75 .9:", np.round(np.quantile(Q["qe"][NP], [.1,.25,.5,.75,.9]), 4))
for lo_band, hi_band in (((0.0, 0.8781), (0.8781, 1.0)), ((0.86, 0.878), (0.885, 0.896))):
    lo = NP[(Q["qe"][NP] >= lo_band[0]) & (Q["qe"][NP] < lo_band[1])]; hi = NP[(Q["qe"][NP] >= hi_band[0]) & (Q["qe"][NP] < hi_band[1])]
    x, y = strat_pair(Q, lo, hi)
    print(f"C2 lo{lo_band} pool {len(lo):,}  hi{hi_band} pool {len(hi):,} -> {len(x):,} rows each")
    for nm, z in (("C2 lowQE", x), ("C2 highQE", y)):
        report(nm, "ft_random", z, f"lexhit={np.nanmean(LQ[z]):.3f} lenmean={Q['len'][z].mean():.2f}")
