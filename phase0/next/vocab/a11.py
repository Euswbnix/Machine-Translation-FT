from arms import *
from a8_lib import nn_match
from scipy.stats import spearmanr
def lexhit_rows(par, idx):
    te, tf = tok(par, "en"), tok(par, "fr"); out = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        fp = {w[:5] for w in tf[i]}; cov = hit = 0
        for w in te[i]:
            trs = lex.get(w)
            if trs: cov += 1; hit += any(f[:5] in fp for f, _ in trs)
        if cov: out[k] = hit/cov
    return out
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
hist = hist_of(T["len"], div)
LT = np.full(len(T["len"]), np.nan); LT[TD] = lexhit_rows("ft_topk", TD)
LQ = np.full(len(Q["len"]), np.nan); LQ[RU0] = lexhit_rows("ft_random", RU0)
np.save("lexhit_top.npy", LT); np.save("lexhit_rnd.npy", LQ)
m = np.isfinite(LQ[RU0]); print("RU0 spearman(qe, lexhit) =", round(spearmanr(Q["qe"][RU0][m], LQ[RU0][m])[0], 3))
m = np.isfinite(LT[TD]); print("TD  spearman(qe, lexhit) =", round(spearmanr(T["qe"][TD][m], LT[TD][m])[0], 3))
# C1: literal vs free halves of the top band (UN, rep=0), exact histogram, rarity NN-matched to div
med = np.nanmedian(LT[TD]); print("TD lexhit median", round(med, 4))
lit_pool = TD[np.nan_to_num(LT[TD], nan=med) > med]; free_pool = TD[np.nan_to_num(LT[TD], nan=med) <= med]
for nm, pool in (("literal", lit_pool), ("free", free_pool)):
    a, s, sh = match_exact(hist, T["len"], pool)
    print(f"C1 {nm} pool {len(pool):,}: slack {s:.3f} short {sh:,}")
for frac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
    h = {L: int(round(frac*n)) for L, n in hist.items()}
    ok = all(match_exact(h, T["len"], p)[0] is not None for p in (lit_pool, free_pool))
    if ok:
        print("C1 feasible at frac", frac, "rows", sum(h.values())); break
tdiv = div
for nm, pool in (("literal", lit_pool), ("free", free_pool)):
    a, _, _ = match_exact(h, T["len"], pool)
    report(f"C1 top {nm} x{frac}", "ft_topk", a, f"lexhit_mean={np.nanmean(LT[a]):.3f} maxrankAUC(vs div)={auc(T['maxrank'][a], T['maxrank'][div]):.3f}")
    np.save(f"C1_{nm}.npy", a)
# C2: QE bands inside the narrow random pool (V_en 40k, V_fr 60k)
tr, tf = tok("ft_random", "en"), tok("ft_random", "fr")
pe, pf = collections.Counter(), collections.Counter()
for i in RU0: pe.update(tr[i]); pf.update(tf[i])
Ve = {w for w, _ in pe.most_common(40000)}; Vf = {w for w, _ in pf.most_common(60000)}
NP = RU0[np.array([all(w in Ve for w in tr[i]) and all(w in Vf for w in tf[i]) for i in RU0])]
print("narrow random pool", len(NP), "QE quantiles", np.round(np.quantile(Q["qe"][NP], [.1,.25,.5,.75,.9]), 4))
for lo_band, hi_band in (((0.85, 0.875), (0.885, 0.8959)), ((0.0, 0.878), (0.878, 1.0))):
    lo = NP[(Q["qe"][NP] >= lo_band[0]) & (Q["qe"][NP] < lo_band[1])]; hi = NP[(Q["qe"][NP] >= hi_band[0]) & (Q["qe"][NP] < hi_band[1])]
    for frac in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2):
        h = {L: int(round(frac*n)) for L, n in hist.items()}
        if all(match_exact(h, Q["len"], p)[0] is not None for p in (lo, hi)): break
    print(f"C2 bands lo{lo_band} ({len(lo):,}) hi{hi_band} ({len(hi):,}): feasible frac {frac} rows {sum(h.values()):,}")
    for nm, p in (("lowQE", lo), ("highQE", hi)):
        a, _, _ = match_exact(h, Q["len"], p)
        report(f"C2 {nm} x{frac}", "ft_random", a, f"maxrankAUC(vs div)={auc(Q['maxrank'][a], T['maxrank'][div]):.3f}")
