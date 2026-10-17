from arms import *
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy"); rnd = np.where(Q["in_ft_un_random"])[0]; rep = np.where(T["in_ft_un_rep"])[0]
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
hist = hist_of(T["len"], div)
report("ft_un_div (E0.5)", "ft_topk", div)
report("ft_un_rep (E0.5)", "ft_topk", rep)
report("ft_un_random (E0.5)", "ft_random", rnd)
report("R0 = rand UN rep=0", "ft_random", R0)
# --- A1: R0 pool restricted to div's vocabulary support
td = tok("ft_topk", "en"); V = {w for i in div for w in td[i]}
tr = tok("ft_random", "en")
inV = np.array([all(w in V for w in tr[i]) for i in range(len(tr))])
A1, s, sh = match_exact(hist, Q["len"], RU0[inV[RU0]])
print("A1 pool (RU0 rows inside div vocab):", int(inV[RU0].sum()), "slack", round(s,3), "short", sh)
if A1: report("A1 rand UN rep=0 in V_div", "ft_random", A1)
# --- A2: maxrank threshold K
for K in (6000, 8000, 10000, 12000, 15000, 20000):
    el = RU0[Q["maxrank"][RU0] < K]
    a, s, sh = match_exact(hist, Q["len"], el)
    if a is None:
        print(f"A2 K={K}: pool {len(el):,} slack {s:.3f} SHORT {sh:,}")
    else:
        report(f"A2 rand rep=0 maxrank<{K}", "ft_random", a, f"slack={s:.3f}")
