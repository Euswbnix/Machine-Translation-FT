from arms import *
import bisect
from a8_lib import nn_match
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]
hist = hist_of(T["len"], div)
tr = tok("ft_random", "en")
pc = collections.Counter()
for i in RU0: pc.update(tr[i])
order = [w for w, _ in pc.most_common()]
print("RU0 pool types", len(order))
for K in (25000, 30000, 33874, 40000, 50000):
    V = set(order[:K])
    el = RU0[np.array([all(w in V for w in tr[i]) for i in RU0])]
    a, s, sh = match_exact(hist, Q["len"], el)
    if a is None:
        print(f"V=top{K} of RU0: pool {len(el):,} slack {s:.3f} SHORT {sh:,}")
        continue
    report(f"A_V{K} random", "ft_random", a, f"slack={s:.3f} maxrankAUC(A>div)={auc(Q['maxrank'][a], T['maxrank'][div]):.3f}")
    try:
        b = nn_match(T, div, Q, el)
        report(f"A_V{K}+nn", "ft_random", b, f"maxrankAUC(A>div)={auc(Q['maxrank'][b], T['maxrank'][div]):.3f} qeAUC(A>R0)={auc(Q['qe'][b], Q['qe'][R0]):.3f}")
        np.save(f"A_V{K}_nn.npy", b)
    except AssertionError as e:
        print("  nn infeasible", e)
