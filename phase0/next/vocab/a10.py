from arms import *
from a8_lib import nn_match
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]
hist = hist_of(T["len"], div)
tr, tf = tok("ft_random", "en"), tok("ft_random", "fr")
pe, pf = collections.Counter(), collections.Counter()
for i in RU0: pe.update(tr[i]); pf.update(tf[i])
oe = [w for w, _ in pe.most_common()]; of = [w for w, _ in pf.most_common()]
Ve = set(oe[:40000])
inE = np.array([all(w in Ve for w in tr[i]) for i in RU0])
for Kf in (40000, 45000, 50000, 60000):
    Vf = set(of[:Kf])
    inF = np.array([all(w in Vf for w in tf[i]) for i in RU0])
    el = RU0[inE & inF]
    try:
        b = nn_match(T, div, Q, el)
    except AssertionError as e:
        print(f"Kf={Kf}: pool {len(el):,} nn infeasible {e}"); continue
    report(f"A_V40k_F{Kf//1000}k+nn", "ft_random", b, f"pool={len(el):,} maxrankAUC(A>div)={auc(Q['maxrank'][b], T['maxrank'][div]):.3f}")
    np.save(f"A_V40k_F{Kf//1000}k_nn.npy", b)
