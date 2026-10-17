from common import *
div = np.where(T["in_ft_un_div"])[0]; rep = np.where(T["in_ft_un_rep"])[0]; rnd = np.where(Q["in_ft_un_random"])[0]
print("sanity types div/rep/rnd:", types("ft_topk","en",div), types("ft_topk","en",rep), types("ft_random","en",rnd))
# pools
TD_all = np.where(T["un"] & (T["rep"] == 0))[0]
RU = np.where(Q["un"])[0]; RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]
print("pool sizes: top UN rep=0", len(TD_all), " random UN", len(RU), " random UN rep=0", len(RU0))
print("random UN rows also in top band (exact pair, QE>=0.89594):", int((Q["un"] & (Q["qe"] >= 0.895936)).sum()))
hist = hist_of(T["len"], div)
R0, slack, short = match_exact(hist, Q["len"], RU0)
print("R0 (random UN rep=0, exact histogram):", len(R0) if R0 is not None else None, "slack", round(slack,3), "short", short)
np.save("R0.npy", np.array(R0))
print("types R0", types("ft_random","en",R0))
keys = ["mlf","minlf","rare30k","rare60k","oov","num","stop","newsy","qe","in_pre_src","in_pre_tgt","mlf_fr","minlf_fr","rare30k_fr","stop_fr","copy","list_end","starts_ing","rep_fr"]
print(f"{'stat':<12}{'un_div':>10}{'un_rep':>10}{'un_rand':>10}{'R0':>10}  AUC(div>rand)  AUC(div>R0)")
for k in keys:
    d, r_, q, z = T[k][div], T[k][rep], Q[k][rnd], Q[k][R0]
    print(f"{k:<12}{d.mean():>10.4f}{r_.mean():>10.4f}{q.mean():>10.4f}{z.mean():>10.4f}  {auc(d,q):>8.3f}      {auc(d,z):>8.3f}")
# length ratio
for nm, P, I in (("div",T,div),("rep",T,rep),("rand",Q,rnd),("R0",Q,R0)):
    lr = P["len_fr"][I]/np.maximum(P["len"][I],1)
    print(nm, "fr/en token ratio mean %.4f sd %.4f  fr tokens %d" % (lr.mean(), lr.std(), P["len_fr"][I].sum()))
