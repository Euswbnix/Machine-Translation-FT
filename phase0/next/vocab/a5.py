from common import *
from scipy.stats import spearmanr
pre = pickle.load(open("pre_en.pkl","rb"))["counts"]
rank = {w: r for r, (w, _) in enumerate(pre.most_common())}
def maxrank(name):
    t = tok(name, "en")
    return np.array([max((rank.get(w, 10**7) for w in x), default=0) for x in t])
T["maxrank"] = maxrank("ft_topk"); Q["maxrank"] = maxrank("ft_random")
np.save("maxrank_top.npy", T["maxrank"]); np.save("maxrank_rnd.npy", Q["maxrank"])
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
print("QE quantiles R0 pool:", np.round(np.quantile(Q["qe"][RU0], [.05,.25,.5,.75,.9,.95,.99]),4))
print("QE quantiles TD pool:", np.round(np.quantile(T["qe"][TD], [.05,.5,.95]),4))
print("share of RU0 with QE >= 0.8959 (would be in top band):", round((Q["qe"][RU0] >= 0.895936).mean(),4))
for nm, P, I in (("RU0 pool", Q, RU0), ("TD pool", T, TD)):
    for k in ("minlf", "rare30k", "maxrank"):
        print(nm, f"spearman(qe, {k}) = {spearmanr(P['qe'][I], P[k][I])[0]:.3f}", end="  ")
    print()
lit = pickle.load(open("lit.pkl","rb"))
print("maxrank quantiles div:", np.quantile(T["maxrank"][div], [.5,.75,.8,.9,.95,.99]).astype(int))
print("maxrank quantiles R0 :", np.quantile(Q["maxrank"][R0], [.5,.75,.8,.9,.95,.99]).astype(int))
print("AUC maxrank div>R0:", round(auc(T["maxrank"][div], Q["maxrank"][R0]),3))
