from common import *
nc = pickle.load(open("nc_en.pkl","rb"))["counts"]
D = os.path.expanduser("~/mt_local/phase0_final/e05_results/data/phase0_e05")
ho = collections.Counter(w for s in read_lines(f"{D}/heldout_un.en") for w in toks(s))
div = np.where(T["in_ft_un_div"])[0]; rep = np.where(T["in_ft_un_rep"])[0]; R0 = np.load("R0.npy"); rnd = np.where(Q["in_ft_un_random"])[0]
def counter(name, idx, side="en"):
    t = tok(name, side); c = collections.Counter()
    for i in idx: c.update(t[i])
    return c
sets = {"un_div": counter("ft_topk", div), "un_rep": counter("ft_topk", rep), "un_random": counter("ft_random", rnd), "R0": counter("ft_random", R0)}
NCt, HOt = sum(nc.values()), sum(ho.values())
print("token coverage of a reference text by the set's source vocabulary (type seen >=k times in the set)")
print(f"{'set':<10}{'NC k>=1':>9}{'NC k>=5':>9}{'NC k>=20':>9}{'HO k>=1':>9}{'HO k>=5':>9}{'HO k>=20':>9}   NC types>=1")
for nm, c in sets.items():
    row = []
    for ref, tot in ((nc, NCt), (ho, HOt)):
        for k in (1, 5, 20):
            row.append(100*sum(n for w, n in ref.items() if c[w] >= k)/tot)
    print(f"{nm:<10}" + "".join(f"{x:>9.3f}" for x in row) + f"   {sum(1 for w in nc if c[w] >= 1)}")
