from common import *
lex = pickle.load(open("lex.pkl","rb"))
nc = pickle.load(open("nc_en.pkl","rb"))["counts"]; NCt = sum(nc.values())
D = os.path.expanduser("~/mt_local/phase0_final/e05_results/data/phase0_e05")
ho = collections.Counter(w for s in read_lines(f"{D}/heldout_un.en") for w in toks(s)); HOt = sum(ho.values())
T["maxrank"] = np.load("maxrank_top.npy"); Q["maxrank"] = np.load("maxrank_rnd.npy")
P = {"ft_topk": T, "ft_random": Q}
def lexhit(par, idx):
    te, tf = tok(par, "en"), tok(par, "fr"); out = []
    for i in idx:
        fp = {w[:5] for w in tf[i]}; cov = hit = 0
        for w in te[i]:
            trs = lex.get(w)
            if trs:
                cov += 1; hit += any(f[:5] in fp for f, _ in trs)
        if cov: out.append(hit/cov)
    return np.mean(out)
def cover(par, idx, ref, tot, k=5):
    t = tok(par, "en"); c = collections.Counter()
    for i in idx: c.update(t[i])
    return 100*sum(n for w, n in ref.items() if c[w] >= k)/tot
def report(nm, par, idx, extra=""):
    A = P[par]; idx = np.asarray(idx)
    ty, nt = types(par, "en", idx)
    print(f"{nm:<26} n={len(idx):>7,} src_tok={nt:>9,} types={ty:>6,} fr_types@6.5M={types_at(par,'fr',idx,6_500_000) or -1:>6,} "
          f"qe={A['qe'][idx].mean():.4f} lexhit={lexhit(par, idx):.3f} newsy={A['newsy'][idx].mean():+.3f} list_end={100*A['list_end'][idx].mean():.1f}% "
          f"in_pre={100*A['in_pre_src'][idx].mean():.1f}% NCcov5={cover(par,idx,nc,NCt):.2f} HOcov5={cover(par,idx,ho,HOt):.2f} {extra}", flush=True)
