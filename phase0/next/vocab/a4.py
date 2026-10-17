from common import *
lex = pickle.load(open("lex.pkl","rb"))
CF = pickle.load(open("pre_fr.pkl","rb"))["counts"]; top100f = {w for w, _ in CF.most_common(100)}
rev = collections.defaultdict(set)
for e, trs in lex.items():
    for f, _ in trs: rev[f[:5]].add(e)
def lit(et, ft):
    """(share of lexicon-covered en tokens whose translation appears in fr, kendall tau of positions or nan,
        share of fr content tokens explained by some en token)"""
    fpos = {}
    for j, w in enumerate(ft): fpos.setdefault(w[:5], j)
    cov = 0; hit = 0; pairs = []; used = set()
    for i, w in enumerate(et):
        trs = lex.get(w)
        if not trs: continue
        cov += 1
        for f, _ in trs:
            j = fpos.get(f[:5])
            if j is not None:
                hit += 1
                if j not in used: pairs.append((i, j)); used.add(j)
                break
    tau = np.nan
    if len(pairs) >= 4:
        c = d = 0
        for a in range(len(pairs)):
            for b in range(a+1, len(pairs)):
                s = (pairs[a][1] - pairs[b][1])
                c += s > 0 if pairs[a][0] > pairs[b][0] else s < 0
                d += s < 0 if pairs[a][0] > pairs[b][0] else s > 0
        tau = (c - d)/(c + d) if c + d else np.nan
    es = set(et); fc = [w for w in ft if len(w) > 3 and w not in top100f and not w.isdigit()]
    expl = sum(1 for w in fc if (rev.get(w[:5], set()) & es) or w in es)/len(fc) if fc else np.nan
    return (hit/cov if cov else np.nan), tau, expl
div = np.where(T["in_ft_un_div"])[0]; rep = np.where(T["in_ft_un_rep"])[0]; R0 = np.load("R0.npy"); rnd = np.where(Q["in_ft_un_random"])[0]
SETS = (("un_div","ft_topk",div),("un_rep","ft_topk",rep),("un_random","ft_random",rnd),("R0","ft_random",R0))
res = {}
for nm, par, idx in SETS:
    te, tf = tok(par, "en"), tok(par, "fr")
    L = np.array([lit(te[i], tf[i]) for i in idx])
    res[nm] = L
    fr_text = text(par, "fr")
    ft_list = [tf[i] for i in idx]
    grams = [[" ".join(t[j:j+8]) for j in range(len(t)-7)] for t in ft_list]
    gc = collections.Counter(g for gs in grams for g in gs)
    r8 = np.array([sum(1 for g in gs if gc[g] > 1)/len(gs) if gs else 0 for gs in grams])
    dup = 1 - len({fr_text[i] for i in idx})/len(idx)
    print(f"{nm:<10} fr types@6.5M {types_at(par,'fr',idx,6_500_000):>6}  fr types {types(par,'fr',idx)}  fr dup {100*dup:.2f}%  fr rep8 within set: rows>0 {100*(r8>0).mean():.1f}% mean {r8.mean():.4f}"
          f" | lexical hit {np.nanmean(L[:,0]):.4f}  tau {np.nanmean(L[:,1]):.4f} (n {np.isfinite(L[:,1]).sum()})  fr explained {np.nanmean(L[:,2]):.4f}")
from common import auc
for j, k in enumerate(("lexical hit","tau","fr explained")):
    a, b, c = res["un_div"][:, j], res["R0"][:, j], res["un_random"][:, j]
    a, b, c = a[np.isfinite(a)], b[np.isfinite(b)], c[np.isfinite(c)]
    print(f"AUC div>R0 {k}: {auc(a,b):.3f}   div>un_random: {auc(a,c):.3f}")
pickle.dump(res, open("lit.pkl","wb"))
