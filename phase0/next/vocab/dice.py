"""Dice translation lexicon (content words, pretraining ranks 100..20000) from 1.5M random pretraining pairs."""
import pickle, random, re, numpy as np, scipy.sparse as sp, os
WORD = re.compile(r"\w+", re.UNICODE)
def toks(s): return [w.lower() for w in WORD.findall(s)]
CE = pickle.load(open("pre_en.pkl","rb"))["counts"]; CF = pickle.load(open("pre_fr.pkl","rb"))["counts"]
ve = [w for w, _ in CE.most_common(20000)[100:] if not w.isdigit()]
vf = [w for w, _ in CF.most_common(20000)[100:] if not w.isdigit()]
ie = {w: i for i, w in enumerate(ve)}; jf = {w: i for i, w in enumerate(vf)}
P = os.path.expanduser("~/mt_local/rebuild/v1_fixed/train.clean")
rng = random.Random(7); N = 9312233; keep = set(rng.sample(range(N), 1500000))
re_, rf_ = [], []
with open(P+".en", encoding="utf-8", newline="\n") as fe, open(P+".fr", encoding="utf-8", newline="\n") as ff:
    for k, (a, b) in enumerate(zip(fe, ff)):
        if k in keep:
            re_.append(sorted({ie[w] for w in toks(a) if w in ie})); rf_.append(sorted({jf[w] for w in toks(b) if w in jf}))
def csr(rows, n):
    ind = np.fromiter((j for r in rows for j in r), dtype=np.int32)
    ptr = np.cumsum([0] + [len(r) for r in rows])
    return sp.csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(rows), n))
X, Y = csr(re_, len(ve)), csr(rf_, len(vf))
C = (X.T @ Y).tocsr()
ce = np.asarray(X.sum(0)).ravel(); cf = np.asarray(Y.sum(0)).ravel()
lex = {}
for i in range(len(ve)):
    s, e = C.indptr[i], C.indptr[i+1]
    if s == e: continue
    js, cs = C.indices[s:e], C.data[s:e]
    d = 2*cs/(ce[i] + cf[js])
    o = np.argsort(-d)[:3]
    tr = [(vf[js[k]], float(d[k])) for k in o if d[k] >= 0.15]
    if tr: lex[ve[i]] = tr
pickle.dump(lex, open("lex.pkl","wb"))
print("entries", len(lex), "sample", [(w, lex[w]) for w in ("poverty","growth","paragraph","shall","vehicle","killed","government") if w in lex])
