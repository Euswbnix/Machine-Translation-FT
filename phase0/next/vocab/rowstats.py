"""Per-row statistics for every row of ft_topk and ft_random (E0.5 parents), saved to rows.npz.
Tokenisation, rep and the E0.5 set membership are reproduced exactly from phase0/e04_split.py / e05_build.py."""
import collections, hashlib, math, os, pickle, re, unicodedata, sys
import numpy as np
WORD = re.compile(r"\w+", re.UNICODE)
WS = re.compile(r"\s+")
S = os.path.expanduser("~/mt_local/phase0_final/e05_src")
D = os.path.expanduser("~/mt_local/phase0_final/e05_results/data/phase0_e05")
def read_lines(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]
def toks(s): return [w.lower() for w in WORD.findall(s)]
def gk(g): return hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
def rep_within(T):
    grams = [[gk(" ".join(t[j:j+8])) for j in range(len(t)-7)] for t in T]
    c = collections.Counter()
    for g in grams: c.update(g)
    return np.array([sum(1 for k in g if c[k] > 1)/len(g) if g else 0.0 for g in grams])
def norm(s): return WS.sub(" ", unicodedata.normalize("NFKC", s).casefold()).strip()
def nh(s): return hashlib.blake2b(norm(s).encode(), digest_size=8).digest()

pre_en = pickle.load(open("pre_en.pkl", "rb")); pre_fr = pickle.load(open("pre_fr.pkl", "rb"))
nc = pickle.load(open("nc_en.pkl", "rb"))["counts"]
CE, CF = pre_en["counts"], pre_fr["counts"]
HE, HF = pre_en["line_hashes"], pre_fr["line_hashes"]
NE, NF, NN = sum(CE.values()), sum(CF.values()), sum(nc.values())
rank_en = {w: r for r, (w, _) in enumerate(CE.most_common())}
rank_fr = {w: r for r, (w, _) in enumerate(CF.most_common())}
top100 = {w for w, _ in CE.most_common(100)}
top100f = {w for w, _ in CF.most_common(100)}
qe = np.load("qe.npz")
out = {}
for name in ("ft_topk", "ft_random"):
    en, fr, lab = (read_lines(f"{S}/{name}.{e}") for e in ("en", "fr", "source"))
    TE = [toks(x) for x in en]; TF = [toks(x) for x in fr]
    rep = rep_within(TE); rep_fr = rep_within(TF)
    n = len(en)
    a = {k: np.zeros(n) for k in ("len","len_fr","mlf","minlf","rare30k","rare60k","oov","num","stop","newsy","mlf_fr","minlf_fr","rare30k_fr","stop_fr","copy","in_pre_src","in_pre_tgt","list_end","starts_ing")}
    for i in range(n):
        t, u = TE[i], TF[i]
        a["len"][i] = len(t); a["len_fr"][i] = len(u)
        if t:
            lf = [math.log10(CE.get(w, 0)+1) for w in t]
            a["mlf"][i] = sum(lf)/len(t); a["minlf"][i] = min(lf)
            a["rare30k"][i] = sum(1 for w in t if rank_en.get(w, 10**9) >= 30000)/len(t)
            a["rare60k"][i] = sum(1 for w in t if rank_en.get(w, 10**9) >= 60000)/len(t)
            a["oov"][i] = sum(1 for w in t if w not in CE)/len(t)
            a["num"][i] = sum(1 for w in t if w.isdigit())/len(t)
            a["stop"][i] = sum(1 for w in t if w in top100)/len(t)
            cw = [w for w in t if w not in top100 and not w.isdigit()]
            a["newsy"][i] = (sum(math.log((nc.get(w,0)+1)/NN) - math.log((CE.get(w,0)+1)/NE) for w in cw)/len(cw)) if cw else 0.0
        if u:
            lf = [math.log10(CF.get(w, 0)+1) for w in u]
            a["mlf_fr"][i] = sum(lf)/len(u); a["minlf_fr"][i] = min(lf)
            a["rare30k_fr"][i] = sum(1 for w in u if rank_fr.get(w, 10**9) >= 30000)/len(u)
            a["stop_fr"][i] = sum(1 for w in u if w in top100f)/len(u)
            st = set(t)
            cand = [w for w in u if len(w) > 3 and not w.isdigit()]
            a["copy"][i] = (sum(1 for w in cand if w in st)/len(cand)) if cand else 0.0
        a["in_pre_src"][i] = nh(en[i]) in HE
        a["in_pre_tgt"][i] = nh(fr[i]) in HF
        s = en[i].rstrip()
        a["list_end"][i] = s.endswith((";", ",", ":"))
        a["starts_ing"][i] = bool(t) and t[0].endswith("ing") and en[i].lstrip()[:1].isupper()
    a["rep"] = rep; a["rep_fr"] = rep_fr
    a["un"] = np.array([l == "un" for l in lab])
    a["qe"] = qe[name]
    # E0.5 membership by exact (en, fr) pair (pairs are unique within each parent)
    idx = {(x, y): i for i, (x, y) in enumerate(zip(en, fr))}
    assert len(idx) == n
    sets = ("ft_un_div", "ft_un_rep", "ft_topk_div", "ft_topk_rep") if name == "ft_topk" else ("ft_un_random", "ft_random_small")
    for s_ in sets:
        m = np.zeros(n, bool)
        for x, y in zip(read_lines(f"{D}/{s_}.en"), read_lines(f"{D}/{s_}.fr")):
            m[idx[(x, y)]] = True
        a["in_" + s_] = m
        print(name, s_, int(m.sum()), flush=True)
    out[name] = a
    print(name, "done", flush=True)
pickle.dump(out, open("rows.pkl", "wb"))
med = np.sort(out["ft_topk"]["rep"])[len(out["ft_topk"]["rep"])//2]
print("median rep", med)
