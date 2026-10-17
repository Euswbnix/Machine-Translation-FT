import collections, os, pickle, random, re
import numpy as np
from scipy.stats import rankdata
WORD = re.compile(r"\w+", re.UNICODE)
S = os.path.expanduser("~/mt_local/phase0_final/e05_src")
def read_lines(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln.rstrip("\n") for ln in f]
def toks(s): return [w.lower() for w in WORD.findall(s)]
R = pickle.load(open("rows.pkl", "rb"))
T, Q = R["ft_topk"], R["ft_random"]
_text = {}
def text(name, side):
    k = (name, side)
    if k not in _text:
        _text[k] = read_lines(f"{S}/{name}.{side}")
    return _text[k]
_tok = {}
def tok(name, side):
    k = (name, side)
    if k not in _tok:
        _tok[k] = [toks(x) for x in text(name, side)]
    return _tok[k]
def auc(x, y):
    """P(X > Y) + 0.5 P(X = Y)"""
    r = rankdata(np.concatenate([x, y]))
    return (r[:len(x)].sum() - len(x)*(len(x)+1)/2) / (len(x)*len(y))
def types(name, side, idx):
    t = tok(name, side)
    return len({w for i in idx for w in t[i]}), sum(len(t[i]) for i in idx)
def types_at(name, side, idx, N, seed=0):
    """distinct words in the first N tokens of the rows in a seeded random order"""
    t = tok(name, side); rng = random.Random(seed); idx = list(idx); rng.shuffle(idx)
    seen = set(); n = 0
    for i in idx:
        for w in t[i]:
            if n >= N: return len(seen)
            seen.add(w); n += 1
    return None
def hist_of(lens, idx): return collections.Counter(int(lens[i]) for i in idx)
def match_exact(hist, lens, eligible, seed=42, frac=1.0):
    """exactly round(frac*hist[L]) rows per length L from eligible; returns (idx, slack, shortfall)"""
    b = collections.defaultdict(list)
    for i in eligible: b[int(lens[i])].append(i)
    need = {L: int(round(frac*n)) for L, n in hist.items()}
    short = sum(max(0, n - len(b[L])) for L, n in need.items())
    slack = min(len(b[L])/n for L, n in need.items() if n)
    if short: return None, slack, short
    rng = random.Random(seed); out = []
    for L in sorted(need): out += rng.sample(b[L], need[L])
    return sorted(out), slack, 0
