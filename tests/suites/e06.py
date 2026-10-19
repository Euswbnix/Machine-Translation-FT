"""E0.6: every new set must be exactly what the draft and the build spec say, or nothing at all.

phase0/e06_build.py is the last step before 40-45 GPU runs on a rented box. If a draw quietly
used a different tie rule, seed or pool, if an undecided OPEN DECISION fell back to a guess,
if a measurement-only directory looked trainable, or if a half-published directory looked
finished, the runs would compare something other than what is registered -- and they would
still train and print a number.

Every fixture is generated here from scratch. E0.5's anchors are built by the real
e04_split.py + e05_build.py, so the anchor byte check is tested against E0.5's own code.
Every new set is recomputed independently: R0 and A_narrow with ports of the vocab memo's
scratch functions (common.match_exact, a12 capped, a8_lib nn_match, freq.py's chunked count),
the other sets from the build spec's text, and the build's output must equal them row for row.
Every STOP the build can reach is made to fire at least once, most of them in-process on a
doctored copy of a valid state, so a check that silently stopped checking turns a test red.
--measure runs on a stub SentencePiece (both interpreters) and, where the real package is
installed, on a model trained here.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import types
import unicodedata
from fractions import Fraction

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)
NV = 1500
V = [f"w{i}" for i in range(NV)]
CW = list(itertools.accumulate(1.0 / (i + 1) for i in range(NV)))
_PERM = random.Random(99).sample(range(NV), NV)
# many-to-one, so French ranks differ from English; six characters ("m00123"), so literalness's
# 5-character prefix is neither the whole word nor a 4-character one (groups of 10 share a
# 5-character prefix, groups of 100 a 4-character one)
FRMAP = {w: f"m{_PERM[i] // 2:05d}" for i, w in enumerate(V)}
FV = sorted(set(FRMAP.values()))
SKIP = "--skip-expected-sha256"
OD2C = {}                     # the registered (c)/(c2) definitions, read from the module under test
STUB_SPM = """import json
__version__ = "stub"
class SentencePieceProcessor:
    def Load(self, path):
        with open(path, encoding="utf-8") as f:
            self.v = json.load(f)["pieces"]
        self.ix = {p: i for i, p in enumerate(self.v)}
        return True
    def get_piece_size(self):
        return len(self.v)
    def is_control(self, i):
        return i in (0, 2, 3)
    def is_unknown(self, i):
        return i == 1
    def encode(self, x, out_type=int, **kw):
        one = lambda s: [self.ix.get(w, 1) for w in s.split(" ")]      # edge spaces become unknown pieces
        return [one(s) for s in x] if isinstance(x, list) else one(x)
"""


def toks(s):
    return [w.lower() for w in WORD.findall(s)]


def fr_line(tokens, extra="", rng=None):
    """Word for word through FRMAP; with rng, one token in six is a free translation (literalness < 1)."""
    return " ".join(rng.choice(FV) if rng is not None and t in FRMAP and rng.random() < 1 / 6 else FRMAP.get(t, t)
                    for t in tokens) + extra


def write(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")


def read(p):
    return [ln.rstrip("\n") for ln in open(p, encoding="utf-8", newline="\n")]


# ---------------------------------------------------------------- fixtures
def make_fixture(root, *, r0_short=False, heldout_leak=False, rep_majority=False, seed=7,
                 n_div=2000, n_rep=300, n_tpl=200, tpl_len=8):
    """ft_topk: UN and Europarl rows; diverse rows are random Zipf word sequences (no repeated
    8-gram), repetitive rows carry one of six 9-word templates. Diverse rows thin out with
    length, so the longest buckets are bound by the diverse side (forced rows, as in the real
    data). Exact duplicate sources among the repetitive rows and among short diverse rows, and
    two 8-token diverse rows that differ only in a number (an OD 4(b) duplicate of 8 tokens).
    ft_random: mostly diverse UN rows, some template rows (rep_rand > 0), Common Crawl rows,
    and 40 exact copies of top-band pairs. A pretraining corpus that leaves 40 words unseen."""
    rng = random.Random(seed)
    seen = set()

    def words(L, hi=NV, unique=True):
        while True:
            t = rng.choices(V[:hi], cum_weights=CW[:hi], k=L)
            if not unique or " ".join(t) not in seen:
                seen.add(" ".join(t))
                return t
    tpl_top = [words(tpl_len, unique=False) for _ in range(n_tpl)]
    tpl_rnd = [words(tpl_len, unique=False) for _ in range(n_tpl)]

    def templ(L, tpl):                                 # a template row; rows themselves stay unique
        while True:
            pre = rng.randint(0, L - tpl_len)
            t = words(pre, unique=False) + rng.choice(tpl) + words(L - tpl_len - pre, unique=False)
            if " ".join(t) not in seen:
                seen.add(" ".join(t))
                return t

    top = []                                           # (tokens, label, fr_extra)
    n_div = 100 if rep_majority else n_div
    for _ in range(n_div):
        L = rng.choices(range(3, 31), weights=[max(1, 31 - L) for L in range(3, 31)])[0]
        top.append((words(L), "un", ""))
    reps = [(templ(rng.randint(10, 30), tpl_top), "un", "") for _ in range(n_rep)]
    top += reps
    for t in rng.sample(reps, 15):                     # exact duplicate sources, another translation
        top.append((t[0], "un", " fbis"))
    shorts = [t for t in top[:n_div] if len(t[0]) < 8]
    for t in rng.sample(shorts, 6):
        top.append((t[0], "un", " fbis"))
    base = words(7)
    top.append((base[:3] + ["12"] + base[3:], "un", ""))
    top.append((base[:3] + ["13"] + base[3:], "un", ""))
    for _ in range(80):
        top.append((words(rng.randint(3, 30)), "europarl", ""))
    for _ in range(40):
        top.append((templ(rng.randint(10, 30), tpl_top), "europarl", ""))
    rng.shuffle(top)
    top_rows = [(" ".join(t), fr_line(t, x, rng), lab) for t, lab, x in top]

    rnd = []
    for _ in range(1300):
        L = rng.randint(3, 30)
        while r0_short and L == 12:
            L = rng.randint(3, 30)
        rnd.append((words(L), "un"))
    for _ in range(150):
        rnd.append((templ(rng.randint(10, 30), tpl_rnd), "un"))
    if r0_short:
        for _ in range(30):
            rnd.append((templ(12, tpl_rnd), "un"))
    for _ in range(400):
        rnd.append((words(rng.randint(3, 30)), "commoncrawl"))
    rnd_rows = [(" ".join(t), fr_line(t, "", rng), lab) for t, lab in rnd]
    cand = [r for r in top_rows if r[2] == "un" and 9 <= len(r[0].split()) and not (r0_short and len(r[0].split()) == 12)]
    rnd_rows += [r for r in rng.sample(cand, 40)]
    rng.shuffle(rnd_rows)

    S = root / "src"
    for name, rows in (("ft_topk", top_rows), ("ft_random", rnd_rows)):
        write(S / f"{name}.en", [e for e, _, _ in rows])
        write(S / f"{name}.fr", [f for _, f, _ in rows])
        write(S / f"{name}.source", [l for _, _, l in rows])
    ho = [words(rng.randint(5, 20)) for _ in range(30)]
    ho_en = [" ".join(t) for t in ho]
    ho_fr = [fr_line(t) for t in ho]
    if heldout_leak:
        ho_en += [e for e, _, l in top_rows if l == "un"]
        ho_fr += [f for _, f, l in top_rows if l == "un"]
    write(S / "heldout_un.en", ho_en)
    write(S / "heldout_un.fr", ho_fr)

    pre = [words(rng.randint(3, 25), hi=NV - 100) for _ in range(2500)]
    pre_pairs = [(" ".join(t), fr_line(t, "", rng)) for t in pre]
    pre_pairs += [(e, f) for e, f, _ in rng.sample(top_rows, 300)] + [(e, f) for e, f, _ in rng.sample(rnd_rows, 200)]
    rng.shuffle(pre_pairs)
    P = root / "pretrain"
    write(P / "train.clean.en", [e for e, _ in pre_pairs])
    write(P / "train.clean.fr", [f for _, f in pre_pairs])
    qe = [f"{rng.uniform(0.90, 0.99):.4f}\t{e}\t{f}" for e, f, _ in top_rows]
    qe += [f"{rng.uniform(0.50, 0.99):.4f}\t{e}\t{f}" for e, f, _ in rnd_rows]
    qe += [f"0.1000\t{e}\t{f}" for e, f, _ in top_rows[:20]]        # conflicting duplicate scores: max wins
    write(root / "v2_scored.tsv", qe)
    write(root / "news.en", [" ".join(words(rng.randint(5, 25))) for _ in range(300)])
    return {"root": root, "S": S, "P": P, "E": root / "e05", "qe": root / "v2_scored.tsv", "news": root / "news.en"}


def build_e05(ctx, fx):
    """E0.5's anchors from E0.5's own code: e04_split.py, then e05_build.py."""
    s, e = fx["S"], fx["E"]
    p = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e04_split.py"), str(s), str(e)], capture_output=True, text=True, timeout=300)
    q = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e05_build.py"), "--dir", str(e), "--src-dir", str(s),
                        "--allow-unpinned-src"], capture_output=True, text=True, timeout=300)
    return p.returncode == 0 and q.returncode == 0, (p.stdout + p.stderr)[-300:] + (q.stdout + q.stderr)[-500:]


def run6(ctx, *args, timeout=600, env=None):
    p = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e06_build.py"), *map(str, args)],
                       capture_output=True, text=True, timeout=timeout,
                       env=None if env is None else {**os.environ, **env})
    return p.returncode, p.stdout + p.stderr


def prod(ctx, D, J, *args, **kw):
    """A production build not tied to a dry run: the tests' only kind besides the dry-run ones."""
    return run6(ctx, "--dir", D, "--decisions", J, SKIP, *args, **kw)


def verify6(ctx, D, test=True):
    return run6(ctx, "--verify", "--dir", D, *(["--allow-unpinned-src"] if test else []))


def base_args(fx, *, pinned=False):
    a = ["--src-dir", fx["S"], "--e05-dir", fx["E"], "--pretrain-en", fx["P"] / "train.clean.en"]
    return a if pinned else a + ["--allow-unpinned-src"]


def full_args(fx):
    a = base_args(fx) + ["--pretrain-fr", fx["P"] / "train.clean.fr", "--qe-tsv", fx["qe"], "--news-en", fx["news"]]
    return a + (["--ref-lex", fx["lex"]] if "lex" in fx else [])


def files_of(D):
    return {p.relative_to(D).as_posix(): p.read_bytes() for p in D.rglob("*") if p.is_file()}


def decisions(path, **over):
    d = {"od1_coverage": "a", "od2_rule": "c", "od2_tie": "larger_k", "tolerance": "1/2",
         "od2_grid": {"a": {"Ke": AB_KE, "Kf": AB_KF},
                      "b": {"Ke": AB_KE, "Kf": AB_KF},
                      "c": {"K": 20}, "c2": {"K": 10, "lambda": [0.0, 0.5, 1.0]}},
         "od3_caps": [42000, 58000], "od4_duplicate": "a", "od4_recompute_rep": False, "od5": "b",
         "od6_topband": "keep", "od7_exposure": "accept", "od10_unidata": unicodedata.unidata_version,
         "od12_draws": "one", "od15_narrowness": "type_count", "od18_register_v": True,
         "fr_budget_214489": 900, "fr_budget_198333": 800, "seeds": [42, 1, 2, 3, 4]}
    for k, v in over.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    if "od2c_definition" not in over and d.get("od2_rule") in OD2C:
        d["od2c_definition"] = OD2C[d["od2_rule"]]
    if "od3_caps_for_reduced_pool" not in over and d.get("od6_topband") == "remove":
        d["od3_caps_for_reduced_pool"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------- independent recomputation
def t_rep_within(T):                                    # vocab/rowstats.py
    gk = lambda g: hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
    grams = [[gk(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in T]
    c = collections.Counter()
    for g in grams:
        c.update(g)
    return np.array([sum(1 for k in g if c[k] > 1) / len(g) if g else 0.0 for g in grams])


def t_match_exact(hist, lens, eligible, seed=42):       # vocab/common.py match_exact, frac = 1.0
    b = collections.defaultdict(list)
    for i in eligible:
        b[int(lens[i])].append(i)
    if any(len(b[L]) < n for L, n in hist.items()):
        return None
    rng = random.Random(seed)
    out = []
    for L in sorted(hist):
        out += rng.sample(b[L], hist[L])
    return sorted(int(i) for i in out)


def t_types_at(t, idx, N, seed=0):                      # vocab/common.py types_at
    rng = random.Random(seed)
    idx = list(idx)
    rng.shuffle(idx)
    seen, n = set(), 0
    for i in idx:
        for w in t[i]:
            if n >= N:
                return len(seen)
            seen.add(w)
            n += 1
    return None


def t_nn_match(tlen, tkey, tidx, plen, pkey, pidx, seed=42):     # vocab/a8_lib.py
    import bisect
    rng = random.Random(seed)
    tb, pb = collections.defaultdict(list), collections.defaultdict(list)
    for i in tidx:
        tb[int(tlen[i])].append(float(np.log1p(tkey[i])))
    for i in pidx:
        pb[int(plen[i])].append((float(np.log1p(pkey[i])), rng.random(), i))
    out = []
    for L, vals in tb.items():
        pool = sorted(pb[L])
        if len(pool) < len(vals):
            return None
        keys = [p[0] for p in pool]
        used = [False] * len(pool)
        med = np.median(keys) if keys else 0
        for v in sorted(vals, key=lambda v: -abs(v - med)):
            j = bisect.bisect_left(keys, v)
            lo, hi = j - 1, j
            while lo >= 0 and used[lo]:
                lo -= 1
            while hi < len(pool) and used[hi]:
                hi += 1
            c = [x for x in (lo, hi) if 0 <= x < len(pool)]
            k = min(c, key=lambda x: abs(keys[x] - v))
            used[k] = True
            out.append(pool[k][2])
    return sorted(int(i) for i in out)


class Ref:
    """The fixture's sets, recomputed without e06_build.py."""

    def __init__(self, fx):
        S, E, P = fx["S"], fx["E"], fx["P"]
        self.T = {e: read(S / f"ft_topk.{e}") for e in ("en", "fr", "source")}
        self.Q = {e: read(S / f"ft_random.{e}") for e in ("en", "fr", "source")}
        self.tt, self.tf = [toks(x) for x in self.T["en"]], [toks(x) for x in self.T["fr"]]
        self.rt, self.rf = [toks(x) for x in self.Q["en"]], [toks(x) for x in self.Q["fr"]]
        self.tlen, self.rlen = np.array([len(t) for t in self.tt]), np.array([len(t) for t in self.rt])
        self.trep, self.rrep = t_rep_within(self.tt), t_rep_within(self.rt)
        self.tun = np.array([l == "un" for l in self.T["source"]])
        self.qun = np.array([l == "un" for l in self.Q["source"]])
        idx = {(x, y): i for i, (x, y) in enumerate(zip(self.T["en"], self.T["fr"]))}
        ridx = {(x, y): i for i, (x, y) in enumerate(zip(self.Q["en"], self.Q["fr"]))}
        self.top_pairs = set(idx)
        mem = lambda nm, ix: sorted(ix[p] for p in zip(read(E / f"{nm}.en"), read(E / f"{nm}.fr")))
        self.div, self.rep, self.rnd = mem("ft_un_div", idx), mem("ft_un_rep", idx), mem("ft_un_random", ridx)
        self.hist = collections.Counter(int(self.tlen[i]) for i in self.div)
        tot = collections.Counter()                     # freq.py: per-chunk counters merged in order
        lines = read(P / "train.clean.en")
        for k in range(0, len(lines), 37):
            c = collections.Counter()
            for s in lines[k:k + 37]:
                c.update(w.lower() for w in WORD.findall(s))
            tot.update(c)
        rank = {w: r for r, (w, _) in enumerate(tot.most_common())}
        self.tmr = np.array([max((rank.get(w, 10 ** 7) for w in x), default=0) for x in self.tt])
        self.rmr = np.array([max((rank.get(w, 10 ** 7) for w in x), default=0) for x in self.rt])
        cf = collections.Counter()
        for s in read(P / "train.clean.fr"):
            cf.update(toks(s))
        rank_f = {w: r for r, (w, _) in enumerate(cf.most_common())}
        self.tmr_f = np.array([max((rank_f.get(w, 10 ** 7) for w in x), default=0) for x in self.tf])
        self.RU0 = [int(i) for i in np.where(self.qun & (self.rrep == 0))[0]]
        self.div_cand = [int(i) for i in np.where(self.tun & (self.trep == 0))[0]]
        self.rep_cand = [int(i) for i in np.where(self.tun & (self.trep > 0))[0]]

    def lexicon(self, P):
        """vocab/dice.py without scipy: ranks 100-19,999 of each side (no isdigit words), Random(7)
        sample of the pretraining pairs, Dice in float32, top 3 >= 0.15 (ties: French rank order)."""
        en, frs = read(P / "train.clean.en"), read(P / "train.clean.fr")
        CE, CF = collections.Counter(), collections.Counter()
        for x in en:
            CE.update(toks(x))
        for y in frs:
            CF.update(toks(y))
        ve = [w for w, _ in CE.most_common(20000)[100:] if not w.isdigit()]
        vf = [w for w, _ in CF.most_common(20000)[100:] if not w.isdigit()]
        ie, jf = {w: i for i, w in enumerate(ve)}, {w: j for j, w in enumerate(vf)}
        keep = set(random.Random(7).sample(range(len(en)), min(1_500_000, len(en))))
        C, ce, cf = collections.Counter(), collections.Counter(), collections.Counter()
        for k, (x, y) in enumerate(zip(en, frs)):
            if k in keep:
                E, F = {ie[w] for w in toks(x) if w in ie}, {jf[w] for w in toks(y) if w in jf}
                ce.update(E)
                cf.update(F)
                C.update((i, j) for i in E for j in F)
        by = collections.defaultdict(list)
        for (i, j), c in C.items():
            by[i].append((j, np.float32(2) * np.float32(c) / (np.float32(ce[i]) + np.float32(cf[j]))))
        lex = {}
        for i, lst in by.items():
            lst.sort(key=lambda t: (-t[1], t[0]))
            tr = [(vf[j], float(dv)) for j, dv in lst[:3] if dv >= np.float32(0.15)]
            if tr:
                lex[ve[i]] = tr
        return lex

    @staticmethod
    def lexhit(lex, te, tf, idx):                       # vocab/arms.py lexhit
        out = []
        for i in idx:
            fp = {w[:5] for w in tf[i]}
            cov = hit = 0
            for w in te[i]:
                trs = lex.get(w)
                if trs:
                    cov += 1
                    hit += any(f[:5] in fp for f, _ in trs)
            if cov:
                out.append(hit / cov)
        return float(np.mean(out))

    def r0(self, seed=42, remove=False):
        pool = [i for i in self.RU0 if not remove or (self.Q["en"][i], self.Q["fr"][i]) not in self.top_pairs]
        return t_match_exact(self.hist, self.rlen, pool, seed)

    def capped(self, Ke, Kf, remove=False):           # vocab/a12.py capped()
        pool0 = [i for i in self.RU0 if not remove or (self.Q["en"][i], self.Q["fr"][i]) not in self.top_pairs]
        pe, pf = collections.Counter(), collections.Counter()
        for i in pool0:
            pe.update(self.rt[i])
            pf.update(self.rf[i])
        oe, of = [w for w, _ in pe.most_common()], [w for w, _ in pf.most_common()]
        Ve, Vf = (set(oe[:Ke]) if Ke else None), (set(of[:Kf]) if Kf else None)
        return [i for i in pool0 if (Ve is None or all(w in Ve for w in self.rt[i])) and (Vf is None or all(w in Vf for w in self.rf[i]))]

    def a_narrow(self, Ke, Kf, seed=42, remove=False):  # vocab/a12.py capped() + nn_match
        return t_nn_match(self.tlen, self.tmr, self.div, self.rlen, self.rmr, self.capped(Ke, Kf, remove), seed)

    def strat(self, a, b, seed=42):                     # e05_build.py lines 255-268
        by = {"a": {}, "b": {}}
        for nm, idx in (("a", a), ("b", b)):
            for i in idx:
                by[nm].setdefault(int(self.tlen[i]), []).append(i)
        rng = random.Random(seed)
        ka, kb = [], []
        for L in sorted(set(by["a"]) | set(by["b"])):
            x, y = by["a"].get(L, []), by["b"].get(L, [])
            k = min(len(x), len(y))
            if k:
                ka += rng.sample(x, k)
                kb += rng.sample(y, k)
        return sorted(ka), sorted(kb)

    def dedup(self, key="raw"):
        kf = (lambda i: self.T["en"][i]) if key == "raw" else \
            (lambda i: " ".join(re.sub(r"\d", "0", w) for w in self.tt[i]))

        def first(rows):
            seen, out, rm = set(), [], []
            for i in rows:
                (rm if kf(i) in seen else out).append(i)
                seen.add(kf(i))
            return out, rm
        rk, rrm = first(self.rep_cand)
        dk, drm = first(self.div_cand)
        dd, rd = self.strat(dk, rk)
        return rd, dd, rrm, drm

    def measure(self, idx, B):
        return {"src_types": len({w for i in idx for w in self.tt[i]}), "fr_types": t_types_at(self.tf, sorted(idx), B)}

    def q_grid(self, target, B, K, lams=(None,)):
        """Spec 6.5 (c) and PROPOSAL c2, from the spec's text."""
        h = collections.Counter(int(self.tlen[i]) for i in target)
        rng = random.Random(42)
        u = {i: rng.random() for i in self.div_cand}
        by = collections.defaultdict(list)
        for i in self.div_cand:
            by[int(self.tlen[i])].append(i)
        lg = lambda v: float(np.log1p(np.int64(v)))
        pts = []
        for lam in lams:
            if lam is None:
                key = lambda i: (int(self.tmr[i]), u[i], i)
            else:
                key = lambda i, lam=lam: ((1 - lam) * lg(self.tmr[i]) + lam * lg(self.tmr_f[i]), u[i], i)
            order = {L: sorted(by[L], key=key) for L in h}
            for k in range(K + 1):
                r = random.Random(42)
                out = []
                for L in sorted(h):
                    m = h[L] + (k * (len(by[L]) - h[L])) // K
                    out += r.sample(sorted(order[L][:m]), h[L])
                pts.append({"k": k, "lambda": lam, "idx": sorted(out), **self.measure(out, B)})
        return pts

    def ab_grid(self, target, B, rule, Kes, Kfs):
        """Spec 6.5 (a)/(b): caps over the diverse candidates (D7), a short point is infeasible."""
        h = collections.Counter(int(self.tlen[i]) for i in target)
        pe, pf = collections.Counter(), collections.Counter()
        for i in self.div_cand:
            pe.update(self.tt[i])
            pf.update(self.tf[i])
        oe, of = [w for w, _ in pe.most_common()], [w for w, _ in pf.most_common()]
        pts = []
        for Ke in Kes:
            for Kf in Kfs:
                Ve, Vf = (set(oe[:Ke]) if Ke else None), (set(of[:Kf]) if Kf else None)
                pool = [i for i in self.div_cand if (Ve is None or all(w in Ve for w in self.tt[i]))
                        and (Vf is None or all(w in Vf for w in self.tf[i]))]
                cnt = collections.Counter(int(self.tlen[i]) for i in pool)
                if any(cnt[L] < n for L, n in h.items()):
                    pts.append({"Ke": Ke, "Kf": Kf, "feasible": False})
                    continue
                idx = (t_nn_match(self.tlen, self.tmr, target, self.tlen, self.tmr, pool) if rule == "a"
                       else t_match_exact(h, self.tlen, pool))
                pts.append({"Ke": Ke, "Kf": Kf, "feasible": True, "idx": idx, **self.measure(idx, B)})
        return pts


def pick(pts, targets, measures, tie):
    best = None
    for p in pts:
        if not p.get("feasible", True) or any(p[m] is None for m in measures):
            continue
        dev = max(Fraction(abs(p[m] - targets[m]), targets[m]) for m in measures)
        key = (dev,) + tie(p)
        if best is None or key < best[0]:
            best = (key, p)
    return best


INF = float("inf")
AB_KE = [600, 1100, 1300, 1350, 1400, None]
AB_KF = [600, 750, None]
CAPS_BIND = [1200, 700]
TIE = {"c": lambda p: (-p["k"],), "c2": lambda p: (-p["k"], p["lambda"]),
       "a": lambda p: (-(INF if p["Ke"] is None else p["Ke"]), -(INF if p["Kf"] is None else p["Kf"]))}
TIE["b"] = TIE["a"]


def _binds(D):
    """The registered caps removed pool rows (the cap path was exercised, not only nn_match)."""
    b = json.loads((D / "manifest.json").read_text(encoding="utf-8"))["e06"]["sets"]["ft_un_a_narrow"]["build"]
    return b["capped_pool_rows"] < b["pool_rows"]


def load_mod(ctx, tag):
    spec = importlib.util.spec_from_file_location(f"e06_build_{tag}", ctx.ROOT / "phase0/e06_build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m




def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edit_manifest(src, dst, fn, name="manifest.json"):
    """A copy of the build src at dst, its manifest changed by fn (the manifest is not in its own table)."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    man = json.loads((dst / name).read_text(encoding="utf-8"))
    fn(man)
    (dst / name).write_text(json.dumps(man), encoding="utf-8")
    return dst


# ---------------------------------------------------------------- the suite
def suite(ctx):
    d, check = ctx.d, ctx.check
    src = lambda name: (ctx.ROOT / "phase0" / name).read_text(encoding="utf-8")
    check("e06_build counts tokens with the same \\w+ rule as e04_split and e05_build",
          'WORD = re.compile(r"\\w+", re.UNICODE)' in src("e06_build.py"))
    m = load_mod(ctx, "unit")
    OD2C.clear()
    OD2C.update(m.OD2C_DEFINITION)
    spec5 = importlib.util.spec_from_file_location("e05_for_e06", ctx.ROOT / "phase0/e05_build.py")
    m5 = importlib.util.module_from_spec(spec5)
    spec5.loader.exec_module(m5)
    check("e06's pins of the E0.3 primary-arm sets and heldout_un are e05_build.py's, character for character",
          m.PINNED_SRC == m5.PINNED_SRC and m.PINNED_HELDOUT == m5.PINNED_HELDOUT)
    sums = {}
    for ln in (ctx.ROOT / "phase0/next/SHA256SUMS").read_text(encoding="utf-8").splitlines():
        mm = re.fullmatch(r"([0-9a-f]{64})\s+(\S+)", ln.strip())
        if mm:
            sums[mm.group(2)] = mm.group(1)
    check("the reference pins (R0.npy, A_narrow_42_58.npy, lex.pkl) are phase0/next/SHA256SUMS's values",
          m.PINNED_REFERENCE == {"R0": sums.get("vocab/R0.npy"), "A_narrow": sums.get("vocab/A_narrow_42_58.npy")}
          and m.PINNED_LEX == sums.get("vocab/lex.pkl"), str(m.PINNED_REFERENCE))
    check("every published set name is one e03_collect.py can evaluate (ft_[a-z0-9_]+)",
          all(re.fullmatch(r"ft_[a-z0-9_]+", v) for v in m.SET_FILE.values()), str(m.SET_FILE))
    unit_tests(ctx, m, d)

    # ---- the fixture, E0.5's anchors from E0.5's own code, the independent recomputation ----
    fx = make_fixture(d / "fx")
    ok5, out5 = build_e05(ctx, fx)
    check("fixture: e04_split + e05_build produce E0.5's anchors", ok5, out5)
    if not ok5:
        return
    ref = Ref(fx)
    lex = ref.lexicon(fx["P"])
    fx["lex"] = fx["root"] / "lex.pkl"
    fx["lex"].write_bytes(pickle.dumps(lex))
    order_rep = sorted(ref.rep_cand, key=lambda i: ref.trep[i])       # e04_split.py's order
    _, kr_e04 = ref.strat(ref.div_cand, order_rep)
    check("fixture: rebuilding the UN split in e04_split's rep-sorted order gives a different ft_un_rep than ascending order",
          kr_e04 != ref.rep and len(ref.rep) > 0)
    nb = len({ref.T["en"][i] for i in ref.rep})
    check("fixture: the repetitive UN half holds duplicate sources and the diverse candidates hold short ones",
          nb < len(ref.rep) and len(ref.div_cand) > len({ref.T["en"][i] for i in ref.div_cand}))
    check("fixture: French words are longer than literalness's 5-character prefix, which groups some of them",
          min(len(w) for w in FV) > 5 and len({w[:5] for w in FV}) < len(FV) and len({w[:4] for w in FV}) < len({w[:5] for w in FV}))
    B = int(0.6 * sum(ref.tlen[i] for i in ref.rep))
    rd_ref, dd_ref, _, _ = ref.dedup()
    Bdd = int(0.6 * sum(ref.tlen[i] for i in rd_ref))
    S = types.SimpleNamespace(m=m, fx=fx, ref=ref, lex=lex, B=B, Bdd=Bdd)

    D = d / "build"
    J = decisions(d / "dec/main.json", fr_budget_214489=B, fr_budget_198333=Bdd)
    rc, out = prod(ctx, D, J, *full_args(fx))
    ok = rc == 0
    check("a production build with every decision set succeeds", ok, out[-1500:])
    if not ok:
        return
    S.D, S.J = D, J
    main_build_checks(ctx, S)
    dry_run_checks(ctx, S)
    verify_checks(ctx, S)
    variant_checks(ctx, S)
    refusal_checks(ctx, S)
    stop_checks(ctx, S)
    publication_checks(ctx, S)
    inprocess_checks(ctx, S)
    feasibility_checks(ctx, S)
    measure_checks(ctx, S)
    reproduce_checks(ctx, S)
    usage_checks(ctx, S)


def main_build_checks(ctx, S):
    d, check, m, fx, ref, D, J, B, Bdd = ctx.d, ctx.check, S.m, S.fx, S.ref, S.D, S.J, S.B, S.Bdd
    man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    idx = lambda nm, d_=D: [int(x) for x in np.load(d_ / "index" / f"{nm}.npy")]
    rows = lambda nm, d_=D: (read(d_ / f"{nm}.en"), read(d_ / f"{nm}.fr"))
    S.idx, S.rows = idx, rows

    # ---- anchors ----
    check("the anchors are E0.5's files byte for byte",
          all((D / f).read_bytes() == (fx["E"] / f).read_bytes() for f in m.ANCHOR_FILES))

    # ---- every set: its parent's rows, its own translation, UN only, its histogram ----
    par = {"ft_topk": ref.T, "ft_random": ref.Q}
    lab_ok = pair_ok = True
    for nm, info in man["sets"].items():
        P = par[info["parent"]]
        ix = idx(nm)
        en, frr = rows(nm)
        pair_ok &= en == [P["en"][i] for i in ix] and frr == [P["fr"][i] for i in ix] and ix == sorted(set(ix))
        lab_ok &= all(P["source"][i] == "un" for i in ix)
    check("every set's text is its index file's rows of its parent, each with its own translation, ascending",
          pair_ok)
    check("no row from another source enters any set", lab_ok)
    H = lambda nm: collections.Counter(len(WORD.findall(x)) for x in rows(nm)[0])
    h214 = H("ft_un_div")
    check("the 214,489-row family shares ft_un_div's exact length histogram",
          all(H(nm) == h214 for nm in ("ft_un_rep", "ft_un_random", "ft_un_r0", "ft_un_a_narrow", "ft_un_div_narrow")))
    check("the dd family (rep_dedup, div_dd, div_narrow_dd) shares rep_dedup's exact histogram",
          H("ft_un_rep_dedup") == H("ft_un_div_dd") == H("ft_un_div_narrow_dd") and H("ft_un_rep_dedup") != h214)

    # ---- R0 and A_narrow: ports of the scratch code ----
    r0 = ref.r0()
    check("R0 = common.match_exact(ft_un_div's histogram, RU0, seed 42), row for row",
          r0 is not None and idx("ft_un_r0") == r0, f"{len(r0 or [])} vs {len(idx('ft_un_r0'))}")
    an = ref.a_narrow(42000, 58000)
    S.r0_ref, S.an_ref = r0, an
    check("A_narrow = a8_lib.nn_match(ft_un_div -> capped RU0, seed 42) at caps 42,000 / 58,000, row for row",
          an is not None and idx("ft_un_a_narrow") == an)
    check("R0 and A_narrow have no repeated 8-gram in ft_random and no duplicate source",
          all(ref.rrep[i] == 0 for i in idx("ft_un_r0") + idx("ft_un_a_narrow"))
          and all(len(set(rows(n)[0])) == len(rows(n)[0]) for n in ("ft_un_r0", "ft_un_a_narrow")))

    # ---- rep_dedup / div_dd ----
    rd_ref, dd_ref, rrm, drm = ref.dedup()
    check("rep_dedup / div_dd = first-occurrence dedup of the raw source, then stratified matching, diverse first",
          idx("ft_un_rep_dedup") == rd_ref and idx("ft_un_div_dd") == dd_ref and len(rd_ref) == len(dd_ref))
    dinfo = man["e06"]["sets"]["ft_un_rep_dedup"]["build"]
    check("the dedup removals are counted, and under OD 4(a) every removed diverse row is under 8 tokens",
          dinfo["repetitive_removed"] == len(rrm) > 0 and dinfo["diverse_removed"] == len(drm) > 0
          and dinfo["diverse_removed_with_8_or_more_tokens"] == 0 and all(ref.tlen[i] < 8 for i in drm), str(dinfo))
    check("rep_dedup has no duplicate source and only rep > 0 rows; div_dd only rep = 0 rows",
          len(set(rows("ft_un_rep_dedup")[0])) == len(rd_ref) and all(ref.trep[i] > 0 for i in rd_ref)
          and all(ref.trep[i] == 0 for i in dd_ref))

    # ---- div_narrow, rule (c), and div_narrow_dd ----
    g = man["e06"]["grids"]["ft_un_div_narrow"]
    tgt = ref.measure(ref.rep, B)
    pts = ref.q_grid(ref.rep, B, 20)
    best = pick(pts, tgt, ["src_types"], TIE["c"])
    S.tgt, S.best_c = tgt, best
    ch = man["e06"]["sets"]["ft_un_div_narrow"]["build"]["chosen"]
    check("div_narrow (c): the coverage target is ft_un_rep's own (source types, French types at B)",
          g["targets"]["src_types"] == tgt["src_types"] and g["targets"]["fr_types"] == tgt["fr_types"], str(g["targets"]))
    check("div_narrow (c): every grid point's coverage equals the spec's q-family recomputed independently",
          [(p["src_types"], p["fr_types"]) for p in g["points"]] == [(p["src_types"], p["fr_types"]) for p in pts])
    check("div_narrow (c): the chosen k minimises the deviation, ties to the larger k, and its rows are that point's",
          best is not None and ch["k"] == best[1]["k"] and idx("ft_un_div_narrow") == best[1]["idx"], f"{ch} vs {best and best[1]['k']}")
    check("div_narrow (c): the manifest names the registered (c) definition it was built under (A15)",
          man["e06"]["sets"]["ft_un_div_narrow"]["build"]["definition"] == m.OD2C_DEFINITION["c"]
          and man["e06"]["decisions"]["od2c_definition"] == m.OD2C_DEFINITION["c"])
    check("div_narrow (c): k = K is exactly match_random over all diverse candidates at ft_un_rep's histogram",
          pts[-1]["idx"] == t_match_exact(collections.Counter(int(ref.tlen[i]) for i in ref.rep), ref.tlen, ref.div_cand))
    forced = g["forced"]
    by = collections.Counter(int(ref.tlen[i]) for i in ref.div_cand)
    hr = collections.Counter(int(ref.tlen[i]) for i in ref.rep)
    fl = sorted(L for L in hr if by[L] == hr[L])
    check("the forced buckets (every diverse candidate used by E0.5) are found, counted, and kept by every point",
          forced["lengths"] == fl and forced["rows"] == sum(hr[L] for L in fl) and len(fl) > 0
          and all(set(i for i in ref.div_cand if ref.tlen[i] in fl) <= set(p["idx"]) for p in pts), str(forced))
    # the smallest caps that keep every forced row, from the candidates' own frequencies (D7)
    pe, pf = collections.Counter(), collections.Counter()
    for i in ref.div_cand:
        pe.update(ref.tt[i])
        pf.update(ref.tf[i])
    re_ = {w: r for r, (w, _) in enumerate(pe.most_common())}
    rf_ = {w: r for r, (w, _) in enumerate(pf.most_common())}
    frows = [i for i in ref.div_cand if ref.tlen[i] in fl]
    ke = max(max(re_[w] for w in ref.tt[i]) for i in frows) + 1
    kf = max(max(rf_[w] for w in ref.tf[i]) for i in frows) + 1
    keeps = lambda Ke, Kf: all(all(re_[w] < Ke for w in ref.tt[i]) and (Kf is None or all(rf_[w] < Kf for w in ref.tf[i]))
                               for i in frows)
    check("the forced rows' smallest caps are counted exactly: they keep every forced row, one less on either side does not",
          forced["smallest_caps_keeping_every_forced_row"] == {"Ke": ke, "Kf": kf} and keeps(ke, kf)
          and not keeps(ke - 1, None) and not all(all(rf_[w] < kf - 1 for w in ref.tf[i]) for i in frows), f"{forced} vs {ke}/{kf}")
    tdd = ref.measure(rd_ref, Bdd)
    pdd = ref.q_grid(rd_ref, Bdd, 20)
    bdd = pick(pdd, tdd, ["src_types"], TIE["c"])
    check("div_narrow_dd: the same rule at rep_dedup's histogram, target and budget, from all diverse candidates",
          bdd is not None and idx("ft_un_div_narrow_dd") == bdd[1]["idx"])
    v = man["e06"]["overlaps"]["ft_un_div & ft_un_div_narrow"]["exact_pairs"]
    check("the manifest records |ft_un_div & div_narrow| (OD 18), at least the forced rows",
          v == len(set(ref.div) & set(idx("ft_un_div_narrow"))) and v >= forced["rows"], str(v))

    # ---- the measures "Measured on the sets" ----
    s6 = man["e06"]["sets"]
    want = ("rows", "source_word_types", "french_word_types_at_budget", "parent_repeated_8gram_share",
            "duplicate_source_fraction", "qe", "literalness", "source_in_pretraining", "pair_in_pretraining")
    check("every set records the draft's columns (rows .. pair in pretraining), plus news and heldout_un coverage",
          all(all(k in s for k in want) and isinstance(s["qe"], dict) and isinstance(s["literalness"], float)
              and isinstance(s["news_coverage"], dict) for s in s6.values()), str(list(s6["ft_un_r0"]))[:300])
    check("with every descriptive input given, the manifest says the table is complete",
          man["e06"]["descriptives"] == {"complete": True, "missing": []}, str(man["e06"]["descriptives"]))
    rr = ref.rep
    check("ft_un_rep's measures equal an independent count (types, French types at B, mean rep, duplicates)",
          s6["ft_un_rep"]["source_word_types"] == tgt["src_types"] and s6["ft_un_rep"]["french_word_types_at_budget"] == tgt["fr_types"]
          and abs(s6["ft_un_rep"]["parent_repeated_8gram_share"]["mean"] - float(np.mean(ref.trep[rr]))) < 1e-12
          and abs(s6["ft_un_rep"]["duplicate_source_fraction"] - (1 - nb_of(ref) / len(rr))) < 1e-12)
    pre = {(e, f) for e, f in zip(read(fx["P"] / "train.clean.en"), read(fx["P"] / "train.clean.fr"))}
    rx = idx("ft_un_r0")
    check("exact-pair exposure and the top-band share of R0 equal an independent count",
          abs(s6["ft_un_r0"]["pair_in_pretraining"] - sum((ref.Q["en"][i], ref.Q["fr"][i]) in pre for i in rx) / len(rx)) < 1e-12
          and s6["ft_un_r0"]["top_band_rows_exact_pair"]["rows"] == sum((ref.Q["en"][i], ref.Q["fr"][i]) in ref.top_pairs for i in rx))
    lits = {n: Ref.lexhit(S.lex, ref.rt, ref.rf, idx(n)) if man["sets"][n]["parent"] == "ft_random"
            else Ref.lexhit(S.lex, ref.tt, ref.tf, idx(n)) for n in man["sets"]}
    check("literalness is vocab/arms.py lexhit (5-character prefixes) over the vocab/dice.py lexicon, recomputed independently, set by set",
          all(abs(s6[n]["literalness"] - lits[n]) < 1e-12 for n in lits) and min(lits.values()) < 0.99
          and man["e06"]["lexicon_entries"] == len(S.lex), f"{ {n: (round(s6[n]['literalness'], 4), round(v, 4)) for n, v in lits.items()} }")
    lits4 = {n: lexhit_prefix(S.lex, ref, man["sets"][n]["parent"], idx(n), 4) for n in man["sets"]}
    check("and the 5-character prefix is the one that counts here (4 characters would give other values)",
          any(abs(lits4[n] - lits[n]) > 1e-9 for n in lits))
    lc = man["e06"]["lexicon_vs_lex_pkl"]
    check("--ref-lex: the rebuilt lexicon is compared with lex.pkl entry by entry, identical here, and literalness "
          "is also recorded over lex.pkl itself",
          lc is not None and lc["identical"] is True and lc["entries_lex_pkl"] == len(S.lex)
          and all(abs(s6[n]["literalness_with_lex_pkl"] - lits[n]) < 1e-12 for n in lits), str(lc)[:300])
    qe = {}
    for ln in read(fx["qe"]):
        sc, e_, f_ = ln.split("\t")
        qe[(e_, f_)] = max(qe.get((e_, f_), -1.0), float(sc))
    check("QE mean is the maximum over conflicting duplicate scores, joined on the exact pair",
          abs(s6["ft_un_div"]["qe"]["mean"] - float(np.mean([qe[(ref.T["en"][i], ref.T["fr"][i])] for i in ref.div]))) < 1e-9)

    # ---- the manifest ----
    w = man["written"]
    check("the manifest lists every file once, manifest.json last, with a sha256 for every other file",
          len(w) == len(set(w)) and w[-1] == "manifest.json" and set(man["sha256"]) == set(w) - {"manifest.json"}
          and all(f"{n}.{e}" in w for n in man["sets"] for e in ("en", "fr")) and all(f"index/{n}.npy" in w for n in man["sets"]))
    check("it marks the build complete, names heldout_un, and records the decisions, the inputs and that they were NOT pinned",
          man.get("e06_complete") is True and "heldout_un" in man.get("heldout_sets", {})
          and man["e06"]["decisions"]["od2_rule"] == "c" and man["e06"]["src_pinned"] is False
          and man["e06"]["decisions_sha256"] == hashlib.sha256(J.read_bytes()).hexdigest()
          and "train.clean.en" in man["e06"]["unpinned_inputs"])
    check("it records that the build was not tied to a dry run (--skip-expected-sha256)",
          man["e06"]["expected_sha256"]["matched"] is False and "skip" in man["e06"]["expected_sha256"]["skipped"])
    names = [c["check"] for c in man["e06"]["checks"]]
    need = ("median rep over ft_topk is 0.0", "E0.5's UN split recomputed in ascending order reproduces all six anchor files",
            "every set: ascending, UN only", "no set shares a source or target line with heldout_un",
            "fast source/French type counts equal the literal definitions", "interpreter unidata",
            "pretraining word types below 10**7", "the rebuilt Dice lexicon was compared with lex.pkl")
    check("the manifest holds the grid table, the RNG register and every named check the build passed",
          len(g["points"]) == 21 and len(man["e06"]["rng_register"]) == 11
          and all(any(x.startswith(n) for x in names) for n in need), str(names))
    spec = importlib.util.spec_from_file_location("cf_e06", ctx.ROOT / "phase0/controls_fingerprint.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    fp = cf.fingerprint(D)
    check("every set and index file is inside the controls fingerprint", all(f in fp["files"] for f in w))
    before = files_of(D)
    rc, out = prod(ctx, D, J, *full_args(fx))
    check("building again reproduces every byte, the manifest included (so the fingerprint is stable)",
          rc == 0 and files_of(D) == before, out[-300:])


def nb_of(ref):
    return len({ref.T["en"][i] for i in ref.rep})


def lexhit_prefix(lex, ref, parent, idx, k):
    te, tf = (ref.rt, ref.rf) if parent == "ft_random" else (ref.tt, ref.tf)
    out = []
    for i in idx:
        fp = {w[:k] for w in tf[i]}
        cov = hit = 0
        for w in te[i]:
            trs = lex.get(w)
            if trs:
                cov += 1
                hit += any(f[:k] in fp for f, _ in trs)
        if cov:
            out.append(hit / cov)
    return float(np.mean(out))


def dry_run_checks(ctx, S):
    d, check, fx, D, J = ctx.d, ctx.check, S.fx, S.D, S.J
    DR = d / "dry"
    rc, out = run6(ctx, "--dry-run", "--dir", DR, "--decisions", J, *full_args(fx))
    ok = rc == 0
    check("a dry run with every descriptive input succeeds", ok, out[-800:])
    if not ok:
        return
    S.DR = DR
    man = json.loads((DR / "dry_run_manifest.json").read_text(encoding="utf-8"))
    pm = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    data = lambda t: {k: v for k, v in t.items() if k != "NOT_FOR_TRAINING"}
    check("the dry run writes the sets, NOT_FOR_TRAINING and dry_run_manifest.json last, never a manifest.json",
          not (DR / "manifest.json").exists() and (DR / "NOT_FOR_TRAINING").is_file()
          and man["written"][-1] == "dry_run_manifest.json" and man["for_training"] is False and man["dry_run"] is True
          and man["e06_complete"] is True)
    check("its set, index and heldout files are the production build's, byte for byte (same decisions, same inputs)",
          data(man["sha256"]) == pm["sha256"], str(sorted(set(data(man["sha256"])) ^ set(pm["sha256"]))))
    check("and its table is complete (QE, literalness, pair in pretraining, news coverage, lexicon check)",
          man["e06"]["descriptives"]["complete"] is True and man["e06"]["lexicon_vs_lex_pkl"]["identical"] is True)
    rc, out = verify6(ctx, DR)
    check("--verify refuses a dry run", rc == 1 and "NOT BUILT" in out and "dry-run" in out, out[-200:])
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(d), "PROBE": "zz", "CONDITIONS": "ft_un_r0",
           "DATA": str(DR), "SFT": str(ctx.ROOT), "PY": ctx.PY}
    q = subprocess.run(["bash", str(ctx.ROOT / "phase0/probe_run.sh")], capture_output=True, text=True, timeout=120, env=env)
    check("probe_run.sh refuses to train on a dry run (it has no manifest.json)",
          q.returncode == 1 and "MISSING" in q.stdout and "manifest.json" in q.stdout, (q.stdout + q.stderr)[-300:])
    rc, out = run6(ctx, "--dry-run", "--dir", d / "dry_noqe", "--decisions", J,
                   *[x for x in full_args(fx) if x not in ("--qe-tsv", fx["qe"])])
    check("a dry run without --qe-tsv is a usage error (exit 64): the registered table may have no n.m. cell",
          rc == 64 and "--qe-tsv" in out and not (d / "dry_noqe").exists(), out[-300:])
    Ju = decisions(d / "dec/unidata.json", fr_budget_214489=S.B, fr_budget_198333=S.Bdd, od10_unidata="0.0.0")
    S.Ju = Ju
    rc, out = run6(ctx, "--dry-run", "--dir", d / "dry_uni", "--decisions", Ju, *full_args(fx))
    mu = json.loads((d / "dry_uni/dry_run_manifest.json").read_text(encoding="utf-8")) if rc == 0 else {}
    check("a dry run on other Unicode tables than od10_unidata runs and records it, so the Mac's dry run and the box "
          "read one decisions file (spec A8)",
          rc == 0 and mu["e06"]["unidata"] == {"interpreter": unicodedata.unidata_version, "od10_unidata": "0.0.0", "equal": False}
          and "NOTE: dry run on Unicode" in out, out[-400:])

    # ---- the box build reproduces the dry run (spec A8) ----
    dm = DR / "dry_run_manifest.json"
    rc, out = run6(ctx, "--dir", D, "--decisions", J, "--expect-sha256-from", dm, *full_args(fx))
    pm = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    rec = pm["e06"]["expected_sha256"] if rc == 0 else {}
    check("--expect-sha256-from accepts a build that reproduces the dry run, and records that manifest's own sha256",
          rc == 0 and rec.get("matched") is True and rec.get("dry_run_manifest_sha256") == sha_file(dm)
          and any(sha_file(dm) in c["check"] for c in pm["e06"]["checks"]), out[-300:])
    before = files_of(D)
    alt = d / "dry_doctored.json"
    mm = json.loads(dm.read_text(encoding="utf-8"))
    mm["sha256"]["index/ft_un_a_narrow.npy"] = "0" * 64
    alt.write_text(json.dumps(mm), encoding="utf-8")
    rc, out = run6(ctx, "--dir", D, "--decisions", J, "--expect-sha256-from", alt, *full_args(fx))
    check("and STOPs, leaving the directory as it was, when one file does not (spec A8)",
          rc == 1 and "ft_un_a_narrow.npy" in out and files_of(D) == before, out[-300:])
    rc, out = run6(ctx, "--dir", D, "--decisions", J, "--expect-sha256-from", d / "dry_uni/dry_run_manifest.json",
                   *full_args(fx))
    check("a dry run built from another decisions file is a STOP before anything is built",
          rc == 1 and "registered decisions file" in out and files_of(D) == before, out[-300:])
    rc, out = run6(ctx, "--dir", D, "--decisions", J, "--expect-sha256-from", D / "manifest.json", *full_args(fx))
    check("a production manifest is not a dry run's: STOP", rc == 1 and "not a finished E0.6 dry run" in out, out[-300:])
    rc, out = run6(ctx, "--dir", d / "no_expect", "--decisions", J, *full_args(fx))
    check("a production build that names no dry run (and does not say --skip-expected-sha256) is a usage error, exit 64",
          rc == 64 and "--expect-sha256-from" in out and not (d / "no_expect").exists(), out[-300:])


def verify_checks(ctx, S):
    d, check, D = ctx.d, ctx.check, S.D
    rc, out = verify6(ctx, D)
    check("--verify (test mode) accepts a finished, untouched build", rc == 0 and "intact" in out, out[-200:])
    rc, out = verify6(ctx, D, test=False)
    check("--verify refuses a build from unpinned inputs unless the tests ask (--allow-unpinned-src)",
          rc == 1 and "NOT FOR TRAINING" in out and "pinned" in out, out[-200:])

    def pinned_like(man):
        man["e06"]["src_pinned"], man["e06"]["allow_unpinned_src"] = True, False
    rc, out = verify6(ctx, edit_manifest(D, d / "v_pinned", pinned_like), test=False)
    check("--verify accepts a build from pinned inputs tied to its dry run (so the refusals are exactly those)",
          rc == 0 and "intact" in out, out[-200:])

    def skipped(man):
        pinned_like(man)
        man["e06"]["expected_sha256"] = {"matched": False, "skipped": "--skip-expected-sha256"}
    rc, out = verify6(ctx, edit_manifest(D, d / "v_skip", skipped), test=False)
    check("--verify refuses a build that did not reproduce a registered dry run",
          rc == 1 and "NOT FOR TRAINING" in out and "dry run" in out, out[-200:])

    def elsewhere(man):
        man["e06"]["built_with"]["machine"] = "pdp11"
    rc, out = verify6(ctx, edit_manifest(D, d / "v_else", elsewhere))
    check("--verify refuses a build made on another machine or interpreter (the run of record is built where it trains)",
          rc == 1 and "BUILT ELSEWHERE" in out and "pdp11" in out, out[-200:])

    def drop_one(man):
        man["sha256"].pop("ft_un_r0.fr")
    rc, out = verify6(ctx, edit_manifest(D, d / "v_table", drop_one))
    check("--verify refuses a manifest whose file list and sha256 table disagree",
          rc == 1 and "disagree" in out, out[-200:])
    Dv = d / "v_cache"
    shutil.rmtree(Dv, ignore_errors=True)
    shutil.copytree(D, Dv, copy_function=shutil.copy2)
    c = Dv / "ft_un_div_narrow.en.cached_256.npz"
    c.write_bytes(b"old cache")
    t = (Dv / "manifest.json").stat().st_mtime
    os.utime(c, (t - 3600, t - 3600))
    rc, out = verify6(ctx, Dv)
    check("--verify refuses a trainer cache older than the build (it would train the replaced text)",
          rc == 1 and "STALE CACHE" in out and c.name in out, out[-200:])
    os.utime(c, (t + 3600, t + 3600))
    rc, out = verify6(ctx, Dv)
    check("and accepts one made after the build", rc == 0, out[-200:])
    tam = d / "tampered"
    shutil.rmtree(tam, ignore_errors=True)
    shutil.copytree(D, tam)
    with open(tam / "ft_un_div_narrow.fr", "a", encoding="utf-8") as f:
        f.write("one extra line\n")
    rc, out = verify6(ctx, tam)
    check("--verify refuses a build in which a set file changed afterwards",
          rc == 1 and "CHANGED" in out and "ft_un_div_narrow.fr" in out, out[-200:])
    shutil.rmtree(tam)
    shutil.copytree(D, tam)
    np.save(tam / "index/ft_un_r0.npy", np.array(S.idx("ft_un_r0")[::-1]))
    rc, out = verify6(ctx, tam)
    check("--verify refuses a changed index file", rc == 1 and "index/ft_un_r0.npy" in out, out[-200:])
    shutil.rmtree(tam)
    shutil.copytree(D, tam)
    (tam / "ft_un_r0.en").unlink()
    rc, out = verify6(ctx, tam)
    check("--verify refuses a build with a listed file missing", rc == 1 and "ft_un_r0.en" in out, out[-200:])

    def no_flag(man):
        del man["e06_complete"]
    rc, out = verify6(ctx, edit_manifest(D, tam, no_flag))
    check("--verify refuses a manifest that lists every hash but does not say the build completed",
          rc == 1 and "NOT BUILT" in out and "e06_complete" in out, out[-200:])


def variant_checks(ctx, S):
    """Per seed, top band removed, OD 4(b), c2 under OD 1(b), rules (a) and (b), caps that bind."""
    d, check, m, fx, ref, B, Bdd, idx, tgt = ctx.d, ctx.check, S.m, S.fx, S.ref, S.B, S.Bdd, S.idx, S.tgt
    D2 = d / "build_seed"
    J2 = decisions(d / "dec/seed.json", fr_budget_214489=B, fr_budget_198333=Bdd, od12_draws="per_seed",
                   od6_topband="remove", od4_duplicate="b", od4_norm_rule=m.A17_RULE, od1_coverage="b",
                   od2_rule="c2", od2_tie="larger_k_then_smaller_lambda", tolerance="1")
    S.J2 = J2
    rc, out = prod(ctx, D2, J2, *full_args(fx))
    ok2 = rc == 0
    check("a build with per-seed draws, top-band rows removed, OD 4(b) and rule c2 under OD 1(b) succeeds", ok2, out[-800:])
    if ok2:
        man2 = json.loads((D2 / "manifest.json").read_text(encoding="utf-8"))
        names = set(man2["sets"])
        check("OD 12 per_seed publishes one R0 and one A_narrow per seed, named _d<seed>, and no single draw",
              {f"ft_un_r0_d{s}" for s in (42, 1, 2, 3, 4)} <= names and {f"ft_un_a_narrow_d{s}" for s in (42, 1, 2, 3, 4)} <= names
              and "ft_un_r0" not in names and "ft_un_a_narrow" not in names)
        check("each seed's draw is the scratch rule at that seed (top-band rows removed), and seeds differ",
              all(idx(f"ft_un_r0_d{s}", D2) == ref.r0(s, remove=True) for s in (42, 1, 2, 3, 4))
              and all(idx(f"ft_un_a_narrow_d{s}", D2) == ref.a_narrow(42000, 58000, s, remove=True) for s in (42, 1, 2))
              and idx("ft_un_r0_d1", D2) != idx("ft_un_r0_d42", D2))
        check("OD 6 remove: no R0 or A_narrow row is a top-band pair, and the manifest records the caps as registered for the reduced pool",
              all((ref.Q["en"][i], ref.Q["fr"][i]) not in ref.top_pairs for s in (42, 1) for n in ("ft_un_r0", "ft_un_a_narrow")
                  for i in idx(f"{n}_d{s}", D2)) and man2["e06"]["decisions"]["od3_caps_for_reduced_pool"] is True)
        rd_b, dd_b, _, drm_b = ref.dedup("a17")
        info_b = man2["e06"]["sets"]["ft_un_rep_dedup"]["build"]
        check("OD 4(b) (A17 key) deduplicates on the digit-masked tokens and reports the diverse row of 8 tokens it removes",
              idx("ft_un_rep_dedup", D2) == rd_b and idx("ft_un_div_dd", D2) == dd_b
              and info_b["diverse_removed_with_8_or_more_tokens"] == sum(1 for i in drm_b if ref.tlen[i] >= 8) >= 1, str(info_b))
        pts2 = ref.q_grid(ref.rep, B, 10, lams=(0.0, 0.5, 1.0))
        b2 = pick(pts2, tgt, ["src_types", "fr_types"], TIE["c2"])
        ch2 = man2["e06"]["sets"]["ft_un_div_narrow"]["build"]["chosen"]
        check("c2 under OD 1(b): the blend of English and French rarest-word ranks, the smallest max deviation over "
              "both sides, ties to the larger k then the smaller lambda",
              b2 is not None and (ch2["k"], ch2["lambda"]) == (b2[1]["k"], b2[1]["lambda"]) and idx("ft_un_div_narrow", D2) == b2[1]["idx"],
              f"{ch2} vs {b2 and (b2[1]['k'], b2[1]['lambda'])}")
        check("c2 at lambda = 0 is rule (c)",
              [p["idx"] for p in pts2 if p["lambda"] == 0.0] == [p["idx"] for p in ref.q_grid(ref.rep, B, 10)])
    for rule in ("a", "b"):
        Dr = d / f"build_{rule}"
        Jr = decisions(d / f"dec/{rule}.json", fr_budget_214489=B, fr_budget_198333=Bdd, od2_rule=rule,
                       od2_tie="larger_Ke_then_larger_Kf", tolerance="1", od5="a",
                       od2_grid={rule: {"Ke": AB_KE, "Kf": [None]}})
        rc, out = prod(ctx, Dr, Jr, *base_args(fx))
        if rc != 0:
            check(f"rule ({rule}) builds", False, out[-800:])
            continue
        manr = json.loads((Dr / "manifest.json").read_text(encoding="utf-8"))
        ptsr = ref.ab_grid(ref.rep, B, rule, AB_KE, [None])
        br = pick(ptsr, tgt, ["src_types"], TIE[rule])
        gp = manr["e06"]["grids"]["ft_un_div_narrow"]["points"]
        chr_ = manr["e06"]["sets"]["ft_un_div_narrow"]["build"]["chosen"]
        check(f"rule ({rule}) under OD 1(a), no French cap (A12): caps over the candidates' own frequencies; short points "
              f"infeasible; the chosen Ke minimises the deviation, ties to the larger Ke; its rows are that point's",
              [p["feasible"] for p in gp] == [p["feasible"] for p in ptsr] and not all(p["feasible"] for p in gp)
              and br is not None and (chr_["Ke"], chr_["Kf"]) == (br[1]["Ke"], None)
              and idx("ft_un_div_narrow", Dr) == br[1]["idx"] and "ft_un_div_narrow_dd" not in manr["sets"],
              f"{chr_} vs {br and (br[1]['Ke'], br[1]['Kf'])}; feasible {[p['feasible'] for p in gp]}")
        check(f"rule ({rule}): the manifest records the descriptives this build could not fill (no --qe-tsv, --pretrain-fr, ...)",
              manr["e06"]["descriptives"]["complete"] is False and "qe (--qe-tsv)" in manr["e06"]["descriptives"]["missing"])
    Dab = d / "build_a_od1b"
    Jab = decisions(d / "dec/a_od1b.json", fr_budget_214489=B, fr_budget_198333=Bdd, od2_rule="a", od1_coverage="b",
                    od2_tie="larger_Ke_then_larger_Kf", tolerance="1", od5="a")
    rc, out = prod(ctx, Dab, Jab, *base_args(fx))
    if rc != 0:
        check("rule (a) under OD 1(b) over a French-cap grid builds", False, out[-800:])
    else:
        mab = json.loads((Dab / "manifest.json").read_text(encoding="utf-8"))
        pab = ref.ab_grid(ref.rep, B, "a", AB_KE, AB_KF)
        bab = pick(pab, tgt, ["src_types", "fr_types"], TIE["a"])
        cab = mab["e06"]["sets"]["ft_un_div_narrow"]["build"]["chosen"]
        check("rule (a) under OD 1(b): Ke and Kf both searched, ties to the larger Ke then the larger Kf; its rows are that point's",
              bab is not None and (cab["Ke"], cab["Kf"]) == (bab[1]["Ke"], bab[1]["Kf"]) and idx("ft_un_div_narrow", Dab) == bab[1]["idx"],
              f"{cab} vs {bab and (bab[1]['Ke'], bab[1]['Kf'])}")
    Dk = d / "build_caps"
    Jk = decisions(d / "dec/caps.json", fr_budget_214489=B, fr_budget_198333=Bdd, od3_caps=CAPS_BIND)
    S.Jk = Jk
    rc, out = prod(ctx, Dk, Jk, *base_args(fx))
    ank = ref.a_narrow(*CAPS_BIND)
    S.ank = ank
    check("A_narrow at registered caps that bind is a12's capped pool + nn_match, row for row",
          rc == 0 and ank is not None and idx("ft_un_a_narrow", Dk) == ank and _binds(Dk), out[-500:])


def refusal_checks(ctx, S):
    d, check, m, fx = ctx.d, ctx.check, S.m, S.fx

    def refused(tag, expect, **over):
        Dx = d / f"refused_{tag}"
        rc, out = prod(ctx, Dx, decisions(d / f"dec/r_{tag}.json", **over), *base_args(fx))
        return rc == 2 and "REFUSED" in out and all(e in out for e in expect) and not Dx.exists(), out[-600:]

    for tag, expect, over in (
            ("od6", ["od6_topband: unset", "R0", "A_narrow"], {"od6_topband": None}),
            ("open", ["od2_rule: unset", "div_narrow"], {"od2_rule": "OPEN"}),
            ("dd_budget", ["fr_budget_198333: unset", "rep_dedup"], {"fr_budget_198333": None}),
            ("tolerance", ["tolerance: unset", "div_narrow"], {"tolerance": None}),
            ("od5c_v", ["od5 = 'c'", "od18_register_v"], {"od5": "c"}),
            ("retune", ["retune"], {"od3_caps": "retune"}),
            ("match", ["od7_exposure = 'match'"], {"od7_exposure": "match"}),
            ("news", ["news_coverage"], {"od15_narrowness": "news_coverage"}),
            ("recompute", ["od4_recompute_rep = true"], {"od4_recompute_rep": True}),
            ("od1c", ["od1_coverage = 'c'", "A28"], {"od1_coverage": "c"}),
            ("od1d", ["od1_coverage = 'd'", "A29"], {"od1_coverage": "d"}),
            ("od4b", ["od4_duplicate = 'b'", "A17"], {"od4_duplicate": "b"}),
            ("tie", ["od2_tie"], {"od2_tie": "smaller_k"}),
            ("grid", ["od2_grid has no entry"], {"od2_grid": {"a": {"Ke": [None], "Kf": [None]}}}),
            ("typo", ["od6_top_band: not a decision key"], {"od6_top_band": "keep"}),
            ("remove", ["od6_topband = 'remove'", "od3_caps_for_reduced_pool", "A21"],
             {"od6_topband": "remove", "od3_caps_for_reduced_pool": None}),
            ("keep_reduced", ["contradicts od6_topband = 'keep'"], {"od3_caps_for_reduced_pool": True}),
            ("a12", ["od1_coverage = 'a' with od2_rule = 'a'", "A12"],
             {"od2_rule": "a", "od2_tie": "larger_Ke_then_larger_Kf", "od2_grid": {"a": {"Ke": [None], "Kf": [600, None]}}}),
            ("c_def", ["od2c_definition: unset", "A15"], {"od2c_definition": None}),
            ("c_def_other", ["od2c_definition = 'my own (c)'", "only definition implemented"], {"od2c_definition": "my own (c)"}),
            ("c_def_ab", ["od2c_definition is set", "contradicts itself"],
             {"od2_rule": "b", "od2_tie": "larger_Ke_then_larger_Kf", "od2c_definition": "x", "od1_coverage": "b"})):
        good, detail = refused(tag, expect, **over)
        check(f"REFUSED with exit 2 and nothing written: {', '.join(expect)}", good, detail)
    rc, out = run6(ctx, "--dir", d / "refused_nofile", SKIP, *base_args(fx))
    check("no decisions file at all is REFUSED (every decision unset)",
          rc == 2 and "REFUSED" in out and not (d / "refused_nofile").exists(), out[-300:])


def stopped(Dx, rc, out, *expect):
    left = [p.name for p in Dx.rglob("*")] if Dx.exists() else []
    return rc == 1 and "STOP" in out and all(e in out for e in expect) and "Traceback" not in out and not any(
        x.endswith((".en", ".fr", ".npy", ".tmp", ".json")) for x in left), f"rc={rc} left={left} {out[-500:]}"


def stop_checks(ctx, S):
    d, check, m, fx, ref, J, B, Bdd = ctx.d, ctx.check, S.m, S.fx, S.ref, S.J, S.B, S.Bdd
    Dp = d / "pins"
    rc, out = prod(ctx, Dp, J, *base_args(fx, pinned=True))
    check("inputs whose sha256 differs from the pins are a STOP unless the deviation is asked for, naming each file",
          *stopped(Dp, rc, out, "not the pinned ones", "ft_topk.en", "train.clean.en", "E0.5 ft_un_div.en"))
    rc, out = prod(ctx, Dp, S.Ju, *base_args(fx))
    check("production refuses an interpreter whose Unicode tables differ from od10_unidata",
          *stopped(Dp, rc, out, "od10_unidata", unicodedata.unidata_version))
    Jt = decisions(d / "dec/tight.json", fr_budget_214489=B, fr_budget_198333=Bdd, tolerance="1/1000000")
    rc, out = prod(ctx, Dp, Jt, *base_args(fx))
    if S.best_c[0][0] == 0:
        ctx.skip("a div_narrow outside the tolerance is a STOP", "the fixture hits the target exactly")
    else:
        check("a div_narrow whose best point is outside the tolerance is a STOP", *stopped(Dp, rc, out, "outside the tolerance"))
    Jc = decisions(d / "dec/tightcaps.json", fr_budget_214489=B, fr_budget_198333=Bdd, od3_caps=[20, 20])
    S.Jc = Jc
    rc, out = prod(ctx, Dp, Jc, *base_args(fx))
    check("an A_narrow pool smaller than its targets in any bucket is a STOP",
          *stopped(Dp, rc, out, "smaller than its targets"))
    Jb = decisions(d / "dec/budget.json", fr_budget_214489=10 ** 7, fr_budget_198333=Bdd)
    rc, out = prod(ctx, Dp, Jb, *base_args(fx))
    check("a set with no more French tokens than its budget is a STOP", *stopped(Dp, rc, out, "budget"))
    Jc2 = decisions(d / "dec/c2_nofr.json", fr_budget_214489=B, fr_budget_198333=Bdd, od2_rule="c2",
                    od2_tie="larger_k_then_smaller_lambda")
    rc, out = prod(ctx, Dp, Jc2, *base_args(fx))
    check("rule c2 without --pretrain-fr is a STOP naming the missing construction input (not a crash)",
          *stopped(Dp, rc, out, "--pretrain-fr"))

    fx_s = make_fixture(d / "fx_short", r0_short=True)
    ok_s, o5 = build_e05(ctx, fx_s)
    rc, out = prod(ctx, d / "short", J, *base_args(fx_s))
    check("an R0 pool that cannot fill a bucket is a STOP", ok_s and stopped(d / "short", rc, out, "R0's pool", "short")[0],
          o5 + out[-400:])
    fx_m = make_fixture(d / "fx_median", rep_majority=True)
    shutil.copytree(fx["E"], fx_m["E"])
    rc, out = prod(ctx, d / "median", J, *base_args(fx_m))
    check("a top band whose median rep is not 0.0 is a STOP (the draft's rep = 0 is E0.5's rule only then)",
          *stopped(d / "median", rc, out, "median"))
    fx_c = d / "fx_changed"
    shutil.copytree(fx["root"], fx_c)
    fc = {"root": fx_c, "S": fx_c / "src", "P": fx_c / "pretrain", "E": fx_c / "e05", "qe": fx_c / "v2_scored.tsv", "news": fx_c / "news.en"}
    en = read(fc["S"] / "ft_topk.en")
    j = ref.rep[0]
    en[j] = en[j] + " zz"
    write(fc["S"] / "ft_topk.en", en)
    rc, out = prod(ctx, d / "changed", J, *base_args(fc))
    check("inputs with one row changed do not reproduce E0.5's anchors: a STOP naming the anchor files",
          *stopped(d / "changed", rc, out, "does not reproduce E0.5's anchors", "ft_un_"))
    fx_h = d / "fx_leak"
    fh = make_fixture(fx_h, heldout_leak=True)
    shutil.copytree(fx["E"], fh["E"])
    for f in ("ft_topk", "ft_random"):
        for e in ("en", "fr", "source"):
            shutil.copy(fx["S"] / f"{f}.{e}", fh["S"] / f"{f}.{e}")
    for f in ("train.clean.en", "train.clean.fr"):
        shutil.copy(fx["P"] / f, fh["P"] / f)
    rc, out = prod(ctx, d / "leak", J, *base_args(fh))
    check("a set that shares a line with heldout_un is a STOP", *stopped(d / "leak", rc, out, "heldout_un"))

    # ---- E0.5's manifest is the source of the anchor pins ----
    for tag, fn, expect in (
            ("e05_sha", lambda mm: mm["sha256"].__setitem__("ft_un_div.en", "0" * 64), "no longer has the sha256 its manifest records"),
            ("e05_flag", lambda mm: mm.pop("e05_complete"), "e05_complete")):
        fe = {**fx, "E": d / f"fx_{tag}"}
        edit_manifest(fx["E"], fe["E"], fn)
        rc, out = prod(ctx, d / tag, J, *base_args(fe))
        check(f"an E0.5 manifest that {'records another sha256 than its file has' if tag == 'e05_sha' else 'does not say E0.5 completed'} is a STOP",
              *stopped(d / tag, rc, out, expect))

    # ---- an output directory that is an input directory, or someone else's build: STOP, untouched ----
    E2 = d / "e05_copy"
    shutil.copytree(fx["E"], E2)
    fe = {**fx, "E": E2}
    before = files_of(E2)
    rc, out = prod(ctx, E2, J, *base_args(fe))
    q = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e05_build.py"), "--verify", "--dir", str(E2)],
                       capture_output=True, text=True, timeout=120)
    check("--dir = --e05-dir is a STOP before anything is read: E0.5's pinned manifest and anchors stay as they were",
          rc == 1 and "is --e05-dir" in out and files_of(E2) == before and q.returncode == 0, out[-300:] + q.stdout[-200:])
    rc, out = prod(ctx, fx["S"], J, *base_args(fx))
    check("--dir = --src-dir is a STOP", rc == 1 and "is --src-dir" in out, out[-300:])
    Df = d / "foreign"
    shutil.copytree(fx["E"], Df)
    before = files_of(Df)
    rc, out = prod(ctx, Df, J, *base_args(fx))
    check("a --dir holding another experiment's manifest (E0.5's) is a STOP, and that manifest is not deleted",
          rc == 1 and "not an E0.6 production manifest" in out and files_of(Df) == before, out[-300:])
    Dj = d / "junk_manifest"
    Dj.mkdir()
    (Dj / "manifest.json").write_text("{not json", encoding="utf-8")
    rc, out = prod(ctx, Dj, J, *base_args(fx))
    check("a --dir whose manifest.json is not JSON is a STOP, and the file is left as it was",
          rc == 1 and "cannot be read as JSON" in out and (Dj / "manifest.json").read_text() == "{not json", out[-300:])


def publication_checks(ctx, S):
    d, check, m, fx, D, J = ctx.d, ctx.check, S.m, S.fx, S.D, S.J
    before = files_of(D)
    rc, out = prod(ctx, D, S.Jc, *base_args(fx))
    check("a STOP leaves an existing build exactly as it was, with no temporary file", rc == 1 and files_of(D) == before)
    Dc = d / "crash"
    shutil.copytree(D, Dc)
    crash = d / "crash.py"
    crash.write_text(
        "import importlib.util, json, os, sys\n"
        "spec = importlib.util.spec_from_file_location('e06_crash', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "log, real = [], os.replace\n"
        "def fake(a, b):\n"
        "    n = os.path.relpath(str(b), sys.argv[2]); log.append(n)\n"
        "    if n == 'manifest.json': raise OSError('simulated crash while publishing the manifest')\n"
        "    return real(a, b)\n"
        "os.replace = fake\n"
        "try:\n    rc = m.main(sys.argv[3:])\nexcept OSError:\n    rc = 'crash'\n"
        "print('LOG ' + json.dumps(log)); print('RC', rc)\n", encoding="utf-8")
    J3 = decisions(d / "dec/crash.json", fr_budget_214489=S.B, fr_budget_198333=S.Bdd, od12_draws="per_seed")
    p = subprocess.run([ctx.PY, str(crash), str(ctx.ROOT / "phase0/e06_build.py"), str(Dc), "--dir", str(Dc),
                        "--decisions", str(J3), SKIP, *map(str, base_args(fx))], capture_output=True, text=True, timeout=600)
    log = json.loads(next((ln[4:] for ln in p.stdout.splitlines() if ln.startswith("LOG ")), "[]"))
    check("publication renames every set and index file first and the manifest last",
          "RC crash" in p.stdout and log and log[-1] == "manifest.json" and "ft_un_r0_d1.en" in log
          and "index/ft_un_r0_d1.npy" in log and log.count("manifest.json") == 1, (p.stdout + p.stderr)[-500:])
    rc, out = verify6(ctx, Dc)
    check("a crash before the manifest is renamed leaves no manifest (the old one is gone) and no temporary file: --verify says NOT BUILT",
          not (Dc / "manifest.json").exists() and not any(x.name.endswith(".tmp") for x in Dc.rglob("*"))
          and rc == 1 and "NOT BUILT" in out, out[-200:])

    # ---- trainer caches and the previous build's files ----
    Dk = d / "cache_rebuild"
    shutil.copytree(D, Dk)
    for c in ("ft_un_div_narrow.en.cached_256.npz", "ft_un_r0.en.cached_128.npz"):
        (Dk / c).write_bytes(b"a cache of the old text")
    rc, out = prod(ctx, Dk, S.J2, *full_args(fx))
    check("rebuilding a directory deletes every trainer cache before publishing (the trainer finds them by name only)",
          rc == 0 and not list(Dk.glob("*.cached_*.npz")) and "trainer caches" in out
          and verify6(ctx, Dk)[0] == 0, out[-400:])
    man = json.loads((Dk / "manifest.json").read_text(encoding="utf-8"))
    check("and removes the previous build's files that this one no longer writes (one draw -> per seed)",
          set(files_of(Dk)) == set(man["written"]) and not (Dk / "ft_un_r0.en").exists()
          and not (Dk / "index/ft_un_a_narrow.npy").exists(), str(sorted(set(files_of(Dk)) - set(man["written"]))))
    rc, out = prod(ctx, Dk, J, *full_args(fx))
    man = json.loads((Dk / "manifest.json").read_text(encoding="utf-8"))
    check("and back (per seed -> one draw): no ft_un_r0_d<seed> file is left under a trainable name",
          rc == 0 and set(files_of(Dk)) == set(man["written"]) and not list(Dk.glob("ft_un_r0_d*"))
          and not list((Dk / "index").glob("ft_un_a_narrow_d*"))
          and {k: v for k, v in files_of(Dk).items() if k != "manifest.json"}
          == {k: v for k, v in files_of(D).items() if k != "manifest.json"}, out[-300:])


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def inproc_builder(m, fx, pinned=None):
    a = types.SimpleNamespace(src_dir=fx["S"], e05_dir=fx["E"], pretrain_en=fx["P"] / "train.clean.en",
                              pretrain_fr=None, qe_tsv=None, news_en=None, ref_lex=None, allow_unpinned_src=True)
    b = m.Builder(a, {}, "production")
    b.check_pins(need_fr=False)
    if pinned is not None:
        b.pinned = pinned
    return b


def inprocess_checks(ctx, S):
    """Every per-set STOP, the rebuild check and the pinned-count table, made to fire in-process."""
    d, check, m, fx, ref, B, Bdd = ctx.d, ctx.check, S.m, S.fx, S.ref, S.B, S.Bdd
    b = inproc_builder(m, fx)
    b.prepare(want_hashes=False, need_fr_ranks=False)
    T, R = b.T, b.R
    r0, _ = b.build_r0("keep", 42)
    rep_dd, div_dd, _ = b.build_dedup("a")
    hist_dd = collections.Counter(T.lens[i] for i in rep_dd)
    base = {"ft_un_r0": {"draft": "R0", "parent": "ft_random", "idx": r0, "budget": B},
            "ft_un_rep_dedup": {"draft": "rep_dedup", "parent": "ft_topk", "idx": rep_dd, "budget": Bdd},
            "ft_un_div_dd": {"draft": "div_dd", "parent": "ft_topk", "idx": div_dd, "budget": Bdd}}

    def stop_of(**chg):
        sets = {k: dict(v) for k, v in base.items()}
        for k, v in chg.items():
            sets[k].update(v)
        try:
            b.check_sets(sets, hist_dd)
            return None
        except m.Stop as e:
            return str(e)

    def swap(idx, old, new):
        return sorted([i for i in idx if i != old] + [new])

    check("in-process: the valid sets pass every per-set check", stop_of() is None, str(stop_of()))
    p0 = next(i for i in r0 if any(R.lab[j] != "un" and R.lens[j] == R.lens[i] for j in range(R.n)))
    j_lab = next(j for j in range(R.n) if R.lab[j] != "un" and R.lens[j] == R.lens[p0])
    p1 = next(i for i in r0 if any(R.lab[j] == "un" and R.rep[j] > 0 and R.lens[j] == R.lens[i] for j in range(R.n)))
    j_rep = next(j for j in range(R.n) if R.lab[j] == "un" and R.rep[j] > 0 and R.lens[j] == R.lens[p1])
    j_len = next(j for j in b.ru0 if j not in set(r0) and R.lens[j] != R.lens[r0[0]])
    q0 = next(i for i in div_dd if any(T.lens[j] == T.lens[i] for j in b.rep_cand))
    j_drep = next(j for j in b.rep_cand if T.lens[j] == T.lens[q0])
    q1 = next(i for i in rep_dd if any(T.lens[j] == T.lens[i] for j in b.div_cand))
    j_div = next(j for j in b.div_cand if T.lens[j] == T.lens[q1])
    cases = (
        ("index list is not strictly ascending", {"ft_un_r0": {"idx": r0[::-1]}}),
        ("holds rows that are not 'un'", {"ft_un_r0": {"idx": swap(r0, p0, j_lab)}}),
        ("length histogram differs", {"ft_un_r0": {"idx": swap(r0, r0[0], j_len)}}),
        ("a row has rep_rand > 0", {"ft_un_r0": {"idx": swap(r0, p1, j_rep)}}),
        ("a row has rep > 0", {"ft_un_div_dd": {"idx": swap(div_dd, q0, j_drep)}}),
        ("a row has rep = 0", {"ft_un_rep_dedup": {"idx": swap(rep_dd, q1, j_div)}}),
        ("not more than its budget", {"ft_un_r0": {"budget": 10 ** 9}}))
    for want, chg in cases:
        got = stop_of(**chg)
        check(f"in-process: a set with one doctored row is a STOP: '{want}'", got is not None and want in got, str(got))
    # two rows of div_dd with one source line (every diverse duplicate in the fixture is shorter than
    # any row div_dd can hold, so the duplicate is made in a copy of the parent's source column)
    i0, k0 = div_dd[0], next(k for k in div_dd[1:] if T.lens[k] == T.lens[div_dd[0]])
    src2 = list(T.src)
    src2[k0] = src2[i0]
    with patched(T, "src", src2):
        got = stop_of()
    check("in-process: a set with one doctored row is a STOP: 'has duplicate source lines'",
          got is not None and "has duplicate source lines" in got, str(got))
    with patched(b, "ho_src", b.ho_src + [R.src[r0[5]]]):
        got = stop_of()
    check("in-process: a set row that is a heldout_un line is a STOP", got is not None and "heldout_un" in got, str(got))
    real_types = m.py_types
    with patched(m, "py_types", lambda idx, t: real_types(idx, t) + 1):
        got = stop_of()
    check("in-process: fast type counts that disagree with the literal definition are a STOP",
          got is not None and "fast type counts disagree" in got, str(got))

    # the chosen point must rebuild to its own measured coverage
    real_rebuild = m.Builder.rebuild_point

    def other_point(self, ns, rule, grid, p, stats=None):
        return real_rebuild(self, ns, rule, grid, {**p, "k": 0 if p["k"] else grid["K"]}, stats)
    buf = io.StringIO()
    with patched(m.Builder, "rebuild_point", other_point), contextlib.redirect_stdout(buf):
        rc = m.main(["--dir", str(d / "inproc_rebuild"), "--decisions", str(S.J), SKIP, *map(str, base_args(fx))])
    check("in-process: a chosen point whose rebuild does not reproduce its measured coverage is a STOP",
          rc == 1 and "did not reproduce its measured coverage" in buf.getvalue() and not (d / "inproc_rebuild").exists(),
          buf.getvalue()[-300:])

    # the counts the pinned inputs must give: each one checked when (and only when) inputs are pinned
    rd, dd, rrm, drm = ref.dedup()
    true = {"diverse UN candidates": len(ref.div_cand), "repetitive UN candidates": len(ref.rep_cand),
            "RU0 (UN rows of ft_random with rep_rand = 0)": len(ref.RU0), "rows of each anchor": len(ref.div),
            "A_narrow capped pool at 42,000 / 58,000 (top-band rows kept)": len(ref.capped(42000, 58000)),
            "repetitive candidates after OD 4(a) dedup": len(ref.rep_cand) - len(rrm),
            "diverse candidates after OD 4(a) dedup": len(ref.div_cand) - len(drm),
            "pretraining lines": len(read(fx["P"] / "train.clean.en"))}

    def pinned_run(table, pinned=True):
        saved = dict(m.EXPECTED_ON_PINNED)
        m.EXPECTED_ON_PINNED.clear()
        m.EXPECTED_ON_PINNED.update(table)
        try:
            bp = inproc_builder(m, fx, pinned=pinned)
            bp.prepare(want_hashes=False, need_fr_ranks=False)
            bp.build_a_narrow("keep", [42000, 58000], 42)
            bp.build_dedup("a")
            return None, bp
        except m.Stop as e:
            return str(e), None
        finally:
            m.EXPECTED_ON_PINNED.clear()
            m.EXPECTED_ON_PINNED.update(saved)
    check("the pinned-count table has exactly the eight counts the build checks", set(true) == set(m.EXPECTED_ON_PINNED),
          str(sorted(set(true) ^ set(m.EXPECTED_ON_PINNED))))
    err, bp = pinned_run(true)
    check("in-process, inputs taken as pinned: the fixture's own counts pass and each is recorded as checked",
          err is None and sum(1 for c in bp.checks if c["check"].endswith("(pinned inputs)")) == len(true), str(err))
    fired = []
    for k in true:
        err, _ = pinned_run({**true, k: true[k] + 1})
        fired.append(err is not None and err.startswith(k) and "the pinned inputs give" in err)
    check("each of the eight pinned counts is a STOP when the inputs give another number", all(fired), str(fired))
    err, _ = pinned_run({k: v + 1 for k, v in true.items()}, pinned=False)
    check("and none is checked when the inputs are not the pinned ones (fixtures give other counts by design)", err is None, str(err))


def feasibility_checks(ctx, S):
    d, check, m, fx, D, J, idx = ctx.d, ctx.check, S.m, S.fx, S.D, S.J, S.idx
    F = d / "feas"
    AB12 = {"Ke": [None], "Kf": [750, None]}     # on the fixture, unrestricted (a)/(b) would pick Kf = 750
    Jf = decisions(d / "dec/feas.json", fr_budget_214489=S.B, fr_budget_198333=S.Bdd,
                   od2_grid={"a": AB12, "b": AB12, "c": {"K": 20}, "c2": {"K": 10, "lambda": [0.0, 0.5, 1.0]}})
    rc, out = run6(ctx, "--feasibility", "--out", F, "--decisions", Jf, *full_args(fx))
    okf = rc == 0
    check("--feasibility builds every option", okf, out[-800:])
    if okf:
        S.F = F
        rep = json.loads((F / "feasibility_report.json").read_text(encoding="utf-8"))
        names = [x.name for x in F.iterdir()]
        check("--feasibility writes no .en/.fr and no manifest.json, a NOT_FOR_TRAINING file, and 'for_training': false",
              not any(x.endswith((".en", ".fr")) for x in names) and "manifest.json" not in names
              and "NOT_FOR_TRAINING" in names and rep["for_training"] is False and rep["decisions"] == "all options")
        check("every feasibility index file is named FEASIBILITY_<set>__<options>.npy and listed with its sha256",
              all(x.startswith("FEASIBILITY_") and "__" in x for x in names if x.endswith(".npy"))
              and set(rep["files"]) == {x for x in names if x != "feasibility_report.json"})
        check("it writes the three anchors' index files too (the contrasts' other side, for --measure)",
              all([int(x) for x in np.load(F / rep["anchors"][n]["file"])] == idx(n) for n in ("ft_un_div", "ft_un_rep", "ft_un_random")))
        res = rep["results"]
        check("it builds R0 and A_narrow under both OD 6 options for every seed",
              all(f"{n}__od6-{o}__seed{s}" in res and "failed" not in res[f"{n}__od6-{o}__seed{s}"]
                  for n in ("ft_un_r0", "ft_un_a_narrow") for o in ("keep", "remove") for s in (42, 1, 2, 3, 4)))
        check("and both duplicate rules, reporting that OD 4(b) removes a diverse row of 8 or more tokens",
              res["ft_un_rep_dedup__od4-a"]["build"]["diverse_removed_with_8_or_more_tokens"] == 0
              and res["ft_un_rep_dedup__od4-b"]["build"]["diverse_removed_with_8_or_more_tokens"] >= 1)
        gr = rep["grids"]["ft_un_div_narrow"]["rules"]
        check("and every narrowing rule over its grid, for div_narrow and div_narrow_dd, with each OD 1 option's choice",
              set(gr) == {"a", "b", "c", "c2"} and all(set(gr[r]["by_od1"]) >= {"a", "b", "c", "d"} for r in gr)
              and set(rep["grids"]["ft_un_div_narrow_dd"]["rules"]) == {"a", "b", "c", "c2"})
        a12 = {}
        for r in ("a", "b"):
            pr = S.ref.ab_grid(S.ref.rep, S.B, r, AB12["Ke"], AB12["Kf"])
            full, res = pick(pr, S.tgt, ["src_types"], TIE[r]), pick([p for p in pr if p["Kf"] is None], S.tgt, ["src_types"], TIE[r])
            a12[r] = (full and (full[1]["Ke"], full[1]["Kf"]), res and (res[1]["Ke"], res[1]["Kf"]),
                      (gr[r]["by_od1"]["a"].get("chosen") or {}).get("Kf", "none chosen"))
        check("under OD 1(a), rules (a) and (b) choose among points without a French cap only (A12), on a grid where the "
              "unrestricted choice would cap the French side",
              all(f is not None and f[1] is not None and x is not None and x[1] is None and c is None for f, x, c in a12.values()),
              str(a12))
        fq = [int(x) for x in np.load(F / gr["c"]["by_od1"]["a"]["file"])]
        check("its (c) / OD 1(a) choice is the production build's div_narrow", fq == idx("ft_un_div_narrow"))
        o3 = rep["od3_cap_report"]
        ankeep = [p for p in o3["keep"] if (p["Ke"], p["Kf"]) == (42000, 58000)]
        check("and the OD 3 cap report around the caps, over the full pool and over the pool without top-band rows (A21)",
              len(o3["keep"]) == 81 and len(o3["remove"]) == 81 and ankeep
              and ankeep[0]["src_types"] == len({w for i in S.an_ref for w in S.ref.rt[i]}), str(ankeep)[:200])
        rc, out = verify6(ctx, F)
        check("--verify refuses feasibility output", rc == 1 and "NOT_FOR_TRAINING" in out, out[-200:])
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(d), "PROBE": "zz", "CONDITIONS": "ft_un_r0",
               "DATA": str(F), "SFT": str(ctx.ROOT), "PY": ctx.PY}
        q = subprocess.run(["bash", str(ctx.ROOT / "phase0/probe_run.sh")], capture_output=True, text=True, timeout=120, env=env)
        check("probe_run.sh refuses to train on a feasibility directory", q.returncode == 1 and "MISSING" in q.stdout,
              (q.stdout + q.stderr)[-300:])
    rc, out = run6(ctx, "--feasibility", "--out", D, *base_args(fx))
    check("--feasibility refuses an --out that holds a manifest.json", rc == 1 and "manifest.json" in out, out[-300:])
    rc, out = run6(ctx, "--feasibility", "--out", d / "feas2", "--dir", d / "feas2", *base_args(fx))
    check("--feasibility refuses an --out that is the production --dir", rc == 1 and "--dir" in out and not (d / "feas2").exists(),
          out[-300:])
    rc, out = run6(ctx, "--feasibility", "--out", fx["E"], *base_args(fx))
    check("--feasibility refuses an --out that is --e05-dir", rc == 1 and "is --e05-dir" in out, out[-300:])
    if okf:
        rc, out = prod(ctx, F, J, *base_args(fx))
        check("a production build refuses a directory that holds feasibility output", rc == 1 and "feasibility" in out, out[-300:])
        rc, out = run6(ctx, "--dry-run", "--dir", F, "--decisions", J, *full_args(fx))
        check("and so does a dry run", rc == 1 and "feasibility" in out, out[-300:])
    if getattr(S, "DR", None) is not None:
        rc, out = prod(ctx, S.DR, J, *base_args(fx))
        check("a production build refuses a dry run's directory", rc == 1 and "--dry-run" in out, out[-300:])


# ---------------------------------------------------------------- --measure
def stub_ids(s, ix):
    return [ix.get(w, 1) for w in s.strip().split(" ")]


def ref_piece(lines, idx, ix, nv):
    cnt = collections.Counter(t for i in idx for t in stub_ids(lines[i], ix))
    real = {t: c for t, c in cnt.items() if t not in (0, 1, 2, 3)}
    prof = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, ">100": 0}
    for c in real.values():
        prof["1" if c == 1 else "2-5" if c <= 5 else "6-20" if c <= 20 else "21-100" if c <= 100 else ">100"] += 1
    return {"piece_tokens": sum(cnt.values()), "unknown_tokens": cnt.get(1, 0), "piece_types": len(real),
            "vocab_share": len(real) / (nv - 4), "profile": prof}


def measure_checks(ctx, S):
    d, check, m, fx, ref, D = ctx.d, ctx.check, S.m, S.fx, S.ref, S.D
    shim = d / "spm_stub"
    shim.mkdir()
    (shim / "sentencepiece.py").write_text(STUB_SPM, encoding="utf-8")
    env = {"PYTHONPATH": str(shim)}
    words = sorted({w for p in ("ft_topk", "ft_random") for e in ("en", "fr") for x in read(fx["S"] / f"{p}.{e}") for w in x.split()})
    pieces = ["<pad>", "<unk>", "<s>", "</s>"] + [w for k, w in enumerate(words) if k % 7]      # every 7th word is unknown
    model = d / "stub.model"
    model.write_text(json.dumps({"pieces": pieces}), encoding="utf-8")
    ix = {p: i for i, p in enumerate(pieces)}
    mt = d / "fake_mt/src/data"
    mt.mkdir(parents=True)
    (mt / "dataset.py").write_text("\n".join("    " + v for v in m.TRAINER_LINES.values()) + "\n", encoding="utf-8")
    nt = d / "newstest.en"
    rng = random.Random(5)
    write(nt, [" ".join(rng.choice(V[:900]) for _ in range(rng.randint(5, 25))) for _ in range(120)])
    M = d / "measure"
    margs = ["--measure", "--from", D, "--out", M, "--src-dir", fx["S"], "--spm-model", model, "--allow-unpinned-src",
             "--max-seq-len", 20, "--newstest-en", nt, "--mt-repo", d / "fake_mt"]
    before = files_of(D)
    rc, out = run6(ctx, *margs, env=env)
    ok = rc == 0
    check("--measure reads a finished build and writes only measure_report.json, never into the build", ok
          and files_of(D) == before and [x.name for x in M.iterdir()] == ["measure_report.json"], out[-600:])
    if ok:
        rep = json.loads((M / "measure_report.json").read_text(encoding="utf-8"))
        man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
        par = {"ft_topk": ref.T, "ft_random": ref.Q}
        good, bad = True, []
        for nm, info in man["sets"].items():
            P, ixs = par[info["parent"]], S.idx(nm)
            r = rep["sets"][nm]
            kept = [i for i in ixs if len(stub_ids(P["en"][i], ix)) + 2 <= 20 and len(stub_ids(P["fr"][i], ix)) + 2 <= 20]
            w = {"source": ref_piece(P["en"], ixs, ix, len(pieces)), "target": ref_piece(P["fr"], ixs, ix, len(pieces))}
            same = (r["source"] == w["source"] and r["target"] == w["target"] and r["rows_kept_by_trainer"] == len(kept)
                    and r["rows_over_max_seq_len"] == len(ixs) - len(kept)
                    and r["source_piece_tokens_kept"] == sum(len(stub_ids(P["en"][i], ix)) for i in kept)
                    and r["target_piece_tokens_kept"] == sum(len(stub_ids(P["fr"][i], ix)) for i in kept)
                    and r["rows_with_carriage_return"] == 0)
            good &= same
            if not same:
                bad.append(nm)
        check("piece types (control and unknown ids excluded), tokens, unknowns, frequency profile and the trainer's "
              "length filter equal an independent count, set by set (spec A28)", good and len(rep["sets"]) == len(man["sets"]), str(bad))
        check("the fixture exercises it: some rows exceed max_seq_len and some pieces are unknown",
              any(r["rows_over_max_seq_len"] > 0 for r in rep["sets"].values())
              and all(r["source"]["unknown_tokens"] > 0 for r in rep["sets"].values()))
        Ntab = [c for c in rep["contrasts"] if c["contrast"] == "N"]
        ta, tb = rep["sets"]["ft_un_r0"]["target_piece_tokens_kept"], rep["sets"]["ft_un_a_narrow"]["target_piece_tokens_kept"]
        check("the contrast table gives each pair's exact piece-token imbalance (N = R0 - A_narrow), and every contrast the build has",
              len(Ntab) == 1 and Ntab[0]["target_imbalance"] == (ta - tb) / tb
              and {c["contrast"] for c in rep["contrasts"]} == {"N", "S", "C1-beyond-coverage", "C1-beyond-duplicates",
                                                               "C1-beyond-both", "P", "Q1'", "Q2'", "V"}, str(Ntab))
        ntc = collections.Counter(w for x in read(nt) for w in toks(x))
        tot = sum(ntc.values())
        sc = collections.Counter(w for i in S.idx("ft_un_div") for w in ref.tt[i])
        want = {f"k{k}": 100 * sum(c for w, c in ntc.items() if sc[w] >= k) / tot for k in (1, 5, 20)}
        check("newstest2014 coverage (descriptive) is arms.cover over the set's source, recomputed independently",
              all(abs(rep["sets"]["ft_un_div"]["newstest2014_coverage"][k] - v) < 1e-9 for k, v in want.items())
              and rep["inputs"]["newstest2014 .en"]["sha256"] == sha_file(nt), str(rep["sets"]["ft_un_div"]["newstest2014_coverage"]))
        check("the report records the trainer code it mirrors and that it is not for training",
              rep["for_training"] is False and rep["trainer_tokenisation"]["sha256"] == sha_file(mt / "dataset.py"))
    if getattr(S, "F", None) is not None:
        MF = d / "measure_feas"
        rc, out = run6(ctx, "--measure", "--from", S.F, "--out", MF, "--src-dir", fx["S"], "--spm-model", model,
                       "--allow-unpinned-src", env=env)
        repf = json.loads((MF / "measure_report.json").read_text(encoding="utf-8")) if rc == 0 else {}
        check("--measure reads a feasibility directory too (every option, the anchors included), with no contrast table",
              rc == 0 and repf["sets"].get("ft_un_div", {}).get("rows") == len(S.idx("ft_un_div")) and "ft_un_a_narrow__od6-remove__seed1" in repf["sets"]
              and isinstance(repf["contrasts"], str), out[-400:])
    rc, out = run6(ctx, *[x if x != M else D for x in margs], env=env)
    check("--measure refuses --out = --from", rc == 1 and "never write into the build" in out and files_of(D) == before, out[-300:])
    rc, out = run6(ctx, *[x if x != M else D / "sub" for x in margs], env=env)
    check("and an --out inside --from", rc == 1 and not (D / "sub").exists(), out[-300:])
    noimp = d / "spm_missing"
    noimp.mkdir()
    (noimp / "sentencepiece.py").write_text("raise ImportError('no sentencepiece in this interpreter')\n", encoding="utf-8")
    rc, out = run6(ctx, *[x if x != M else d / "m_noimp" for x in margs], env={"PYTHONPATH": str(noimp)})
    check("without sentencepiece --measure is a STOP with a plain message", *stopped(d / "m_noimp", rc, out, "sentencepiece is not importable"))
    rc, out = run6(ctx, *[x if x != M else d / "m_pin" for x in margs if x != "--allow-unpinned-src"], env=env)
    check("an SPM model other than the pinned one is a STOP unless the deviation is asked for",
          *stopped(d / "m_pin", rc, out, "not the pinned ones", "spm model"))
    Dt = d / "measure_tampered"
    shutil.copytree(D, Dt)
    with open(Dt / "ft_un_r0.fr", "a", encoding="utf-8") as f:
        f.write("extra\n")
    rc, out = run6(ctx, *[Dt if x == D else (d / "m_tam" if x == M else x) for x in margs], env=env)
    check("--measure refuses a build that changed since it was built", *stopped(d / "m_tam", rc, out, "changed since", "ft_un_r0.fr"))
    S2 = d / "src_other"
    shutil.copytree(fx["S"], S2)
    with open(S2 / "ft_topk.en", "a", encoding="utf-8") as f:
        f.write("one more row\n")
    rc, out = run6(ctx, *[S2 if x == fx["S"] else (d / "m_src" if x == M else x) for x in margs], env=env)
    check("--measure refuses parents other than the ones the build was made from", *stopped(d / "m_src", rc, out, "is not the file"))
    (mt / "dataset.py").write_text("sp.encode(src_lines, out_type=int)\n", encoding="utf-8")
    rc, out = run6(ctx, *[x if x != M else d / "m_mt" for x in margs], env=env)
    check("--measure STOPs if the trainer no longer tokenises the way it mirrors (A28)",
          *stopped(d / "m_mt", rc, out, "no longer tokenises"))
    (mt / "dataset.py").write_text("\n".join("    " + v for v in m.TRAINER_LINES.values()) + "\n", encoding="utf-8")

    # a row holding "\r" would be split by the trainer's text-mode read: counted and flagged (in-process)
    spec = importlib.util.spec_from_file_location("spm_stub_inproc", shim / "sentencepiece.py")
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)
    prev = sys.modules.get("sentencepiece")
    sys.modules["sentencepiece"] = stub
    real_read = m.read_lines
    j = S.idx("ft_un_div")[3]

    def with_cr(p):
        rows = real_read(p)
        return [x + ("\rx" if (k == j and p.name == "ft_topk.en") else "") for k, x in enumerate(rows)] if p.name == "ft_topk.en" else rows
    buf = io.StringIO()
    try:
        with patched(m, "read_lines", with_cr), contextlib.redirect_stdout(buf):
            rc = m.main(["--measure", "--from", str(D), "--out", str(d / "m_cr"), "--src-dir", str(fx["S"]),
                         "--spm-model", str(model), "--allow-unpinned-src"])
    finally:
        if prev is None:
            sys.modules.pop("sentencepiece", None)
        else:
            sys.modules["sentencepiece"] = prev
    rcr = json.loads((d / "m_cr/measure_report.json").read_text(encoding="utf-8")) if rc == 0 else {}
    check("in-process: a set row holding a carriage return is counted and flagged (the trainer would misalign it)",
          rc == 0 and rcr["sets"]["ft_un_div"]["rows_with_carriage_return"] == 1 and "WARNING" in buf.getvalue(), buf.getvalue()[-300:])

    j2 = S.idx("ft_un_div")[7]

    def with_spaces(p):
        rows = real_read(p)
        return [("  " + x + "  ") if (k == j2 and p.name == "ft_topk.en") else x for k, x in enumerate(rows)]
    sys.modules["sentencepiece"] = stub
    buf = io.StringIO()
    try:
        with patched(m, "read_lines", with_spaces), contextlib.redirect_stdout(buf):
            rc = m.main([str(x) for x in margs[:3]] + ["--out", str(d / "m_sp")] + [str(x) for x in margs[5:]])
    finally:
        if prev is None:
            sys.modules.pop("sentencepiece", None)
        else:
            sys.modules["sentencepiece"] = prev
    rsp = json.loads((d / "m_sp/measure_report.json").read_text(encoding="utf-8")) if rc == 0 else {}
    check("in-process: rows are stripped as the trainer strips them (edge spaces change no piece count)",
          rc == 0 and ok and rsp["sets"] == json.loads((M / "measure_report.json").read_text(encoding="utf-8"))["sets"],
          buf.getvalue()[-300:])

    # the real package, where this interpreter has it (py310 on the Mac; not the system python3)
    try:
        import sentencepiece as spm_real
    except ImportError:
        spm_real = None
    if spm_real is None or getattr(spm_real, "__version__", "") == "stub":
        ctx.skip("--measure with the real sentencepiece package", "sentencepiece is not installed in this interpreter")
        return
    txt = d / "spm_train.txt"
    write(txt, read(fx["S"] / "ft_topk.en")[:1500] + read(fx["S"] / "ft_topk.fr")[:1500])
    spm_real.SentencePieceTrainer.train(input=str(txt), model_prefix=str(d / "real"), vocab_size=400, model_type="unigram",
                                        pad_id=0, unk_id=1, bos_id=2, eos_id=3, minloglevel=2, hard_vocab_limit=False)
    MR = d / "measure_real"
    rc, out = run6(ctx, "--measure", "--from", D, "--out", MR, "--src-dir", fx["S"], "--spm-model", d / "real.model",
                   "--allow-unpinned-src")
    if rc != 0:
        check("--measure runs with the real sentencepiece package", False, out[-500:])
        return
    rr = json.loads((MR / "measure_report.json").read_text(encoding="utf-8"))
    sp = spm_real.SentencePieceProcessor()
    sp.Load(str(d / "real.model"))
    sp_special = {i for i in range(sp.get_piece_size()) if sp.is_control(i) or sp.is_unknown(i)}
    ixs = S.idx("ft_un_div")
    ids = sp.encode([ref.T["en"][i].strip() for i in ixs], out_type=int)
    check("--measure with the real sentencepiece package: source piece types and tokens equal the library's own encoding",
          rr["sets"]["ft_un_div"]["source"]["piece_types"] == len({t for e in ids for t in e} - sp_special)
          and rr["sets"]["ft_un_div"]["source"]["piece_tokens"] == sum(len(e) for e in ids), str(rr["sets"]["ft_un_div"]["source"]))


def reproduce_checks(ctx, S):
    d, check, m, fx, ref = ctx.d, ctx.check, S.m, S.fx, S.ref
    rdir = d / "refs"
    rdir.mkdir()
    np.save(rdir / "R0.npy", np.array(ref.r0(), dtype=np.int64))
    np.save(rdir / "A_250_140.npy", np.array(S.ank, dtype=np.int64))
    np.save(rdir / "A_42_58.npy", np.array(S.an_ref, dtype=np.int64))
    rargs = ["--reproduce-check", "--ref-r0", rdir / "R0.npy", "--allow-other-interpreter", *base_args(fx)]
    rc, out = run6(ctx, *rargs, "--ref-a-narrow", rdir / "A_42_58.npy")
    check("--reproduce-check passes when R0 and A_narrow (42,000 / 58,000) equal the reference files row for row",
          rc == 0 and out.count("REPRODUCED") == 2, out[-400:])
    rc, out = run6(ctx, *rargs, "--ref-a-narrow", rdir / "A_250_140.npy", "--ref-caps", *map(str, CAPS_BIND))
    check("and at caps that bind (the cap path, not only the nn_match path)", rc == 0 and out.count("REPRODUCED") == 2, out[-400:])
    bad = np.array(ref.r0(), dtype=np.int64)
    alt_rows = sorted(set(ref.RU0) - set(bad.tolist()))
    bad[3] = alt_rows[0]
    bad.sort()
    np.save(rdir / "R0_bad.npy", bad)
    rc, out = run6(ctx, "--reproduce-check", "--ref-r0", rdir / "R0_bad.npy", "--ref-a-narrow", rdir / "A_42_58.npy",
                   "--allow-other-interpreter", *base_args(fx))
    check("--reproduce-check fails (exit 1) when one R0 row differs, and says where",
          rc == 1 and "R0 DIFFERS" in out and "first difference" in out and "A_narrow: REPRODUCED" in out, out[-400:])
    rc, out = run6(ctx, *rargs, "--ref-a-narrow", rdir / "A_250_140.npy")
    check("--reproduce-check fails when A_narrow differs (here: built at other caps than the reference)",
          rc == 1 and "A_narrow DIFFERS" in out, out[-400:])
    np.save(rdir / "short.npy", np.array(ref.r0()[:-1], dtype=np.int64))
    rc, out = run6(ctx, "--reproduce-check", "--ref-r0", rdir / "short.npy", "--ref-a-narrow", rdir / "A_42_58.npy",
                   "--allow-other-interpreter", *base_args(fx))
    check("--reproduce-check fails on a reference with a row missing", rc == 1 and "R0 DIFFERS" in out, out[-300:])
    check("--reproduce-check writes nothing", not any(x.name not in ("R0.npy", "A_250_140.npy", "A_42_58.npy", "R0_bad.npy", "short.npy")
                                                     for x in rdir.iterdir()))
    rc, out = run6(ctx, "--reproduce-check", "--ref-r0", rdir / "R0.npy", "--ref-a-narrow", rdir / "A_42_58.npy",
                   "--allow-other-interpreter", *base_args(fx, pinned=True))
    check("--reproduce-check refuses inputs other than the pinned ones, the two reference files included",
          rc == 1 and "not the pinned ones" in out and "reference R0" in out and "reference A_narrow" in out
          and "ft_topk.en" in out, out[-400:])
    rc, out = run6(ctx, "--reproduce-check", "--ref-r0", rdir / "R0.npy", "--ref-a-narrow", rdir / "A_250_140.npy",
                   "--ref-caps", *map(str, CAPS_BIND), "--allow-other-interpreter", *base_args(fx, pinned=True))
    check("--reproduce-check refuses caps other than 42,000 / 58,000 without --allow-unpinned-src",
          rc == 1 and "--ref-caps other than" in out, out[-300:])
    rc, out = run6(ctx, "--reproduce-check", "--ref-r0", rdir / "R0.npy", "--ref-a-narrow", rdir / "A_42_58.npy", *base_args(fx))
    me = m.interpreter()
    if all(me[k] == v for k, v in m.REFERENCE_INTERPRETER.items()):
        check("on the interpreter that wrote the scratch files (3.14.3, Unicode 16.0.0, x86_64), --reproduce-check needs "
              "no interpreter override", rc == 0 and out.count("REPRODUCED") == 2, out[-300:])
    else:
        check("on another interpreter, --reproduce-check refuses unless the override is asked for",
              rc == 1 and "scratch index files were written by Python 3.14.3" in out and "x86_64" in out, out[-300:])


def usage_checks(ctx, S):
    d, check, fx, J = ctx.d, ctx.check, S.fx, S.J
    for what, args in (("--verify without --dir", ["--verify"]),
                       ("a production build without --src-dir", ["--dir", d / "u1", "--decisions", J, SKIP]),
                       ("--skip-expected-sha256 beside --expect-sha256-from",
                        ["--dir", d / "u2", "--decisions", J, SKIP, "--expect-sha256-from", d / "x.json", *base_args(fx)]),
                       ("--ref-lex without --pretrain-fr", ["--dir", d / "u3", "--decisions", J, SKIP, "--ref-lex", fx["lex"],
                                                            *base_args(fx)]),
                       ("two modes", ["--verify", "--feasibility", "--dir", d / "u4"])):
        rc, out = run6(ctx, *args)
        check(f"a usage error ({what}) exits 64, never 2 (REFUSED)", rc == 64 and "usage error" in out, out[-300:])


def unit_tests(ctx, m, d):
    check = ctx.check
    lg = lambda v: float(v)                       # a key function that makes exact ties easy to build
    L = {i: 5 for i in range(100)}
    # lower neighbour on an equal distance
    out = m.nn_match([0], L, {0: 1.0}, [10, 11], L, {10: 0.0, 11: 2.0}, 42, "t", lg)
    check("nn_match: on an equal distance the lower neighbour wins", out == [10], str(out))
    # ascending target order on equal |v - median|: S (key 2) is nearest for both targets
    pk = {20: -1.0, 21: 2.0, 22: 4.5}
    o1 = m.nn_match([0, 1], L, {0: 1.0, 1: 3.0}, [20, 21, 22], L, pk, 42, "t", lg)
    o2 = m.nn_match([0, 1], L, {0: 3.0, 1: 1.0}, [20, 21, 22], L, pk, 42, "t", lg)
    check("nn_match: targets equally far from the median are taken in ascending target-row order",
          o1 == [21, 22] and o2 == [20, 21], f"{o1} {o2}")
    # key ties: tie key, drawn once per pool row in ascending order, also at lengths without targets
    # rows 30, 32, 34 (length 5) share one key; rows 31, 33 sit at length 7, where there is no target
    def smallest_tie_key(sd, pool):
        r = random.Random(sd)
        u = {i: r.random() for i in pool}              # one draw per pool row, ascending
        return min((30, 32, 34), key=u.get)
    sd = next(s for s in range(500) if smallest_tie_key(s, (30, 31, 32, 33, 34)) not in (30, smallest_tie_key(s, (30, 32, 34))))
    lens = {30: 5, 31: 7, 32: 5, 33: 7, 34: 5, 0: 5}
    keys = {30: 1.0, 31: 9.0, 32: 1.0, 33: 9.0, 34: 1.0, 0: 1.0}
    out = m.nn_match([0], lens, keys, [30, 31, 32, 33, 34], lens, keys, sd, "t", lg)
    out2 = m.nn_match([0], lens, keys, [30, 32, 34], lens, keys, sd, "t", lg)
    check("nn_match: equal keys go to the smallest tie key (not the lowest index), and tie keys are drawn for every "
          "pool row, also at lengths without targets (removing those rows changes the result)",
          out == [smallest_tie_key(sd, (30, 31, 32, 33, 34))] and out2 == [smallest_tie_key(sd, (30, 32, 34))]
          and out != out2 and out != [30], f"{out} {out2} seed {sd}")
    # D7: frequency ties follow first occurrence inside the pool
    et = {0: ["y", "a"], 1: ["x", "a"], 2: ["y"], 3: ["x"], 4: ["a"]}
    ft = {i: ["f"] for i in et}
    po = m.pool_order([0, 1, 2, 3, 4], et, ft)
    check("caps: equal counts rank by first occurrence in the pool (a, y before x), so a cap of 2 keeps y and drops x",
          m.caps(po, 2, None) == [0, 2, 4] and m.caps(po, 3, None) == [0, 1, 2, 3, 4] and m.caps(po, None, None) == [0, 1, 2, 3, 4])
    rk = m.ranks_of(collections.Counter(["b", "a", "a", "c", "b", "d"]), "t")
    check("pretraining ranks: ties broken by first occurrence in corpus order", rk == {"b": 0, "a": 1, "c": 2, "d": 3}, str(rk))
    kept, rm = m.dedup_first([5, 1, 7, 9], {5: "s", 1: "t", 7: "s", 9: "t"}.get)
    check("dedup keeps the first occurrence", kept == [5, 1] and rm == [7, 9])
    check("the A17 key lowercases \\w+ tokens and masks every digit",
          m.a17_key("Item 12 of A/61/250") == m.a17_key("item 13 of a/62/251") == "item 00 of a 00 000")
    pts = [{"feasible": True, "src_types": 98, "Ke": 1000, "Kf": None}, {"feasible": True, "src_types": 102, "Ke": 2000, "Kf": 5},
           {"feasible": True, "src_types": 102, "Ke": None, "Kf": 5}, {"feasible": False, "Ke": 9, "Kf": 9}]
    b, dev = m.choose(pts, ["src_types"], {"src_types": 100}, "a")
    check("choice: equal deviations go to the larger Ke (no cap is largest)", b is pts[2] and dev == Fraction(1, 50))
    pts = [{"feasible": True, "src_types": 102, "Ke": 3000, "Kf": 5}, {"feasible": True, "src_types": 98, "Ke": 3000, "Kf": 7},
           {"feasible": True, "src_types": 102, "Ke": 3000, "Kf": 6}]
    pts_none = pts + [{"feasible": True, "src_types": 98, "Ke": 3000, "Kf": None}]
    check("choice: at an equal deviation and an equal Ke, the larger Kf wins, and no French cap is the largest "
          "(the tie that is the rule under OD 1(a), A12)",
          m.choose(pts, ["src_types"], {"src_types": 100}, "a")[0] is pts[1]
          and m.choose(pts_none, ["src_types"], {"src_types": 100}, "b")[0] is pts_none[3])
    pts = [{"feasible": True, "src_types": 98, "k": 3, "lambda": 0.5}, {"feasible": True, "src_types": 102, "k": 3, "lambda": 0.0},
           {"feasible": True, "src_types": 102, "k": 2, "lambda": 0.0}]
    check("choice: under c2 ties go to the larger k, then the smaller lambda", m.choose(pts, ["src_types"], {"src_types": 100}, "c2")[0] is pts[1])
    check("choice: under (c) ties go to the larger k",
          m.choose([{"feasible": True, "s": 9, "k": 1}, {"feasible": True, "s": 11, "k": 4}], ["s"], {"s": 10}, "c")[0]["k"] == 4)
    check("the +-2% rule is inclusive and exact: 102 of 100 is within, 103 is not",
          m.within(m.rel_dev(102, 100), Fraction(1, 50)) and not m.within(m.rel_dev(103, 100), Fraction(1, 50))
          and m.within(m.rel_dev(98, 100), Fraction(1, 50)))
    # q-family at k = K is match_random
    rng = random.Random(3)
    lens = {i: rng.randint(3, 6) for i in range(200)}
    cand = list(range(200))
    h = collections.Counter({3: 5, 4: 7, 5: 2, 6: 9})
    byL = {}
    for i in cand:
        byL.setdefault(lens[i], []).append(i)
    ns = {"h": h, "nL": {Lx: len(byL[Lx]) for Lx in h}, "by": byL}
    orders = {Lx: sorted(byL[Lx], key=lambda i: -i) for Lx in h}
    check("q-family: k = K is exactly match_random over every candidate",
          m.Builder.q_point(ns, orders, 7, 7) == m.match_random(h, lens, cand, "t")[0]
          and m.Builder.q_point(ns, orders, 0, 7) == sorted(i for Lx in sorted(h) for i in orders[Lx][:h[Lx]]))
    # the fast measures are the literal definitions
    tl = [[f"w{rng.randint(0, 40)}" for _ in range(rng.randint(0, 9))] for _ in range(300)]
    voc = {}
    side = m.Side(tl, voc)
    sub = sorted(rng.sample(range(300), 120))
    check("fast type counts equal the literal set count and vocab/common.py types_at, at and around the budget",
          side.types(sub) == m.py_types(sub, tl)
          and all(side.types_at(sub, bb) == m.py_types_at(sub, tl, bb) == t_types_at(tl, sub, bb)
                  for bb in (1, 50, 200, side.tokens(sub) - 1, side.tokens(sub), side.tokens(sub) + 5)))
    arr = np.array([3, 9, 214488], dtype=np.int64)
    blob = m.npy_bytes(arr)
    check("index files are int64 .npy that numpy loads, with a header that does not depend on the numpy version",
          np.array_equal(np.load(io.BytesIO(blob)), arr) and (blob.index(b"\n") + 1) % 64 == 0 and np.load(io.BytesIO(blob)).dtype == np.int64)
    # the lexicon comparison and the reader that never runs code from a pickle
    built = {"a": [("x", 0.5), ("y", 0.25)], "b": [("x", 0.3)], "c": [("z", 0.2), ("w", 0.2)], "d": [("q", 0.9)]}
    refl = {"a": [("x", 0.5), ("y", 0.25)], "b": [("v", 0.3)], "c": [("w", 0.2), ("z", 0.2)], "e": [("q", 0.9)]}
    c = m.compare_lex(built, refl)
    check("the lexicon comparison counts entries only on one side, other translation sets, and third-place reorderings apart",
          (c["only_in_built"], c["only_in_lex_pkl"], c["entries_with_other_translations"],
           c["entries_with_same_translations_in_another_order"], c["identical"]) == (1, 1, 1, 1, False)
          and m.compare_lex(built, built)["identical"] is True, str(c))
    pk_ok, pk_bad = d / "lex_ok.pkl", d / "lex_bad.pkl"
    pk_ok.write_bytes(pickle.dumps(built))
    pk_bad.write_bytes(pickle.dumps({"a": [("x", np.float32(0.5))]}))
    try:
        m.load_lex(pk_bad)
        refused_bad = False
    except m.Stop as e:
        refused_bad = "numpy" in str(e)
    check("lex.pkl is read without running any class it names (a pickle that needs numpy is refused, not executed)",
          m.load_lex(pk_ok) == built and refused_bad)
    check("the registered (c)/(c2) definitions name the spec section and every parameter of the q-family (A15)",
          all(all(x in v for x in ("A15", "maxrank", "m_L = h_L + (k*(n_L-h_L))//K", "G8a", "G8b", "seed 42"))
              for v in m.OD2C_DEFINITION.values()) and "lambda" in m.OD2C_DEFINITION["c2"])
