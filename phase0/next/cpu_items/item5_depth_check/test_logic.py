#!/usr/bin/env python3
"""Toy test of chk.py's vectorised rep / types_at against naive versions written from the
definitions (e04_split.py statistic; vocab/common.py types_at semantics)."""
import collections, random, re, sys
import numpy as np
sys.path.insert(0, ".")
import chk

WORD = re.compile(r"\w+", re.UNICODE)
rng = random.Random(7)
words = [f"w{i}" for i in range(12)]
sents = []
for i in range(3000):
    L = rng.choice([0, 3, 7, 8, 9, 12, 20, 30])
    sents.append(" ".join(rng.choice(words[:rng.choice([2, 4, 12])]) for _ in range(L)))
sents += sents[:50]                       # exact repeats
toks = [[w.lower() for w in WORD.findall(s)] for s in sents]


def naive_rep(T):
    grams = [[" ".join(t[j:j + 8]) for j in range(len(t) - 7)] for t in T]
    c = collections.Counter(g for gs in grams for g in gs)
    return [sum(1 for g in gs if c[g] > 1) / len(gs) if gs else 0.0 for gs in grams]


def naive_types_at(T, N, seed):
    idx = list(range(len(T))); random.Random(seed).shuffle(idx)
    seen, n = set(), 0
    for i in idx:
        for w in T[i]:
            if n >= N:
                return len(seen)
            seen.add(w); n += 1
    return None


voc = {}
ids = np.array([voc.setdefault(w, len(voc)) for t in toks for w in t], np.uint32)
lens = np.array([len(t) for t in toks], np.int64)
off = np.concatenate([[0], np.cumsum(lens)[:-1]])
ng = np.maximum(lens - 7, 0)
H, ostart = chk.gram_hashes(ids, off, ng)
bad = 0
for c in (100, 777, 2000, len(toks)):
    flag = chk.rep_flags(H[:int(ostart[c])])
    rows = np.flatnonzero(ng[:c] > 0)
    nrep = np.zeros(c, np.int64)
    nrep[rows] = np.add.reduceat(flag.astype(np.int32), ostart[rows])
    mine = np.where(ng[:c] > 0, nrep / np.maximum(ng[:c], 1), 0.0)
    ref = np.array(naive_rep(toks[:c]))
    bad += int(np.abs(mine - ref).max() > 1e-12)
    print("rep prefix", c, "maxdiff", float(np.abs(mine - ref).max()), "rep>0", int((ref > 0).sum()))
for c in (500, len(toks)):
    for seed in range(3):
        perm = list(range(c)); random.Random(seed).shuffle(perm); perm = np.array(perm)
        for B in (50, 999, 5000):
            a = chk.first_B_types(ids, off, lens, perm, B)
            b = naive_types_at(toks[:c], B, seed)
            if a != b:
                bad += 1
                print("types mismatch", c, seed, B, a, b)
print("ALL OK" if bad == 0 else f"{bad} FAILURES")
