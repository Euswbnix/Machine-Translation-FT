#!/usr/bin/env python3
"""en-de step 1: per-row source labels by exact (src, tgt) match against the statmt
originals, after the corpus's own whitespace handling (strip(), then internal CR -> space,
as phase0/rebuild_corpus.py --mode fixed); TSV alignment check; per-row arrays.

Writes (scratch only): rows.npz with score, label, ambig_mask, pair_key, src_key, src_len.
"""
import hashlib, json, re, sys, time
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
CORP = Path.home() / "mt_local/rebuild/ende_fixed"
TSV = Path.home() / "mt_local/phase0_final/ende_scored.tsv"
ST = Path.home() / "mt_local/statmt"
SOURCES = [  # order = HF wmt14 de-en subset order; ties go to the first listed
    ("europarl", ST / "europarl/training/europarl-v7.de-en.en", ST / "europarl/training/europarl-v7.de-en.de"),
    ("commoncrawl", ST / "commoncrawl/commoncrawl.de-en.en", ST / "commoncrawl/commoncrawl.de-en.de"),
    ("news-commentary", ST / "news-commentary/training/news-commentary-v9.de-en.en",
     ST / "news-commentary/training/news-commentary-v9.de-en.de"),
]
WORD = re.compile(r"\w+", re.UNICODE)
UNM = 255


def k64(s, t):
    return int.from_bytes(hashlib.blake2b(f"{s}\x00{t}".encode("utf-8"), digest_size=8).digest(), "little")


def ks(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def corpus_form(x):
    # download_wmt_*: strip() + replace('\n',' '); rebuild fixed mode: internal CR -> space
    return x.strip().replace("\n", " ").replace("\r", " ")


t0 = time.time()
H, L, Hraw, stat = [], [], [], {}
for i, (name, fe, fd) in enumerate(SOURCES):
    h, hr = [], []
    bad_utf = 0
    with open(fe, "rb") as a, open(fd, "rb") as b:
        for x, y in zip(a, b, strict=True):
            try:
                xs, ys = x.decode("utf-8"), y.decode("utf-8")
            except UnicodeDecodeError:
                bad_utf += 1
                xs, ys = x.decode("utf-8", "replace"), y.decode("utf-8", "replace")
            xs, ys = xs[:-1] if xs.endswith("\n") else xs, ys[:-1] if ys.endswith("\n") else ys
            hr.append(k64(xs, ys))
            h.append(k64(corpus_form(xs), corpus_form(ys)))
    stat[name] = {"statmt_lines": len(h), "undecodable_lines": bad_utf}
    H.append(np.asarray(h, np.uint64)); Hraw.append(np.asarray(hr, np.uint64))
    L.append(np.full(len(h), i, np.uint8))
    print(f"indexed {name}: {len(h):,} ({time.time()-t0:.0f}s)", flush=True)

H = np.concatenate(H); L = np.concatenate(L); Hraw = np.concatenate(Hraw)
o = np.argsort(H, kind="stable"); H, L = H[o], L[o]
u, inv = np.unique(H, return_inverse=True)
mask = np.zeros(len(u), np.uint8)
np.bitwise_or.at(mask, inv.reshape(-1), (1 << L).astype(np.uint8))
Hraw = np.unique(Hraw)

# ---- stream corpus + TSV in lockstep ----
n = 0
score, pk, sk, sl, rawhit = [], [], [], [], []
mis = 0; mis_ex = []
with open(CORP / "train.clean.en", encoding="utf-8", newline="\n") as fe, \
     open(CORP / "train.clean.de", encoding="utf-8", newline="\n") as fd, \
     open(TSV, encoding="utf-8", newline="\n") as ft:
    for e, d, t in zip(fe, fd, ft, strict=True):
        e, d, t = e[:-1], d[:-1], t[:-1]
        sc, rest = t.split("\t", 1)
        if rest != e + "\t" + d:
            mis += 1
            if len(mis_ex) < 5:
                mis_ex.append(n)
        score.append(float(sc))
        pk.append(k64(e, d)); sk.append(ks(e)); sl.append(len(WORD.findall(e)))
        n += 1
        if n % 1_000_000 == 0:
            print(f"  {n:,} rows ({time.time()-t0:.0f}s)", flush=True)
score = np.asarray(score, np.float64); pk = np.asarray(pk, np.uint64)
sk = np.asarray(sk, np.uint64); sl = np.asarray(sl, np.int32)

i = np.searchsorted(u, pk); ic = np.minimum(i, len(u) - 1); hit = u[ic] == pk
m = np.where(hit, mask[ic], 0).astype(np.uint8)
# first-listed source among the bits
lab = np.full(n, UNM, np.uint8)
for b in (2, 1, 0):
    lab[(m >> b) & 1 == 1] = b
ambig = hit & ((m & (m - 1)) != 0)
j = np.searchsorted(Hraw, pk); jc = np.minimum(j, len(Hraw) - 1); rawhit = Hraw[jc] == pk

names = [s[0] for s in SOURCES]
rep = {"rows": n, "tsv_alignment_mismatches": mis, "tsv_mismatch_examples": mis_ex,
       "statmt": stat, "statmt_total_lines": int(sum(v["statmt_lines"] for v in stat.values())),
       "match_rate_corpus_form": float(hit.mean()),
       "match_rate_raw_line_exact": float(rawhit.mean()),
       "labels": {nm: int((lab == c).sum()) for c, nm in enumerate(names)} | {"unmatched": int((lab == UNM).sum())},
       "ambiguous_rows": int(ambig.sum()),
       "ambiguous_by_mask": {str(int(x)): int(c) for x, c in zip(*np.unique(m[ambig], return_counts=True))},
       "label_contiguity": None,
       "python": sys.version.split()[0]}
# is the corpus in HF subset order (europarl block, commoncrawl block, nc block)?
first = {nm: int(np.flatnonzero(lab == c)[0]) if (lab == c).any() else None for c, nm in enumerate(names)}
last = {nm: int(np.flatnonzero(lab == c)[-1]) if (lab == c).any() else None for c, nm in enumerate(names)}
rep["label_contiguity"] = {"first_row": first, "last_row": last}
np.savez(OUT / "rows.npz", score=score, label=lab, ambig_mask=m, pair_key=pk, src_key=sk, src_len=sl)
json.dump(rep, open(OUT / "s1_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
print(f"done {time.time()-t0:.0f}s")
