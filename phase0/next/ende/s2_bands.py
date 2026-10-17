#!/usr/bin/env python3
"""en-de step 2: QE distribution, candidate top bands, E0.4 split, E0.5 single-source
feasibility, vocabulary and duplicate descriptives. Definitions copied from
phase0/e03_build_controls.py (top-N then exact-pair dedup, first copy kept; ft_random =
RandomState(42).choice over the unique pool), phase0/e04_split.py (rep, median split,
stratified length matching with random.Random(42)) and phase0/e05_build.py (match_random,
single-source halves, lexicon). Read-only on the corpus; writes JSON to the scratch dir.
"""
import collections, hashlib, json, random, re, sys, time
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
BANDS = [int(x) for x in sys.argv[2].split(",")]
CORP = Path.home() / "mt_local/rebuild/ende_fixed"
NAMES = ["europarl", "commoncrawl", "news-commentary"]
WORD = re.compile(r"\w+", re.UNICODE)
SEED = 42
t0 = time.time()

z = np.load(OUT / "rows.npz")
score, lab, pk, sk, slen = z["score"], z["label"], z["pair_key"], z["src_key"], z["src_len"]
N = len(score)
with open(CORP / "train.clean.en", encoding="utf-8", newline="\n") as f:
    EN = [ln[:-1] for ln in f]
assert len(EN) == N
print(f"loaded {N:,} rows ({time.time()-t0:.0f}s)", flush=True)


def lname(c):
    return NAMES[c] if c < len(NAMES) else "unmatched"


def comp(idx):
    c = collections.Counter(lab[idx].tolist()) if len(idx) else {}
    return {lname(k): {"n": int(v), "share": round(v / len(idx), 4)} for k, v in sorted(c.items(), key=lambda x: -x[1])}


def q(x, ps=(0, 1, 5, 10, 25, 50, 75, 90, 94, 95, 99, 100)):
    return {f"p{p}": round(float(v), 4) for p, v in zip(ps, np.percentile(x, ps))}


def toks(s):
    return [w.lower() for w in WORD.findall(s)]


def gram_key(g):
    return hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()


def rep_within(tl):
    grams = [[gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in tl]
    cnt = collections.Counter()
    for g in grams:
        cnt.update(g)
    return [sum(1 for k in g if cnt[k] > 1) / len(g) if g else 0.0 for g in grams]


def lexicon(rows):
    tl = [toks(EN[i]) for i in rows]
    return {"source_words": sum(map(len, tl)), "source_word_types": len({w for t in tl for w in t}),
            "duplicate_source_fraction": round(1 - len({EN[i] for i in rows}) / len(rows), 5)}


def describe(rows, repmap=None):
    rows = list(rows)
    k = len(rows)
    ls = sorted(int(slen[i]) for i in rows)
    d = {"rows": k, "len_mean": round(sum(ls) / k, 3), "len_median": ls[k // 2],
         "qe_mean": round(float(score[rows].mean()), 4), "sources": comp(np.asarray(rows))}
    if repmap is not None:
        r = [repmap[i] for i in rows]
        d["rep_mean"] = round(sum(r) / k, 4)
        d["rep_positive_fraction"] = round(sum(1 for x in r if x > 0) / k, 4)
    d.update(lexicon(rows))
    return d


def stratified(a_idx, b_idx, seed=SEED):
    """E0.4 / E0.5 stratified matching: per length bucket min(n_a, n_b), random within the
    bucket. Lists keep the caller's order (it matters for rng.sample)."""
    by = {"a": {}, "b": {}}
    for nm, idx in (("a", a_idx), ("b", b_idx)):
        for i in idx:
            by[nm].setdefault(int(slen[i]), []).append(i)
    rng = random.Random(seed)
    ka, kb = [], []
    for L in sorted(set(by["a"]) | set(by["b"])):
        x, y = by["a"].get(L, []), by["b"].get(L, [])
        k = min(len(x), len(y))
        if k:
            ka += rng.sample(x, k)
            kb += rng.sample(y, k)
    return sorted(ka), sorted(kb)


def match_random(hist, eligible):
    buckets = {}
    for i in eligible:
        buckets.setdefault(int(slen[i]), []).append(i)
    short = {L: (n, len(buckets.get(L, []))) for L, n in hist.items() if len(buckets.get(L, [])) < n}
    slack = {L: len(buckets.get(L, [])) / n for L, n in hist.items()}
    tight = sorted(slack.items(), key=lambda x: x[1])[:3]
    res = {"buckets": len(hist), "short_buckets": len(short),
           "shortfall_rows": int(sum(n - h for n, h in short.values())),
           "tightest_bucket_slack": round(min(slack.values()), 3),
           "tightest_3": [{"len": L, "need": hist[L], "have": len(buckets.get(L, [])), "slack": round(s, 3)} for L, s in tight]}
    if short:
        return res, None
    rng = random.Random(SEED)
    kept = []
    for L in sorted(hist):
        kept += rng.sample(buckets[L], hist[L])
    return res, sorted(kept)


out = {"rows": N, "python": sys.version.split()[0]}

# ---------- QE distribution ----------
out["qe_overall"] = q(score)
out["qe_by_source"] = {lname(c): {"n": int((lab == c).sum()), **q(score[lab == c], (1, 10, 50, 90, 99)),
                                   "mean": round(float(score[lab == c].mean()), 4)} for c in range(3)}
out["corpus_sources"] = comp(np.arange(N))
_, first_all = np.unique(pk, return_index=True)
uniq = np.sort(first_all)
out["unique_pairs"] = int(len(uniq))
_, first_src = np.unique(sk, return_index=True)
out["unique_sources"] = int(len(first_src))
# QE histogram for the top tail
order = np.argsort(-score, kind="stable")
out["score_at_rank"] = {str(r): round(float(score[order[r - 1]]), 4)
                        for r in (1, 1000, 10000, 50000, 100000, 250000, 254596, 500000, 1000000, 2000000)}

# ---------- emulated ft_random (unique pool, RandomState(42)) per band size k ----------
bands_out = {}
for NB in BANDS:
    tb = time.time()
    top_sel = order[:NB]
    _, first = np.unique(pk[top_sel], return_index=True)
    band = top_sel[np.sort(first)]                       # score order, first copy kept
    k = len(band)
    B = {"n_selected": NB, "n_unique": k, "share_of_pool": round(NB / N, 4),
         "dups_removed": int(NB - k),
         "score_min": round(float(score[band].min()), 4), "score_max": round(float(score[band].max()), 4),
         "score_mean": round(float(score[band].mean()), 4),
         "sources": comp(band)}
    # tie at the boundary
    B["rows_at_boundary_score"] = int((score == score[order[NB - 1]]).sum())

    band_l = band.tolist()
    btoks = [toks(EN[i]) for i in band_l]
    rep = rep_within(btoks)
    repmap = dict(zip(band_l, rep))
    n = k
    srt = sorted(range(n), key=lambda j: rep[j])
    med = rep[srt[n // 2]]
    div_all = [band_l[j] for j in srt if rep[j] <= med]       # e04 order (rep-sorted, stable)
    rep_all = [band_l[j] for j in srt if rep[j] > med]
    B["rep"] = {"median": med, "mean": round(sum(rep) / n, 4),
                "share_rep_gt_0": round(sum(1 for x in rep if x > 0) / n, 4),
                "share_rep_gt_median": round(len(rep_all) / n, 4),
                "rep_by_source": {lname(c): {"n": int(cnt),
                                             "share_rep_gt_0": round(sum(1 for j in range(n) if lab[band_l[j]] == c and rep[j] > 0) / max(cnt, 1), 4)}
                                  for c, cnt in collections.Counter(lab[band].tolist()).items()}}
    B["halves_before_matching"] = {"diverse": {"rows": len(div_all), "sources": comp(np.asarray(div_all)),
                                               "len_mean": round(float(slen[div_all].mean()), 2)},
                                   "repetitive": {"rows": len(rep_all), "sources": comp(np.asarray(rep_all)),
                                                  "len_mean": round(float(slen[rep_all].mean()), 2) if rep_all else None}}
    # ---- all-source E0.4 halves ----
    d4, r4 = stratified(div_all, rep_all)
    B["e04_all_source"] = {"rows_each": len(d4)}
    sets = {}
    if d4:
        hist = collections.Counter(int(slen[i]) for i in d4)
        assert hist == collections.Counter(int(slen[i]) for i in r4)
        sets["topk_div"] = describe(d4, repmap)
        sets["topk_rep"] = describe(r4, repmap)
        # random arm: emulated ft_random of size k, as e05 did
        rnd = np.random.RandomState(SEED).choice(uniq, k, replace=False)
        rnd_l = rnd.tolist()
        rres, rk = match_random(hist, rnd_l)
        B["e04_all_source"]["random_from_ft_random_k"] = rres
        rres2, _ = match_random(hist, uniq.tolist())
        B["e04_all_source"]["random_from_whole_unique_pool"] = rres2
        rnd_rep = dict(zip(rnd_l, rep_within([toks(EN[i]) for i in rnd_l])))
        if rk:
            sets["random_small"] = describe(rk, rnd_rep)
            sets["random_small"]["overlap_with_band_rows"] = int(len(set(rk) & set(band_l)))
        B["ft_random_k_sources"] = comp(rnd)
    # ---- single-source halves, every source ----
    inband = np.zeros(N, bool); inband[band] = True
    B["single_source"] = {}
    for c in range(3):
        sd = [band_l[j] for j in range(n) if lab[band_l[j]] == c and rep[j] <= med]   # e05 order: ascending
        sr = [band_l[j] for j in range(n) if lab[band_l[j]] == c and rep[j] > med]
        kd, kr = stratified(sd, sr)
        S = {"band_rows": int((lab[band] == c).sum()), "div_before": len(sd), "rep_before": len(sr),
             "rows_each": len(kd)}
        if kd:
            h = collections.Counter(int(slen[i]) for i in kd)
            elig_rnd = rnd[lab[rnd] == c].tolist()
            elig_pool = uniq[lab[uniq] == c].tolist()
            elig_pool_nb = uniq[(lab[uniq] == c) & ~inband[uniq]].tolist()
            r1, k1 = match_random(h, elig_rnd)
            r2, k2 = match_random(h, elig_pool)
            r3, k3 = match_random(h, elig_pool_nb)
            S["random_from_ft_random_k_same_source"] = r1
            S["random_from_whole_pool_same_source"] = r2
            S["random_from_pool_same_source_excluding_band"] = r3
            S["sets"] = {"div": describe(kd, repmap), "rep": describe(kr, repmap)}
            if k1:
                S["sets"]["random_ft_random"] = describe(k1, rnd_rep)
            if k2:
                S["sets"]["random_whole_pool"] = describe(k2)
                S["sets"]["random_whole_pool"]["overlap_with_band_rows"] = int(sum(1 for i in k2 if inband[i]))
            if k3:
                S["sets"]["random_pool_excl_band"] = describe(k3)
            S["_idx"] = {"div": kd, "rep": kr, "rnd_pool": k2, "rnd_pool_excl_band": k3, "rnd_ftr": k1}
        B["single_source"][lname(c)] = S
    B["sets_all_source"] = sets
    B["_idx"] = {"band": band_l, "div": d4, "rep": r4}
    bands_out[str(NB)] = B
    print(f"band {NB:,}: k={k:,} med={med} rows_each={len(d4):,} ({time.time()-tb:.0f}s)", flush=True)

# strip indices into a separate file
idx = {b: {"all": v.pop("_idx"), "single": {s: vv.pop("_idx", None) for s, vv in v["single_source"].items()}}
       for b, v in bands_out.items()}
out["bands"] = bands_out
json.dump(out, open(OUT / f"s2_report_{'_'.join(map(str, BANDS))}.json", "w"), indent=1)
json.dump(idx, open(OUT / f"s2_idx_{'_'.join(map(str, BANDS))}.json", "w"))
print(f"done {time.time()-t0:.0f}s")
