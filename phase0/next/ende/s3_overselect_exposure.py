#!/usr/bin/env python3
"""en-de step 3.
A. Does the QE top band over-select repetitive text? rep (e04 statistic) computed within the
   band vs within same-size comparison sets (uniform random, length-matched random, bottom),
   all sources and inside Europarl.
B. Pretraining exposure alternatives: leave-out base (pretrain on corpus minus the FT arms),
   residual exact-source and 8-gram exposure per arm; random 50/50 split feasibility.
"""
import collections, hashlib, json, random, re, sys, time
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])
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
IDX = json.load(open(OUT / "s2_idx_254596_250000_500000_1000000.json"))
print(f"loaded ({time.time()-t0:.0f}s)", flush=True)


def toks(s):
    return [w.lower() for w in WORD.findall(s)]


def gram_key(g):
    return hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()


def rep_within(rows):
    tl = [toks(EN[i]) for i in rows]
    grams = [[gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in tl]
    cnt = collections.Counter()
    for g in grams:
        cnt.update(g)
    rep = [sum(1 for k in g if cnt[k] > 1) / len(g) if g else 0.0 for g in grams]
    tok_rep = sum(sum(1 for k in g if cnt[k] > 1) for g in grams) / max(1, sum(map(len, grams)))
    ge8 = [r for r, t in zip(rep, tl) if len(t) >= 8]
    return {"rows": len(rows), "rep_gt0_share": round(sum(1 for r in rep if r > 0) / len(rep), 4),
            "rep_gt0_share_rows_ge8_tokens": round(sum(1 for r in ge8 if r > 0) / max(1, len(ge8)), 4),
            "rows_lt8_tokens_share": round(1 - len(ge8) / len(rep), 4),
            "rep_mean": round(sum(rep) / len(rep), 4), "repeated_8gram_token_share": round(tok_rep, 4),
            "len_mean": round(float(np.mean(slen[rows])), 2),
            "dup_source_fraction": round(1 - len({EN[i] for i in rows}) / len(rows), 5),
            "word_types": len({w for t in tl for w in t}), "words": sum(map(len, tl))}


def hist_match(hist, eligible, seed=SEED):
    b = {}
    for i in eligible:
        b.setdefault(int(slen[i]), []).append(i)
    short = sum(max(0, n - len(b.get(L, []))) for L, n in hist.items())
    if short:
        return None, short
    rng = random.Random(seed)
    kept = []
    for L in sorted(hist):
        kept += rng.sample(b[L], hist[L])
    return sorted(kept), 0


_, fa = np.unique(pk, return_index=True)
uniq = np.sort(fa)
order_u = uniq[np.argsort(-score[uniq], kind="stable")]
res = {"A_overselection": {}, "B_exposure": {}, "C_split": {}}

# ---------------- A ----------------
for b in ("254596", "1000000"):
    band = IDX[b]["all"]["band"]
    k = len(band)
    A = {}
    A["band"] = rep_within(band)
    rnd = np.random.RandomState(SEED).choice(uniq, k, replace=False).tolist()
    A["uniform_random_same_size"] = rep_within(rnd)
    h = collections.Counter(int(slen[i]) for i in band)
    lm, short = hist_match(h, uniq.tolist())
    A["random_same_length_histogram"] = rep_within(lm) if lm else {"short": short}
    A["bottom_same_size"] = rep_within(order_u[-k:].tolist())
    ep_band = [i for i in band if lab[i] == 0]
    hep = collections.Counter(int(slen[i]) for i in ep_band)
    ep_pool = uniq[lab[uniq] == 0].tolist()
    lm_ep, short = hist_match(hep, ep_pool)
    A["europarl_band_rows_only"] = rep_within(ep_band)
    A["europarl_random_same_length_histogram"] = rep_within(lm_ep) if lm_ep else {"short": short}
    cc_band = [i for i in band if lab[i] == 1]
    hcc = collections.Counter(int(slen[i]) for i in cc_band)
    lm_cc, short = hist_match(hcc, uniq[lab[uniq] == 1].tolist())
    A["commoncrawl_band_rows_only"] = rep_within(cc_band)
    A["commoncrawl_random_same_length_histogram"] = rep_within(lm_cc) if lm_cc else {"short": short}
    res["A_overselection"][b] = A
    print(f"A {b} done ({time.time()-t0:.0f}s)", flush=True)

# ---------------- B: 8-gram index over the whole corpus ----------------
offs = np.zeros(N + 1, np.int64)
chunks = []
for i, s in enumerate(EN):
    t = [w.lower() for w in WORD.findall(s)]
    g = [hash(" ".join(t[j:j + 8])) for j in range(len(t) - 7)]
    offs[i + 1] = offs[i] + len(g)
    chunks.append(g)
G = np.fromiter((x for g in chunks for x in g), np.int64, int(offs[-1]))
del chunks
print(f"8-grams indexed: {len(G):,} ({time.time()-t0:.0f}s)", flush=True)
row_of = np.repeat(np.arange(N), np.diff(offs))

designs = {}
for b in ("254596", "1000000"):
    S = IDX[b]["single"]["europarl"]
    A_ = IDX[b]["all"]
    designs[b] = {"ep_div": S["div"], "ep_rep": S["rep"], "ep_random_ftr": S["rnd_ftr"],
                  "ep_random_pool": S["rnd_pool"], "all_div": A_["div"], "all_rep": A_["rep"]}
for b, arms in designs.items():
    excl_pairs = np.unique(pk[np.concatenate([np.asarray(v) for v in arms.values()])])
    in_arm = np.isin(pk, excl_pairs)
    P = ~in_arm
    B = {"arm_rows": {a: len(v) for a, v in arms.items()},
         "rows_removed_from_pretraining_exact_pair": int(in_arm.sum()),
         "pretraining_rows_left": int(P.sum()), "share_removed": round(float(in_arm.mean()), 4),
         "removed_by_source": {NAMES[c]: int((in_arm & (lab == c)).sum()) for c in range(3)},
         "removed_share_of_each_source": {NAMES[c]: round(float((in_arm & (lab == c)).sum() / (lab == c).sum()), 4) for c in range(3)}}
    psrc = np.unique(sk[P])
    pg = np.unique(G[P[row_of]])
    B["per_arm"] = {}
    for a, v in arms.items():
        v = np.asarray(v)
        src_left = float(np.isin(sk[v], psrc).mean())
        sel = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in v])
        gg = G[sel]
        B["per_arm"][a] = {"rows_whose_exact_source_remains_in_pretraining": round(src_left, 4),
                           "8gram_tokens_seen_in_pretraining_leaveout": round(float(np.isin(gg, pg).mean()), 4),
                           "8gram_tokens": int(len(gg))}
    res["B_exposure"][b] = B
    print(f"B {b} done ({time.time()-t0:.0f}s)", flush=True)

# ---------------- C: random 50/50 split ----------------
perm = np.random.RandomState(SEED).permutation(uniq)
F = np.sort(perm[: len(perm) // 2])
oF = F[np.argsort(-score[F], kind="stable")]
for share, nm in ((1_000_000 / 16_646_992, "6pct_of_F"), (1_000_000 / N, "23.6pct_of_F")):
    kb = int(round(share * len(F)))
    band = sorted(oF[:kb].tolist())
    tl = [toks(EN[i]) for i in band]
    grams = [[gram_key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in tl]
    cnt = collections.Counter()
    for g in grams:
        cnt.update(g)
    rep = [sum(1 for k in g if cnt[k] > 1) / len(g) if g else 0.0 for g in grams]
    out = {"F_rows": int(len(F)), "band_rows": kb, "score_min": round(float(score[oF[kb - 1]]), 4),
           "rep_gt0_share": round(sum(1 for r in rep if r > 0) / kb, 4)}
    for c, cn in ((None, "all"), (0, "europarl")):
        d = [band[j] for j in range(kb) if rep[j] <= 0 and (c is None or lab[band[j]] == c)]
        r = [band[j] for j in range(kb) if rep[j] > 0 and (c is None or lab[band[j]] == c)]
        bd, br = collections.Counter(int(slen[i]) for i in d), collections.Counter(int(slen[i]) for i in r)
        out[f"{cn}_rows_each_after_matching"] = int(sum(min(bd[L], br[L]) for L in set(bd) | set(br)))
    res["C_split"][nm] = out
    print(f"C {nm} done ({time.time()-t0:.0f}s)", flush=True)

json.dump(res, open(OUT / "s3_report.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print(f"done {time.time()-t0:.0f}s")
