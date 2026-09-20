"""E0.4: split ft_topk by within-set 8-gram repetitiveness, then length-match the halves.

The statistic and the rule are fixed in PROTOCOL.md 'E0.4'; this only executes them.
Usage: e04_split.py <controls-dir> <out-dir>

Implementation note (not part of the pre-registered rule): 8-grams are keyed by a
64-bit BLAKE2b digest, not by builtin hash(), so the split is reproducible across
processes -- builtin hash() is salted per process by PYTHONHASHSEED.
"""
import collections, hashlib, json, re, sys
from pathlib import Path

WORD = re.compile(r"\w+", re.UNICODE)


def key(gram):
    return hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()


src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

src = [ln.rstrip("\n") for ln in open(src_dir / "ft_topk.en", encoding="utf-8", newline="\n")]
tgt = [ln.rstrip("\n") for ln in open(src_dir / "ft_topk.fr", encoding="utf-8", newline="\n")]
assert len(src) == len(tgt), (len(src), len(tgt))
n = len(src)
print(f"ft_topk: {n:,} rows", flush=True)

toks = [[w.lower() for w in WORD.findall(s)] for s in src]
grams = [[key(" ".join(t[j:j + 8])) for j in range(len(t) - 7)] for t in toks]
counts = collections.Counter()
for g in grams:
    counts.update(g)
print(f"8-gram types: {len(counts):,}", flush=True)

rep = [sum(1 for c in (counts[k] for k in g) if c > 1) / len(g) if g else 0.0 for g in grams]
del grams

order = sorted(range(n), key=lambda i: rep[i])
med = rep[order[n // 2]]
div = [i for i in order if rep[i] <= med]
repet = [i for i in order if rep[i] > med]
print(f"median rep = {med:.4f}; diverse {len(div):,}  repetitive {len(repet):,}", flush=True)


def lens(idx):
    return [len(toks[i]) for i in idx]


def summ(idx):
    L = sorted(lens(idx))
    return sum(L) / len(L), L[len(L) // 2]


# stratified length matching (PROTOCOL E0.4, amended): equal counts per length bucket
import random
rng = random.Random(42)
by_len = {"div": {}, "rep": {}}
for name, idx in (("div", div), ("rep", repet)):
    for i in idx:
        by_len[name].setdefault(len(toks[i]), []).append(i)
kept = {"div": [], "rep": []}
for L in sorted(set(by_len["div"]) | set(by_len["rep"])):
    a, b = by_len["div"].get(L, []), by_len["rep"].get(L, [])
    k = min(len(a), len(b))
    if not k:
        continue
    kept["div"] += rng.sample(a, k)
    kept["rep"] += rng.sample(b, k)
div, repet = sorted(kept["div"]), sorted(kept["rep"])
k = len(div)
(am, amed), (bm, bmed) = summ(div), summ(repet)
print(f"after stratified matching: {k:,} rows each | diverse mean {am:.2f} median {amed} | "
      f"repetitive mean {bm:.2f} median {bmed}", flush=True)

meta = {"source_set": "ft_topk", "rows_each": k, "median_rep": med,
        "gram_key": "blake2b-64",
        "diverse": {"len_mean": am, "len_median": amed, "rep_mean": sum(rep[i] for i in div) / k},
        "repetitive": {"len_mean": bm, "len_median": bmed, "rep_mean": sum(rep[i] for i in repet) / k}}
written = []
for name, idx in (("ft_topk_div", div), ("ft_topk_rep", repet)):
    with open(out_dir / f"{name}.en", "w", encoding="utf-8", newline="\n") as a, \
         open(out_dir / f"{name}.fr", "w", encoding="utf-8", newline="\n") as b:
        for i in idx:
            a.write(src[i] + "\n")
            b.write(tgt[i] + "\n")
    written += [f"{name}.en", f"{name}.fr"]
for f in ("heldout_un.en", "heldout_un.fr"):
    (out_dir / f).write_bytes((src_dir / f).read_bytes())
    written.append(f)
written.append("manifest.json")
json.dump({**meta, "written": written, "heldout_sets": {"heldout_un": {"n": 2000}}},
          open(out_dir / "manifest.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
