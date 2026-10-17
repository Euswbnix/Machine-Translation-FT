"""QE score (CometKiwi-22, v2_scored.tsv) for every row of ft_topk and ft_random, joined on the exact (en, fr) pair."""
import numpy as np, sys, os
S = os.path.expanduser("~/mt_local/phase0_final/e05_src")
keys = {}
sets = {}
for name in ("ft_topk", "ft_random"):
    en = open(f"{S}/{name}.en", "rb").read().split(b"\n")[:-1]
    fr = open(f"{S}/{name}.fr", "rb").read().split(b"\n")[:-1]
    assert len(en) == len(fr)
    sets[name] = np.full(len(en), np.nan)
    for i, (a, b) in enumerate(zip(en, fr)):
        keys.setdefault(a + b"\t" + b, []).append((name, i))
print("keys", len(keys), flush=True)
hits = 0; n = 0; multi = 0
with open(os.path.expanduser("~/mt_local/phase0_final/v2_scored.tsv"), "rb", buffering=1 << 24) as f:
    for ln in f:
        n += 1
        sc, _, rest = ln.partition(b"\t")
        v = keys.get(rest.rstrip(b"\n"))
        if v is not None:
            s = float(sc)
            for name, i in v:
                if not np.isnan(sets[name][i]) and sets[name][i] != s: multi += 1
                sets[name][i] = s if np.isnan(sets[name][i]) else max(sets[name][i], s)
            hits += 1
print("pool lines", n, "hits", hits, "conflicting duplicate scores", multi)
for name, a in sets.items():
    print(name, "matched", int((~np.isnan(a)).sum()), "of", len(a), "mean", np.nanmean(a), "min", np.nanmin(a), "max", np.nanmax(a))
np.savez("qe.npz", **sets)
