from arms import *
import heapq, bisect
def nn_match(tA, tidx, pA, pidx, key="maxrank", seed=42, n_target=None):
    """exact length histogram; within each length, each target row takes the unused pool row with the nearest log(key)"""
    rng = random.Random(seed)
    tb = collections.defaultdict(list); pb = collections.defaultdict(list)
    for i in tidx: tb[int(tA["len"][i])].append(float(np.log1p(tA[key][i])))
    for i in pidx: pb[int(pA["len"][i])].append((float(np.log1p(pA[key][i])), rng.random(), i))
    out = []
    for L, vals in tb.items():
        pool = sorted(pb[L]); assert len(pool) >= len(vals), (L, len(pool), len(vals))
        keys = [p[0] for p in pool]; used = [False]*len(pool)
        # process targets from extremes inward so tails get first pick
        med = np.median(keys) if keys else 0
        for v in sorted(vals, key=lambda v: -abs(v - med)):
            j = bisect.bisect_left(keys, v); lo, hi = j-1, j
            while lo >= 0 and used[lo]: lo -= 1
            while hi < len(pool) and used[hi]: hi += 1
            c = [x for x in (lo, hi) if 0 <= x < len(pool)]
            k = min(c, key=lambda x: abs(keys[x]-v)); used[k] = True; out.append(pool[k][2])
    return np.array(sorted(out))
