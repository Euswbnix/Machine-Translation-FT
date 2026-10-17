from arms import *
import heapq, bisect
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
hist = hist_of(T["len"], div)
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
def greedy_types(A, par, pidx, hist, maximize=True):
    """lazy greedy over the pool, exact length quotas: add rows with the most (maximize) / fewest new types"""
    t = tok(par, "en"); quota = dict(hist); seen = set(); out = []
    sign = -1 if maximize else 1
    h = [(sign*len(set(t[i])), int(i)) for i in pidx if int(A["len"][i]) in quota]; heapq.heapify(h)
    while h and len(out) < sum(hist.values()):
        g, i = heapq.heappop(h)
        L = int(A["len"][i])
        if quota.get(L, 0) == 0: continue
        new = len(set(t[i]) - seen)
        if h and sign*new > h[0][0]:      # stale: push back with updated gain
            heapq.heappush(h, (sign*new, i)); continue
        out.append(i); seen |= set(t[i]); quota[L] -= 1
    return np.array(sorted(out)), sum(quota.values())
A_nn = nn_match(T, div, Q, RU0); np.save("A_nn.npy", A_nn)
report("A_nn rand UN rep=0 ~div rarity", "ft_random", A_nn, f"AUC(maxrank A>div)={auc(Q['maxrank'][A_nn], T['maxrank'][div]):.3f} qeAUC(A>R0)={auc(Q['qe'][A_nn], Q['qe'][R0]):.3f}")
B_nn = nn_match(Q, R0, T, TD); np.save("B_nn.npy", B_nn)
report("B_nn top UN rep=0 ~R0 rarity", "ft_topk", B_nn, f"AUC(maxrank B>R0)={auc(T['maxrank'][B_nn], Q['maxrank'][R0]):.3f} overlap_div={len(np.intersect1d(B_nn, div)):,}")
Bmax, left = greedy_types(T, "ft_topk", TD, hist, True); np.save("B_greedy.npy", Bmax)
report("B_greedy max types (top)", "ft_topk", Bmax, f"unfilled={left} overlap_div={len(np.intersect1d(Bmax, div)):,}")
Amin, left = greedy_types(Q, "ft_random", RU0, hist, False); np.save("A_greedy.npy", Amin)
report("A_greedy min types (rand)", "ft_random", Amin, f"unfilled={left}")
