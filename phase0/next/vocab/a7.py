from arms import *
div = np.where(T["in_ft_un_div"])[0]; R0 = np.load("R0.npy")
RU0 = np.where(Q["un"] & (Q["rep"] == 0))[0]; TD = np.where(T["un"] & (T["rep"] == 0))[0]
TDx = np.setdiff1d(TD, div)        # top-band UN rep=0 rows NOT already in ft_un_div
EDGES = [0, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000, 20000, 30000, 50000, 100000, 300000, 10**8]
def cellkey(A, i): return (int(A["len"][i]), int(np.searchsorted(EDGES, A["maxrank"][i], side="right")))
def joint_match(target_A, target_idx, pool_A, pool_idx, seed=42, frac=1.0):
    need = collections.Counter(cellkey(target_A, i) for i in target_idx)
    have = collections.defaultdict(list)
    for i in pool_idx: have[cellkey(pool_A, i)].append(i)
    need = {c: int(round(frac*n)) for c, n in need.items()}
    short = sum(max(0, n - len(have[c])) for c, n in need.items())
    ratios = sorted(len(have[c])/n for c, n in need.items() if n)
    if short: return None, ratios[0], short
    rng = random.Random(seed); out = []
    for c in sorted(need): out += rng.sample(have[c], need[c])
    return np.array(sorted(out)), ratios[0], 0
def max_frac(target_A, target_idx, pool_A, pool_idx):
    lo, hi = 0.0, 1.0
    for _ in range(14):
        m = (lo+hi)/2
        a, _, sh = joint_match(target_A, target_idx, pool_A, pool_idx, frac=m)
        lo, hi = (m, hi) if a is not None else (lo, m)
    return lo
print("bins:", EDGES)
for nm, A_, I in (("div", T, div), ("R0", Q, R0)):
    b = np.bincount(np.searchsorted(EDGES, A_["maxrank"][I], side="right"), minlength=len(EDGES)+1)
    print(nm, "bin shares %:", np.round(100*b/len(I), 2)[1:])
# A3: random UN rep=0 rows, joint (length x maxrank-bin) histogram of ft_un_div
a, s, sh = joint_match(T, div, Q, RU0)
print("A3 full size: short", sh)
f = max_frac(T, div, Q, RU0); print("A3 max feasible fraction", round(f, 4), "->", int(f*len(div)), "rows")
a, s, sh = joint_match(T, div, Q, RU0, frac=f)
report(f"A3 rand->div profile x{f:.3f}", "ft_random", a)
np.save("A3.npy", a)
# B3: top-band UN rep=0 rows, joint histogram of R0
for nm, pool in (("TD (all)", TD), ("TD minus ft_un_div", TDx)):
    a, s, sh = joint_match(Q, R0, T, pool)
    print(f"B3 from {nm} ({len(pool):,}) full size: short {sh:,}")
    f = max_frac(Q, R0, T, pool); print("  max feasible fraction", round(f, 4), "->", int(f*len(R0)), "rows")
    a, s, sh = joint_match(Q, R0, T, pool, frac=f)
    report(f"B3 top->R0 profile x{f:.3f}", "ft_topk", a)
    np.save(f"B3_{'all' if pool is TD else 'disjoint'}.npy", a)
