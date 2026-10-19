"""Step 3c: is anything drawn from Python's random between the P1, P2 and P3 shuffles?

With K = 208 fixed for P1, the P2 order is regenerated with x extra words drawn after the P1 shuffle
(x = 0..2000), and the session-1 P2-window loss correlation (03's T1 statistic restricted to P2
windows) is computed for each x. x = 0 is the prediction (no draws between epochs). Same for P3 with
x2 extra words after the P2 shuffle (only 16 windows, weak).
"""
import json, os, random, re, time
import numpy as np
B = os.path.expanduser("~/replay_check/build")
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
t0 = time.time()
comp = np.load(f"{B}/comp_np201.npz")
Bs = [len(comp[f"starts_{e}"]) - 1 for e in (1, 2, 3)]
c = np.load(f"{D}/train.clean.en.cached_256.npz"); tt, to = c["tgt_tokens"], c["tgt_offsets"]
is_bos = np.zeros(len(tt), bool); is_bos[to[:-1]] = True
cnt = np.bincount(tt[~is_bos], minlength=32000).astype(np.float64)
tok_s = (-np.log(np.maximum(cnt, 1) / cnt.sum()))[tt]; tok_s[is_bos] = 0
row_S = np.add.reduceat(tok_s, to[:-1]); row_N = (np.diff(to) - 1).astype(np.float64)
bS = [np.add.reduceat(row_S[comp[f"idx_{e}"]], comp[f"starts_{e}"][:-1]) for e in (1, 2, 3)]
bN = [np.add.reduceat(row_N[comp[f"idx_{e}"]], comp[f"starts_{e}"][:-1]) for e in (1, 2, 3)]
rep = open(f"{B}/logs/training_report.txt").read().split("TRAINING LOSS HISTORY")[1].split("VALIDATION")[0]
hist = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s*$", rep, re.M)}
L = np.array([hist[s] for s in range(100, 100001, 100)])
def hp(x, k=5):
    out = np.full(len(x), np.nan); cs = np.concatenate([[0], np.cumsum(x)])
    for w in range(k, len(x) - k):
        out[w] = x[w] - (cs[w + k + 1] - cs[w - k] - x[w]) / (2 * k)
    return out
Lh = hp(L); Lh[:50] = np.nan
P3_USED = 100000 - Bs[0] - Bs[1]

def orders(x1, x2):
    r = random.Random(42)
    for _ in range(208): r.getrandbits(32)
    out = []
    for e, b in enumerate(Bs):
        if e == 1:
            for _ in range(x1): r.getrandbits(32)
        if e == 2:
            for _ in range(x2): r.getrandbits(32)
        p = list(range(b)); r.shuffle(p); out.append(np.asarray(p))
    return out

def stat(pm, segment):
    S = np.concatenate([bS[0][pm[0]], bS[1][pm[1]], bS[2][pm[2][:P3_USED]]])
    N = np.concatenate([bN[0][pm[0]], bN[1][pm[1]], bN[2][pm[2][:P3_USED]]])
    F = hp(S.reshape(1000, 100).sum(1) / N.reshape(1000, 100).sum(1))
    a, b = (Bs[0] // 100 + 1, (Bs[0] + Bs[1]) // 100) if segment == 2 else ((Bs[0] + Bs[1]) // 100 + 1, 1000)
    f, l = F[a:b], Lh[a:b]; m = ~(np.isnan(f) | np.isnan(l))
    return float(np.corrcoef(f[m], l[m])[0, 1])

def job(x, seg):
    return stat(orders(x, 0) if seg == 2 else orders(0, x), seg)
import multiprocessing as mp
with mp.get_context("fork").Pool(12) as pool:
    r2 = pool.starmap(job, [(x, 2) for x in range(0, 2001)], chunksize=16)
    r3 = pool.starmap(job, [(x, 3) for x in range(0, 2001)], chunksize=16)
r2 = np.array(r2); r3 = np.array(r3)
def summ(a):
    o = a[1:]
    return {"x0": float(a[0]), "rank_of_x0": int((a > a[0]).sum()) + 1, "others_mean": float(o.mean()),
            "others_sd": float(o.std()), "best_x": int(a.argmax()), "best": float(a.max())}
res = {"P2_extra_words": summ(r2), "P3_extra_words": summ(r3), "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{B}/04b_epoch_boundary_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
