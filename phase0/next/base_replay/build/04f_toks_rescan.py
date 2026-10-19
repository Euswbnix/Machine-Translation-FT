"""Step 3f: 04d's throughput statistic E re-scanned over K = 0..2048 for session 1's P1 windows, leaving out
steps 20,001-30,000 (04e: throughput there is disturbed, coefficient of variation 11.5% against 2-3%
elsewhere, so the window time no longer follows the batches); Spearman as well as Pearson."""
import json, os, sys
import numpy as np
B = os.path.expanduser("~/replay_check/build")
exec(open(f"{B}/04d_toks_test.py").read().split("import multiprocessing as mp")[0])
def spear(a, b):
    ra = a.argsort().argsort(); rb = b.argsort().argsort(); return float(np.corrcoef(ra, rb)[0, 1])
mm = ok1 & seg["P1"] & ~((ends1 > 20000) & (ends1 <= 30100))
def sK(K):
    W1 = windows_s1(perm_seq(K, 0, 0)); e = eff(W1)
    return corr(e[mm], y1[mm]), spear(e[mm], y1[mm])
import multiprocessing as mp
with mp.get_context("fork").Pool(12) as pool:
    r = np.array(pool.map(sK, range(0, 2049), chunksize=16))
out = {"n_windows": int(mm.sum())}
for j, nm in enumerate(("pearson", "spearman")):
    a = r[:, j]; o = np.delete(a, 208)
    top = np.argsort(-a)[:6]
    out[nm] = {"K208": float(a[208]), "rank": int((a > a[208]).sum()) + 1, "others_mean": float(o.mean()),
               "others_sd": float(o.std()), "top6": [[int(k), float(a[k])] for k in top]}
json.dump(out, open(f"{B}/04f_toks_rescan.json", "w"), indent=1); print(json.dumps(out, indent=1))
