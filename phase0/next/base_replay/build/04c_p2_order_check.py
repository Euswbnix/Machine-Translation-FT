"""Step 3d: a sharper test of the P2 order (extra words x drawn between the P1 and P2 shuffles).

A multi-feature window-loss model is fitted on session 1's P1 windows (order validated by 03), then used
unchanged to predict the P2 windows under each candidate x = 0..2000; x = 0 is the prediction. Also
reports how many P2 positions the x = 0 and best-x permutations share.
Batch features: token-weighted mean target and source unigram surprisal, mean label tokens per row,
share of rows with <= 6 label tokens, rows in batch, batch max length. Window features are the window's
token-weighted means; all series high-passed as in 03.
"""
import json, os, random, re, time
import numpy as np
B = os.path.expanduser("~/replay_check/build"); D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
t0 = time.time()
comp = np.load(f"{B}/comp_np201.npz"); Bs = [len(comp[f"starts_{e}"]) - 1 for e in (1, 2, 3)]
c = np.load(f"{D}/train.clean.en.cached_256.npz")
def surprisal(tok, off):
    is_bos = np.zeros(len(tok), bool); is_bos[off[:-1]] = True
    cnt = np.bincount(tok[~is_bos], minlength=32000).astype(np.float64)
    s = (-np.log(np.maximum(cnt, 1) / cnt.sum()))[tok]; s[is_bos] = 0
    return np.add.reduceat(s, off[:-1]), (np.diff(off) - 1).astype(np.float64)
tS, tN = surprisal(c["tgt_tokens"], c["tgt_offsets"]); sS, sN = surprisal(c["src_tokens"], c["src_offsets"])
sl = np.load(f"{B}/src_lens.npy"); tl = np.load(f"{B}/tgt_lens.npy"); ml = np.maximum(sl, tl).astype(float)
short = (tN <= 6).astype(float)
def batch_feats(e):
    idx, st = comp[f"idx_{e}"], comp[f"starts_{e}"]
    R = lambda v: np.add.reduceat(v[idx], st[:-1])
    nrows = np.diff(st).astype(float)
    return {"tS": R(tS), "tN": R(tN), "sS": R(sS), "sN": R(sN), "short": R(short), "nrows": nrows,
            "maxlen": np.maximum.reduceat(ml[idx], st[:-1])}
BF = [batch_feats(e) for e in (1, 2, 3)]
rep = open(f"{B}/logs/training_report.txt").read().split("TRAINING LOSS HISTORY")[1].split("VALIDATION")[0]
hist = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s*$", rep, re.M)}
L = np.array([hist[s] for s in range(100, 100001, 100)])
def hp(x, k=5):
    out = np.full(len(x), np.nan); cs = np.concatenate([[0], np.cumsum(x)])
    for w in range(k, len(x) - k):
        out[w] = x[w] - (cs[w + k + 1] - cs[w - k] - x[w]) / (2 * k)
    return out
Lh = hp(L); Lh[:50] = np.nan
P3U = 100000 - Bs[0] - Bs[1]
def orders(x):
    r = random.Random(42)
    for _ in range(208): r.getrandbits(32)
    out = []
    for e, b in enumerate(Bs):
        if e == 1:
            for _ in range(x): r.getrandbits(32)
        p = list(range(b)); r.shuffle(p); out.append(np.asarray(p))
    return out
def window_X(pm):
    seq = {k: np.concatenate([BF[0][k][pm[0]], BF[1][k][pm[1]], BF[2][k][pm[2][:P3U]]]) for k in BF[0]}
    W = {k: v.reshape(1000, 100).sum(1) for k, v in seq.items()}
    X = np.stack([W["tS"] / W["tN"], W["sS"] / W["sN"], W["tN"] / W["nrows"], W["short"] / W["nrows"],
                  W["nrows"], seq["maxlen"].reshape(1000, 100).max(1)], 1)
    return np.stack([hp(X[:, j]) for j in range(X.shape[1])], 1)
w1 = Bs[0] // 100; w2 = (Bs[0] + Bs[1]) // 100
pm0 = orders(0); X0 = window_X(pm0)
m = ~np.isnan(Lh[:w1]) & ~np.isnan(X0[:w1]).any(1)
Xa = np.c_[np.ones(m.sum()), X0[:w1][m]]; beta, *_ = np.linalg.lstsq(Xa, Lh[:w1][m], rcond=None)
fit_r = float(np.corrcoef(Xa @ beta, Lh[:w1][m])[0, 1])
def p2_r(x):
    X = window_X(orders(x))[w1 + 1:w2]; l = Lh[w1 + 1:w2]
    mm = ~np.isnan(l) & ~np.isnan(X).any(1)
    return float(np.corrcoef(np.c_[np.ones(mm.sum()), X[mm]] @ beta, l[mm])[0, 1])
import multiprocessing as mp
with mp.get_context("fork").Pool(12) as pool:
    r = np.array(pool.map(p2_r, range(0, 2001), chunksize=16))
best = int(r.argmax()); pmb = orders(best)
res = {"fit_on_P1_windows_r": fit_r, "beta": beta.tolist(), "P2_r_x0": float(r[0]), "rank_of_x0": int((r > r[0]).sum()) + 1,
       "others_mean": float(r[1:].mean()), "others_sd": float(r[1:].std()), "best_x": best, "best_r": float(r[best]),
       "P2_positions_shared_x0_vs_best": float((pm0[1] == pmb[1]).mean()),
       "P2_positions_shared_x0_vs_x53": float((pm0[1] == orders(53)[1]).mean()),
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{B}/04c_p2_order_check.json", "w"), indent=1); print(json.dumps(res, indent=1))
