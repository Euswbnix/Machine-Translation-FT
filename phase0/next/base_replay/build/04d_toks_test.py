"""Step 3e: a stronger order test from the logged throughput (Tok/s) of each 100-step window.

Tok/s = (non-pad target label tokens in the window) / (wall time of the window). The numerator is fixed
by which batches the window holds, and the wall time grows with the padded size of those batches, so
throughput is a near-deterministic function of the batch order. Windows whose time includes a
checkpoint save (the window after every multiple of 500 steps), an evaluation, or an epoch change are
excluded. Statistics, each over candidate orders:
  E   (no fitting): corr(log Tok/s, log efficiency) with efficiency = label tokens / padded src+tgt slots;
  M   (fitted on session 1's P1 windows at K = 208, then frozen): Tok/s ~ label tokens, padded src and
      tgt slots, attention-size term, rows (OLS), correlation of prediction with log Tok/s elsewhere.
Candidates: K = 0..2048 words before the P1 shuffle (P1 windows of session 1, and session 2);
x = 0..2000 extra words between the P1 and P2 shuffles (P2 windows); x3 = 0..2000 between P2 and P3.
"""
import json, os, random, re, time
import numpy as np
B = os.path.expanduser("~/replay_check/build")
t0 = time.time()
comp = np.load(f"{B}/comp_np201.npz"); Bs = [len(comp[f"starts_{e}"]) - 1 for e in (1, 2, 3)]
sl = np.load(f"{B}/src_lens.npy").astype(np.int64); tl = np.load(f"{B}/tgt_lens.npy").astype(np.int64)
def bfeat(e):
    idx, st = comp[f"idx_{e}"], comp[f"starts_{e}"]
    r = np.diff(st).astype(float)
    S = np.maximum.reduceat(sl[idx], st[:-1]).astype(float); Tm = np.maximum.reduceat(tl[idx], st[:-1]).astype(float) - 1
    N = np.add.reduceat(tl[idx] - 1, st[:-1]).astype(float)
    return np.stack([N, r * S, r * Tm, r * (S * S + Tm * Tm + S * Tm), r], 1)   # per sorted batch
BF = [bfeat(e) for e in (1, 2, 3)]

def parse(path):
    txt = open(path, "rb").read().decode("utf-8", "replace")
    toks = {int(s): int(v) for s, v in re.findall(r"Step\s+(\d+) \| Loss [\d.]+ \| LR [\d.e+-]+ \| Tok/s (\d+)", txt)}
    evals = {int(s) for s in re.findall(r"Step\s+(\d+) \| Valid BLEU", txt)}
    return toks, evals
t1, e1 = parse(f"{B}/logs/s1_backup.swanlab"); t2, e2 = parse(f"{B}/logs/s2_backup.swanlab")
P3U = 100000 - Bs[0] - Bs[1]

def clean(step_end, evals, starts_of_epochs, first):
    a = step_end - 99
    if step_end == first: return False
    if (a - 1) % 500 == 0: return False                  # previous step saved emergency.pt / step_*.pt
    if (a - 1) in evals: return False                    # evaluation ran after the previous log
    if any(a <= s0 <= step_end for s0 in starts_of_epochs): return False
    return True

def perm_seq(K, x2, x3):
    r = random.Random(42)
    for _ in range(K): r.getrandbits(32)
    out = []
    for e, b in enumerate(Bs):
        for _ in range((0, x2, x3)[e]): r.getrandbits(32)
        p = list(range(b)); r.shuffle(p); out.append(np.asarray(p))
    return out

def windows_s1(pm):
    F = np.concatenate([BF[0][pm[0]], BF[1][pm[1]], BF[2][pm[2][:P3U]]])     # 100,000 x 5
    return F.reshape(1000, 100, 5).sum(1)
def windows_s2(pm):
    return BF[0][pm[0][:22200]].reshape(222, 100, 5).sum(1)

ends1 = np.arange(100, 100001, 100); ends2 = np.arange(100100, 122201, 100)
ok1 = np.array([s in t1 and clean(s, e1, [1, Bs[0] + 1, Bs[0] + Bs[1] + 1], 100) for s in ends1])
ok2 = np.array([s in t2 and clean(s, e2, [100001], 100100) for s in ends2])
y1 = np.array([np.log(t1.get(s, 1)) for s in ends1]); y2 = np.array([np.log(t2.get(s, 1)) for s in ends2])
seg = {"P1": (ends1 <= Bs[0]) & (ends1 >= 5000), "P2": (ends1 - 99 > Bs[0]) & (ends1 <= Bs[0] + Bs[1]),
       "P3": ends1 - 99 > Bs[0] + Bs[1]}
def eff(W): return np.log(W[:, 0] / (W[:, 1] + W[:, 2]))
def design(W): return np.c_[np.ones(len(W)), W[:, 0], W[:, 1], W[:, 2], W[:, 3], W[:, 4]]
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])

W0 = windows_s1(perm_seq(208, 0, 0)); m = ok1 & seg["P1"]
X = design(W0[m]); tsec = np.exp(-y1[m]) * W0[m, 0]               # implied seconds per window at K=208
beta, *_ = np.linalg.lstsq(X, tsec, rcond=None)                    # time model fitted on P1 only
def pred_logtoks(W): return np.log(W[:, 0] / np.maximum(design(W) @ beta, 1e-6))

def score_K(K):
    pm = perm_seq(K, 0, 0); W1 = windows_s1(pm); W2 = windows_s2(pm); mm = ok1 & seg["P1"]
    return (corr(eff(W1[mm]), y1[mm]), corr(pred_logtoks(W1[mm]), y1[mm]),
            corr(eff(W2[ok2]), y2[ok2]), corr(pred_logtoks(W2[ok2]), y2[ok2]))
def score_x(x):
    W1 = windows_s1(perm_seq(208, x, 0)); mm = ok1 & seg["P2"]
    return corr(eff(W1[mm]), y1[mm]), corr(pred_logtoks(W1[mm]), y1[mm])
def score_x3(x):
    W1 = windows_s1(perm_seq(208, 0, x)); mm = ok1 & seg["P3"]
    return corr(eff(W1[mm]), y1[mm]), corr(pred_logtoks(W1[mm]), y1[mm])
import multiprocessing as mp
with mp.get_context("fork").Pool(12) as pool:
    rK = np.array(pool.map(score_K, range(0, 2049), chunksize=16))
    rx = np.array(pool.map(score_x, range(0, 2001), chunksize=16))
    rx3 = np.array(pool.map(score_x3, range(0, 2001), chunksize=16))
def summ(a, k):
    o = np.delete(a, k)
    return {"value": float(a[k]), "rank": int((a > a[k]).sum()) + 1, "others_mean": float(o.mean()),
            "others_sd": float(o.std()), "best": int(a.argmax()), "best_value": float(a.max())}
res = {"clean_windows": {"s1_P1": int((ok1 & seg["P1"]).sum()), "s1_P2": int((ok1 & seg["P2"]).sum()),
                         "s1_P3": int((ok1 & seg["P3"]).sum()), "s2": int(ok2.sum())},
       "toks_lines": {"s1": len(t1), "s2": len(t2)}, "time_model_beta": beta.tolist(),
       "P1_K208_E": summ(rK[:, 0], 208), "P1_K208_M_in_sample": summ(rK[:, 1], 208),
       "S2_K208_E": summ(rK[:, 2], 208), "S2_K208_M": summ(rK[:, 3], 208),
       "P2_x0_E": summ(rx[:, 0], 0), "P2_x0_M": summ(rx[:, 1], 0),
       "P3_x0_E": summ(rx3[:, 0], 0), "P3_x0_M": summ(rx3[:, 1], 0),
       "P2_x53": {"E": float(rx[53, 0]), "M": float(rx[53, 1])},
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{B}/04d_toks_test.json", "w"), indent=1); print(json.dumps(res, indent=1))
