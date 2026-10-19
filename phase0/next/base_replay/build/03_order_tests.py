"""Step 3b: batch ORDER (Python's random.shuffle(batches)) and tests of it against the April training logs.

Python's global random is seeded by set_seed(42); before the first sampler iteration swanlab.init (cloud
mode, swanlab 0.7.15) builds 6 hardware collectors, and the constructors of four of them call
random_index() (8 x random.choice-style draws = 16 32-bit words each): NVIDIA 6, AMD 5, disk 1,
network 1 -> K_pred = 13 x 16 = 208 words, the same in both sessions (both metadata files: _monitor 6,
identical hardware). Nothing else in the main process draws from it before the sampler (checked:
torch DataLoader, multiprocessing, tensorboard, the trainer, eval). This script does NOT assume K: it
scores every K in 0..2048 words and reports where K_pred falls.

Tests (all computed for every K; K_pred is the pre-stated prediction):
  T1 window-loss test: the logged train loss is a token-weighted mean over each 100-step window. A
     content proxy, the token-weighted mean unigram surprisal of the window's target label tokens
     (tgt[1:], i.e. EOS kept, BOS dropped, as in the loss), is computed for each window under order K.
     Both series are high-passed (x_w minus the mean of its 10 neighbours w-5..w+5) and correlated.
     Separately for session 1 (windows 1-1000: P1, P2, first 1,649 batches of P3) and session 2
     (windows 1001-1222: P1 positions 0-22,199).
  T2 spike test: session 1's two spike-guard hits (steps 10,443 and 12,074 -> P1 positions 10,442 and
     12,073): the percentile, among all P1 batches, of those two batches' mean surprisal under K.
  T0 simulation-free replay test: session 1's windows 1-222 against session 2's windows 1-222 (the
     same P1 positions if session 2 replayed P1 in the same order), high-passed loss against loss;
     control: session 2 against session 1's windows 492-713 (P2 positions).
"""
import json, os, random, re, time
import numpy as np

B = os.path.expanduser("~/replay_check/build")
t0 = time.time()
comp = np.load(f"{B}/comp_np201.npz")
Bs = [len(comp[f"starts_{e}"]) - 1 for e in (1, 2, 3)]
LOG = {"end_epoch1_step": 49174, "end_epoch2_step": 98351, "s1_steps": 100000, "s2_steps": 22294,
       "spike_steps": [10443, 12074]}
assert Bs[0] == LOG["end_epoch1_step"] and Bs[0] + Bs[1] == LOG["end_epoch2_step"], Bs
P3_USED = LOG["s1_steps"] - LOG["end_epoch2_step"]

# ---- per-row content proxy: unigram surprisal of the target label tokens -----------------------
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
c = np.load(f"{D}/train.clean.en.cached_256.npz")
tt, to = c["tgt_tokens"], c["tgt_offsets"]
n = len(to) - 1
is_bos = np.zeros(len(tt), bool); is_bos[to[:-1]] = True        # position 0 of each row = BOS
cnt = np.bincount(tt[~is_bos], minlength=32000).astype(np.float64)
sur = -np.log(np.maximum(cnt, 1) / cnt.sum())
tok_s = sur[tt]; tok_s[is_bos] = 0.0
row_S = np.add.reduceat(tok_s, to[:-1])                          # sum over label tokens
row_N = (np.diff(to) - 1).astype(np.float64)                     # label tokens per row
del tok_s, is_bos, tt
print("row features", round(time.time() - t0, 1), "s", flush=True)

# per sorted-order batch sums, per epoch
bS, bN = [], []
for e in (1, 2, 3):
    idx, st = comp[f"idx_{e}"], comp[f"starts_{e}"]
    bS.append(np.add.reduceat(row_S[idx], st[:-1])); bN.append(np.add.reduceat(row_N[idx], st[:-1]))

# ---- logged losses --------------------------------------------------------------------------------
rep = open(f"{B}/logs/training_report.txt").read()
sec = rep.split("TRAINING LOSS HISTORY")[1].split("VALIDATION BLEU HISTORY")[0]
hist = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s*$", sec, re.M)}
steps = np.arange(100, 122201, 100)
assert all(s in hist for s in steps), "loss history incomplete"
L = np.array([hist[s] for s in steps])

def hp(x, k=5):
    """x_w minus the mean of its neighbours w-k..w+k (w excluded); NaN where the window is incomplete"""
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0], np.cumsum(x)])
    for w in range(k, len(x) - k):
        out[w] = x[w] - (cs[w + k + 1] - cs[w - k] - x[w]) / (2 * k)
    return out

def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())

# session-1 windows 1..1000 and session-2 windows 1..222 are high-passed separately (the resume is a jump)
L1, L2 = hp(L[:1000]), hp(L[1000:])
L1[:50] = np.nan          # steps < 5,000: warm-up, the loss trend's curvature swamps any content signal

def step_map():
    """(epoch, position) of session-1 steps 1..100000 and session-2 steps 100001..122200"""
    s = np.arange(1, 100001)
    ep = np.where(s <= Bs[0], 1, np.where(s <= Bs[0] + Bs[1], 2, 3))
    pos = np.where(ep == 1, s - 1, np.where(ep == 2, s - 1 - Bs[0], s - 1 - Bs[0] - Bs[1]))
    return ep, pos
EP1, POS1 = step_map()
POS2 = np.arange(0, 22200)

def perms(K):
    r = random.Random(42)                       # == random.seed(42) on the module's instance
    for _ in range(K):
        r.getrandbits(32)                       # one 32-bit word each, as random_index() consumes them
    out = []
    for b in Bs:
        p = list(range(b)); r.shuffle(p); out.append(np.asarray(p, np.int64))
    return out

def window_feature(pm):
    S1 = np.empty(100000); N1 = np.empty(100000)
    for e in (1, 2, 3):
        m = EP1 == e
        j = pm[e - 1][POS1[m]]
        S1[m] = bS[e - 1][j]; N1[m] = bN[e - 1][j]
    F1 = S1.reshape(1000, 100).sum(1) / N1.reshape(1000, 100).sum(1)
    j = pm[0][POS2]
    F2 = bS[0][j].reshape(222, 100).sum(1) / bN[0][j].reshape(222, 100).sum(1)
    return hp(F1), hp(F2)

p1_mean_s = bS[0] / bN[0]                         # per P1 batch mean surprisal
ranks = p1_mean_s.argsort().argsort() / (len(p1_mean_s) - 1)
K_PRED = 208
res = {"B": Bs, "log": LOG, "P3_batches_used": P3_USED, "K_pred": K_PRED, "per_K": []}
def score(K):
    pm = perms(K)
    F1, F2 = window_feature(pm)
    r1, n1 = corr(F1, L1); r2, n2 = corr(F2, L2)
    sp = [float(ranks[pm[0][s - 1]]) for s in LOG["spike_steps"]]
    return {"K": K, "r_s1": r1, "n_s1": n1, "r_s2": r2, "n_s2": n2, "spike_pct": sp}
import multiprocessing as mp
with mp.get_context("fork").Pool(12) as pool:
    res["per_K"] = pool.map(score, range(0, 2049), chunksize=16)
print("scored", round(time.time() - t0, 1), "s", flush=True)

# T0: simulation-free check that session 2 replayed P1's order
r0, n0 = corr(hp(L[:222]), L2)                  # S1 windows 1..222 (P1 0..22,199) vs S2 windows
rc, nc = corr(hp(L[491:713]), L2)               # S1 windows 492..713 (P2 positions) vs S2 (control)
res["T0"] = {"r_s1P1_vs_s2": r0, "n": n0, "r_s1P2_vs_s2_control": rc, "n_control": nc}

pk = res["per_K"]
r1 = np.array([d["r_s1"] for d in pk]); r2 = np.array([d["r_s2"] for d in pk])
sm = np.array([min(d["spike_pct"]) for d in pk])
def summ(arr, k):
    others = np.delete(arr, k)
    return {"value": float(arr[k]), "rank_among_2049_K": int((arr > arr[k]).sum()) + 1,
            "others_mean": float(others.mean()), "others_sd": float(others.std()),
            "others_max": float(others.max()), "z_vs_others": float((arr[k] - others.mean()) / others.std())}
res["summary_K_pred"] = {"T1_session1": summ(r1, K_PRED), "T1_session2": summ(r2, K_PRED),
                         "T2_min_spike_percentile": summ(sm, K_PRED),
                         "T2_spike_percentiles_at_K_pred": pk[K_PRED]["spike_pct"]}
res["best_K_session1"] = int(r1.argmax()); res["best_K_session2"] = int(r2.argmax())
res["best_K_spike"] = int(sm.argmax())
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{B}/03_order_tests.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_K"}, indent=1))
