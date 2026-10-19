"""Auditor check 2: independent re-implementation of the April sampler (bde629a TokenBatchSampler semantics),
Python's random advanced by swanlab 0.7.15's own get_metadata() (HOME redirected into this folder), then
three epochs; compare with the builder's masks. Also: stable-sort variant (tie-order sensitivity), and a
simulation-free Tok/s test session 1 vs session 2."""
import json, os, random, re, sys, time
H = os.path.expanduser("~"); V = f"{H}/replay_check/verify_method"
os.environ["HOME"] = f"{V}/home"
import numpy as np
D = f"{H}/workspace/Machine_translation/data_enfr_v1"; B = f"{H}/replay_check/build"
SW = f"{H}/workspace/Machine_translation/swanlog"
t0 = time.time(); res = {"numpy": np.__version__}
c = np.load(f"{D}/train.clean.en.cached_256.npz")
sl = np.diff(c["src_offsets"]).astype(np.int32); tl = np.diff(c["tgt_offsets"]).astype(np.int32); n = len(sl)

import swanlab  # noqa: F401  (train.py imports it before set_seed)
from swanlab.data.run.metadata import get_metadata
random.seed(42); np.random.seed(42)
md, mon = get_metadata()
res["swanlab_monitor"] = md["swanlab"]["_monitor"]; res["mt_index_after_swanlab"] = random.getstate()[1][-1]
r = random.Random(42)
for _ in range(208): r.getrandbits(32)
res["state_equals_208_words"] = r.getstate() == random.getstate()

def compose(kind=None):
    idx = np.arange(n)
    np.random.shuffle(idx)
    cs = 192 * 100
    for i in range(0, n, cs):
        ch = idx[i:i + cs]
        idx[i:i + cs] = ch[np.argsort(sl[ch], kind=kind)] if kind else ch[np.argsort(sl[ch])]
    seq = np.maximum(sl[idx], tl[idx]).tolist()
    starts = [0]; cnt = 0; mx = 0
    for j, s in enumerate(seq):
        nm = mx if mx > s else s
        if (cnt + 1) * nm > 24576 or cnt >= 192:
            starts.append(j); cnt = 1; mx = s
        else:
            cnt += 1; mx = nm
    starts.append(n)
    return idx, np.asarray(starts, np.int64)

E = []
for e in (1, 2, 3):
    idx, st = compose()
    p = list(range(len(st) - 1)); random.shuffle(p)
    E.append((idx, st, np.asarray(p, np.int64)))
res["B"] = [len(x[1]) - 1 for x in E]
res["log_epoch_ends_match"] = res["B"][0] == 49174 and res["B"][0] + res["B"][1] == 98351
P3U = 100000 - res["B"][0] - res["B"][1]; res["P3_used"] = P3U
def rows_at(e, k):
    idx, st, p = E[e]
    return np.concatenate([idx[st[j]:st[j + 1]] for j in p[:k]])
rep = np.zeros(n, bool); rep[rows_at(0, 22294)] = True
p3 = np.zeros(n, bool); p3[rows_at(2, P3U)] = True
expo = (2 + rep.astype(np.uint8) + p3.astype(np.uint8)).astype(np.uint8)
res["replayed_rows"] = int(rep.sum()); res["p3_rows"] = int(p3.sum())
res["rows_by_exposure"] = {int(k): int(v) for k, v in zip(*np.unique(expo, return_counts=True))}
b_rep = np.load(f"{B}/replayed_row_mask.npy"); i2l = np.load(f"{B}/index_to_line.npy")
res["replayed_mask_equal_builder"] = bool(np.array_equal(rep, b_rep))
res["p3_mask_equal_builder"] = bool(np.array_equal(p3, np.load(f"{B}/p3prefix_line_mask.npy")[i2l]))
res["exposure_equal_builder"] = bool(np.array_equal(expo, np.load(f"{B}/exposure_count_by_line.npy")[i2l]))
np.save(f"{V}/my_replayed_row.npy", rep); np.save(f"{V}/my_p3_row.npy", p3)
# P1 sorted-batch id per row, for re-draws later
idx1, st1, p1 = E[0]
bid1 = np.empty(n, np.int64); bid1[idx1] = np.repeat(np.arange(len(st1) - 1), np.diff(st1))
np.save(f"{V}/my_bid1_row.npy", bid1)
idx3, st3, _ = E[2]
bid3 = np.empty(n, np.int64); bid3[idx3] = np.repeat(np.arange(len(st3) - 1), np.diff(st3))
np.save(f"{V}/my_bid3_row.npy", bid3)
res["t_main_s"] = round(time.time() - t0, 1)

# ---- simulation-free order test: logged Tok/s, session 1 windows vs session 2 windows (same P1 positions) ----
def parse(path):
    txt = open(path, "rb").read().decode("utf-8", "replace")
    toks = {int(s): int(v) for s, v in re.findall(r"Step\s+(\d+) \| Loss [\d.]+ \| LR [\d.e+-]+ \| Tok/s (\d+)", txt)}
    ev = {int(s) for s in re.findall(r"Step\s+(\d+) \| Valid BLEU", txt)}
    return toks, ev
t1, e1 = parse(f"{SW}/run-20260424_000243-n00ar93oggh8om85j3o5f/backup.swanlab")
t2, e2 = parse(f"{SW}/run-20260424_021306-2fo9021j4lyiq27t2jghh/backup.swanlab")
res["toks_lines"] = [len(t1), len(t2)]; res["evals"] = [sorted(e1)[:50], sorted(e2)]
def rank(x): return np.argsort(np.argsort(x)).astype(float)
def corrs(a, b): return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(rank(a), rank(b))[0, 1]), "n": int(len(a))}
def clean(w, off, ev):          # window w covers steps off+100(w-1)+1 .. off+100w
    a = off + 100 * (w - 1)       # the step before the window
    return w > 1 and a % 500 != 0 and a not in ev
def pair(ws, off1):
    x, y = [], []
    for w in ws:
        s1, s2 = off1 + 100 * w, 100000 + 100 * w
        if s1 in t1 and s2 in t2 and clean(w, off1, e1) and clean(w, 100000, e2):
            x.append(np.log(t1[s1])); y.append(np.log(t2[s2]))
    return np.array(x), np.array(y)
out = {}
x, y = pair(range(1, 223), 0); out["s1P1_vs_s2_all_w2_222"] = corrs(x, y)
x, y = pair(range(1, 200), 0); out["s1P1_vs_s2_w2_199_(steps<20k)"] = corrs(x, y)
off = (res["B"][0] // 100 + 1) * 100            # control: session-1 windows inside P2 (other batches)
x, y = pair(range(1, 200), off); out["control_s1P2_vs_s2"] = corrs(x, y)
# model-based on session 2 with MY reconstruction: window efficiency = label tokens / padded slots
idx, st, p = E[0]
S = np.maximum.reduceat(sl[idx], st[:-1]).astype(float); T = np.maximum.reduceat(tl[idx], st[:-1]).astype(float) - 1
R = np.diff(st).astype(float); N = np.add.reduceat((tl[idx] - 1).astype(float), st[:-1])
seqN, seqP = N[p[:22200]].reshape(222, 100).sum(1), (R * (S + T))[p[:22200]].reshape(222, 100).sum(1)
effw = np.log(seqN / seqP)
ws = [w for w in range(1, 223) if 100000 + 100 * w in t2 and clean(w, 100000, e2)]
out["s2_logtoks_vs_my_efficiency"] = corrs(effw[np.array(ws) - 1], np.array([np.log(t2[100000 + 100 * w]) for w in ws]))
# same for session 1 P1 windows w<200 and 300<=w<=491 (steps 20k-30k excluded, as the builder did)
seqN1, seqP1 = N[p].astype(float), (R * (S + T))[p]
nw = len(p) // 100
e1w = np.log(seqN1[:nw * 100].reshape(nw, 100).sum(1) / seqP1[:nw * 100].reshape(nw, 100).sum(1))
ws1 = [w for w in list(range(50, 200)) + list(range(301, nw + 1)) if 100 * w in t1 and clean(w, 0, e1)]
out["s1P1_logtoks_vs_my_efficiency_excl_20k_30k"] = corrs(e1w[np.array(ws1) - 1], np.array([np.log(t1[100 * w]) for w in ws1]))
ws1b = [w for w in range(50, nw + 1) if 100 * w in t1 and clean(w, 0, e1)]
out["s1P1_logtoks_vs_my_efficiency_all_w50+"] = corrs(e1w[np.array(ws1b) - 1], np.array([np.log(t1[100 * w]) for w in ws1b]))
res["toks_tests"] = out

# ---- tie-order sensitivity: same streams, stable argsort ----
random.seed(42); np.random.seed(42); get_metadata()
Es = []
for e in (1, 2, 3):
    idx, st = compose("stable"); p = list(range(len(st) - 1)); random.shuffle(p); Es.append((idx, st, np.asarray(p)))
res["B_stable_sort"] = [len(x[1]) - 1 for x in Es]
idx, st, p = Es[0]; rep_s = np.zeros(n, bool); rep_s[np.concatenate([idx[st[j]:st[j + 1]] for j in p[:22294]])] = True
res["stable_sort_replayed_rows"] = int(rep_s.sum()); res["stable_sort_overlap_with_real"] = int((rep_s & rep).sum())
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{V}/v2_sampler.json", "w"), indent=1); print(json.dumps(res, indent=1))
