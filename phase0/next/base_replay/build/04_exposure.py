"""Step 4: replayed rows and per-line exposure counts, for K = 208 (the order validated in 03).

1. Verbatim check: run the April TokenBatchSampler unchanged with the REAL Python random module
   (random.seed(42); np.random.seed(42); 208 words drawn as swanlab's collectors draw them) for three
   epochs and require batch-for-batch equality with comp_np201 + the permutation used in 03.
2. Session 1 = P1 (49,174 batches) + P2 (49,177) + P3 positions 0..1,648; session 2 = P1 positions
   0..22,293. Each dataset row's exposure = 1 (P1) + 1 (P2) + [in P3 prefix] + [replayed].
3. Map dataset rows to corpus lines (index_to_line.npy, validated in 01).
Outputs: replayed_line_mask.npy (bool, per corpus line), exposure_count_by_line.npy (uint8, per corpus
line; 0 for the 2,261 lines the length filter drops), p1_position_by_line.npy (int32, P1 batch position
of each line's row, -1 if dropped), p3prefix_line_mask.npy, 04_exposure.json.
"""
import ast, hashlib, json, os, random, re, time
import numpy as np

B = os.path.expanduser("~/replay_check/build")
K = 208
t0 = time.time()
comp = np.load(f"{B}/comp_np201.npz")
Bs = [len(comp[f"starts_{e}"]) - 1 for e in (1, 2, 3)]
n = len(comp["idx_1"])

def perms(k):
    r = random.Random(42)
    for _ in range(k):
        r.getrandbits(32)
    out = []
    for b in Bs:
        p = list(range(b)); r.shuffle(p); out.append(np.asarray(p, np.int64))
    return out
pm = perms(K)
res = {"K_words": K, "B": Bs, "perm_K207_equals_K208": all(np.array_equal(a, b) for a, b in zip(perms(207), pm))}

# ---- 1. verbatim run with the real random module -------------------------------------------------
SRC = open(f"{B}/april_dataset.py", "rb").read()
assert hashlib.sha256(SRC).hexdigest().startswith("593d4550c323c71c")
cls = [x for x in ast.parse(SRC).body if isinstance(x, ast.ClassDef) and x.name == "TokenBatchSampler"][0]
ns = {"np": np, "random": random, "Sampler": object, "TranslationDataset": object}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "april_dataset.py:TokenBatchSampler", "exec"), ns)
class DS:
    src_lens = np.load(f"{B}/src_lens.npy"); tgt_lens = np.load(f"{B}/tgt_lens.npy")
    def __len__(self): return len(self.src_lens)
sampler = ns["TokenBatchSampler"](DS(), 24576, 192, True)
random.seed(42); np.random.seed(42)
for _ in range(K // 16):                       # 13 x swanlab random_index(): 8 x random() = 16 words each
    "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))
verb_ok = []
for e in (1, 2, 3):
    batches = list(iter(sampler))
    idx, st, p = comp[f"idx_{e}"], comp[f"starts_{e}"], pm[e - 1]
    ok = len(batches) == Bs[e - 1] and all(
        np.array_equal(np.asarray(batches[i], np.int64), idx[st[p[i]]:st[p[i] + 1]]) for i in range(len(batches)))
    verb_ok.append(bool(ok))
res["verbatim_equal_per_epoch"] = verb_ok
assert all(verb_ok), "verbatim run differs from the reconstruction"
print("verbatim", verb_ok, round(time.time() - t0, 1), "s", flush=True)

# ---- 2. replayed rows, P3 prefix, exposure --------------------------------------------------------
def rows_at(e, positions):
    idx, st, p = comp[f"idx_{e}"], comp[f"starts_{e}"], pm[e - 1]
    j = p[positions]
    return np.concatenate([idx[st[a]:st[a + 1]] for a in j])
S2_USED, P3_USED = 22294, 100000 - (Bs[0] + Bs[1])
replayed = np.zeros(n, bool); replayed[rows_at(1, np.arange(S2_USED))] = True
p3pre = np.zeros(n, bool); p3pre[rows_at(3, np.arange(P3_USED))] = True
expo = (2 + p3pre.astype(np.uint8) + replayed.astype(np.uint8)).astype(np.uint8)
# P1 batch position of each row
p1pos = np.empty(n, np.int32)
inv = np.empty(Bs[0], np.int64); inv[pm[0]] = np.arange(Bs[0])           # sorted batch id -> position
st1 = comp["starts_1"]
bid = np.repeat(np.arange(Bs[0]), np.diff(st1))                            # sorted batch id per slot
p1pos[comp["idx_1"]] = inv[bid]
assert np.array_equal(replayed, p1pos < S2_USED)

# gradient caveats (rows whose forward/backward ran but whose gradient was never applied)
spike_groups = [(4 * ((s + 3) // 4) - 3, 4 * ((s + 3) // 4)) for s in (10443, 12074)]   # steps, 1-based
s1_dropped_pos = [p for a, b in spike_groups for p in range(a - 1, b)]             # P1 positions
s2_unstepped_pos = list(range(22294 - 22294 % 4, 22294))                            # 22,292 and 22,293
res["session1_spike_groups_steps"] = spike_groups
res["session1_rows_grad_dropped"] = int(np.isin(p1pos, s1_dropped_pos).sum())
res["session2_unstepped_positions"] = s2_unstepped_pos
res["session2_rows_unstepped_at_interrupt"] = int(np.isin(p1pos, s2_unstepped_pos).sum())

# ---- 3. per corpus line ---------------------------------------------------------------------------
i2l = np.load(f"{B}/index_to_line.npy"); status = np.load(f"{B}/line_status.npy")
NL = len(status)
rep_line = np.zeros(NL, bool); rep_line[i2l] = replayed
expo_line = np.zeros(NL, np.uint8); expo_line[i2l] = expo
p3_line = np.zeros(NL, bool); p3_line[i2l] = p3pre
pos_line = np.full(NL, -1, np.int32); pos_line[i2l] = p1pos
np.save(f"{B}/replayed_line_mask.npy", rep_line)
np.save(f"{B}/exposure_count_by_line.npy", expo_line)
np.save(f"{B}/p3prefix_line_mask.npy", p3_line)
np.save(f"{B}/p1_position_by_line.npy", pos_line)
np.save(f"{B}/replayed_row_mask.npy", replayed)           # per dataset index (cache row order)
res.update({
    "rows": int(n), "corpus_lines": int(NL), "dropped_lines": int((status == 1).sum()),
    "replayed_rows": int(replayed.sum()), "replayed_share_of_rows": float(replayed.mean()),
    "p3_prefix_rows": int(p3pre.sum()),
    "rows_by_exposure": {int(k): int(v) for k, v in zip(*np.unique(expo, return_counts=True))},
    "replayed_and_p3_prefix_rows": int((replayed & p3pre).sum()),
    "files_sha256": {f: hashlib.sha256(open(f"{B}/{f}", "rb").read()).hexdigest() for f in
                     ("replayed_line_mask.npy", "exposure_count_by_line.npy", "p3prefix_line_mask.npy",
                      "p1_position_by_line.npy")},
})

# ---- per-segment window-loss correlation at K (P1 / P2 / P3 windows of session 1, and session 2) ---
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
c = np.load(f"{D}/train.clean.en.cached_256.npz"); tt, to = c["tgt_tokens"], c["tgt_offsets"]
is_bos = np.zeros(len(tt), bool); is_bos[to[:-1]] = True
cnt = np.bincount(tt[~is_bos], minlength=32000).astype(np.float64)
tok_s = (-np.log(np.maximum(cnt, 1) / cnt.sum()))[tt]; tok_s[is_bos] = 0
row_S = np.add.reduceat(tok_s, to[:-1]); row_N = (np.diff(to) - 1).astype(np.float64)
rep = open(f"{B}/logs/training_report.txt").read().split("TRAINING LOSS HISTORY")[1].split("VALIDATION")[0]
hist = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s*$", rep, re.M)}
L = np.array([hist[s] for s in range(100, 122201, 100)])
def hp(x, k=5):
    out = np.full(len(x), np.nan); cs = np.concatenate([[0], np.cumsum(x)])
    for w in range(k, len(x) - k):
        out[w] = x[w] - (cs[w + k + 1] - cs[w - k] - x[w]) / (2 * k)
    return out
seq_S, seq_N = [], []
for e, cnt_b in ((1, Bs[0]), (2, Bs[1]), (3, P3_USED)):
    idx, st, p = comp[f"idx_{e}"], comp[f"starts_{e}"], pm[e - 1]
    bS = np.add.reduceat(row_S[idx], st[:-1]); bN = np.add.reduceat(row_N[idx], st[:-1])
    seq_S.append(bS[p[:cnt_b]]); seq_N.append(bN[p[:cnt_b]])
S1 = np.concatenate(seq_S); N1 = np.concatenate(seq_N)
F1 = hp(S1.reshape(1000, 100).sum(1) / N1.reshape(1000, 100).sum(1)); L1 = hp(L[:1000]); L1[:50] = np.nan
F2 = hp(seq_S[0][:22200].reshape(222, 100).sum(1) / seq_N[0][:22200].reshape(222, 100).sum(1)); L2 = hp(L[1000:])
def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); return [float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())]
w_end1 = Bs[0] // 100; w_end2 = (Bs[0] + Bs[1]) // 100
res["window_corr_at_K"] = {
    "session1_all": corr(F1, L1),
    "session1_P1_windows": corr(F1[:w_end1], L1[:w_end1]),
    "session1_P2_windows": corr(F1[w_end1 + 1:w_end2], L1[w_end1 + 1:w_end2]),
    "session1_P3_windows": corr(F1[w_end2 + 1:], L1[w_end2 + 1:]),
    "session2_P1_prefix": corr(F2, L2)}
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{B}/04_exposure.json", "w"), indent=1)
print(json.dumps(res, indent=1))
