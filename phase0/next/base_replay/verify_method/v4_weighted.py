"""Auditor check 4: the released base is the mean of checkpoints at steps 105,000/110,000/115,000/120,000/122,294
(average_checkpoints.py, last 5 matches of 'step_(\\d+).pt', interrupted_step_122294.pt included). A replayed batch at
session-2 position q (step 100,001+q) is inside the fraction w(q) of those checkpoints (0 for q = 22,292-22,293, never
stepped). Per set: mean over exposed rows of the summed w over the pair's corpus lines; re-draw null over P1 orders."""
import collections, json, os, random, re, time
import numpy as np
H = os.path.expanduser("~"); V = f"{H}/replay_check/verify_method"; B = f"{H}/replay_check/build"
D = f"{H}/workspace/Machine_translation/data_enfr_v1"; IN = f"{H}/replay_check/inputs"
t0 = time.time()
c = np.load(f"{D}/train.clean.en.cached_256.npz")
sl = np.diff(c["src_offsets"]).astype(np.int32); tl = np.diff(c["tgt_offsets"]).astype(np.int32); n = len(sl)
np.random.seed(42); r = random.Random(42)
for _ in range(208): r.getrandbits(32)
idx = np.arange(n); np.random.shuffle(idx)
for i in range(0, n, 19200):
    ch = idx[i:i + 19200]; idx[i:i + 19200] = ch[np.argsort(sl[ch])]
seq = np.maximum(sl[idx], tl[idx]).tolist(); starts = [0]; cnt = mx = 0
for j, s in enumerate(seq):
    nm = mx if mx > s else s
    if (cnt + 1) * nm > 24576 or cnt >= 192: starts.append(j); cnt = 1; mx = s
    else: cnt += 1; mx = nm
starts.append(n); st = np.asarray(starts); NB = len(st) - 1
p = list(range(NB)); r.shuffle(p); p = np.asarray(p)
pos_of_bid = np.empty(NB, np.int64); pos_of_bid[p] = np.arange(NB)
bid = np.empty(n, np.int64); bid[idx] = np.repeat(np.arange(NB), np.diff(st))
pos_row = pos_of_bid[bid]
i2l = np.load(f"{B}/index_to_line.npy"); NL = 9312233
pos_line = np.full(NL, -1, np.int64); pos_line[i2l] = pos_row
res = {"p1_position_equal_builder": bool(np.array_equal(pos_line, np.load(f"{B}/p1_position_by_line.npy")))}
CK = np.array([105000, 110000, 115000, 120000, 122292])   # last optimizer step inside each checkpoint
def w_of_pos(q):  # fraction of the 5 averaged checkpoints that contain the update from position q
    step = 100001 + q
    return ((step[:, None] <= CK[None, :]).sum(1) / 5.0) * (q < 22294)
wq = w_of_pos(np.arange(NB))
res["mean_w_over_replayed_positions"] = float(wq[:22294].mean())
def read_lines(pth):
    with open(pth, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]
SIX = ("ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small")
sets = {nm: list(zip(read_lines(f"{IN}/phase0_e05/{nm}.en"), read_lines(f"{IN}/phase0_e05/{nm}.fr"))) for nm in SIX}
fe, ff = read_lines(f"{IN}/e05_src/ft_random.en"), read_lines(f"{IN}/e05_src/ft_random.fr")
for nm, f in (("R0", "R0.npy"), ("A_narrow", "A_narrow_42_58.npy")):
    sets[nm] = [(fe[i], ff[i]) for i in np.load(f"{IN}/e06_idx/{f}")]
want = set().union(*[set(v) for v in sets.values()])
occ = collections.defaultdict(list)
with open(f"{D}/train.clean.en", encoding="utf-8", newline="\n") as a, open(f"{D}/train.clean.fr", encoding="utf-8", newline="\n") as b:
    for ln, (x, y) in enumerate(zip(a, b)):
        k = (x[:-1], y[:-1])
        if k in want: occ[k].append(ln)
flat = {}
for nm, pairs in sets.items():
    L = [occ.get(pq, []) for pq in pairs]; nl = np.array([len(x) for x in L])
    lines = np.array([l for x in L for l in x], np.int64); rows_o = np.repeat(np.arange(len(pairs)), nl)
    flat[nm] = (rows_o, None, lines, nl > 0, len(pairs))
bid_line = np.full(NL, -1, np.int64); bid_line[i2l] = bid
def stat(nm, posb):            # posb: P1 position of every sorted batch id
    rows_o, _, lines, ex, N = flat[nm]
    b_ = bid_line[lines]; k = b_ >= 0
    w = np.zeros(len(lines)); w[k] = wq[posb[b_[k]]]
    tot = np.bincount(rows_o, weights=w, minlength=N)
    return float(tot[ex].mean()), float(tot.mean())
obs = {nm: stat(nm, pos_of_bid) for nm in sets}
res["weighted_replay_per_set_(exposed_mean, all_rows_mean)"] = obs
rng = np.random.default_rng(11); ND = 500
dr = {nm: np.empty((ND, 2)) for nm in sets}
for i in range(ND):
    pb = rng.permutation(NB)
    for nm in sets: dr[nm][i] = stat(nm, pb)
con = []
for a_, b_ in (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("ft_topk_rep", "ft_topk_div"), ("R0", "ft_un_div"), ("A_narrow", "R0")):
    for j, mname in enumerate(("weighted_replay_exposed_mean", "weighted_replay_all_rows_mean")):
        d = dr[a_][:, j] - dr[b_][:, j]; o = obs[a_][j] - obs[b_][j]
        con.append({"a": a_, "b": b_, "metric": mname, "diff": o, "null_mean": float(d.mean()), "null_sd": float(d.std(ddof=1)),
                    "p": float((np.abs(d - d.mean()) >= abs(o - d.mean())).mean())})
res["contrasts"] = con; res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{V}/v4_weighted.json", "w"), indent=1); print(json.dumps(res, indent=1))
