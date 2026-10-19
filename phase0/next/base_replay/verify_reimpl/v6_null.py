"""v6: design-based null for the replayed share of exposed rows.

Given the epoch-1 composition, the replayed batches are the first 22,294 positions of a uniform
random permutation, i.e. a uniform random subset of 22,294 of the 49,174 batches. Re-draw that
subset R times (same subset for every set in a draw) and compare each observed contrast with the
null distribution. Rule for a row: replayed if any corpus line carrying its pair is in a chosen batch.
"""
import json, os
import numpy as np

OUT = os.path.expanduser("~/replay_check/verify_reimpl")
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
E05 = os.path.expanduser("~/replay_check/inputs/phase0_e05")
SRC5 = os.path.expanduser("~/replay_check/inputs/e05_src")
IDX6 = os.path.expanduser("~/replay_check/inputs/e06_idx")
R, REPLAY = 2000, 22294


def read_pairs(stem):
    with open(stem + ".en", "rb") as a, open(stem + ".fr", "rb") as b:
        return list(zip([x[:-1] if x.endswith(b"\n") else x for x in a], [x[:-1] if x.endswith(b"\n") else x for x in b]))


C = np.load(f"{OUT}/v2_comp_np226.npz")
B1 = C["starts1"].size
i2l = np.load(f"{OUT}/v1_index_to_line.npy")
n_lines = np.load(f"{OUT}/v1_line_status.npy").size
batch_of_row = np.empty(i2l.size, np.int64); batch_of_row[C["idx1"]] = np.repeat(np.arange(B1), C["rows1"])
batch_of_line = np.full(n_lines, -1, np.int64); batch_of_line[i2l] = batch_of_row
pos_of_batch = np.empty(B1, np.int64); pos_of_batch[C["perm1"]] = np.arange(B1)
obs_sel = pos_of_batch < REPLAY

sets = {s: read_pairs(f"{E05}/{s}") for s in ("ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep")}
ftr = read_pairs(f"{SRC5}/ft_random")
sets["R0"] = [ftr[i] for i in np.load(f"{IDX6}/R0.npy")]
sets["A_narrow"] = [ftr[i] for i in np.load(f"{IDX6}/A_narrow_42_58.npy")]
want = {p for P in sets.values() for p in P}
hits = {}
with open(f"{D}/train.clean.en", "rb") as a, open(f"{D}/train.clean.fr", "rb") as b:
    for ln, (x, y) in enumerate(zip(a, b)):
        k = (x[:-1] if x.endswith(b"\n") else x, y[:-1] if y.endswith(b"\n") else y)
        if k in want:
            hits.setdefault(k, []).append(ln)

flat, offs, nexp, single = {}, {}, {}, {}
for s, P in sets.items():
    f, o, sg = [], [], []
    for p in P:
        L = hits.get(p)
        if not L:
            continue
        bl = batch_of_line[np.array(L)]
        bl = bl[bl >= 0]
        assert bl.size > 0
        o.append(len(f)); f.extend(bl.tolist()); sg.append(len(L) == 1)
    flat[s], offs[s], nexp[s], single[s] = np.array(f), np.array(o), len(o), np.array(sg)


def shares(sel):
    out = {}
    for s in sets:
        anyr = np.logical_or.reduceat(sel[flat[s]], offs[s])
        out[s] = (anyr.mean(), anyr[single[s]].mean())
    return out


obs = shares(obs_sel)
rng = np.random.default_rng(20261018)
null = {s: np.empty((R, 2)) for s in sets}
for r in range(R):
    sel = np.zeros(B1, bool); sel[rng.choice(B1, REPLAY, replace=False)] = True
    sh = shares(sel)
    for s in sets:
        null[s][r] = sh[s]
rep = {"R": R, "observed_replayed_share_of_exposed": {s: round(float(obs[s][0]), 5) for s in sets}, "contrasts": []}
for a, b in (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("ft_topk_rep", "ft_topk_div"),
             ("R0", "ft_un_div"), ("A_narrow", "R0")):
    for k, lab in ((0, "all exposed rows"), (1, "single-line rows")):
        o = obs[a][k] - obs[b][k]
        nd = null[a][:, k] - null[b][:, k]
        m, sd = nd.mean(), nd.std()
        p = float((np.abs(nd - m) >= abs(o - m)).mean())
        rep["contrasts"].append({"a": a, "b": b, "rows": lab, "observed_diff": round(float(o), 5),
                                 "null_mean": round(float(m), 5), "null_sd": round(float(sd), 5), "p_two_sided": round(p, 3)})
json.dump(rep, open(f"{OUT}/v6_null.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
