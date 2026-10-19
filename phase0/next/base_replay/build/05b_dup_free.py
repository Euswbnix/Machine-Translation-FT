"""Step 6b: separate the sampler's balance from the corpus's duplicates.

05's per-row replay rule ("replayed if any occurrence is replayed") and summed exposure count make a
pair that sits on k corpus lines more likely to be replayed (1-(1-p)^k) and more exposed (sum over k).
Here the same statistics are computed (a) on exposed rows whose pair occurs on exactly ONE corpus line
and (b) per occurrence line, with the same 2,000-draw design-based intervals as 05 (same seed).
"""
import json, os
import numpy as np
B = os.path.expanduser("~/replay_check/build")
occ = np.load(f"{B}/set_occurrences.npz")
status = np.load(f"{B}/line_status.npy"); kept = status == 0
rep_line = np.load(f"{B}/replayed_line_mask.npy"); expo_line = np.load(f"{B}/exposure_count_by_line.npy")
comp = np.load(f"{B}/comp_np201.npz"); i2l = np.load(f"{B}/index_to_line.npy"); NL = len(status)
def bid_by_line(e):
    st = comp[f"starts_{e}"]; idx = comp[f"idx_{e}"]
    bid = np.empty(len(idx), np.int64); bid[idx] = np.repeat(np.arange(len(st) - 1), np.diff(st))
    out = np.full(NL, -1, np.int64); out[i2l] = bid; return out
bid1, bid3 = bid_by_line(1), bid_by_line(3)
NB1, NB3 = len(comp["starts_1"]) - 1, len(comp["starts_3"]) - 1
names = ["ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small", "R0", "A_narrow"]
single, lines_all = {}, {}
for nm in names:
    r, l = occ[f"{nm}_rows"], occ[f"{nm}_lines"]
    c = np.bincount(r); one = np.where(c == 1)[0]
    m1 = np.isin(r, one)
    single[nm] = l[m1]; lines_all[nm] = l
def stats(z1, z3, lines):
    k = kept[lines]
    r1 = np.zeros(len(lines), bool); r1[k] = z1[bid1[lines[k]]]
    r3 = np.zeros(len(lines), bool); r3[k] = z3[bid3[lines[k]]]
    return r1.mean(), (k * (2 + r1 + r3)).mean()
z1o = np.zeros(NB1, bool); z1o[np.unique(bid1[rep_line])] = True
p3l = np.load(f"{B}/p3prefix_line_mask.npy"); z3o = np.zeros(NB3, bool); z3o[np.unique(bid3[p3l])] = True
obs = {nm: {"single": stats(z1o, z3o, single[nm]), "per_line": stats(z1o, z3o, lines_all[nm])} for nm in names}
for nm in names:
    assert abs(obs[nm]["single"][0] - rep_line[single[nm]].mean()) < 1e-12
rng = np.random.default_rng(20261018)
NDRAW = 2000
dr = {nm: np.empty((NDRAW, 4)) for nm in names}
for i in range(NDRAW):
    z1 = np.zeros(NB1, bool); z1[rng.choice(NB1, 22294, replace=False)] = True
    z3 = np.zeros(NB3, bool); z3[rng.choice(NB3, 1649, replace=False)] = True
    for nm in names:
        dr[nm][i] = (*stats(z1, z3, single[nm]), *stats(z1, z3, lines_all[nm]))
per_set = [{"set": nm, "exposed_rows_single_line": int(len(single[nm])),
            "replayed_share_single_line": float(obs[nm]["single"][0]),
            "mean_exposure_single_line": float(obs[nm]["single"][1]),
            "occurrence_lines": int(len(lines_all[nm])),
            "replayed_share_per_occurrence_line": float(obs[nm]["per_line"][0]),
            "mean_exposure_per_occurrence_line": float(obs[nm]["per_line"][1])} for nm in names]
contr = []
for a, b in (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("ft_topk_rep", "ft_topk_div"),
             ("R0", "ft_un_div"), ("A_narrow", "R0")):
    for j, (m, key, col) in enumerate((("replayed_share_single_line", "single", 0), ("mean_exposure_single_line", "single", 1),
                                       ("replayed_share_per_occurrence_line", "per_line", 0))):
        o = obs[a][key][col] - obs[b][key][col]
        cc = {"single": (0, 1), "per_line": (2, 3)}[key][col]
        d = dr[a][:, cc] - dr[b][:, cc]; sd = float(d.std(ddof=1))
        contr.append({"a": a, "b": b, "metric": m, "diff": float(o), "ci95_design": [o - 1.96 * sd, o + 1.96 * sd],
                      "redraw_null_mean_diff": float(d.mean()),
                      "p_two_sided_redraw": float((np.abs(d - d.mean()) >= abs(o - d.mean())).mean())})
res = {"per_set": per_set, "contrasts": contr, "n_redraws": NDRAW}
json.dump(res, open(f"{B}/05b_dup_free.json", "w"), indent=1)
for s in per_set: print(s)
for c in contr: print(c)
