"""Auditor check 3: exact-pair exposure (string-exact, no hashing), replay share and exposure counts per set
from MY replay masks (v2), R0/A_narrow mapping checks, single-line subsets, re-draw nulls for the contrasts."""
import collections, json, os, pickle, re, time
import numpy as np
H = os.path.expanduser("~"); V = f"{H}/replay_check/verify_method"; B = f"{H}/replay_check/build"
D = f"{H}/workspace/Machine_translation/data_enfr_v1"; IN = f"{H}/replay_check/inputs"
t0 = time.time(); res = {}
def read_lines(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]
def cr_count(p): return open(p, "rb").read().count(b"\r")
SIX = ("ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small")
sets = {nm: list(zip(read_lines(f"{IN}/phase0_e05/{nm}.en"), read_lines(f"{IN}/phase0_e05/{nm}.fr"))) for nm in SIX}
res["cr_in_set_files"] = {nm: cr_count(f"{IN}/phase0_e05/{nm}.en") + cr_count(f"{IN}/phase0_e05/{nm}.fr") for nm in SIX}
fe, ff, fs = (read_lines(f"{IN}/e05_src/ft_random.{x}") for x in ("en", "fr", "source"))
R0 = np.load(f"{IN}/e06_idx/R0.npy"); AN = np.load(f"{IN}/e06_idx/A_narrow_42_58.npy")
WORD = re.compile(r"\w+", re.UNICODE)
Rt = pickle.load(open(f"{B}/rows.pkl", "rb"))["ft_random"]
lens_file = np.array([len(WORD.findall(x)) for x in fe])
chk = {"pkl_rows": int(len(Rt["len"])), "file_rows": len(fe),
       "pkl_len_equals_file_wordcount": bool(np.array_equal(Rt["len"], lens_file)),
       "pkl_un_equals_file_source": bool(np.array_equal(Rt["un"], np.array([s == "un" for s in fs])))}
fpos = {}
for i, pq in enumerate(zip(fe, ff)): fpos.setdefault(pq, i)
chk["ft_random_distinct_pairs"] = len(fpos)
m = np.zeros(len(fe), bool); m[[fpos[pq] for pq in sets["ft_un_random"]]] = True
chk["pkl_in_ft_un_random_equals_file"] = bool(np.array_equal(Rt["in_ft_un_random"], m))
div_hist = collections.Counter(len(WORD.findall(a)) for a, _ in sets["ft_un_div"])
for nm, ix in (("R0", R0), ("A_narrow", AN)):
    chk[f"{nm}_n"] = int(len(ix)); chk[f"{nm}_unique"] = int(len(np.unique(ix)))
    chk[f"{nm}_all_un_file"] = all(fs[i] == "un" for i in ix)
    chk[f"{nm}_rep0_pkl"] = bool((Rt["rep"][ix] == 0).all())
    chk[f"{nm}_hist_equals_ft_un_div_hist"] = collections.Counter(int(lens_file[i]) for i in ix) == div_hist
    sets[nm] = [(fe[i], ff[i]) for i in ix]
chk["un_share_of_ft_random"] = float(np.mean([s == "un" for s in fs]))
res["mapping_checks"] = chk
del Rt
# ---- string-exact matching against all corpus lines ----
want = set()
for v in sets.values(): want.update(v)
occ = collections.defaultdict(list)
with open(f"{D}/train.clean.en", encoding="utf-8", newline="\n") as a, open(f"{D}/train.clean.fr", encoding="utf-8", newline="\n") as b:
    NL = 0
    for ln, (x, y) in enumerate(zip(a, b)):
        NL += 1
        k = (x[:-1] if x.endswith("\n") else x, y[:-1] if y.endswith("\n") else y)
        if k in want: occ[k].append(ln)
res["corpus_lines"] = NL; res["t_match_s"] = round(time.time() - t0, 1)
i2l = np.load(f"{B}/index_to_line.npy")
rep_row = np.load(f"{V}/my_replayed_row.npy"); p3_row = np.load(f"{V}/my_p3_row.npy")
kept = np.zeros(NL, bool); kept[i2l] = True
rep_line = np.zeros(NL, bool); rep_line[i2l] = rep_row
p3_line = np.zeros(NL, bool); p3_line[i2l] = p3_row
expo_line = np.zeros(NL, np.int64); expo_line[i2l] = 2 + rep_row + p3_row
bid1 = np.full(NL, -1, np.int64); bid1[i2l] = np.load(f"{V}/my_bid1_row.npy")
bid3 = np.full(NL, -1, np.int64); bid3[i2l] = np.load(f"{V}/my_bid3_row.npy")
per, flat = {}, {}
for nm, pairs in sets.items():
    L = [occ.get(pq, []) for pq in pairs]
    nlin = np.array([len(x) for x in L]); ex = nlin > 0
    lines = np.array([l for x in L for l in x], np.int64)
    rows_o = np.repeat(np.arange(len(pairs)), nlin)
    rep_any = np.zeros(len(pairs), bool); rep_any[rows_o[rep_line[lines]]] = True
    expo = np.bincount(rows_o, weights=expo_line[lines], minlength=len(pairs))
    single = ex & (nlin == 1)
    per[nm] = {"rows": len(pairs), "exposed_rows": int(ex.sum()), "exposed_share": float(ex.mean()),
               "exposed_on_dropped_line_only": int(sum(1 for x in L if x and not any(kept[x]))),
               "exposed_and_replayed": int((ex & rep_any).sum()),
               "replayed_share_of_exposed": float(rep_any[ex].mean()), "replayed_share_of_all_rows": float(rep_any.mean()),
               "mean_exposure_count_of_exposed": float(expo[ex].mean()),
               "exposed_rows_multi_line": int((nlin > 1).sum()), "mean_lines_per_exposed": float(nlin[ex].mean()),
               "single_line_rows": int(single.sum()), "single_line_replayed_share": float(rep_any[single].mean()),
               "replayed_share_per_corpus_line": float(rep_line[lines].mean()),
               "mean_exposure_per_corpus_line": float(expo_line[lines].mean())}
    flat[nm] = (rows_o, lines, ex, single)
res["per_set"] = per
# ---- re-draw null: content-random replay with batch composition fixed ----
rng = np.random.default_rng(7)
NDRAW = 1000
NB1, NB3 = int(bid1.max()) + 1, int(bid3.max()) + 1
res["NB1_NB3"] = [NB1, NB3]
def stat(nm, z1, z3):
    rows_o, lines, ex, single = flat[nm]; N = len(ex)
    k = kept[lines]
    r1 = np.zeros(len(lines), bool); r1[k] = z1[bid1[lines[k]]]
    r3 = np.zeros(len(lines), bool); r3[k] = z3[bid3[lines[k]]]
    ra = np.zeros(N, bool); ra[rows_o[r1]] = True
    e = np.bincount(rows_o, weights=k * (2 + r1 + r3), minlength=N)
    return ra[ex].mean(), ra.mean(), e[ex].mean(), ra[single].mean()
z1o = np.zeros(NB1, bool); z1o[np.unique(bid1[rep_line])] = True
z3o = np.zeros(NB3, bool); z3o[np.unique(bid3[p3_line])] = True
res["observed_batches_replayed_p3"] = [int(z1o.sum()), int(z3o.sum())]
obs = {nm: stat(nm, z1o, z3o) for nm in sets}
dr = {nm: np.empty((NDRAW, 4)) for nm in sets}
for i in range(NDRAW):
    z1 = np.zeros(NB1, bool); z1[rng.choice(NB1, 22294, replace=False)] = True
    z3 = np.zeros(NB3, bool); z3[rng.choice(NB3, 1649, replace=False)] = True
    for nm in sets: dr[nm][i] = stat(nm, z1, z3)
names = ["replayed_share_of_exposed", "replayed_share_of_all_rows", "mean_exposure_count_of_exposed", "single_line_replayed_share"]
con = []
for a, b in (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("ft_topk_rep", "ft_topk_div"), ("R0", "ft_un_div"), ("A_narrow", "R0")):
    for j, mn in enumerate(names):
        d = dr[a][:, j] - dr[b][:, j]; o = obs[a][j] - obs[b][j]
        con.append({"a": a, "b": b, "metric": mn, "diff": float(o), "null_mean": float(d.mean()), "null_sd": float(d.std(ddof=1)),
                    "p_two_sided": float((np.abs(d - d.mean()) >= abs(o - d.mean())).mean()),
                    "obs_minus_null_mean": float(o - d.mean())})
res["contrasts"] = con
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{V}/v3_sets.json", "w"), indent=1); print(json.dumps(res, indent=1))
