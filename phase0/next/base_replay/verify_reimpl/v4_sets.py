"""v4: pretraining exposure of the E0.5 / E0.6 sets and their replayed share; comparison with the builder.

Exposure rule (phase0/e03_build_controls.py lines 270-290, the drafts' "pair in pretraining"):
a set row is exposed if its exact (en, fr) pair, with only the trailing newline removed, equals
some line pair of train.clean.{en,fr}. Here the match is on the raw bytes themselves (a dict of
byte-string pairs), not on a hash, so there are no collisions to rule out.
Exposed-and-replayed: at least one corpus line carrying the pair was a kept dataset row whose
batch is among the first 22,294 of the epoch-1 order. Also reported: the "all copies" variant
and rows whose pair sits on exactly one corpus line.
"""
import json, os, pickle, re, sys
import numpy as np

OUT = os.path.expanduser("~/replay_check/verify_reimpl")
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
E05 = os.path.expanduser("~/replay_check/inputs/phase0_e05")
SRC5 = os.path.expanduser("~/replay_check/inputs/e05_src")
IDX6 = os.path.expanduser("~/replay_check/inputs/e06_idx")
BLD = os.path.expanduser("~/replay_check/build")
SETS = ["ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small"]


def read_pairs(stem):
    with open(stem + ".en", "rb") as a, open(stem + ".fr", "rb") as b:
        A = [x[:-1] if x.endswith(b"\n") else x for x in a]
        B = [x[:-1] if x.endswith(b"\n") else x for x in b]
    assert len(A) == len(B), stem
    return list(zip(A, B))


def main():
    rep = {}
    i2l = np.load(f"{OUT}/v1_index_to_line.npy")
    status = np.load(f"{OUT}/v1_line_status.npy")
    replayed_row = np.load(f"{OUT}/v2_replayed_row_mask_np226.npy")
    p3pre_row = np.load(f"{OUT}/v2_p3prefix_row_mask_np226.npy")
    n_lines = status.size
    replayed_line = np.zeros(n_lines, bool); replayed_line[i2l[replayed_row]] = True
    expo_line = np.zeros(n_lines, np.int8)
    expo_line[i2l] = 2 + replayed_row.astype(np.int8) + p3pre_row.astype(np.int8)

    # ---- sets ----
    sets = {s: read_pairs(f"{E05}/{s}") for s in SETS}
    ftr = read_pairs(f"{SRC5}/ft_random")
    lab = open(f"{SRC5}/ft_random.source", "rb").read().split(b"\n")
    lab = lab[:-1] if lab[-1] == b"" else lab
    assert len(lab) == len(ftr)
    for nm, f in (("R0", "R0.npy"), ("A_narrow", "A_narrow_42_58.npy")):
        ix = np.load(f"{IDX6}/{f}")
        assert ix.min() >= 0 and ix.max() < len(ftr) and np.unique(ix).size == ix.size
        rep.setdefault("e06_index_files", {})[nm] = {"rows": int(ix.size), "all_un": bool(all(lab[i] == b"un" for i in ix))}
        sets[nm] = [ftr[i] for i in ix]

    # rows.pkl order check (box copy, sha256 equal to the Mac original ~/mt_local/phase0_next/vocab/rows.pkl)
    R = pickle.load(open(f"{BLD}/rows.pkl", "rb"))["ft_random"]
    WORD = re.compile(r"\w+", re.UNICODE)
    en_txt = [p[0].decode("utf-8") for p in ftr]; fr_txt = [p[1].decode("utf-8") for p in ftr]
    wl = np.array([len(WORD.findall(x)) for x in en_txt]); wf = np.array([len(WORD.findall(x)) for x in fr_txt])
    un = np.array([l == b"un" for l in lab])
    ur = set(sets["ft_un_random"]); rs = set(sets["ft_random_small"])
    m_ur = np.array([p in ur for p in ftr]); m_rs = np.array([p in rs for p in ftr])
    rep["rows_pkl_order_check"] = {
        "rows": int(len(R["len"])), "file_rows": len(ftr),
        "len_en_equal": bool(np.array_equal(R["len"].astype(int), wl)),
        "len_fr_equal": bool(np.array_equal(R["len_fr"].astype(int), wf)),
        "un_equal": bool(np.array_equal(R["un"].astype(bool), un)),
        "in_ft_un_random_equal": bool(np.array_equal(R["in_ft_un_random"].astype(bool), m_ur)),
        "in_ft_random_small_equal": bool(np.array_equal(R["in_ft_random_small"].astype(bool), m_rs)),
    }
    del R, en_txt, fr_txt

    # ---- one pass over the corpus ----
    want = {}
    for s, P in sets.items():
        for p in P:
            want.setdefault(p, None)
    hits = {}
    with open(f"{D}/train.clean.en", "rb") as a, open(f"{D}/train.clean.fr", "rb") as b:
        ln = -1
        for ln, (x, y) in enumerate(zip(a, b)):
            k = (x[:-1] if x.endswith(b"\n") else x, y[:-1] if y.endswith(b"\n") else y)
            if k in want:
                hits.setdefault(k, []).append(ln)
    assert ln + 1 == n_lines

    per = {}
    for s, P in sets.items():
        n = len(P)
        nlines = np.zeros(n, np.int32); anyrep = np.zeros(n, bool); allrep = np.zeros(n, bool)
        anykept = np.zeros(n, bool); ecount = np.zeros(n, np.int32)
        for i, p in enumerate(P):
            L = hits.get(p)
            if not L:
                continue
            L = np.array(L)
            nlines[i] = L.size
            anyrep[i] = replayed_line[L].any(); allrep[i] = replayed_line[L].all()
            anykept[i] = (status[L] == 0).any(); ecount[i] = expo_line[L].sum()
        ex = nlines > 0
        single = nlines == 1
        per[s] = {
            "rows": n, "distinct_pairs": len(set(P)),
            "exposed_rows": int(ex.sum()), "exposed_share": round(float(ex.mean()), 5),
            "exposed_only_via_dropped_lines": int((ex & ~anykept).sum()),
            "exposed_and_replayed_any_copy": int(anyrep.sum()),
            "exposed_and_replayed_all_copies": int(allrep.sum()),
            "replayed_share_of_exposed": round(float(anyrep.sum() / ex.sum()), 5),
            "replayed_share_of_all_rows": round(float(anyrep.mean()), 5),
            "mean_exposure_count_of_exposed": round(float(ecount[ex].mean()), 4),
            "exposed_rows_on_more_than_one_line": int((nlines > 1).sum()),
            "single_line_rows": int(single.sum()),
            "single_line_replayed_share": round(float(anyrep[single].mean()), 5),
        }
    rep["per_set"] = per

    # ---- comparison with the builder's arrays (read only) ----
    cmpb = {}
    b_i2l = np.load(f"{BLD}/index_to_line.npy")
    cmpb["index_to_line_equal"] = bool(np.array_equal(b_i2l, i2l))
    b_row = np.load(f"{BLD}/replayed_row_mask.npy")
    cmpb["replayed_row_mask"] = {"shape_equal": b_row.shape == replayed_row.shape,
                                 "builder_true": int(b_row.sum()), "mine_true": int(replayed_row.sum()),
                                 "disagreeing_rows": int((b_row != replayed_row).sum())}
    b_line = np.load(f"{BLD}/replayed_line_mask.npy")
    cmpb["replayed_line_mask_disagreeing_lines"] = int((b_line.astype(bool) != replayed_line).sum())
    b_exp = np.load(f"{BLD}/exposure_count_by_line.npy")
    cmpb["exposure_count_by_line_disagreeing"] = int((b_exp.astype(np.int64) != expo_line.astype(np.int64)).sum())
    b_p1 = np.load(f"{BLD}/p1_position_by_line.npy")
    my_p1_row = np.load(f"{OUT}/v2_p1_pos_by_row_np226.npy")
    my_p1_line = np.full(n_lines, -1, np.int64); my_p1_line[i2l] = my_p1_row
    cmpb["p1_position_by_line"] = {"builder_dtype": str(b_p1.dtype), "builder_shape": list(b_p1.shape),
                                   "disagreeing_lines": int((b_p1.astype(np.int64) != my_p1_line).sum())}
    b_p3 = np.load(f"{BLD}/p3prefix_line_mask.npy")
    my_p3_line = np.zeros(n_lines, bool); my_p3_line[i2l[p3pre_row]] = True
    cmpb["p3prefix_line_mask_disagreeing"] = int((b_p3.astype(bool) != my_p3_line).sum())
    try:
        bj = json.load(open(f"{BLD}/results.json"))
        bps = {d["set"]: d for d in bj["per_set"]}
        cmpb["per_set_vs_builder"] = {s: {"exposed": [per[s]["exposed_rows"], bps[s]["exposed_rows"]],
                                          "exposed_and_replayed": [per[s]["exposed_and_replayed_any_copy"], bps[s]["exposed_and_replayed"]],
                                          "mean_exposure": [per[s]["mean_exposure_count_of_exposed"], bps[s]["mean_exposure_count_of_exposed"]]}
                                      for s in per if s in bps}
    except Exception as e:
        cmpb["per_set_vs_builder"] = f"could not read builder results.json: {e!r}"
    rep["builder_comparison"] = cmpb
    json.dump(rep, open(f"{OUT}/v4_sets.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
