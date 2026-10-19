"""Steps 5-6: pretraining exposure, replay and exposure counts of the E0.5 sets, R0 and A_narrow; contrasts.

Matching rule (the drafts'): a set row is "exposed" if its exact (en, fr) pair occurs on at least one line
of the v1.1 corpus (data_enfr_v1/train.clean.{en,fr}, all 9,312,233 lines), both files read with
newline="\n" and only the trailing "\n" removed, as e03_build_controls.py reads the pretraining corpus
and the phase0/next scripts read the sets. Keyed by blake2b-64 of f"{en}\\x00{fr}" (e03's h64) and
confirmed by a second, independent 64-bit blake2b key. A pair on several corpus lines is replayed if any
occurrence is replayed; its exposure count is the sum of its occurrences' per-line counts (a dropped
line counts 0). Also reported: the share exposed through lines the trainer actually kept, and the same
statistics under the trainer's own view of a line (str.strip() on both sides).

Contrast intervals: rows that share a batch share their replay status, so row-level intervals are too
narrow. The design-based interval is diff +/- 1.96 x the sd of diff over 2,000 re-draws of the
content-random replay that the sampler implements (a uniformly random set of 22,294 of P1's 49,174
batches, and of 1,649 of P3's 49,180 batches, batch composition fixed); the naive row-level interval is
reported beside it.
"""
import hashlib, json, multiprocessing as mp, os, pickle, re, time
import numpy as np

B = os.path.expanduser("~/replay_check/build")
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
IN = os.path.expanduser("~/replay_check/inputs")
t0 = time.time(); T = {}

def k1(s, t): return int.from_bytes(hashlib.blake2b(f"{s}\x00{t}".encode("utf-8"), digest_size=8).digest(), "little")
def k2(s, t): return int.from_bytes(hashlib.blake2b(f"{s}\x00{t}".encode("utf-8"), digest_size=8, person=b"rc-verify").digest(), "little")
def keys(pairs):
    a = np.fromiter((k1(s, t) for s, t in pairs), np.uint64, len(pairs))
    b = np.fromiter((k2(s, t) for s, t in pairs), np.uint64, len(pairs))
    c = np.fromiter((k1(s.strip(), t.strip()) for s, t in pairs), np.uint64, len(pairs))
    return a, b, c
def read_lines(p):
    with open(p, encoding="utf-8", newline="\n") as f:
        return [ln[:-1] if ln.endswith("\n") else ln for ln in f]

def corpus_keys():
    def chunks():
        buf = []
        with open(f"{D}/train.clean.en", encoding="utf-8", newline="\n") as a, \
             open(f"{D}/train.clean.fr", encoding="utf-8", newline="\n") as b:
            for s, t in zip(a, b):
                buf.append((s[:-1] if s.endswith("\n") else s, t[:-1] if t.endswith("\n") else t))
                if len(buf) == 250_000:
                    yield buf; buf = []
        if buf: yield buf
    A, Bk, C = [], [], []
    with mp.get_context("fork").Pool(16) as pool:
        for a, b, c in pool.imap(keys, chunks(), chunksize=1):
            A.append(a); Bk.append(b); C.append(c)
    return np.concatenate(A), np.concatenate(Bk), np.concatenate(C)

if __name__ == "__main__":
    ck1, ck2, ck1s = corpus_keys()
    NL = len(ck1); assert NL == 9312233
    T["corpus_keys_s"] = round(time.time() - t0, 1)
    status = np.load(f"{B}/line_status.npy"); kept = status == 0
    rep_line = np.load(f"{B}/replayed_line_mask.npy"); expo_line = np.load(f"{B}/exposure_count_by_line.npy")
    p3_line = np.load(f"{B}/p3prefix_line_mask.npy")
    # sorted-order batch id of each line in P1 and P3 (for the re-draws)
    comp = np.load(f"{B}/comp_np201.npz"); i2l = np.load(f"{B}/index_to_line.npy")
    def bid_by_line(e):
        st = comp[f"starts_{e}"]; idx = comp[f"idx_{e}"]
        bid = np.empty(len(idx), np.int64); bid[idx] = np.repeat(np.arange(len(st) - 1), np.diff(st))
        out = np.full(NL, -1, np.int64); out[i2l] = bid; return out
    bid1, bid3 = bid_by_line(1), bid_by_line(3)
    NB1, NB3 = len(comp["starts_1"]) - 1, len(comp["starts_3"]) - 1
    # the realised replay / P3 prefix, as batch indicators (consistency check with 04's masks)
    z1_obs = np.zeros(NB1, bool); z1_obs[np.unique(bid1[rep_line])] = True
    z3_obs = np.zeros(NB3, bool); z3_obs[np.unique(bid3[p3_line])] = True
    assert z1_obs.sum() == 22294 and z3_obs.sum() == 1649
    assert np.array_equal(rep_line[kept], z1_obs[bid1[kept]]) and np.array_equal(p3_line[kept], z3_obs[bid3[kept]])

    order = np.argsort(ck1, kind="stable"); sk = ck1[order]
    order_s = np.argsort(ck1s, kind="stable"); sks = ck1s[order_s]

    # ---- sets --------------------------------------------------------------------------------------
    sets = {}
    for nm in ("ft_un_div", "ft_un_rep", "ft_un_random", "ft_topk_div", "ft_topk_rep", "ft_random_small"):
        sets[nm] = list(zip(read_lines(f"{IN}/phase0_e05/{nm}.en"), read_lines(f"{IN}/phase0_e05/{nm}.fr")))
    fr_en, fr_fr = read_lines(f"{IN}/e05_src/ft_random.en"), read_lines(f"{IN}/e05_src/ft_random.fr")
    fr_src = read_lines(f"{IN}/e05_src/ft_random.source")
    # rows.pkl's ft_random table must be in ft_random's line order before its indices are used
    R = pickle.load(open(f"{B}/rows.pkl", "rb"))["ft_random"]
    WORD = re.compile(r"\w+", re.UNICODE)
    nQ = len(R["len"])
    pk = {"rows_pkl_rows": int(nQ), "ft_random_lines": len(fr_en)}
    pk["len_equal"] = bool(nQ == len(fr_en) and np.array_equal(R["len"], [len(WORD.findall(x)) for x in fr_en]))
    pk["len_fr_equal"] = bool(nQ == len(fr_fr) and np.array_equal(R["len_fr"], [len(WORD.findall(x)) for x in fr_fr]))
    pk["un_equal"] = bool(np.array_equal(R["un"].astype(bool), np.array([s == "un" for s in fr_src])))
    pos = {p: i for i, p in enumerate(zip(fr_en, fr_fr))}
    for nm in ("ft_un_random", "ft_random_small"):
        m = np.zeros(nQ, bool); m[[pos[p] for p in sets[nm]]] = True
        pk[f"in_{nm}_equal"] = bool(np.array_equal(R[f"in_{nm}"].astype(bool), m))
    pk["row_order_is_line_order"] = all(v for k, v in pk.items() if k.endswith("_equal"))
    assert pk["row_order_is_line_order"], pk
    del R
    for nm, f in (("R0", "R0.npy"), ("A_narrow", "A_narrow_42_58.npy")):
        ix = np.load(f"{IN}/e06_idx/{f}")
        assert ix.min() >= 0 and ix.max() < nQ and len(np.unique(ix)) == len(ix)
        sets[nm] = [(fr_en[i], fr_fr[i]) for i in ix]
        pk[f"{nm}_all_un"] = bool(all(fr_src[i] == "un" for i in ix))
    T["sets_s"] = round(time.time() - t0, 1)

    # ---- matching -----------------------------------------------------------------------------------
    occ = {}            # set name -> (row index per occurrence, corpus line per occurrence)
    per_set = []
    for nm, pairs in sets.items():
        a, b, c = keys(pairs)
        lo = np.searchsorted(sk, a, "left"); hi = np.searchsorted(sk, a, "right")
        cntocc = hi - lo
        rows_o = np.repeat(np.arange(len(pairs)), cntocc)
        lines_o = order[np.concatenate([np.arange(x, y) for x, y in zip(lo, hi)])] if cntocc.sum() else np.empty(0, np.int64)
        assert np.array_equal(ck2[lines_o], b[rows_o]), f"{nm}: second key disagrees"   # no collision
        occ[nm] = (rows_o, lines_o)
        exposed = cntocc > 0
        exposed_kept = np.zeros(len(pairs), bool); exposed_kept[rows_o[kept[lines_o]]] = True
        rep_row = np.zeros(len(pairs), bool); rep_row[rows_o[rep_line[lines_o]]] = True
        expo_row = np.bincount(rows_o, weights=expo_line[lines_o].astype(float), minlength=len(pairs))
        p3_row = np.zeros(len(pairs), bool); p3_row[rows_o[p3_line[lines_o]]] = True
        # trainer's view (strip both sides): exposure only
        los = np.searchsorted(sks, c, "left"); his = np.searchsorted(sks, c, "right")
        ne = int(exposed.sum()); N = len(pairs)
        d = {"set": nm, "rows": N, "exposed_rows": ne, "exposed_share": ne / N,
             "exposed_rows_on_kept_lines": int(exposed_kept.sum()),
             "exposed_rows_strip_rule": int(((his - los) > 0).sum()),
             "exposed_and_replayed": int((exposed & rep_row).sum()),
             "replayed_share_of_exposed": float(rep_row[exposed].mean()) if ne else None,
             "replayed_share_of_all_rows": float(rep_row.mean()),
             "mean_exposure_count_of_exposed": float(expo_row[exposed].mean()) if ne else None,
             "mean_occurrences_of_exposed": float(cntocc[exposed].mean()) if ne else None,
             "exposed_rows_with_several_lines": int((cntocc > 1).sum()),
             "exposed_rows_in_p3_prefix": int((exposed & p3_row).sum()),
             "exposed_rows_by_count": {int(k): int(v) for k, v in zip(*np.unique(expo_row[exposed].astype(int), return_counts=True))},
             "distinct_pairs": int(len(np.unique(a)))}
        per_set.append(d)
        print(nm, {k: d[k] for k in ("rows", "exposed_rows", "exposed_share", "replayed_share_of_exposed", "mean_exposure_count_of_exposed")}, flush=True)
    T["match_s"] = round(time.time() - t0, 1)
    overlap = {f"{x}&{y}": len(set(sets[x]) & set(sets[y])) for x, y in
               (("R0", "A_narrow"), ("R0", "ft_un_random"), ("A_narrow", "ft_un_random"), ("ft_un_div", "ft_un_rep"),
                ("ft_un_div", "ft_un_random"), ("ft_un_rep", "ft_un_random"), ("ft_topk_div", "ft_topk_rep"), ("R0", "ft_un_div"))}

    # ---- re-draws of the content-random replay ---------------------------------------------------------
    rng = np.random.default_rng(20261018)
    NDRAW = 2000
    S = {d["set"]: d for d in per_set}
    def stats_from(z1, z3, nm):
        rows_o, lines_o = occ[nm]; N = S[nm]["rows"]
        k = kept[lines_o]
        r1 = np.zeros(len(lines_o), bool); r1[k] = z1[bid1[lines_o[k]]]
        r3 = np.zeros(len(lines_o), bool); r3[k] = z3[bid3[lines_o[k]]]
        exposed = np.zeros(N, bool); exposed[rows_o] = True
        rep_row = np.zeros(N, bool); rep_row[rows_o[r1]] = True
        expo_row = np.bincount(rows_o, weights=(k * (2 + r1 + r3)).astype(float), minlength=N)
        return rep_row[exposed].mean(), rep_row.mean(), expo_row[exposed].mean()
    for nm in S:                                    # the realised draw reproduces the reported statistics
        a, b, c = stats_from(z1_obs, z3_obs, nm)
        assert abs(a - S[nm]["replayed_share_of_exposed"]) < 1e-12 and abs(c - S[nm]["mean_exposure_count_of_exposed"]) < 1e-9
    draws = {nm: np.empty((NDRAW, 3)) for nm in S}
    for i in range(NDRAW):
        z1 = np.zeros(NB1, bool); z1[rng.choice(NB1, 22294, replace=False)] = True
        z3 = np.zeros(NB3, bool); z3[rng.choice(NB3, 1649, replace=False)] = True
        for nm in S:
            draws[nm][i] = stats_from(z1, z3, nm)
    T["redraws_s"] = round(time.time() - t0, 1)
    np.savez(f"{B}/05_redraws.npz", **draws)

    def naive_prop(p1, n1, p2, n2): return 1.96 * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    contrasts = []
    metric_names = ["replayed_share_of_exposed", "replayed_share_of_all_rows", "mean_exposure_count_of_exposed"]
    for a, b in (("ft_un_rep", "ft_un_div"), ("ft_un_div", "ft_un_random"), ("ft_topk_rep", "ft_topk_div"),
                 ("R0", "ft_un_div"), ("A_narrow", "R0")):
        for mi, m in enumerate(metric_names):
            obs = S[a][m] - S[b][m]
            dd = draws[a][:, mi] - draws[b][:, mi]
            sd = float(dd.std(ddof=1))
            p_two = float((np.abs(dd - dd.mean()) >= abs(obs - dd.mean())).mean())
            if mi < 2:
                na, nb = (S[a]["exposed_rows"], S[b]["exposed_rows"]) if mi == 0 else (S[a]["rows"], S[b]["rows"])
                hw = naive_prop(S[a][m], na, S[b][m], nb)
            else:
                def vals(nm):
                    rows_o, lines_o = occ[nm]; N = S[nm]["rows"]
                    v = np.bincount(rows_o, weights=expo_line[lines_o].astype(float), minlength=N)
                    ex = np.zeros(N, bool); ex[rows_o] = True; return v[ex]
                va, vb = vals(a), vals(b)
                hw = 1.96 * np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))
            contrasts.append({"a": a, "b": b, "metric": m, "diff": obs,
                              "ci95_design": [obs - 1.96 * sd, obs + 1.96 * sd], "redraw_sd": sd,
                              "redraw_null_mean_diff": float(dd.mean()), "p_two_sided_redraw": p_two,
                              "ci95_naive_rowlevel": [obs - hw, obs + hw]})
    out = {"matching_rule": "exact (en, fr) pair, trailing newline removed, any of the 9,312,233 corpus lines",
           "rows_pkl_check": pk, "per_set": per_set, "set_overlaps_pairs": overlap, "contrasts": contrasts,
           "draft_exposure_pct": {"ft_un_div": 36.7, "ft_un_rep": 36.7, "ft_un_random": 37.9,
                                  "ft_topk_div": 52.9, "ft_topk_rep": 42.4, "ft_random_small": 58.5},
           "n_redraws": NDRAW, "timings_s": T, "seconds": round(time.time() - t0, 1)}
    np.savez(f"{B}/set_occurrences.npz", **{f"{nm}_rows": occ[nm][0] for nm in occ}, **{f"{nm}_lines": occ[nm][1] for nm in occ})
    json.dump(out, open(f"{B}/05_sets.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("rows_pkl_check", "set_overlaps_pairs", "timings_s")}, indent=1))
