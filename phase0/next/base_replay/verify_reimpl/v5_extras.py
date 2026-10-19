"""v5: sensitivity of the composition to argsort tie order, and a few row counts at the edges.

Usage: python -B v5_extras.py KIND OUT_TAG
  KIND = default | stable   (argsort kind used in the chunk sort)
  Run with NPY_DISABLE_CPU_FEATURES set to see a non-AVX-512 dispatch.
"""
import json, os, random, sys
import numpy as np

OUT = os.path.expanduser("~/replay_check/verify_reimpl")
CACHE = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1/train.clean.en.cached_256.npz")
MAX_TOK, MAX_SENT, CHUNK = 24576, 192, 19200
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def batch_starts(L):
    starts = [0]; cnt = mx = 0
    for p, l in enumerate(L):
        nm = l if l > mx else mx
        if (cnt + 1) * nm > MAX_TOK or cnt >= MAX_SENT:
            if cnt:
                starts.append(p)
            cnt, mx = 1, l
        else:
            cnt, mx = cnt + 1, nm
    return np.array(starts)


kind, tag = sys.argv[1], sys.argv[2]
c = np.load(CACHE)
src = np.diff(c["src_offsets"]).astype(np.int32); tgt = np.diff(c["tgt_offsets"]).astype(np.int32)
seq = np.maximum(src, tgt); n = src.size
random.seed(42); np.random.seed(42)
for _ in range(13):
    random.choices(ALPHABET, k=8)
rep = {"kind": kind, "disabled_features": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "numpy": np.__version__}
ref = np.load(f"{OUT}/v2_comp_np226.npz")
for e in (1, 2, 3):
    idx = np.arange(n); np.random.shuffle(idx)
    for i in range(0, n, CHUNK):
        ch = idx[i:i + CHUNK]
        idx[i:i + CHUNK] = ch[np.argsort(src[ch]) if kind == "default" else np.argsort(src[ch], kind="stable")]
    st = batch_starts(seq[idx].tolist())
    p = list(range(st.size)); random.shuffle(p)
    rep[f"B{e}"] = int(st.size)
    rep[f"same_composition_as_main_e{e}"] = bool(np.array_equal(idx, ref[f"idx{e}"]) and np.array_equal(st, ref[f"starts{e}"]))
    if e == 1:
        rows = np.diff(np.append(st, n)); bop = np.repeat(np.arange(st.size), rows)
        pob = np.empty(st.size, np.int64); pob[np.array(p)] = np.arange(st.size)
        p1 = np.empty(n, np.int64); p1[idx] = pob[bop]
        mine = np.load(f"{OUT}/v2_replayed_row_mask_np226.npy")
        alt = p1 < 22294
        rep["replayed_rows"] = int(alt.sum())
        rep["replayed_rows_overlap_with_main"] = int((alt & mine).sum())
        rep["jaccard_with_main"] = round(float((alt & mine).sum() / (alt | mine).sum()), 4)
rep["logged"] = {"B1": 49174, "B1+B2": 98351}
rep["matches_log"] = rep["B1"] == 49174 and rep["B1"] + rep["B2"] == 98351

if kind == "default" and not rep["disabled_features"]:
    # edge counts on the main reconstruction
    rows1, perm1 = ref["rows1"], ref["perm1"]
    mxt1, mxs1 = ref["mxt1"], ref["mxs1"]
    def rows_at(positions):
        return int(sum(int(rows1[perm1[q]]) for q in positions))
    # spike guard (session 1): accumulation groups of 4 containing steps 10,443 and 12,074 were dropped
    groups = []
    for s in (10443, 12074):
        g_end = ((s + 3) // 4) * 4
        groups.append(list(range(g_end - 3, g_end + 1)))
    rep["spike_groups_steps"] = groups
    rep["spike_groups_rows"] = rows_at([g - 1 for G in groups for g in G])
    for s in (10443, 12074):
        b = perm1[s - 1]
        rep[f"spike_{s}_max_tgt_percentile"] = round(float((mxt1 < mxt1[b]).mean() * 100), 3)
        rep[f"spike_{s}_max_src_percentile"] = round(float((mxs1 < mxs1[b]).mean() * 100), 3)
    # session 2 ended at 122,294: steps 122,293-122,294 never reached an optimizer step (122,292 % 4 == 0)
    rep["unstepped_tail_positions"] = [22292, 22293]
    rep["unstepped_tail_rows"] = rows_at([22292, 22293])
    rep["replayed_rows_check"] = rows_at(range(22294))
json.dump(rep, open(f"{OUT}/v5_{tag}.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
