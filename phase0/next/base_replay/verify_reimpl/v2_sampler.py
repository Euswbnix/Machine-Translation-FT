"""v2: independent re-implementation of the April TokenBatchSampler for epochs 1-3.

Usage: python -B v2_sampler.py TAG [K_WORDS]
  TAG     label for the numpy in use (run with PYTHONPATH pointing at the April numpy for 'np201')
  K_WORDS optional: skip this many 32-bit MT words instead of emulating swanlab's 13 random_index() calls

RNG facts used (all checked elsewhere):
  train.py set_seed(42): random.seed(42); np.random.seed(42)
  swanlab.init -> get_hardware_info(): 13 x random_index() = 13 x random.choices(ALPHABET, k=8)
  (NVIDIA 6, AMD 5, disk 1, network 1); their strings are recorded in backup.swanlab and are
  asserted here. np.random is untouched until the first sampler pass.
Sampler (dataset.py @ bde629a): per epoch np.random.shuffle(arange(n)); within chunks of
  192*100 positions reorder by np.argsort(src_lens[chunk]) (default kind, int32 keys);
  greedy batches with (len+1)*max(max(src,tgt)) <= 24576 and <= 192 rows; random.shuffle(batches).
"""
import hashlib, json, os, random, sys, time
import numpy as np

OUT = os.path.expanduser("~/replay_check/verify_reimpl")
CACHE = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1/train.clean.en.cached_256.npz")
MAX_TOK, MAX_SENT = 24576, 192
CHUNK = MAX_SENT * 100
LOGGED = ["xaji0y6d", "pbhsahxt", "hv3a3zmf", "8mdd4v30", "t9nt3w5u", "zbikcidk",   # NVIDIA
          "wnnhj7xv", "g0fn9xuy", "41ibljh7", "5lxo6qji", "ujv6oh9s",               # AMD
          "dbdw2pcn", "9t84azyt"]                                                   # disk, network
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
REPLAY = 22_294          # batches trained in session 2 (steps 100,001..122,294)
SESSION1 = 100_000


def batch_starts(L):
    """L: list of per-position max(src,tgt) lengths in sampler order. Returns batch start positions."""
    starts = [0]
    cnt, mx = 0, 0
    for p, l in enumerate(L):
        nm = l if l > mx else mx
        if (cnt + 1) * nm > MAX_TOK or cnt >= MAX_SENT:
            if cnt:
                starts.append(p)
            cnt, mx = 1, l
        else:
            cnt, mx = cnt + 1, nm
    return np.array(starts, dtype=np.int64)


def main():
    tag = sys.argv[1]
    kwords = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t0 = time.time()
    c = np.load(CACHE)
    src_lens = np.diff(c["src_offsets"]).astype(np.int32)
    tgt_lens = np.diff(c["tgt_offsets"]).astype(np.int32)
    n = src_lens.size
    seq = np.maximum(src_lens, tgt_lens)

    random.seed(42)
    np.random.seed(42)
    if kwords is None:
        got = ["".join(random.choices(ALPHABET, k=8)) for _ in LOGGED]
        assert got == LOGGED, (got, LOGGED)
    else:
        for _ in range(kwords):
            random.getrandbits(32)
    mt_pos = random.getstate()[1][-1]

    rep = {"tag": tag, "numpy": np.__version__, "numpy_file": np.__file__, "n_rows": int(n),
           "python_rng": "emulated swanlab draws (asserted)" if kwords is None else f"skipped {kwords} words",
           "mt_position_before_first_shuffle": int(mt_pos), "epochs": {}}
    save = {}
    for e in (1, 2, 3):
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, CHUNK):
            ch = idx[i:i + CHUNK]
            idx[i:i + CHUNK] = ch[np.argsort(src_lens[ch])]
        starts = batch_starts(seq[idx].tolist())
        B = starts.size
        perm = list(range(B))
        random.shuffle(perm)
        perm = np.array(perm, dtype=np.int64)              # perm[j] = batch id at epoch position j
        bounds = np.append(starts, n)
        rows = np.diff(bounds)
        # per-batch stats in batch-id order
        s_sorted, t_sorted = src_lens[idx], tgt_lens[idx]
        mxs = np.maximum.reduceat(s_sorted, starts)
        mxt = np.maximum.reduceat(t_sorted, starts)
        ntok = np.add.reduceat(t_sorted.astype(np.int64) - 1, starts)   # trainer's n_tokens
        save[f"idx{e}"] = idx
        save[f"starts{e}"] = starts
        save[f"perm{e}"] = perm
        save[f"rows{e}"] = rows
        save[f"mxs{e}"] = mxs
        save[f"mxt{e}"] = mxt
        save[f"ntok{e}"] = ntok
        assert np.all(rows <= MAX_SENT) and np.all(rows * np.maximum(mxs, mxt) <= MAX_TOK)
        rep["epochs"][e] = {"batches": int(B), "comp_sha": hashlib.sha256(idx.tobytes() + starts.tobytes()).hexdigest(),
                            "order_sha": hashlib.sha256(perm.tobytes()).hexdigest()}
        print(f"epoch {e}: B={B} ({time.time()-t0:.1f}s)", flush=True)

    B1, B2, B3 = (rep["epochs"][e]["batches"] for e in (1, 2, 3))
    rep["logged_epoch_ends_predicted"] = [B1, B1 + B2]
    p3_batches = SESSION1 - B1 - B2
    rep["p3_batches_in_session1"] = int(p3_batches)

    # row -> epoch-1 position of its batch
    batch_of_pos1 = np.repeat(np.arange(B1), save["rows1"])      # batch id for each sorted position
    pos_of_batch1 = np.empty(B1, np.int64); pos_of_batch1[save["perm1"]] = np.arange(B1)
    p1_pos = np.empty(n, np.int32); p1_pos[save["idx1"]] = pos_of_batch1[batch_of_pos1]
    replayed = p1_pos < REPLAY
    batch_of_pos3 = np.repeat(np.arange(B3), save["rows3"])
    pos_of_batch3 = np.empty(B3, np.int64); pos_of_batch3[save["perm3"]] = np.arange(B3)
    p3_pos = np.empty(n, np.int32); p3_pos[save["idx3"]] = pos_of_batch3[batch_of_pos3]
    p3pre = p3_pos < p3_batches
    expo = 2 + replayed.astype(np.int8) + p3pre.astype(np.int8)
    rep["replayed_rows"] = int(replayed.sum())
    rep["replayed_share"] = float(replayed.mean())
    rep["p3_prefix_rows"] = int(p3pre.sum())
    rep["exposure_hist"] = {int(k): int(v) for k, v in zip(*np.unique(expo, return_counts=True))}
    rep["replayed_mask_sha"] = hashlib.sha256(np.packbits(replayed).tobytes()).hexdigest()
    rep["seconds"] = round(time.time() - t0, 1)
    sfx = tag if kwords is None else f"{tag}_K{kwords}"
    np.savez(f"{OUT}/v2_comp_{sfx}.npz", **save)
    np.save(f"{OUT}/v2_replayed_row_mask_{sfx}.npy", replayed)
    np.save(f"{OUT}/v2_p1_pos_by_row_{sfx}.npy", p1_pos)
    np.save(f"{OUT}/v2_p3prefix_row_mask_{sfx}.npy", p3pre)
    json.dump(rep, open(f"{OUT}/v2_{sfx}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
