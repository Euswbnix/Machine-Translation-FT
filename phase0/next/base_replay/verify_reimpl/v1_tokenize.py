"""v1: independent re-tokenisation of the v1.1 corpus, the way the April dataset did it.

April rule (src/data/dataset.py @ bde629a, _build_from_text):
  text-mode open (universal newlines), zip(src, tgt), strip() both,
  skip the pair if either side is empty, sp.encode(add_bos=True, add_eos=True),
  drop the pair if either side has more than 256 ids. Dataset index i = i-th surviving pair.
Checks the result against the trainer's own cache (train.clean.en.cached_256.npz).
Writes: v1_index_to_line.npy (int64), v1_line_status.npy (uint8: 0 kept, 1 empty, 2 too long),
        v1_tokenize.json
"""
import hashlib, json, os, sys, time
import multiprocessing as mp
import numpy as np

D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
OUT = os.path.expanduser("~/replay_check/verify_reimpl")
SRC, TGT = f"{D}/train.clean.en", f"{D}/train.clean.fr"
SPM, CACHE = f"{D}/spm_enfr_v1_fixed.model", f"{D}/train.clean.en.cached_256.npz"
MAXTOK, CH, NPROC = 256, 50_000, 16

_sp = None


def _init(path):
    global _sp
    import sentencepiece as spm
    _sp = spm.SentencePieceProcessor(model_file=path)


def _work(args):
    ci, s_lines, t_lines = args
    a = _sp.encode(s_lines, add_bos=True, add_eos=True, out_type=int, num_threads=1)
    b = _sp.encode(t_lines, add_bos=True, add_eos=True, out_type=int, num_threads=1)
    sl = np.array([len(x) for x in a], dtype=np.int32)
    tl = np.array([len(x) for x in b], dtype=np.int32)
    keep = (sl <= MAXTOK) & (tl <= MAXTOK)
    ks = [np.asarray(x, dtype=np.int64) for x, k in zip(a, keep) if k]
    kt = [np.asarray(x, dtype=np.int64) for x, k in zip(b, keep) if k]
    st = np.concatenate(ks) if ks else np.empty(0, np.int64)
    tt = np.concatenate(kt) if kt else np.empty(0, np.int64)
    assert st.size == 0 or (st.min() >= 0 and st.max() < 65536)
    assert tt.size == 0 or (tt.min() >= 0 and tt.max() < 65536)
    return ci, sl, tl, keep, st.astype(np.uint16), tt.astype(np.uint16)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def raw_counts(path):
    nl = cr = 0
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            nl += blk.count(b"\n"); cr += blk.count(b"\r")
    return nl, cr


def main():
    t0 = time.time()
    rep = {"inputs": {}}
    for p in (SRC, TGT, SPM):
        rep["inputs"][os.path.basename(p)] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    rep["inputs"][os.path.basename(CACHE)] = {"bytes": os.path.getsize(CACHE)}
    for p in (SRC, TGT):
        nl, cr = raw_counts(p)
        rep["inputs"][os.path.basename(p)].update({"newlines": nl, "cr_bytes": cr})
    print("hashes/raw counts", round(time.time() - t0, 1), "s", flush=True)

    # line numbers of non-empty pairs, in order; universal-newline text mode as in April
    status_parts = []
    line_no = [0]

    def chunks():
        sc, tc, ci = [], [], 0
        st = []
        with open(SRC, "r", encoding="utf-8") as fs, open(TGT, "r", encoding="utf-8") as ft:
            for s, t in zip(fs, ft):
                s, t = s.strip(), t.strip()
                if not s or not t:
                    st.append(1)
                else:
                    st.append(0)
                    sc.append(s); tc.append(t)
                    if len(sc) >= CH:
                        yield (ci, sc, tc); ci += 1; sc, tc = [], []
                line_no[0] += 1
        if sc:
            yield (ci, sc, tc)
        status_parts.append(np.array(st, dtype=np.uint8))

    sls, tls, keeps, sts, tts = [], [], [], [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(NPROC, initializer=_init, initargs=(SPM,)) as pool:
        for ci, sl, tl, keep, st, tt in pool.imap(_work, chunks(), chunksize=1):
            assert ci == len(sls)
            sls.append(sl); tls.append(tl); keeps.append(keep); sts.append(st); tts.append(tt)
    print("tokenised", round(time.time() - t0, 1), "s", flush=True)

    n_lines = line_no[0]
    status = status_parts[0]
    assert status.size == n_lines
    sl = np.concatenate(sls); tl = np.concatenate(tls); keep = np.concatenate(keeps)
    nonempty_lines = np.flatnonzero(status == 0)
    assert nonempty_lines.size == sl.size
    status[nonempty_lines[~keep]] = 2
    index_to_line = nonempty_lines[keep].astype(np.int64)
    src_tok = np.concatenate(sts); tgt_tok = np.concatenate(tts)

    c = np.load(CACHE)
    keys = sorted(c.files)
    so, to = c["src_offsets"], c["tgt_offsets"]
    cs_len, ct_len = np.diff(so), np.diff(to)
    cmp = {
        "cache_members": keys,
        "cache_dtypes": {k: str(c[k].dtype) for k in keys},
        "cache_rows": int(so.size - 1),
        "my_rows": int(index_to_line.size),
        "src_len_equal": bool(cs_len.size == keep.sum() and np.array_equal(cs_len, sl[keep])),
        "tgt_len_equal": bool(ct_len.size == keep.sum() and np.array_equal(ct_len, tl[keep])),
    }
    cmp["src_tokens_equal"] = bool(np.array_equal(c["src_tokens"], src_tok))
    cmp["tgt_tokens_equal"] = bool(np.array_equal(c["tgt_tokens"], tgt_tok))
    rep["lines"] = int(n_lines)
    rep["empty_pairs_skipped"] = int((status == 1).sum())
    rep["dropped_over_256"] = int((status == 2).sum())
    rep["kept"] = int((status == 0).sum())
    rep["cache_check"] = cmp
    rep["index_to_line_sha256_rawbytes"] = hashlib.sha256(index_to_line.tobytes()).hexdigest()
    rep["seconds"] = round(time.time() - t0, 1)
    np.save(f"{OUT}/v1_index_to_line.npy", index_to_line)
    np.save(f"{OUT}/v1_line_status.npy", status)
    json.dump(rep, open(f"{OUT}/v1_tokenize.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
