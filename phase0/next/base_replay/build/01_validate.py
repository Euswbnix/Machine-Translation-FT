"""Step 1-2: rebuild the trainer's tokenised dataset from the corpus and require it to equal the cache.

Replicates src/data/dataset.py TranslationDataset._build_from_text at bde629a (the commit both April-24
sessions ran, per swanlab metadata; byte-identical to the box's working copy, mtime 2026-04-19):
  - read both sides with open(path, "r") (text mode, universal newlines, locale encoding) and zip();
  - line.strip(); skip the pair if either side is empty (skipped lines get no dataset index);
  - chunks of 50,000 kept pairs, each encoded by a SentencePieceProcessor rebuilt from the
    serialized proto, sp.encode(lines, add_bos=True, add_eos=True, out_type=int);
  - drop the pair if len(src_ids) > 256 or len(tgt_ids) > 256 (max_tokens = model.max_seq_len = 256);
  - kept pairs are concatenated in order (imap is ordered), so dataset index i = i-th kept pair.
Outputs: index_to_line.npy (int64, 0-based corpus line of each dataset index), line_status.npy
(int8 per corpus line: 0 kept, 1 dropped by the length filter, 2 skipped as empty), 01_validate.json.
"""
import hashlib, json, locale, multiprocessing as mp, os, sys, time
import numpy as np
import sentencepiece as spm

B = os.path.expanduser("~/replay_check/build")
D = os.path.expanduser("~/workspace/Machine_translation/data_enfr_v1")
SRC, TGT = f"{D}/train.clean.en", f"{D}/train.clean.fr"
SPM, CACHE = f"{D}/spm_enfr_v1_fixed.model", f"{D}/train.clean.en.cached_256.npz"
MAX_TOKENS, CHUNK, NPROC = 256, 50_000, 16

_SP = None
def _init(proto):
    global _SP
    _SP = spm.SentencePieceProcessor()
    _SP.LoadFromSerializedProto(proto)

def _work(args):
    ci, lines_no, src_lines, tgt_lines = args
    s = _SP.encode(src_lines, add_bos=True, add_eos=True, out_type=int)
    t = _SP.encode(tgt_lines, add_bos=True, add_eos=True, out_type=int)
    sl = np.fromiter((len(x) for x in s), dtype=np.int32, count=len(s))
    tl = np.fromiter((len(x) for x in t), dtype=np.int32, count=len(t))
    keep = (sl <= MAX_TOKENS) & (tl <= MAX_TOKENS)
    ks = [x for x, k in zip(s, keep) if k]
    kt = [x for x, k in zip(t, keep) if k]
    sa = np.fromiter((v for x in ks for v in x), dtype=np.uint16) if ks else np.empty(0, np.uint16)
    ta = np.fromiter((v for x in kt for v in x), dtype=np.uint16) if kt else np.empty(0, np.uint16)
    return ci, np.asarray(lines_no, np.int64), keep, sl, tl, sa, ta

def chunks(status):
    sb, tb, nb, ci = [], [], [], 0
    with open(SRC, "r") as fs, open(TGT, "r") as ft:   # exactly as the dataset opens them
        for ln, (a, b) in enumerate(zip(fs, ft)):
            a, b = a.strip(), b.strip()
            if not a or not b:
                status[ln] = 2
                continue
            sb.append(a); tb.append(b); nb.append(ln)
            if len(sb) >= CHUNK:
                yield (ci, nb, sb, tb); ci += 1; sb, tb, nb = [], [], []
    if sb:
        yield (ci, nb, sb, tb)

def main():
    t0 = time.time()
    enc = locale.getpreferredencoding(False)
    n_lines = {p: sum(1 for _ in open(p, "rb")) for p in (SRC, TGT)}
    n_text = {p: sum(1 for _ in open(p, "r")) for p in (SRC, TGT)}
    N = n_text[SRC]
    status = np.zeros(N, np.int8)
    proto = open(SPM, "rb").read()
    sp = spm.SentencePieceProcessor(); sp.LoadFromSerializedProto(proto)
    sp_ref = spm.SentencePieceProcessor(); sp_ref.Load(SPM)       # what Tokenizer() does
    assert sp.serialized_model_proto() == sp_ref.serialized_model_proto()
    idx_parts, s_parts, t_parts, sl_parts, tl_parts = [], [], [], [], []
    dropped = []
    with mp.get_context("fork").Pool(NPROC, initializer=_init, initargs=(proto,)) as pool:
        for ci, lines_no, keep, sl, tl, sa, ta in pool.imap(_work, chunks(status), chunksize=1):
            idx_parts.append(lines_no[keep]); s_parts.append(sa); t_parts.append(ta)
            sl_parts.append(sl[keep]); tl_parts.append(tl[keep])
            for ln, a, b in zip(lines_no[~keep], sl[~keep], tl[~keep]):
                dropped.append((int(ln), int(a), int(b)))
    t_tok = time.time() - t0
    idx = np.concatenate(idx_parts); my_sl = np.concatenate(sl_parts); my_tl = np.concatenate(tl_parts)
    my_s = np.concatenate(s_parts); my_t = np.concatenate(t_parts)
    for ln, _, _ in dropped:
        status[ln] = 1
    # the cache, read exactly as TranslationDataset.__init__ does
    c = np.load(CACHE)
    c_so, c_to = c["src_offsets"], c["tgt_offsets"]
    c_sl = np.diff(c_so).astype(np.int32); c_tl = np.diff(c_to).astype(np.int32)
    res = {
        "code": "TranslationDataset._build_from_text @ bde629a (box copy sha256 593d4550c323c71c...)",
        "locale_encoding_now": enc,
        "corpus_lines_binary": {os.path.basename(k): v for k, v in n_lines.items()},
        "corpus_lines_text_mode": {os.path.basename(k): v for k, v in n_text.items()},
        "kept_rows": int(len(idx)), "cache_rows": int(len(c_sl)),
        "skipped_empty_after_strip": int((status == 2).sum()),
        "dropped_over_max_tokens": len(dropped),
        "dropped_src_only": sum(1 for _, a, b in dropped if a > MAX_TOKENS and b <= MAX_TOKENS),
        "dropped_tgt_only": sum(1 for _, a, b in dropped if a <= MAX_TOKENS and b > MAX_TOKENS),
        "dropped_both": sum(1 for _, a, b in dropped if a > MAX_TOKENS and b > MAX_TOKENS),
        "cache_dtypes": {k: str(c[k].dtype) for k in c.files},
        "cache_keys": list(c.files),
        "index_to_line_increasing": bool(np.all(np.diff(idx) > 0)),
    }
    same_n = len(idx) == len(c_sl)
    res["src_lengths_equal_all_rows"] = bool(same_n and np.array_equal(my_sl, c_sl))
    res["tgt_lengths_equal_all_rows"] = bool(same_n and np.array_equal(my_tl, c_tl))
    if same_n:
        res["src_len_mismatch_rows"] = int((my_sl != c_sl).sum()); res["tgt_len_mismatch_rows"] = int((my_tl != c_tl).sum())
    res["src_tokens_equal"] = bool(np.array_equal(my_s, c["src_tokens"]))
    res["tgt_tokens_equal"] = bool(np.array_equal(my_t, c["tgt_tokens"]))
    res["offsets_start_at_zero"] = bool(c_so[0] == 0 and c_to[0] == 0)
    res["validated"] = bool(res["src_lengths_equal_all_rows"] and res["tgt_lengths_equal_all_rows"]
                            and res["src_tokens_equal"] and res["tgt_tokens_equal"])
    res["dropped_examples_line_srclen_tgtlen"] = dropped[:10]
    res["length_stats"] = {"src_mean": float(c_sl.mean()), "tgt_mean": float(c_tl.mean()),
                           "maxlen_gt_128_rows": int((np.maximum(c_sl, c_tl) > 128).sum())}
    np.save(f"{B}/index_to_line.npy", idx)
    np.save(f"{B}/line_status.npy", status)
    np.save(f"{B}/src_lens.npy", c_sl); np.save(f"{B}/tgt_lens.npy", c_tl)
    res["index_to_line_sha256"] = hashlib.sha256(idx.tobytes()).hexdigest()
    res["seconds_tokenise"] = round(t_tok, 1); res["seconds_total"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{B}/01_validate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not res["validated"]:
        sys.exit("VALIDATION FAILED")

if __name__ == "__main__":
    main()
