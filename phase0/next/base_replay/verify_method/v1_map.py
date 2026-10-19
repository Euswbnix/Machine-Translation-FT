"""Auditor check 1: dataset index -> corpus line mapping (builder's index_to_line.npy / line_status.npy),
verified by sampled re-tokenisation against the trainer's cache, done the April dataset's way."""
import json, os, random, time
import numpy as np, sentencepiece as spm
H = os.path.expanduser("~"); D = f"{H}/workspace/Machine_translation/data_enfr_v1"; B = f"{H}/replay_check/build"
V = f"{H}/replay_check/verify_method"
t0 = time.time()
i2l = np.load(f"{B}/index_to_line.npy"); status = np.load(f"{B}/line_status.npy")
c = np.load(f"{D}/train.clean.en.cached_256.npz")
so, to = c["src_offsets"], c["tgt_offsets"]; n = len(so) - 1
res = {"n_cache_rows": int(n), "n_i2l": int(len(i2l)), "n_lines": int(len(status)),
       "i2l_strictly_increasing": bool(np.all(np.diff(i2l) > 0)),
       "kept_lines_equal_i2l": bool(np.array_equal(np.nonzero(status == 0)[0], i2l)),
       "dropped": int((status == 1).sum()), "empty": int((status == 2).sum())}
rng = random.Random(20261018)
sample = sorted(set(rng.sample(range(n), 100_000)) | {0, 1, n - 2, n - 1})
want = {int(i2l[i]): i for i in sample}
dropped = set(np.nonzero(status == 1)[0].tolist())
need = set(want) | dropped
got = {}
with open(f"{D}/train.clean.en", "r") as fe, open(f"{D}/train.clean.fr", "r") as ff:
    nl = 0
    for ln, (a, b) in enumerate(zip(fe, ff)):
        nl += 1
        if ln in need:
            got[ln] = (a.strip(), b.strip())
res["text_mode_line_count"] = nl
sp = spm.SentencePieceProcessor(); sp.Load(f"{D}/spm_enfr_v1_fixed.model")
st, tt = c["src_tokens"], c["tgt_tokens"]
mis = 0
for ln, i in want.items():
    a, b = got[ln]
    s = sp.encode(a, add_bos=True, add_eos=True, out_type=int); t = sp.encode(b, add_bos=True, add_eos=True, out_type=int)
    if s != st[so[i]:so[i + 1]].tolist() or t != tt[to[i]:to[i + 1]].tolist():
        mis += 1
res["sampled_rows"] = len(want); res["sampled_token_mismatches"] = mis
bad_drop = 0
for ln in dropped:
    a, b = got[ln]
    if not (len(sp.encode(a, add_bos=True, add_eos=True)) > 256 or len(sp.encode(b, add_bos=True, add_eos=True)) > 256):
        bad_drop += 1
res["dropped_lines_not_over_256"] = bad_drop
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"{V}/v1_map.json", "w"), indent=1); print(json.dumps(res, indent=1))
