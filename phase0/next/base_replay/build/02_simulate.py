"""Step 3: rebuild the batch composition of epochs P1, P2, P3 with the April sampler code, verbatim.

TokenBatchSampler is taken from the box's src/data/dataset.py (sha256 593d4550..., = bde629a) by AST and
executed unchanged. Its __iter__ draws from two generators:
  - numpy's legacy global RandomState (np.random.shuffle of arange(n)), then np.argsort per chunk of
    max_sentences*100 rows, then token-budget batching  -> batch COMPOSITION;
  - Python's global random (random.shuffle(batches))                                    -> batch ORDER.
This script fixes the composition: it runs __iter__ three times after np.random.seed(42) with Python's
random replaced by a stub whose shuffle is the identity, so the numpy stream is consumed exactly as in
training and the batches come out in their pre-shuffle (sorted) order. Order is applied in 03.
Run once per numpy build (PYTHONPATH selects the April 2.0.1 copy or the env's 2.2.6).
Writes comp_<tag>.npz: for each epoch e, idx_e (dataset indices after the chunk sort) and
starts_e (batch start offsets into idx_e, B_e+1 entries); also the pre-sort shuffle's sha256.
"""
import ast, hashlib, json, os, random, sys, time
import numpy as np

B = os.path.expanduser("~/replay_check/build")
tag = sys.argv[1]
SRC = open(f"{B}/april_dataset.py", "rb").read()
assert hashlib.sha256(SRC).hexdigest().startswith("593d4550c323c71c"), "not the April dataset.py"
cls = [n for n in ast.parse(SRC).body if isinstance(n, ast.ClassDef) and n.name == "TokenBatchSampler"][0]

class _StubRandom:                       # stands in for the `random` module inside __iter__ only
    @staticmethod
    def shuffle(x):
        return None

ns = {"np": np, "random": _StubRandom, "Sampler": object, "TranslationDataset": object}  # annotation only
exec(compile(ast.Module(body=[cls], type_ignores=[]), "april_dataset.py:TokenBatchSampler", "exec"), ns)
TBS = ns["TokenBatchSampler"]

class DS:
    def __init__(self):
        self.src_lens = np.load(f"{B}/src_lens.npy"); self.tgt_lens = np.load(f"{B}/tgt_lens.npy")
        assert self.src_lens.dtype == np.int32 and self.tgt_lens.dtype == np.int32
    def __len__(self):
        return len(self.src_lens)

# also capture the raw permutation from np.random.shuffle (before the chunk sort), to compare builds
_orig_shuffle = np.random.shuffle
_shuf_hashes = []
def _spy_shuffle(a):
    _orig_shuffle(a)
    _shuf_hashes.append(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
np.random.shuffle = _spy_shuffle

ds = DS(); n = len(ds)
sampler = TBS(ds, 24576, 192, True)            # config: batch_size 24576 tokens, max_sentences 192
print("numpy", np.__version__, np.__file__, "rows", n, "len(sampler)", len(sampler), flush=True)
np.random.seed(42)                              # set_seed(42); nothing else touches numpy's stream
out, info = {}, {"numpy": np.__version__, "numpy_file": np.__file__, "rows": n,
                 "trainer_len_sampler": len(sampler), "epochs": []}
for e in (1, 2, 3):
    t0 = time.time()
    batches = list(iter(sampler))
    lens = np.fromiter((len(b) for b in batches), dtype=np.int64, count=len(batches))
    starts = np.zeros(len(batches) + 1, np.int64); np.cumsum(lens, out=starts[1:])
    idx = np.fromiter((i for b in batches for i in b), dtype=np.int32, count=n)
    assert starts[-1] == n and np.array_equal(np.sort(idx), np.arange(n, dtype=np.int32))
    out[f"idx_{e}"] = idx; out[f"starts_{e}"] = starts
    info["epochs"].append({"epoch": e, "B": len(batches), "seconds": round(time.time() - t0, 1),
                           "presort_shuffle_sha256": _shuf_hashes[-1],
                           "sorted_order_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
                           "rows_per_batch_min_max": [int(lens.min()), int(lens.max())],
                           "batches_lt_192_rows": int((lens < 192).sum())})
    print(info["epochs"][-1], flush=True)
np.savez(f"{B}/comp_{tag}.npz", **out)
json.dump(info, open(f"{B}/02_simulate_{tag}.json", "w"), indent=1)
