"""Score parallel corpus with CometKiwi-22 (reference-free QE).

Output: TSV `<score>\\t<src>\\t<tgt>` per line, in the SAME order as input
(tabs inside src/tgt are replaced by spaces).

CometKiwi-22 (Unbabel/wmt22-cometkiwi-da; InfoXLM-large backbone, HF base_model
microsoft/infoxlm-large). Throughput (derived from the paper's run, not re-measured):
30,129,500 pairs in 13.8 h on one RTX 5090 at batch_size=64, ~600 pairs/s. See
PROTOCOL.md D7.

ONE GPU PER PROCESS. --gpus > 1 is refused: COMET's predict() starts a DDP
Trainer on every call, and calling it once per chunk hangs from the second chunk
on (ranks > 0 exit() inside the first predict). For several GPUs use
phase0/score_sharded.py, which splits the input into contiguous shards and runs
one `--gpus 1 --resume` process of this script per device.

Line handling: --src/--tgt and the output are read and written with
newline="\\n" and strict UTF-8, so only LF ends a line. CR, VT, NEL, U+2028 etc.
stay inside the text. Both input sides must have the same line count.

Resume: pass --resume to continue an interrupted run. Any partial last output
line (bytes after the final LF, e.g. from a kill mid-write) is truncated first,
then the complete lines are counted, that many input lines are skipped, and the
output is appended. Each chunk is flushed and fsync'ed.

Single instance: the scorer takes a non-blocking flock on <out>.lock before touching
<out> and exits 2 if another process holds it, so a second scorer (e.g. an orphan of a
killed score_sharded.py parent) can never append duplicate rows.

Stack identity on resume: with --resume, an existing partial output and --meta-out, the
earlier meta's packages / torch_cuda / gpu_names / python / model / checkpoint_path /
batch_size / model_revision / encoder_revision must equal the current stack, otherwise exit 3
(rows 0..skip-1 and skip.. would come from different stacks); an unreadable earlier meta with
rows already scored is also exit 3. With --resume and NO scored rows yet (a shard that died
before its first chunk), a readable earlier meta that differs still exits 3, so a restarted
shard cannot silently switch stack relative to shards that already finished; an unreadable
one is only warned about. --allow-stack-change accepts the change and records it. The meta
keeps a `segments` list with one entry per (re)start that scored rows, never replaced; a
restart from zero rows under --allow-stack-change moves the old segments to
`restarted_from_empty`.

Exit codes: 0 ok; 1 scoring failed; 2 bad arguments/inputs, lock held, or the resolved hub
revision differs from the requested one; 3 scoring-stack refusal (see above).

Pinned hub revisions: the CometKiwi snapshot is downloaded at --model-revision (default
1ad785194e391eebc6c53e2d0776cada8f83179a, HF API sha of Unbabel/wmt22-cometkiwi-da read
2026-09-13) and loaded with local_files_only=True. COMET builds the InfoXLM tokenizer/config
from the hub name microsoft/infoxlm-large, which with local_files_only resolves the cache's
refs/main; so the scorer first resolves that repo's main online (tokenizer/config files
only) and refuses (exit 2) unless it is --encoder-revision (default
d616d637f0720deda963cebbfc630657d2b7d3ae, HF API read 2026-09-13). Both resolved revisions
are recorded in the meta and are identity fields. --model-revision "" restores the old
unpinned download_model() path (revisions recorded as null). Not executed against the real
COMET/huggingface_hub here (neither is installed); tests use fakes.

comet, torch and tqdm are imported lazily, so --help, the --gpus refusal and the
helpers work without them.
"""

import argparse
import datetime
import fcntl
import json
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"
DEFAULT_MODEL_REVISION = "1ad785194e391eebc6c53e2d0776cada8f83179a"
ENCODER_REPO = "microsoft/infoxlm-large"
DEFAULT_ENCODER_REVISION = "d616d637f0720deda963cebbfc630657d2b7d3ae"
ENCODER_FILES = ["*.json", "*.model", "*.txt"]      # tokenizer + config; weights come from the kiwi checkpoint
EXIT_STACK = 3
META_PACKAGES = ("unbabel-comet", "torch", "transformers", "pytorch-lightning",
                 "lightning", "sentencepiece", "numpy")


def format_line(score, src: str, tgt: str) -> str:
    """The exact output line format (kept byte-identical to the paper's run)."""
    src_safe = src.replace("\t", " ")
    tgt_safe = tgt.replace("\t", " ")
    return f"{score:.6f}\t{src_safe}\t{tgt_safe}\n"


def open_text(path, mode="r"):
    return open(path, mode, encoding="utf-8", newline="\n")


def count_lines(path) -> int:
    """Lines as this script reads them: LF-terminated, strict UTF-8. A final line
    without LF counts as a line."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with open_text(path) as f:
        for _ in f:
            n += 1
    return n


def truncate_partial_last_line(path) -> int:
    """Drop bytes after the final LF (all bytes if there is none). Returns the
    number of bytes dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "rb+") as f:
        size = f.seek(0, os.SEEK_END)
        pos = size
        block = 1 << 16
        keep = 0
        while pos > 0:
            start = max(0, pos - block)
            f.seek(start)
            buf = f.read(pos - start)
            i = buf.rfind(b"\n")
            if i >= 0:
                keep = start + i + 1
                break
            pos = start
        if keep != size:
            f.truncate(keep)
            f.flush()
            os.fsync(f.fileno())
        return size - keep


def fast_forward(file_handle, n: int):
    for _ in range(n):
        if not file_handle.readline():
            return


def chunked(src_f, tgt_f, chunk_size):
    buf = []
    while True:
        s = src_f.readline()
        t = tgt_f.readline()
        if not s or not t:
            if s or t:
                raise RuntimeError("src and tgt ran out at different lines")
            break
        buf.append({"src": s.rstrip("\n"), "mt": t.rstrip("\n")})
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def load_model(args):
    """Returns (model, resolved checkpoint path, {"model_revision", "encoder_revision"}).
    Tests replace this function (and exercise it with fake huggingface_hub/comet modules)."""
    if not args.model_revision:
        from comet import download_model, load_from_checkpoint
        ckpt_path = download_model(args.model)
        return load_from_checkpoint(ckpt_path), ckpt_path, {"model_revision": None, "encoder_revision": None}
    from huggingface_hub import snapshot_download
    kiwi_dir = Path(snapshot_download(repo_id=args.model, revision=args.model_revision))
    # COMET's encoder calls from_pretrained("microsoft/infoxlm-large", local_files_only=...);
    # resolving "main" here writes refs/main, which is what that local lookup then uses.
    enc_dir = Path(snapshot_download(repo_id=ENCODER_REPO, revision="main", allow_patterns=ENCODER_FILES))
    from comet import load_from_checkpoint
    ckpt_path = kiwi_dir / "checkpoints" / "model.ckpt"
    revs = {"model_revision": kiwi_dir.name, "encoder_revision": enc_dir.name}
    if revision_mismatch(args, revs):          # refuse before the (slow) model load
        return None, ckpt_path, revs
    return load_from_checkpoint(str(ckpt_path), local_files_only=True), ckpt_path, revs


def revision_mismatch(args, revs) -> list[str]:
    """Requested vs resolved hub revisions; empty when pinning is off (--model-revision '')."""
    if not args.model_revision:
        return []
    want = {"model_revision": args.model_revision, "encoder_revision": args.encoder_revision}
    return [f"{k}: requested {want[k]}, resolved {revs.get(k)}" for k in want if want[k] and revs.get(k) != want[k]]


def collect_meta(args, ckpt_path, extra=None) -> dict:
    """Scoring-stack metadata. Every field is best effort: unavailable -> None."""
    meta = {"script": "scripts/score_with_comet.py",
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "model": args.model, "checkpoint_path": str(ckpt_path) if ckpt_path else None,
            "batch_size": args.batch_size, "chunk_size": args.chunk_size, "gpus": args.gpus,
            "precision": "not passed to predict() (Lightning default)",
            "length_batching": "not passed to predict() (COMET default; applies only when gpus < 2)",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "src": str(args.src), "tgt": str(args.tgt), "out": str(args.out),
            "packages": {}, "torch_cuda": None, "gpu_names": None}
    try:
        from importlib import metadata as md
        for p in META_PACKAGES:
            try:
                meta["packages"][p] = md.version(p)
            except Exception:
                meta["packages"][p] = None
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433
        meta["torch_cuda"] = getattr(torch.version, "cuda", None)
        try:
            if torch.cuda.is_available():
                meta["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            else:
                meta["gpu_names"] = []
        except Exception:
            meta["gpu_names"] = None
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


IDENTITY_KEYS = ("packages", "torch_cuda", "gpu_names", "python", "model", "checkpoint_path", "batch_size",
                 "model_revision", "encoder_revision")


def stack_diff(old: dict, new: dict) -> list[str]:
    return [k for k in IDENTITY_KEYS
            if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]


def segment_of(meta: dict, allowed: bool) -> dict:
    return {**{k: meta.get(k) for k in IDENTITY_KEYS}, "utc": meta.get("utc"),
            "resume_skip": meta.get("resume_skip"), "stack_change_allowed": allowed}


def take_lock(path):
    """Non-blocking exclusive flock on <path>; returns the open fd, or None if held."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())          # a host crash must not leave a zero-length meta after the rename
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Source side text file")
    ap.add_argument("--tgt", required=True, help="Target side text file")
    ap.add_argument("--out", required=True, help="Output TSV: score\\tsrc\\ttgt")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id; CometKiwi-22 is the standard reference-"
                         "free QE model. Gated — request access at "
                         "https://huggingface.co/Unbabel/wmt22-cometkiwi-da "
                         "and run `hf auth login` first.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpus", type=int, default=1,
                    help="0 (CPU) or 1. Values > 1 are refused: use phase0/score_sharded.py")
    ap.add_argument("--chunk-size", type=int, default=50000,
                    help="Process this many pairs per predict() call")
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted run: truncate a partial last output "
                         "line, count complete lines, skip them in the input, append.")
    ap.add_argument("--meta-out", default="",
                    help="Write scoring-stack metadata JSON here (package versions, "
                         "CUDA, GPU names, batch/chunk size, model, checkpoint path)")
    ap.add_argument("--allow-stack-change", action="store_true",
                    help="resume even if --meta-out records a different scoring stack (recorded in segments)")
    ap.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                    help="hub commit of --model to download ('' = unpinned legacy download_model())")
    ap.add_argument("--encoder-revision", default=DEFAULT_ENCODER_REVISION,
                    help=f"required commit of {ENCODER_REPO} main (tokenizer/config); checked, exit 2 if main moved")
    args = ap.parse_args(argv)

    if args.gpus > 1:
        print(f"ERROR: --gpus {args.gpus} refused. COMET predict() with gpus>1 runs DDP per call "
              "and hangs from the second chunk on. Run one process per GPU instead: "
              "python phase0/score_sharded.py --src ... --tgt ... --out ... --gpus N "
              "(or --devices 0,1,...).", file=sys.stderr)
        return 2
    if args.gpus < 0 or args.chunk_size < 1 or args.batch_size < 1:
        print("ERROR: --gpus must be 0 or 1; --chunk-size and --batch-size must be >= 1", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(args.src).is_file() or not Path(args.tgt).is_file():
        print("ERROR: --src/--tgt missing", file=sys.stderr)
        return 2
    try:
        n_src = count_lines(args.src)
        n_tgt = count_lines(args.tgt)
    except UnicodeDecodeError as e:
        print(f"ERROR: input is not valid UTF-8: {e}", file=sys.stderr)
        return 2
    if n_src != n_tgt:
        print(f"ERROR: --src has {n_src:,} lines but --tgt has {n_tgt:,}; refusing", file=sys.stderr)
        return 2

    lock_fd = take_lock(out_path.with_name(out_path.name + ".lock"))
    if lock_fd is None:
        print(f"ERROR: another scorer holds {out_path}.lock; refusing to touch {out_path} "
              "(a second process would append duplicate rows)", file=sys.stderr)
        return 2

    skip = 0
    if args.resume:
        dropped = truncate_partial_last_line(out_path)
        print(f"Resume: dropped {dropped:,} bytes of a partial last output line", file=sys.stderr)
        try:
            skip = count_lines(out_path)
        except UnicodeDecodeError as e:
            print(f"ERROR: existing output {out_path} is not valid UTF-8: {e}", file=sys.stderr)
            return 2
        if skip > n_src:
            print(f"ERROR: {out_path} already has {skip:,} lines but the input has {n_src:,}; "
                  "this output belongs to a different input", file=sys.stderr)
            return 2
        print(f"Resume: {skip:,} of {n_src:,} pairs already scored — skipping ahead", file=sys.stderr)
        open_mode = "a"
        if skip == n_src:
            print(f"Nothing to do: {out_path} is complete ({n_src:,} lines)", file=sys.stderr)
            return 0
    else:
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"WARNING: {out_path} exists and is non-empty. "
                  "Pass --resume to continue, or delete the file first.",
                  file=sys.stderr)
            return 2
        open_mode = "w"

    print(f"Downloading / loading {args.model} ...", file=sys.stderr)
    model, ckpt_path, revs = load_model(args)
    bad_rev = revision_mismatch(args, revs)
    if bad_rev:
        print("ERROR: hub revision is not the pinned one (" + "; ".join(bad_rev) + "). Refusing to score with "
              "unpinned weights/tokenizer. Pass the new sha explicitly if the change is intended.", file=sys.stderr)
        return 2
    segments = []
    if args.meta_out:
        cur = collect_meta(args, ckpt_path, {"resume_skip": skip, "n_input": n_src, **revs})
        mp = Path(args.meta_out)
        old, old_err = None, None
        if args.resume and mp.exists():
            try:
                old = json.load(open(mp, encoding="utf-8"))
                if not isinstance(old, dict):
                    raise ValueError("not an object")
            except (OSError, ValueError) as e:
                old, old_err = None, e
        if args.resume and skip > 0:
            if old is None and not args.allow_stack_change:
                why = old_err if old_err is not None else "missing"
                print(f"ERROR: resuming {skip:,} scored rows but {mp} is unreadable ({why}); the stack that "
                      "scored them cannot be verified. Restore the meta, delete the output to rescore, or pass "
                      "--allow-stack-change to accept.", file=sys.stderr)
                return EXIT_STACK
            if old is not None:
                changed = stack_diff(old, cur)
                if changed and not args.allow_stack_change:
                    print(f"ERROR: the scoring stack changed since rows 0..{skip - 1:,} were scored "
                          f"(differs: {', '.join(changed)}). Refusing to mix stacks in one output. Restore the "
                          "stack, delete the output to rescore, or pass --allow-stack-change.", file=sys.stderr)
                    return EXIT_STACK
                segments = list(old.get("segments") or [segment_of(old, False)])
        elif args.resume and old is not None:           # skip == 0: restarted before its first chunk
            changed = stack_diff(old, cur)
            if changed and not args.allow_stack_change:
                print(f"ERROR: no rows scored yet, but {mp} records a different scoring stack from an earlier "
                      f"start (differs: {', '.join(changed)}). Other shards of the same run may have finished under "
                      "that stack. Restore the stack, delete the output and its meta to rescore under the new "
                      "one, or pass --allow-stack-change.", file=sys.stderr)
                return EXIT_STACK
            if changed:
                cur["restarted_from_empty"] = list(old.get("segments") or [segment_of(old, False)])
        elif args.resume and old_err is not None:
            print(f"WARNING: {mp} is unreadable ({old_err}); no rows were scored, so it is replaced", file=sys.stderr)
        segments.append(segment_of(cur, bool(args.allow_stack_change)))
        cur["segments"] = segments
        write_json_atomic(mp, cur)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_):
            return it

    n_total = skip
    with open_text(args.src) as src_f, open_text(args.tgt) as tgt_f, \
         open_text(out_path, open_mode) as f_out:
        if skip:
            fast_forward(src_f, skip)
            fast_forward(tgt_f, skip)
        for chunk in tqdm(chunked(src_f, tgt_f, args.chunk_size),
                          desc="scoring", unit="chunk", initial=skip // args.chunk_size):
            preds = model.predict(
                chunk,
                batch_size=args.batch_size,
                gpus=args.gpus,
                progress_bar=False,
            )
            scores = list(preds["scores"])
            if len(scores) != len(chunk):
                print(f"ERROR: model returned {len(scores):,} scores for a chunk of {len(chunk):,}",
                      file=sys.stderr)
                return 1
            f_out.write("".join(format_line(s, d["src"], d["mt"]) for d, s in zip(chunk, scores)))
            f_out.flush()
            os.fsync(f_out.fileno())  # a host-level loss can only drop whole chunks
            n_total += len(chunk)

    n_out = count_lines(out_path)
    if n_total != n_src or n_out != n_src:
        print(f"ERROR: output has {n_out:,} lines (wrote up to {n_total:,}) but input has {n_src:,}",
              file=sys.stderr)
        return 1
    if args.meta_out:
        write_json_atomic(args.meta_out, collect_meta(args, ckpt_path, {
            "resume_skip": skip, "n_input": n_src, "n_output": n_out, "complete": True, "segments": segments,
            **revs, **({"restarted_from_empty": cur["restarted_from_empty"]} if "restarted_from_empty" in cur else {})}))
    print(f"Wrote {n_total:,} scored pairs total to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
