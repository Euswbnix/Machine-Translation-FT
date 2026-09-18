#!/usr/bin/env python3
"""Score a parallel corpus on several GPUs: one single-GPU scorer process per device.

COMET's own multi-GPU predict() hangs when called once per chunk (see
scripts/score_with_comet.py), so this script shards instead:

  1. split --src/--tgt (LF is the only line break, strict UTF-8) into contiguous shards
     of ceil(rows / n_shards) lines each (the last one shorter; --shards defaults to the
     number of devices, and more shards than devices simply queue); unequal side line
     counts are refused;
  2. write <work-dir>/manifest.json: input sha256 + rows, per shard its start row,
     line count and the sha256 of both shard files;
  3. run the incomplete shards as a work queue over the given devices: at most one scorer
     per device, `CUDA_VISIBLE_DEVICES=<dev> <python> <scorer> --src shard --tgt shard --out
     shard.tsv --gpus 1 --resume --meta-out shard.meta.json` (log in <work-dir>/score.<k>.log);
     when a scorer exits 0 its device takes the next shard. A failure is printed at once
     ("shard k FAILED ...") and its device leaves the pool for this run (suspect GPU); the
     other shards keep going. A stack refusal (scorer exit 3) stops launching new shards;
  4. verify every shard output: exactly the shard's line count, every line is
     `<finite float>\\t<src>\\t<tgt>` whose src/tgt equal the shard input line with tabs
     replaced by spaces;
  5. concatenate in shard order to <out>.tmp, fsync, os.replace to --out;
  6. merge the per-shard meta JSON into --meta-out.

Re-running with the same inputs skips shards whose output is complete and verified, and
resumes the rest (the scorer truncates a partial last line). A shard output that has the
full line count but fails verification is never overwritten: the run stops and names it.
The shard boundaries are fixed by the manifest; a rerun may use a different or smaller
device list (e.g. without a failed GPU), and the output is byte-identical.

Single instance: run() takes a non-blocking flock on <work-dir>/lock and exits 2 if another
score_sharded run holds it (each scorer also locks its own <shard>.tsv.lock). Scorers run in
their own session; SIGTERM/SIGINT/SIGHUP to this parent terminate them (then kill after
30 s), and on Linux each scorer gets PR_SET_PDEATHSIG so it dies with the parent. Run it
inside tmux; do not kill only the parent.

Stack changes: merged meta lists `stack_changed_within_shard` (shards whose scorer
segments differ on the identity fields), `stack_identities` (each distinct identity tuple
over every segment of every shard, with its shard indices), `stack_mixed` (more than one
distinct tuple, i.e. also a change BETWEEN shards) and `stack_unverified` (shards whose meta
is missing or unreadable). gpu_names is an identity field, so a box with mixed GPU models
reports stack_mixed. --allow-stack-change is passed to the scorers only when given here.

Exit codes: 0 ok; 1 a scorer failed or verification failed; 2 bad inputs/arguments or
another run holds the lock; 3 a scorer refused a scoring-stack change (nothing mixed; the
message lists the shards and the recovery options; a plain rerun fails the same way).

    python phase0/score_sharded.py --src to_score.en --tgt to_score.fr \\
        --out new_scores.tsv --gpus 4 --meta-out new_scores.meta.json
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SCORER = REPO / "scripts" / "score_with_comet.py"


class Refuse(Exception):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def open_text(path, mode="r"):
    return open(path, mode, encoding="utf-8", newline="\n")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def count_lines(path) -> int:
    n = 0
    with open_text(path) as f:
        for _ in f:
            n += 1
    return n


def norm(line: str) -> str:
    return line.rstrip("\n").replace("\t", " ")


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shard_paths(work: Path, k: int, src_ext: str, tgt_ext: str) -> dict:
    return {"src": work / f"shard.{k:03d}.{src_ext}", "tgt": work / f"shard.{k:03d}.{tgt_ext}",
            "out": work / f"shard.{k:03d}.tsv", "meta": work / f"shard.{k:03d}.meta.json",
            "log": work / f"score.{k:03d}.log"}


def split_shards(src, tgt, work: Path, n_shards: int, src_ext: str, tgt_ext: str) -> dict:
    try:
        n = count_lines(src)
        n_t = count_lines(tgt)
    except UnicodeDecodeError as e:
        raise Refuse(f"input is not valid UTF-8: {e}")
    if n != n_t:
        raise Refuse(f"--src has {n:,} lines but --tgt has {n_t:,}; refusing")
    if n == 0:
        raise Refuse("inputs are empty")
    per = math.ceil(n / n_shards)
    n_shards = math.ceil(n / per)
    shards = []
    with open_text(src) as fs, open_text(tgt) as ft:
        for k in range(n_shards):
            start = k * per
            count = min(per, n - start)
            p = shard_paths(work, k, src_ext, tgt_ext)
            for side, fin in (("src", fs), ("tgt", ft)):
                tmp = p[side].with_name(p[side].name + ".tmp")
                with open_text(tmp, "w") as fo:
                    for _ in range(count):
                        line = fin.readline()
                        if not line:
                            raise Refuse(f"{side} ended early while writing shard {k}")
                        fo.write(line if line.endswith("\n") else line + "\n")
                os.replace(tmp, p[side])
            shards.append({"index": k, "start": start, "count": count,
                           "src_file": p["src"].name, "tgt_file": p["tgt"].name,
                           "src_sha256": sha256(p["src"]), "tgt_sha256": sha256(p["tgt"]),
                           "out_file": p["out"].name})
    return {"n_rows": n, "per_shard": per, "n_shards": n_shards, "n_devices": n_shards, "shards": shards}


def verify_shard_output(p: dict, count: int) -> tuple[str, str]:
    """-> (state, detail); state in missing | partial | complete | bad."""
    out = p["out"]
    if not out.exists():
        return "missing", "no output yet"
    n = 0
    try:
        with open_text(p["src"]) as fs, open_text(p["tgt"]) as ft, open_text(out) as fo:
            for line in fo:
                if not line.endswith("\n"):
                    return "partial", f"partial last line after {n:,} lines"
                if n >= count:
                    return "bad", f"more than {count:,} lines"
                parts = line[:-1].split("\t")
                s_in, t_in = fs.readline(), ft.readline()
                if len(parts) != 3:
                    return "bad", f"line {n:,}: {len(parts)} tab-separated fields, expected 3"
                try:
                    sc = float(parts[0])
                except ValueError:
                    return "bad", f"line {n:,}: score {parts[0]!r} is not a number"
                if not math.isfinite(sc):
                    return "bad", f"line {n:,}: non-finite score {parts[0]!r}"
                if parts[1] != norm(s_in) or parts[2] != norm(t_in):
                    return "bad", f"line {n:,} (shard row, 0-based): text differs from the shard input"
                n += 1
    except UnicodeDecodeError as e:
        return "bad", f"not valid UTF-8: {e}"
    if n < count:
        return "partial", f"{n:,} of {count:,} lines"
    return "complete", f"{n:,} lines verified"


def parse_devices(a) -> list[str]:
    if a.devices and a.gpus:
        raise Refuse("give --devices or --gpus, not both")
    if a.devices:
        devs = [d.strip() for d in a.devices.split(",") if d.strip()]
    elif a.gpus:
        devs = [str(i) for i in range(a.gpus)]
    else:
        raise Refuse("give --devices 0,1,... or --gpus N")
    if not devs or len(set(devs)) != len(devs):
        raise Refuse(f"bad device list {devs}")
    return devs


IDENTITY_KEYS = ("packages", "torch_cuda", "gpu_names", "python", "model", "checkpoint_path", "batch_size",
                 "model_revision", "encoder_revision")
EXIT_STACK = 3          # score_with_comet.py's stack-refusal exit code


def _pdeathsig():                      # preexec_fn, Linux only
    import ctypes
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG = 1


def terminate_all(procs, grace=30.0):
    live = [pr for _, pr, _ in procs if pr.poll() is None]
    for pr in live:
        try:
            pr.terminate()
        except ProcessLookupError:
            pass
    t_end = time.time() + grace
    for pr in live:
        try:
            pr.wait(timeout=max(0.0, t_end - time.time()))
        except subprocess.TimeoutExpired:
            pr.kill()
            pr.wait()


def identity_of(d: dict) -> str:
    return json.dumps({k: d.get(k) for k in IDENTITY_KEYS}, sort_keys=True)


def merge_meta(manifest: dict, work: Path, devs: list[str], src_ext: str, tgt_ext: str) -> dict:
    per_shard, common, keys_seen = [], None, set()
    changed_within, unverified = [], []
    identities: dict[str, list[int]] = {}
    for sh in manifest["shards"]:
        p = shard_paths(work, sh["index"], src_ext, tgt_ext)
        m = None
        if p["meta"].exists():
            try:
                m = json.load(open(p["meta"], encoding="utf-8"))
            except Exception as e:  # keep going: metadata must never block scores
                m = {"error": f"unreadable meta: {e}"}
        ids = set()
        if isinstance(m, dict) and "error" not in m:
            segs = [g for g in (m.get("segments") or []) if isinstance(g, dict)]
            ids = {identity_of(g) for g in segs} if segs else {identity_of(m)}
            if len(ids) > 1:
                changed_within.append(sh["index"])
        else:
            unverified.append(sh["index"])
        for i in sorted(ids):
            identities.setdefault(i, []).append(sh["index"])
        per_shard.append({"index": sh["index"], "device": m.get("cuda_visible_devices") if isinstance(m, dict) else None,
                          "start": sh["start"], "count": sh["count"], "meta": m})
        if isinstance(m, dict):
            flat = {k: json.dumps(v, sort_keys=True) for k, v in m.items()}
            keys_seen |= set(flat)
            common = flat if common is None else {k: v for k, v in common.items() if flat.get(k) == v}
    varying = sorted(keys_seen - set(common or {}))
    return {"script": "phase0/score_sharded.py", "n_rows": manifest["n_rows"],
            "src_sha256": manifest["src_sha256"], "tgt_sha256": manifest["tgt_sha256"],
            "devices": devs, "per_shard": manifest["per_shard"],
            "common": {k: json.loads(v) for k, v in (common or {}).items()},
            "varying_keys": varying, "stack_changed_within_shard": changed_within,
            "stack_identities": [{"identity": json.loads(i), "shards": v} for i, v in identities.items()],
            "stack_mixed": len(identities) > 1, "stack_unverified": unverified, "shards": per_shard}


def run(a) -> int:
    devs = parse_devices(a)
    src, tgt, out = Path(a.src), Path(a.tgt), Path(a.out)
    for p in (src, tgt):
        if not p.is_file():
            raise Refuse(f"{p} missing")
    scorer = Path(a.scorer)
    if not scorer.is_file():
        raise Refuse(f"scorer {scorer} missing")
    work = Path(a.work_dir) if a.work_dir else out.with_name(out.name + ".shards")
    work.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(work / "lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise Refuse(f"another score_sharded run holds {work / 'lock'}; not launching anything "
                     "(check `ps`/tmux for the running one)", code=2)
    try:
        return _run_locked(a, devs, src, tgt, out, scorer, work)
    finally:
        os.close(lock_fd)


def _run_locked(a, devs, src, tgt, out, scorer, work) -> int:
    src_ext = src.name.rsplit(".", 1)[-1] if "." in src.name else "src"
    tgt_ext = tgt.name.rsplit(".", 1)[-1] if "." in tgt.name else "tgt"
    if src_ext == tgt_ext:
        src_ext, tgt_ext = src_ext + ".src", tgt_ext + ".tgt"

    mpath = work / "manifest.json"
    src_sha, tgt_sha = sha256(src), sha256(tgt)
    manifest = None
    if mpath.exists():
        manifest = json.load(open(mpath, encoding="utf-8"))
        if (manifest.get("src_sha256"), manifest.get("tgt_sha256")) != (src_sha, tgt_sha):
            raise Refuse(f"{mpath} was made for different inputs; use a fresh --work-dir")
        have = manifest.get("n_shards", manifest.get("n_devices"))
        if a.shards and a.shards != have:
            raise Refuse(f"--shards {a.shards} but this work dir already holds {have} shards; shard boundaries "
                         f"are fixed once scoring starts. Pass --shards {have}, or use a new --work-dir.")
        if have != len(devs):
            print(f"note: manifest has {len(manifest['shards'])} shards; running the incomplete ones on "
                  f"{len(devs)} device(s) {','.join(devs)}", file=sys.stderr)
        for sh in manifest["shards"]:
            p = shard_paths(work, sh["index"], src_ext, tgt_ext)
            for side in ("src", "tgt"):
                if not p[side].exists() or sha256(p[side]) != sh[f"{side}_sha256"]:
                    raise Refuse(f"shard file {p[side]} is missing or differs from the manifest; "
                                 "use a fresh --work-dir")
        print(f"manifest ok: {len(manifest['shards'])} shards of {manifest['per_shard']:,} rows "
              f"({manifest['n_rows']:,} rows)", file=sys.stderr)
    else:
        manifest = split_shards(src, tgt, work, a.shards or len(devs), src_ext, tgt_ext)
        manifest.update({"src": str(src), "tgt": str(tgt), "src_sha256": src_sha, "tgt_sha256": tgt_sha})
        write_json_atomic(mpath, manifest)
        print(f"split {manifest['n_rows']:,} rows into {len(manifest['shards'])} shards of "
              f"{manifest['per_shard']:,}", file=sys.stderr)

    procs = []
    caught = []

    def on_signal(signum, _frame):
        caught.append(signum)
        raise KeyboardInterrupt

    old_handlers = {sg: signal.signal(sg, on_signal) for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        return _launch_and_finish(a, devs, out, scorer, work, manifest, src_ext, tgt_ext, procs)
    except KeyboardInterrupt:
        terminate_all(procs)
        sig = caught[0] if caught else signal.SIGINT
        raise Refuse(f"interrupted by signal {sig}; scorers terminated. Re-run this command to resume", code=1)
    finally:
        if any(pr.poll() is None for _, pr, _ in procs):
            terminate_all(procs)
        for sg, h in old_handlers.items():
            signal.signal(sg, h)


def _log_tail_differs(log_path) -> str:
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace")[-20000:]
    except OSError:
        return "?"
    import re
    hits = re.findall(r"differs: ([^)]*)\)", txt)
    if hits:
        return hits[-1]
    return "meta unreadable" if "unreadable" in txt else "?"


def _launch_and_finish(a, devs, out, scorer, work, manifest, src_ext, tgt_ext, procs) -> int:
    queue = []
    for sh in manifest["shards"]:
        k = sh["index"]
        p = shard_paths(work, k, src_ext, tgt_ext)
        state, detail = verify_shard_output(p, sh["count"])
        if state == "complete":
            print(f"shard {k}: complete, skipping ({detail})", file=sys.stderr)
            continue
        if state == "bad":
            raise Refuse(f"shard {k} output {p['out']} is inconsistent with its input: {detail}. "
                         "Not touching it; inspect or delete it and re-run.", code=1)
        # Resume on the device this shard already ran on when it is still in the list:
        # gpu_names is an identity field, so a shard that moves to another GPU is refused
        # as a stack change (exit 3). Its recorded device is in its own meta.
        prev = None
        try:
            prev = json.loads(p["meta"].read_text(encoding="utf-8")).get("cuda_visible_devices")
        except (OSError, ValueError, AttributeError):
            prev = None
        queue.append((k, detail, prev if prev in devs else None))

    free = list(devs)              # devices with no scorer running
    running = {}                   # device -> (k, Popen, log)
    failed, stack_refused = [], []
    stop_launching = False
    while queue or running:
        while queue and free and not stop_launching:
            i = next((j for j, (_, _, prev) in enumerate(queue) if prev in free), 0)
            k, detail, prev = queue.pop(i)
            dev = free.pop(free.index(prev)) if prev in free else free.pop(0)
            p = shard_paths(work, k, src_ext, tgt_ext)
            cmd = [a.python, str(scorer), "--src", str(p["src"]), "--tgt", str(p["tgt"]),
                   "--out", str(p["out"]), "--gpus", "1", "--resume",
                   "--batch-size", str(a.batch_size), "--chunk-size", str(a.chunk_size), "--model", a.model,
                   "--model-revision", a.model_revision, "--encoder-revision", a.encoder_revision,
                   "--meta-out", str(p["meta"])] + (["--allow-stack-change"] if a.allow_stack_change else [])
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev}
            print(f"shard {k}: {detail}; launching on device {dev} (log {p['log']})", file=sys.stderr)
            log = open(p["log"], "ab")
            pr = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  preexec_fn=_pdeathsig if sys.platform.startswith("linux") else None)
            procs.append((k, pr, log))
            running[dev] = (k, pr, log)
        if not running:
            break
        time.sleep(a.poll_interval)
        for dev in list(running):
            k, pr, log = running[dev]
            rc = pr.poll()
            if rc is None:
                continue
            log.close()
            del running[dev]
            logp = shard_paths(work, k, src_ext, tgt_ext)["log"]
            if rc == 0:
                free.append(dev)
            elif rc == EXIT_STACK:
                stack_refused.append((k, _log_tail_differs(logp)))
                free.append(dev)                 # the stack, not the GPU, is the problem
                stop_launching = True
                print(f"shard {k} REFUSED a scoring-stack change (differs: {stack_refused[-1][1]}; log {logp}); "
                      "launching no further shards", file=sys.stderr, flush=True)
            else:
                failed.append((k, rc))
                print(f"shard {k} FAILED on device {dev} (exit {rc}, log {logp}); device {dev} taken out of "
                      "this run's pool", file=sys.stderr, flush=True)
    not_run = [k for k, _, _ in queue]

    if stack_refused:
        lines = [f"  shard {k}: out {shard_paths(work, k, src_ext, tgt_ext)['out']}, meta "
                 f"{shard_paths(work, k, src_ext, tgt_ext)['meta']}, differs: {why}" for k, why in stack_refused]
        raise Refuse("scoring stack changed relative to what these shards recorded; nothing was mixed:\n"
                     + "\n".join(lines) + "\nA plain re-run fails the same way. Options: (1) restore the stack the "
                     "meta records; (2) delete those shards' .tsv and .meta.json to rescore them under the new "
                     "stack (other finished shards keep the old stack: the merged meta will report stack_mixed); "
                     "(3) re-run with --allow-stack-change (recorded in segments)."
                     + (f" Also failed: {failed}." if failed else "")
                     + (f" Not launched: shards {not_run}." if not_run else ""), code=EXIT_STACK)
    if failed or not_run:
        raise Refuse("scorer failed on " + ", ".join(
            f"shard {k} (exit {rc}, log {shard_paths(work, k, src_ext, tgt_ext)['log']})" for k, rc in failed)
            + (f"; shards {not_run} not launched (no healthy device left)" if not_run else "")
            + "; re-run this command to resume (optionally with --devices excluding a failed GPU)", code=1)

    for sh in manifest["shards"]:
        p = shard_paths(work, sh["index"], src_ext, tgt_ext)
        state, detail = verify_shard_output(p, sh["count"])
        if state != "complete":
            raise Refuse(f"shard {sh['index']} output not complete after scoring: {state}: {detail}", code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    total = 0
    with open(tmp, "wb") as fo:
        for sh in manifest["shards"]:
            with open(shard_paths(work, sh["index"], src_ext, tgt_ext)["out"], "rb") as fi:
                for line in fi:
                    fo.write(line)
                    total += 1
        fo.flush()
        os.fsync(fo.fileno())
    if total != manifest["n_rows"]:
        tmp.unlink()
        raise Refuse(f"concatenated {total:,} lines but the input has {manifest['n_rows']:,}", code=1)
    os.replace(tmp, out)
    print(f"wrote {out}: {total:,} rows from {len(manifest['shards'])} verified shards", file=sys.stderr)

    if a.meta_out:
        write_json_atomic(a.meta_out, merge_meta(manifest, work, devs, src_ext, tgt_ext))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--out", required=True, help="final scored TSV (written via <out>.tmp + os.replace)")
    ap.add_argument("--devices", default="", help="comma list of CUDA device ids, e.g. 0,1,2,3")
    ap.add_argument("--shards", type=int, default=0,
                    help="number of shards (default: one per device). More shards than devices queue; "
                         "each is verified as it completes, so a long run reports progress and a resume "
                         "re-checks less. Fixed for a work dir once scoring starts.")
    ap.add_argument("--gpus", type=int, default=0, help="use devices 0..N-1")
    ap.add_argument("--work-dir", default="", help="shards, logs, manifest (default <out>.shards)")
    ap.add_argument("--meta-out", default="", help="merged scoring-stack metadata JSON")
    ap.add_argument("--scorer", default=str(DEFAULT_SCORER), help="scorer script with the score_with_comet.py CLI")
    ap.add_argument("--python", default=sys.executable, help="python executable for the scorer processes")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--model", default="Unbabel/wmt22-cometkiwi-da")
    ap.add_argument("--model-revision", default="1ad785194e391eebc6c53e2d0776cada8f83179a",
                    help="passed to the scorer (pinned hub commit of --model; '' = unpinned)")
    ap.add_argument("--encoder-revision", default="d616d637f0720deda963cebbfc630657d2b7d3ae",
                    help="passed to the scorer (required microsoft/infoxlm-large commit)")
    ap.add_argument("--poll-interval", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--allow-stack-change", action="store_true",
                    help="pass --allow-stack-change to the scorers (a resumed shard may mix stacks; recorded)")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except Refuse as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
