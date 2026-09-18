"""Scoring lane: score_with_comet.py (no comet needed), score_sharded.py, rescore_plan.py
merge text/atomicity guards, --allow-missing and calibrate.

A fake scorer (generated in ctx.d) imports the REAL scripts/score_with_comet.py and only
replaces load_model() with a deterministic hash model, so resume, truncation, line
counting and the output format under test are the production code. It can be told to
die after k lines, leaving a partial last line (FAKE_DIE_AFTER, FAKE_DIE_DEVICE,
FAKE_DIE_MARKER so it dies only once).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os

FAKE = r'''
import hashlib, importlib.util, os, sys
spec = importlib.util.spec_from_file_location("swc_real", os.environ["REAL_SCORER"])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def score(s, t):
    return int.from_bytes(hashlib.blake2b((s + "\0" + t).encode("utf-8"), digest_size=8).digest(), "little") / 2**64

if os.environ.get("FAKE_PIDDIR"):
    open(os.path.join(os.environ["FAKE_PIDDIR"], str(os.getpid())), "w").close()

if os.environ.get("FAKE_INTERVALS"):
    import time
    _t0 = time.time()
    import atexit
    def _iv():
        with open(os.environ["FAKE_INTERVALS"], "a") as f:
            f.write(f"{os.environ.get('CUDA_VISIBLE_DEVICES')} {_t0} {time.time()}\n")
    atexit.register(_iv)

class Fake:
    def __init__(self):
        self.done = 0
    def predict(self, chunk, batch_size, gpus, progress_bar):
        sd = os.environ.get("FAKE_SLEEP_DEVICES")
        if os.environ.get("FAKE_SLEEP") and (sd is None or os.environ.get("CUDA_VISIBLE_DEVICES") in sd.split(",")):
            import time; time.sleep(float(os.environ["FAKE_SLEEP"]))
        assert gpus == 1 or gpus == 0
        sc = [score(d["src"], d["mt"]) for d in chunk]
        die = os.environ.get("FAKE_DIE_AFTER")
        dev = os.environ.get("FAKE_DIE_DEVICE")
        marker = os.environ.get("FAKE_DIE_MARKER")
        if die is not None and (dev is None or dev == os.environ.get("CUDA_VISIBLE_DEVICES")) \
                and not (marker and os.path.exists(marker)):
            k = int(die)
            if self.done + len(chunk) > k:
                out = sys.argv[sys.argv.index("--out") + 1]
                j = k - self.done
                with open(out, "ab") as f:
                    for d, s in list(zip(chunk, sc))[:j]:
                        f.write(m.format_line(s, d["src"], d["mt"]).encode("utf-8"))
                    nl = m.format_line(sc[j], chunk[j]["src"], chunk[j]["mt"]).encode("utf-8")
                    f.write(nl[:max(1, len(nl) // 2)])
                if marker:
                    open(marker, "w").close()
                os._exit(17)
        self.done += len(chunk)
        return {"scores": sc}

def _load(args):
    return Fake(), "/fake/ckpt/" + args.model, {
        "model_revision": os.environ.get("FAKE_MODEL_REV", args.model_revision),
        "encoder_revision": os.environ.get("FAKE_ENC_REV", args.encoder_revision)}
m.load_model = _load
sys.exit(m.main())
'''


def rb(p) -> bytes:
    """read_bytes that returns b"" for a missing file, so a broken script fails a check, not the suite."""
    return p.read_bytes() if p.exists() else b""


def hscore(s, t):
    return int.from_bytes(hashlib.blake2b((s + "\0" + t).encode("utf-8"), digest_size=8).digest(), "little") / 2**64


def expected_tsv(src_lines, tgt_lines) -> bytes:
    """Independent re-statement of the scorer's output format."""
    out = []
    for s, t in zip(src_lines, tgt_lines):
        out.append("%.6f\t%s\t%s\n" % (hscore(s, t), s.replace("\t", " "), t.replace("\t", " ")))
    return "".join(out).encode("utf-8")


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(x + "\n" for x in lines))


def suite(ctx):
    d, check, run, ROOT, np = ctx.d, ctx.check, ctx.run, ctx.ROOT, ctx.np
    swc = ROOT / "scripts/score_with_comet.py"
    sharded = ROOT / "phase0/score_sharded.py"
    rp = ROOT / "phase0/rescore_plan.py"
    fake = d / "fake_scorer.py"
    fake.write_text(FAKE, encoding="utf-8")
    env = {"REAL_SCORER": str(swc)}

    # ---- corpus with awkward characters (LF is the only line break) -----------
    n = 1001
    src, tgt = [], []
    for i in range(n):
        s, t = f"src {i} words", f"tgt {i} mots"
        if i % 7 == 0:
            s += " after-LS"
        if i % 11 == 0:
            t += "\x85nel\x0bvt"
        if i % 13 == 0:
            s += "\ttab"
        if i % 17 == 0:
            t += "\rcr-inside"
        src.append(s); tgt.append(t)
    write_lines(d / "c.en", src); write_lines(d / "c.fr", tgt)
    exp = expected_tsv(src, tgt)

    # ---- score_with_comet.py without comet -----------------------------------
    rc_c, _ = run(["-c", "import comet"])
    rc, o = run([swc, "--src", d / "nope.en", "--tgt", d / "nope.fr", "--out", d / "x.tsv", "--gpus", "2"])
    check("score_with_comet --gpus 2 is refused (pointing to score_sharded.py) without comet installed",
          rc == 2 and "score_sharded.py" in o and "Traceback" not in o and rc_c != 0, f"rc={rc} comet_rc={rc_c} {o[-300:]}")
    rc, o = run([swc, "--help"])
    check("score_with_comet --help works without comet", rc == 0 and "--meta-out" in o, o[-200:])

    spec = importlib.util.spec_from_file_location("swc_under_test", swc)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    (d / "ls.txt").write_bytes("a b\x85c\x0bd\x0ce\x1cf\n2nd\n".encode("utf-8"))
    check("a line holding U+2028/NEL/VT/FF/FS counts as ONE line in score_with_comet.count_lines",
          mod.count_lines(d / "ls.txt") == 2, str(mod.count_lines(d / "ls.txt")))
    (d / "cr.txt").write_bytes(b"a\rb\n")
    check("a CR inside a line does not split it (newline='\\n')", mod.count_lines(d / "cr.txt") == 1)

    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "ref.tsv", "--chunk-size", "50",
                 "--meta-out", d / "ref.meta.json"], env=env)
    ref = rb(d / "ref.tsv") if (d / "ref.tsv").exists() else b""
    check("single-process scorer output equals the independently formatted expectation byte for byte",
          rc == 0 and ref == exp, f"rc={rc} {o[-300:]}")
    check("scorer output has exactly one line per input pair (U+2028/NEL/VT/CR rows included)",
          ref.count(b"\n") == n, str(ref.count(b"\n")))
    meta = json.load(open(d / "ref.meta.json")) if (d / "ref.meta.json").exists() else {}
    check("--meta-out writes stack metadata without comet/torch (missing fields are null)",
          meta.get("checkpoint_path") == "/fake/ckpt/Unbabel/wmt22-cometkiwi-da" and meta.get("batch_size") == 64
          and meta.get("chunk_size") == 50 and "unbabel-comet" in meta.get("packages", {})
          and "torch_cuda" in meta and "gpu_names" in meta and meta.get("complete") is True, str(meta)[:300])

    # resume truncates a partial last line
    lines = exp.split(b"\n")
    (d / "part.tsv").write_bytes(b"\n".join(lines[:5]) + b"\n" + lines[5][:9])
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "part.tsv", "--chunk-size", "64",
                 "--resume"], env=env)
    check("--resume drops the partial last line (reports 9 bytes) and finishes byte-identical",
          rc == 0 and "dropped 9 bytes" in o and rb(d / "part.tsv") == exp, o[-300:])
    (d / "die.tsv").unlink(missing_ok=True)
    rc1, o1 = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "die.tsv", "--chunk-size", "40"],
                  env={**env, "FAKE_DIE_AFTER": "123"})
    raw = rb(d / "die.tsv") if (d / "die.tsv").exists() else b""
    rc2, o2 = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "die.tsv", "--chunk-size", "40",
                   "--resume"], env=env)
    check("kill mid-line then --resume produces the correct file",
          rc1 == 17 and not raw.endswith(b"\n") and raw.count(b"\n") == 123 and rc2 == 0
          and rb(d / "die.tsv") == exp, f"rc1={rc1} rc2={rc2} {o2[-200:]}")
    write_lines(d / "short.fr", tgt[:-1])
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "short.fr", "--out", d / "uneq.tsv"], env=env)
    check("scorer refuses src/tgt with different line counts before loading the model",
          rc == 2 and "refusing" in o and not (d / "uneq.tsv").exists(), o[-200:])

    # ---- score_sharded.py ------------------------------------------------------
    def shard_run(work, out, devices="0,1,2", extra_env=None, extra=()):
        return run([sharded, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", out, "--work-dir", work,
                    "--devices", devices, "--scorer", fake, "--chunk-size", "50", *extra],
                   env={**env, **(extra_env or {})})

    rc, o = shard_run(d / "w1", d / "sh1.tsv", extra=("--meta-out", d / "sh1.meta.json"))
    man = json.load(open(d / "w1/manifest.json")) if (d / "w1/manifest.json").exists() else {"shards": []}
    got = rb(d / "sh1.tsv") if (d / "sh1.tsv").exists() else b""
    check("sharded run over 3 fake devices (1001 rows) reproduces the single-process output byte for byte",
          rc == 0 and got == ref and ref, f"rc={rc} {o[-400:]}")
    ok_man = [(s["start"], s["count"]) for s in man["shards"]] == [(0, 334), (334, 334), (668, 333)]
    if ok_man:
        for s in man["shards"]:
            b = (d / "w1" / s["src_file"]).read_bytes()
            ok_man &= hashlib.sha256(b).hexdigest() == s["src_sha256"] and \
                b == "".join(x + "\n" for x in src[s["start"]:s["start"] + s["count"]]).encode("utf-8")
    check("manifest records contiguous start/count and correct sha256 per shard", ok_man, str(man["shards"])[:300])
    mm = json.load(open(d / "sh1.meta.json")) if (d / "sh1.meta.json").exists() else {}
    check("merged meta has one entry per shard with its own CUDA_VISIBLE_DEVICES",
          [s["meta"]["cuda_visible_devices"] for s in mm.get("shards", []) if s.get("meta")] == ["0", "1", "2"]
          and mm.get("common", {}).get("model") == "Unbabel/wmt22-cometkiwi-da", str(mm)[:300])
    check("no .tmp left next to the sharded output", not (d / "sh1.tsv.tmp").exists())

    rc, o = shard_run(d / "w1", d / "sh1.tsv")
    check("re-run skips complete verified shards", rc == 0 and o.count("complete, skipping") == 3
          and rb(d / "sh1.tsv") == ref, o[-300:])

    marker = d / "died.marker"
    rc1, o1 = shard_run(d / "w2", d / "sh2.tsv",
                        extra_env={"FAKE_DIE_AFTER": "77", "FAKE_DIE_DEVICE": "1", "FAKE_DIE_MARKER": str(marker)})
    part = rb(d / "w2/shard.001.tsv") if (d / "w2/shard.001.tsv").exists() else b"\n"
    out_after_fail = (d / "sh2.tsv").exists()
    check("sharded kill-and-resume: failing shard named, its partial last line left, no --out written",
          rc1 == 1 and "shard 1" in o1 and not part.endswith(b"\n") and not out_after_fail, f"rc1={rc1} {o1[-300:]}")
    rc2, o2 = shard_run(d / "w2", d / "sh2.tsv")
    log1 = (d / "w2/score.001.log").read_text(errors="replace") if (d / "w2/score.001.log").exists() else ""
    import re
    dropped = [int(x.replace(",", "")) for x in re.findall(r"dropped ([\d,]+) bytes", log1)]
    check("sharded kill-and-resume: re-run resumes only shard 1, drops its partial line, output identical",
          rc2 == 0 and o2.count("complete, skipping") == 2 and any(x > 0 for x in dropped)
          and rb(d / "sh2.tsv") == ref, f"rc2={rc2} dropped={dropped} {o2[-300:]}")
    # gpu_names is an identity field, so a resumed shard must go back to its own device:
    # taking the first free one instead makes the scorer refuse the resume (exit 3). That
    # is what a nightly stop hits on a multi-GPU box, and it made this test order-dependent.
    check("sharded kill-and-resume: shard 1 resumes on device 1, the device it ran on",
          "shard 1:" in o2 and "launching on device 1" in o2, o2[-300:])

    # corrupt a completed shard with the same line count (swap two lines)
    sp = d / "w1/shard.002.tsv"
    ls = rb(sp).split(b"\n")
    if len(ls) > 4:
        ls[3], ls[4] = ls[4], ls[3]
        sp.write_bytes(b"\n".join(ls))
    (d / "sh1.tsv").unlink(missing_ok=True)
    rc, o = shard_run(d / "w1", d / "sh1.tsv")
    check("sharded refuses a shard output whose text does not match its input (swapped lines), no --out",
          rc == 1 and "shard 2" in o and "line 3" in o and not (d / "sh1.tsv").exists(), o[-300:])

    # ---- work queue: a rerun on fewer devices; early failure reported at once; one scorer per device
    wq = d / "wq"
    rc1, o1 = shard_run(wq, d / "q.tsv", devices="0,1,2", extra_env={"FAKE_DIE_AFTER": "10"},
                        extra=("--poll-interval", "0.05"))
    iv = d / "intervals.txt"; iv.write_text("")
    rc2, o2 = shard_run(wq, d / "q.tsv", devices="0,1",
                        extra_env={"FAKE_INTERVALS": str(iv), "FAKE_SLEEP": "0.15"},
                        extra=("--meta-out", d / "q.meta.json", "--poll-interval", "0.05"))
    qm = json.load(open(d / "q.meta.json")) if (d / "q.meta.json").exists() else {}
    check("a 3-shard manifest whose shards all failed is finished by a rerun on 2 devices, byte-identical, merged meta ok",
          rc1 == 1 and rc2 == 0 and rb(d / "q.tsv") == ref and len(qm.get("shards", [])) == 3
          and qm.get("stack_mixed") is False, f"rc1={rc1} rc2={rc2} {o2[-400:]}")
    ivs = {}
    for ln in iv.read_text().splitlines():
        dv, a_, b_ = ln.split()
        ivs.setdefault(dv, []).append((float(a_), float(b_)))
    overlap = [dv for dv, xs in ivs.items() for (a1, b1) in xs for (a2, b2) in xs if (a1, b1) != (a2, b2) and a1 < b2 and a2 < b1]
    check("the queue never runs two scorers on one device at the same time (3 shards, 2 devices)",
          sum(len(x) for x in ivs.values()) == 3 and set(ivs) <= {"0", "1"} and not overlap, f"{ivs} overlap={overlap}")

    import subprocess as _sp
    import time as _time
    early = _sp.Popen([ctx.PY, str(sharded), "--src", str(d / "c.en"), "--tgt", str(d / "c.fr"), "--out", str(d / "e.tsv"),
                       "--work-dir", str(d / "we"), "--devices", "0,1,2", "--scorer", str(fake), "--chunk-size", "50",
                       "--poll-interval", "0.05"],
                      env={**os.environ, **env, "FAKE_DIE_AFTER": "5", "FAKE_DIE_DEVICE": "2", "FAKE_SLEEP": "1.0",
                           "FAKE_SLEEP_DEVICES": "0,1"}, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
    t_fail, early_out = None, []
    for ln in early.stdout:
        early_out.append(ln)
        if "shard 2 FAILED" in ln and t_fail is None:
            t_fail = _time.time()
    early.wait()
    t_end = _time.time()
    check("a shard that dies early is reported ('shard 2 FAILED') while the other shards are still scoring",
          early.returncode == 1 and t_fail is not None and t_end - t_fail > 3.0,
          f"rc={early.returncode} gap={None if t_fail is None else round(t_end - t_fail, 2)} {''.join(early_out)[-300:]}")
    rc, o = run([sharded, "--src", d / "c.en", "--tgt", d / "short.fr", "--out", d / "sh3.tsv", "--work-dir", d / "w3",
                 "--gpus", "3", "--scorer", fake], env=env)
    check("sharded refuses unequal src/tgt line counts", rc == 2 and "refusing" in o and not (d / "sh3.tsv").exists(),
          o[-200:])

    # ---- single-instance locks (audit F2) ------------------------------------------
    import fcntl
    wl = d / "wlock"; wl.mkdir()
    fd = os.open(str(wl / "lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc, o = shard_run(wl, d / "shl.tsv")
    finally:
        os.close(fd)
    check("score_sharded refuses (exit 2) while another run holds <work-dir>/lock; launches no scorer",
          rc == 2 and "holds" in o and not list(wl.glob("score.*.log")) and not (d / "shl.tsv").exists(), o[-300:])
    lo = d / "locked.tsv"; lo.write_bytes(exp[:40])
    fd = os.open(str(d / "locked.tsv.lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", lo, "--resume"], env=env)
    finally:
        os.close(fd)
    check("score_with_comet refuses (exit 2) while another process holds <out>.lock; output untouched",
          rc == 2 and "lock" in o and rb(lo) == exp[:40], o[-300:])

    # ---- SIGTERM to the score_sharded parent terminates its scorers (audit F2) -------
    import signal
    import subprocess
    import sys as _sys
    import time
    piddir = d / "pids"; piddir.mkdir()
    sig_env = {**os.environ, **env, "FAKE_PIDDIR": str(piddir), "FAKE_SLEEP": "30"}
    par = subprocess.Popen([ctx.PY, str(sharded), "--src", str(d / "c.en"), "--tgt", str(d / "c.fr"),
                            "--out", str(d / "sig.tsv"), "--work-dir", str(d / "wsig"), "--devices", "0,1",
                            "--scorer", str(fake), "--chunk-size", "50"], env=sig_env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t_end = time.time() + 30
    while time.time() < t_end and len(list(piddir.iterdir())) < 2:
        time.sleep(0.2)
    kids = [int(x.name) for x in piddir.iterdir()]
    par.send_signal(signal.SIGTERM)
    try:
        sig_out = par.communicate(timeout=60)[0]
    except subprocess.TimeoutExpired:
        par.kill(); sig_out = par.communicate()[0]

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    time.sleep(0.5)
    survivors = [k for k in kids if alive(k)]
    for k in survivors:
        try:
            os.kill(k, signal.SIGKILL)
        except OSError:
            pass
    check("SIGTERM to score_sharded terminates both running scorers before the parent exits (no orphans)",
          len(kids) == 2 and not survivors and par.returncode != 0 and "interrupted by signal" in (sig_out or ""),
          f"kids={kids} survivors={survivors} rc={par.returncode} {(sig_out or '')[-200:]}")

    spec_sh = importlib.util.spec_from_file_location("sharded_under_test", sharded)
    shm = importlib.util.module_from_spec(spec_sh); spec_sh.loader.exec_module(shm)
    wm = d / "wmeta"; wm.mkdir()
    seg = {"packages": {"torch": "1"}, "torch_cuda": None, "gpu_names": None, "python": "3", "model": "m",
           "checkpoint_path": "c", "batch_size": 64}
    (wm / "shard.000.meta.json").write_text(json.dumps({"segments": [seg, dict(seg, packages={"torch": "2"})]}))
    (wm / "shard.001.meta.json").write_text(json.dumps({"segments": [seg, dict(seg, utc="later")]}))
    man_ = {"n_rows": 2, "src_sha256": "a", "tgt_sha256": "b", "per_shard": 1,
            "shards": [{"index": 0, "start": 0, "count": 1}, {"index": 1, "start": 1, "count": 1}]}
    mmeta = shm.merge_meta(man_, wm, ["0", "1"], "en", "fr")
    check("score_sharded merge_meta lists shards whose resume segments changed stack (only shard 0)",
          mmeta.get("stack_changed_within_shard") == [0], str(mmeta.get("stack_changed_within_shard")))
    # Count only fsyncs of regular files: the directory fsync after os.replace must not
    # stand in for the file fsync (a mutation removing the file fsync once went uncaught).
    import stat as _stat
    fs_calls = []
    real_fsync = os.fsync
    try:
        os.fsync = lambda fd: (fs_calls.append(_stat.S_ISREG(os.fstat(fd).st_mode)), real_fsync(fd))[1]
        shm.write_json_atomic(d / "fs1.json", {"a": 1})
        n_sh = sum(fs_calls)
        k = len(fs_calls)
        mod.write_json_atomic(d / "fs2.json", {"a": 1})
        n_swc = sum(fs_calls[k:])
    finally:
        os.fsync = real_fsync
    check("both write_json_atomic copies fsync the file before os.replace (no zero-length meta after a host crash)",
          n_sh >= 1 and n_swc >= 1 and json.load(open(d / "fs1.json")) == {"a": 1} and json.load(open(d / "fs2.json")) == {"a": 1},
          f"score_sharded={n_sh} score_with_comet={n_swc}")
    # cross-shard: each shard uniform, but the two shards differ (audit round 2, finding 1)
    wx = d / "wmeta_x"; wx.mkdir()
    (wx / "shard.000.meta.json").write_text(json.dumps({"segments": [dict(seg, checkpoint_path="/snap/aaa")]}))
    (wx / "shard.001.meta.json").write_text(json.dumps({"segments": [dict(seg, checkpoint_path="/snap/bbb")]}))
    mx = shm.merge_meta(man_, wx, ["0"], "en", "fr")
    (wx / "shard.001.meta.json").write_text(json.dumps({"segments": [dict(seg, checkpoint_path="/snap/aaa", utc="x")]}))
    ms = shm.merge_meta(man_, wx, ["0"], "en", "fr")
    check("merge_meta: shard 0 [stack A], shard 1 [stack B] -> stack_mixed true (within-shard list empty); [A],[A] -> false",
          mx.get("stack_mixed") is True and mx.get("stack_changed_within_shard") == [] and len(mx.get("stack_identities", [])) == 2
          and ms.get("stack_mixed") is False and ms.get("stack_unverified") == [], f"{mx.get('stack_mixed')} {ms.get('stack_mixed')}")
    (wx / "shard.001.meta.json").write_text("")
    mu = shm.merge_meta(man_, wx, ["0"], "en", "fr")
    check("merge_meta: a zero-length shard meta is listed in stack_unverified (and merge_meta does not raise)",
          mu.get("stack_unverified") == [1], str(mu.get("stack_unverified")))
    (wx / "shard.001.meta.json").write_text(json.dumps({"packages": {"torch": "1"}, "torch_cuda": None, "gpu_names": None,
                                                         "python": "3", "model": "m", "checkpoint_path": "c", "batch_size": 64,
                                                         "model_revision": "r2"}))
    (wx / "shard.000.meta.json").write_text(json.dumps({**seg, "model_revision": "r1"}))
    mt_ = shm.merge_meta(man_, wx, ["0"], "en", "fr")
    check("merge_meta: metas without segments are compared on their top-level identity (model_revision differs -> mixed)",
          mt_.get("stack_mixed") is True, str(mt_.get("stack_identities"))[:200])

    # ---- resume under a different scoring stack (audit F4) --------------------------
    def die_then_meta(name):
        p_ = d / f"{name}.tsv"; m_ = d / f"{name}.meta.json"
        p_.unlink(missing_ok=True); m_.unlink(missing_ok=True)
        r1, _ = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40",
                     "--meta-out", m_], env={**env, "FAKE_DIE_AFTER": "123"})
        return p_, m_, r1

    p_, m_, r1 = die_then_meta("stk")
    mm_ = json.load(open(m_)) if m_.exists() else {"packages": {}}
    mm_["packages"]["torch"] = "9.9.9"                    # as an earlier stack would have written it
    for sg in mm_.get("segments") or []:
        sg["packages"]["torch"] = "9.9.9"
    json.dump(mm_, open(m_, "w"))
    n_before = rb(p_).count(b"\n")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40", "--resume",
                 "--meta-out", m_], env=env)
    check("--resume refuses (exit 3) when --meta-out records a different stack (packages); no rows appended",
          r1 == 17 and rc == 3 and "packages" in o and rb(p_).count(b"\n") == n_before, f"rc={rc} {o[-300:]}")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40", "--resume",
                 "--meta-out", m_, "--allow-stack-change"], env=env)
    mm2 = json.load(open(m_)) if m_.exists() else {}
    segs = mm2.get("segments", [])
    check("--allow-stack-change resumes, finishes byte-identical and keeps 2 segments with the two torch versions",
          rc == 0 and rb(p_) == exp and len(segs) == 2 and segs[0]["packages"].get("torch") == "9.9.9"
          and segs[1]["packages"].get("torch") != "9.9.9" and segs[1].get("stack_change_allowed") is True, f"rc={rc} {str(segs)[:300]}")
    p_, m_, r1 = die_then_meta("stk_same")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40", "--resume",
                 "--meta-out", m_], env=env)
    mm3 = json.load(open(m_)) if m_.exists() else {}
    check("--resume under the same stack succeeds and appends a segment (2 total)",
          rc == 0 and rb(p_) == exp and len(mm3.get("segments", [])) == 2, f"rc={rc} {o[-200:]}")

    # resume at skip == 0 (died before its first chunk) against an old meta from another stack
    z_, zm_ = d / "z0.tsv", d / "z0.meta.json"
    z_.write_bytes(b""); zm_.unlink(missing_ok=True)
    run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "z0pre.tsv", "--chunk-size", "40",
         "--meta-out", zm_], env={**env, "FAKE_DIE_AFTER": "0"})
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", z_, "--chunk-size", "40", "--resume",
                 "--meta-out", zm_, "--encoder-revision", "e" * 40], env={**env, "FAKE_ENC_REV": "e" * 40})
    rcz, oz = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", z_, "--chunk-size", "40", "--resume",
                   "--meta-out", zm_, "--allow-stack-change", "--encoder-revision", "e" * 40], env={**env, "FAKE_ENC_REV": "e" * 40})
    zm = json.load(open(zm_)) if zm_.exists() else {}
    check("--resume with 0 rows scored still refuses (exit 3) an old meta with another encoder_revision; "
          "--allow-stack-change proceeds, keeps one segment and records restarted_from_empty",
          rc == 3 and "encoder_revision" in o and rcz == 0 and rb(z_) == exp and len(zm.get("segments", [])) == 1
          and len(zm.get("restarted_from_empty", [])) == 1, f"rc={rc} rcz={rcz} {o[-300:]}")
    p_, m_, r1 = die_then_meta("stk_rev")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40", "--resume",
                 "--meta-out", m_, "--encoder-revision", "e" * 40], env={**env, "FAKE_ENC_REV": "e" * 40})
    check("--resume with rows scored refuses (exit 3) when only encoder_revision changed", rc == 3 and "encoder_revision" in o,
          f"rc={rc} {o[-200:]}")
    p_, m_, r1 = die_then_meta("stk_empty_meta")
    m_.write_text("")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", p_, "--chunk-size", "40", "--resume",
                 "--meta-out", m_], env=env)
    check("--resume with rows scored and a zero-length meta refuses with exit 3 (stack unverifiable)",
          rc == 3 and "unreadable" in o, f"rc={rc} {o[-200:]}")
    rc, o = run([fake, "--src", d / "c.en", "--tgt", d / "c.fr", "--out", d / "revx.tsv", "--meta-out", d / "revx.meta.json"],
                env={**env, "FAKE_MODEL_REV": "0" * 40})
    check("a resolved model revision that differs from the requested pin refuses (exit 2) before writing output",
          rc == 2 and "model_revision" in o and not rb(d / "revx.tsv"), f"rc={rc} {o[-200:]}")
    rm = json.load(open(d / "ref.meta.json")) if (d / "ref.meta.json").exists() else {}
    check("scorer meta records the pinned model_revision and encoder_revision",
          rm.get("model_revision") == mod.DEFAULT_MODEL_REVISION and rm.get("encoder_revision") == mod.DEFAULT_ENCODER_REVISION,
          str({k: rm.get(k) for k in ("model_revision", "encoder_revision")}))

    # the real load_model with fake huggingface_hub / comet modules: pinned revision, local-only load
    import sys as _sys2
    import types
    calls = []
    hub = types.ModuleType("huggingface_hub")
    def _snap(repo_id, revision=None, allow_patterns=None, **k):
        calls.append(("snap", repo_id, revision, allow_patterns))
        rev = revision if revision != "main" else os.environ.get("FAKE_MAIN_REV", mod.DEFAULT_ENCODER_REVISION)
        return str(d / "hub" / repo_id.replace("/", "--") / "snapshots" / rev)
    hub.snapshot_download = _snap
    cm = types.ModuleType("comet")
    def _lfc(path, **k):
        calls.append(("load", path, k))
        return "MODEL"
    cm.load_from_checkpoint = _lfc
    cm.download_model = lambda *a, **k: calls.append(("legacy",)) or "legacy.ckpt"
    saved = {n: _sys2.modules.get(n) for n in ("huggingface_hub", "comet")}
    _sys2.modules["huggingface_hub"], _sys2.modules["comet"] = hub, cm
    try:
        A = types.SimpleNamespace(model="Unbabel/wmt22-cometkiwi-da", model_revision=mod.DEFAULT_MODEL_REVISION,
                                  encoder_revision=mod.DEFAULT_ENCODER_REVISION)
        mdl, ck, revs = mod.load_model(A)
        ok_pin = (mdl == "MODEL" and ("snap", A.model, A.model_revision, None) in calls
                  and any(c[0] == "load" and c[2].get("local_files_only") is True for c in calls)
                  and str(ck).endswith(f"{A.model_revision}/checkpoints/model.ckpt")
                  and revs == {"model_revision": A.model_revision, "encoder_revision": A.encoder_revision}
                  and not mod.revision_mismatch(A, revs))
        calls.clear(); os.environ["FAKE_MAIN_REV"] = "f" * 40
        mdl2, _, revs2 = mod.load_model(A)
        moved = mdl2 is None and not any(c[0] == "load" for c in calls) and mod.revision_mismatch(A, revs2)
    finally:
        os.environ.pop("FAKE_MAIN_REV", None)
        for n, v in saved.items():
            if v is None:
                _sys2.modules.pop(n, None)
            else:
                _sys2.modules[n] = v
    check("load_model downloads the kiwi snapshot at the pinned revision and loads it local_files_only",
          ok_pin, str(calls)[:400])
    check("load_model refuses to load when microsoft/infoxlm-large main no longer resolves to the pinned encoder revision",
          moved, str(calls)[:300])

    # ---- rescore_plan merge ----------------------------------------------------
    r = d / "rs"; r.mkdir()
    cs = [f"pair {i}" + (" ls" if i == 2 else "") for i in range(8)]
    ct = [f"paire {i}" for i in range(8)]
    write_lines(r / "new.en", cs); write_lines(r / "new.fr", ct)
    old_rows = [0, 3, 4, 7]
    (r / "old.tsv").write_text("".join(f"0.{i}00000\t{cs[i]}\t{ct[i]}\n" for i in old_rows), encoding="utf-8")
    rc, o = run([rp, "plan", "--old-scored", r / "old.tsv", "--new-src", r / "new.en", "--new-tgt", r / "new.fr",
                 "--out-dir", r / "plan"])
    rc2, o2 = run([fake, "--src", r / "plan/to_score.en", "--tgt", r / "plan/to_score.fr",
                   "--out", r / "new_scores.tsv"], env=env)
    base = ["--plan-dir", r / "plan", "--new-src", r / "new.en", "--new-tgt", r / "new.fr"]
    rc3, o3 = run([rp, "merge", *base, "--new-scores", r / "new_scores.tsv", "--out", r / "merged.tsv"])
    want = "".join((f"0.{i}00000" if i in old_rows else "%.6f" % hscore(cs[i], ct[i])) + f"\t{cs[i]}\t{ct[i]}\n"
                   for i in range(8))
    good = (r / "merged.tsv").read_text(encoding="utf-8") if (r / "merged.tsv").exists() else ""
    check("merge with verified new scores writes the expected TSV (float32 rounding aside)",
          rc == 0 and rc2 == 0 and rc3 == 0 and
          [ln.split("\t")[1:] for ln in good.split("\n")[:-1]] == [ln.split("\t")[1:] for ln in want.split("\n")[:-1]]
          and all(abs(float(a.split("\t")[0]) - float(b.split("\t")[0])) < 1e-6
                  for a, b in zip(good.split("\n")[:-1], want.split("\n")[:-1])) and good.count("\n") == 8,
          f"{o[-200:]} {o2[-200:]} {o3[-200:]}")
    good_bytes = rb(r / "merged.tsv") if (r / "merged.tsv").exists() else b""

    ns = rb(r / "new_scores.tsv").split(b"\n")
    if len(ns) > 2:
        ns[1], ns[2] = ns[2], ns[1]
    (r / "swapped.tsv").write_bytes(b"\n".join(ns))
    rc, o = run([rp, "merge", *base, "--new-scores", r / "swapped.tsv", "--out", r / "merged.tsv"])
    check("merge refuses swapped new_scores lines (same count), naming row 1; no .tmp; good output untouched",
          rc != 0 and "to_score row 1" in o and not (r / "merged.tsv.tmp").exists()
          and rb(r / "merged.tsv") == good_bytes, o[-300:])

    # stale to_score (same row count, other pairs) + scores consistent with it: only the corpus check catches it
    st = r / "stale"; st.mkdir()
    write_lines(st / "to_score.en", [f"other {i}" for i in range(4)])
    write_lines(st / "to_score.fr", [f"autre {i}" for i in range(4)])
    rc, o = run([fake, "--src", st / "to_score.en", "--tgt", st / "to_score.fr", "--out", st / "ns.tsv"], env=env)
    rc, o = run([rp, "merge", *base, "--new-scores", st / "ns.tsv", "--to-score-dir", st, "--out", r / "merged.tsv"])
    check("merge refuses scores for stale to_score files (text != corpus at missing_rows); no .tmp; output untouched",
          rc != 0 and "refusing" in o and not (r / "merged.tsv.tmp").exists()
          and rb(r / "merged.tsv") == good_bytes, o[-300:])
    (r / "trunc.tsv").write_bytes(rb(r / "new_scores.tsv")[:-3])
    rc, o = run([rp, "merge", *base, "--new-scores", r / "trunc.tsv", "--out", r / "merged2.tsv"])
    check("merge refuses a partial last line in new_scores and writes nothing",
          rc != 0 and not (r / "merged2.tsv").exists() and not (r / "merged2.tsv.tmp").exists(), o[-200:])

    rc, o = run([rp, "merge", *base, "--allow-missing", "--out", r / "reused_only.tsv"])
    ro = (r / "reused_only.tsv").read_text(encoding="utf-8").split("\n")[:-1] if (r / "reused_only.tsv").exists() else []
    check("merge --allow-missing writes literal 'nan' for unscored rows and reused scores elsewhere",
          rc == 0 and len(ro) == 8 and [x.split("\t")[0] for x in ro] ==
          [f"0.{i}00000" if i in old_rows else "nan" for i in range(8)] and ro[2].split("\t")[1] == cs[2], str(ro)[:300])
    rc, o = run([rp, "merge", *base, "--allow-missing", "--new-scores", r / "new_scores.tsv", "--out", r / "x.tsv"])
    check("merge --allow-missing refuses --new-scores", rc != 0 and not (r / "x.tsv").exists(), o[-200:])
    rc, o = run([rp, "merge", *base, "--out", r / "x.tsv"])
    check("merge without --new-scores and without --allow-missing refuses", rc != 0 and not (r / "x.tsv").exists(),
          o[-200:])

    # ---- rescore_plan calibrate --------------------------------------------------
    spec = importlib.util.spec_from_file_location("rp_under_test", rp)
    rpm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpm)
    doc = rpm.__doc__ or ""
    proto = (ROOT / "PROTOCOL.md").read_text()
    check("rescore_plan docstring states the rebuilt-corpus figures (21,628,292 / 38,275,284, 326 MB) that PROTOCOL D7 uses",
          "21,628,292" in doc and "38,275,284" in doc and "326" in doc and "6 GPU-hours" not in doc and "230 MB" not in doc
          and "21,628,292" in proto)
    rng = np.random.default_rng(7)
    x = rng.integers(0, 6, 40).astype(float)
    naive = np.array([np.sum(x < v) + (np.sum(x == v) + 1) / 2 for v in x])
    check("avg_ranks equals a naive average-rank computation with ties", np.allclose(rpm.avg_ranks(x), naive))

    cal = d / "cal"; (cal / "plan").mkdir(parents=True)
    N = 9000
    labels = np.array([0, 1, 2, 255], dtype=np.uint8)[rng.integers(0, 4, N)]
    old = np.round(rng.random(N), 6).astype(np.float32)
    old[rng.random(N) < 0.2] = np.nan
    np.save(cal / "plan/reuse_scores.npy", old)
    (cal / "plan/plan.json").write_text(json.dumps({"new_rows": N, "to_score": int(np.isnan(old).sum())}))
    np.save(cal / "labels.npy", labels)
    (cal / "report.json").write_text(json.dumps({"sources": ["europarl", "un", "giga-fren"]}))
    newsc = np.where(np.isnan(old), rng.random(N), old.astype(np.float64))

    def write_scored(path, vals):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(f"{v:.6f}\ts{i}\tt{i}\n" for i, v in enumerate(vals)))

    def calib(scored, out, *extra):
        return run([rp, "calibrate", "--plan-dir", cal / "plan", "--new-scored", scored, "--provenance",
                    cal / "labels.npy", "--provenance-report", cal / "report.json", "--out", out, *extra])

    write_scored(cal / "same.tsv", newsc)
    rc, o = calib(cal / "same.tsv", cal / "same.json")
    rep = json.load(open(cal / "same.json")) if (cal / "same.json").exists() else {}
    check("calibrate passes (exit 0) on identical scores, per source incl. unmatched",
          rc == 0 and rep.get("verdict") == "pass" and rep.get("compared_rows") == int((~np.isnan(old)).sum())
          and set(rep.get("per_source", {})) == {"europarl", "un", "giga-fren", "unmatched"}, o[-300:])
    shifted = newsc.copy(); shifted[labels == 1] += 0.001
    write_scored(cal / "shift.tsv", shifted)
    rc, o = calib(cal / "shift.tsv", cal / "shift.json")
    rep = json.load(open(cal / "shift.json")) if (cal / "shift.json").exists() else {}
    ps = rep.get("per_source", {})
    check("calibrate fails (exit 1) on a +0.001 offset in one source, and blames only that source",
          rc == 1 and ps.get("un", {}).get("status") == "fail" and ps.get("europarl", {}).get("status") == "pass"
          and ps.get("giga-fren", {}).get("status") == "pass", o[-300:])
    rc, o = calib(cal / "same.tsv", cal / "ins.json", "--min-rows", "100000")
    check("calibrate exits 2 when there are too few compared rows", rc == 2, o[-200:])
    rc, o = calib(cal / "same.tsv", cal / "jmk0.json", "--min-jaccard", "1.01")
    rc2, o2 = calib(cal / "same.tsv", cal / "jmk.json", "--min-jaccard", "1.01", "--jaccard-min-k", "100000")
    rep = json.load(open(cal / "jmk.json")) if (cal / "jmk.json").exists() else {}
    check("calibrate --jaccard-min-k: Jaccard of groups with k below it is reported but not gated (default 0 gates)",
          rc == 1 and rc2 == 0 and rep.get("verdict") == "pass" and "topk_jaccard" in rep.get("overall", {})
          and rep["overall"].get("jaccard_not_gated") == ["50%", "30%", "15%", "5%"], f"rc={rc} rc2={rc2} {o2[-200:]}")
    (cal / "badrep.json").write_text("{not json")
    rc, o = run([rp, "calibrate", "--plan-dir", cal / "plan", "--new-scored", cal / "same.tsv", "--provenance",
                 cal / "labels.npy", "--provenance-report", cal / "badrep.json", "--out", cal / "badrep_out.json"])
    check("calibrate: an unreadable provenance report is invalid input (exit 3), not a failed calibration (1)",
          rc == 3 and "Traceback" not in o, f"rc={rc} {o[-200:]}")
    write_scored(cal / "short.tsv", newsc[:-1])
    rc, o = calib(cal / "short.tsv", cal / "short.json")
    check("calibrate rejects a scored file whose row count differs from the plan (exit 3)",
          rc == 3 and "rows" in o and not (cal / "short.json").exists(), o[-200:])
