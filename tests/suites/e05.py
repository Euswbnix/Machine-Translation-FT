"""E0.5: the six sets must be exactly what PROTOCOL.md says, or nothing at all.

e05_build.py draws a random control with the SAME size and the SAME source-length histogram
as the two halves of the QE top band, and then builds all three again inside one source, so
that only the selection differs. If a draw quietly came up short in a bucket, lost the
pairing between a sentence and its translation, let a non-UN row into a "UN only" set, or
left a half-built directory looking finished, the experiment would compare something other
than what was registered -- and it would still train and still print a number.
e05_report.py then has to name the pre-stated outcome, with the exact 0.05 critical value,
and refuse every shape of incomplete result.
"""
import collections
import importlib.util
import json
import re
import shutil
import subprocess

WORD = re.compile(r"\w+", re.UNICODE)
PHRASE = "the general assembly adopted the resolution without a vote"     # 9 words, shared


def write(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(x + "\n" for x in lines), encoding="utf-8")


def read(p):
    return [ln.rstrip("\n") for ln in open(p, encoding="utf-8", newline="\n")]


def uniq(tag, i, n=10):
    return " ".join(f"{tag}{i}w{j}" for j in range(n - 1)) + f" {tag}{i}"


def fr_of(en):
    return "FR<" + en.split()[-1] + ">"


def make_src(s, un_random=9):
    """20 top-band rows of 10 words: 12 with no repeated 8-gram (8 UN, 4 Europarl) and 8 that
    share one (6 UN, 2 Europarl). 14 random rows of 10 words plus noise at another length."""
    top = [(uniq("ud", i), "un") for i in range(8)] + [(uniq("ed", i), "europarl") for i in range(4)] \
        + [(f"{PHRASE} ur{i}", "un") for i in range(6)] + [(f"{PHRASE} er{i}", "europarl") for i in range(2)]
    rnd = [(uniq("ru", i), "un") for i in range(un_random)] + [(uniq("rc", i), "commoncrawl") for i in range(5)] \
        + [(uniq("rx", i, 7), "un") for i in range(4)]
    for name, rows in (("ft_topk", top), ("ft_random", rnd)):
        write(s / f"{name}.en", [e for e, _ in rows])
        write(s / f"{name}.fr", [fr_of(e) for e, _ in rows])
        write(s / f"{name}.source", [l for _, l in rows])
    write(s / "heldout_un.en", ["h one two three"])
    write(s / "heldout_un.fr", ["h un deux trois"])


def split(ctx, s, d):
    p = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e04_split.py"), str(s), str(d)],
                       capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def build(ctx, d, s, *extra):
    args = [ctx.PY, str(ctx.ROOT / "phase0/e05_build.py"), "--dir", str(d)]
    if s is not None:
        args += ["--src-dir", str(s)]
    p = subprocess.run(args + list(extra), capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


NEW = ("ft_random_small", "ft_un_div", "ft_un_rep", "ft_un_random")
SIX = ("ft_topk_div", "ft_topk_rep") + NEW


def tsv(p, vals, baseline=(("newstest2014", 35.0), ("heldout_un", 46.0))):
    lines = ["condition\tseed\ttestset\tbleu"]
    lines += [f"baseline\t-\t{ts}\t{b:.2f}" for ts, b in baseline]
    for cond, xs in vals.items():
        lines += [f"{cond}\t{i}\tnewstest2014\t{v:.2f}" for i, v in enumerate(xs)]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def report(ctx, path, *extra):
    p = subprocess.run([ctx.PY, str(ctx.ROOT / "phase0/e05_report.py"), "--results", str(path),
                        *extra], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def suite(ctx):
    d = ctx.d
    src = lambda name: (ctx.ROOT / "phase0" / name).read_text(encoding="utf-8")
    lit = 'WORD = re.compile(r"\\w+", re.UNICODE)'
    ctx.check("e05_build counts length in the same tokens e04_split matched the halves on",
              lit in src("e04_split.py") and lit in src("e05_build.py"))

    # ---- exact critical values ----
    spec = importlib.util.spec_from_file_location("probe_stats_t", ctx.ROOT / "phase0/probe_stats.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 30: 2.042}
    ctx.check("t critical values reproduce the 0.05 table at whole degrees of freedom",
              all(abs(ps.t_crit(k) - v) < 5e-4 for k, v in table.items()),
              str({k: round(ps.t_crit(k), 4) for k in table}))
    ctx.check("and fall strictly between the table rows at a fractional df (3.71 -> about 2.86)",
              2.776 < ps.t_crit(3.71) < 3.182 and abs(ps.t_crit(3.71) - 2.864) < 2e-3, str(ps.t_crit(3.71)))
    c = ps.contrast([32.57, 32.77, 32.97], [33.05, 33.20, 33.35])
    ctx.check("a p = 0.045 contrast is DETECTED (the floor(df) table called it not significant)",
              c is not None and c["clear"] and abs(c["p"] - 0.045) < 2e-3 and 3 < c["df"] < 4, str(c))
    ctx.check("zero variance on both sides, or one value a side, is 'undefined', not a verdict",
              ps.contrast([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None and ps.contrast([1.0], [2.0, 3.0]) is None)
    ctx.check("the detection bound is never below the seed-noise floor",
              c["bound"] >= c["noise"] and abs(c["bound"] - max(c["noise"], c["crit"] * c["se"])) < 1e-12)

    # ---- build ----
    s, e = d / "ok/src", d / "ok/e05"
    make_src(s)
    rc0, out0 = split(ctx, s, e)
    rc, out = build(ctx, e, s, "--allow-unpinned-src")
    ok = rc0 == 0 and rc == 0
    ctx.check("e04_split then e05_build produce all six sets", ok and all((e / f"{n}.en").is_file() for n in SIX),
              f"split rc={rc0} build rc={rc} {out0[-200:]} {out[-400:]}")
    rows = {n: (read(e / f"{n}.en"), read(e / f"{n}.fr")) for n in SIX} if ok else {n: ([], []) for n in SIX}
    hist = lambda n: collections.Counter(len(WORD.findall(x)) for x in rows[n][0])
    ctx.check("the all-source random control has the halves' size and length histogram",
              ok and len(rows["ft_random_small"][0]) == 8 and hist("ft_random_small") == hist("ft_topk_div") == hist("ft_topk_rep"),
              str({n: dict(hist(n)) for n in SIX[:3]}))
    ctx.check("the three single-source sets share one size and one length histogram",
              ok and len(rows["ft_un_div"][0]) == 6 and hist("ft_un_div") == hist("ft_un_rep") == hist("ft_un_random"),
              str({n: dict(hist(n)) for n in SIX[3:]}))
    ctx.check("every row of a new set keeps its own translation",
              ok and all(f == fr_of(x) for n in NEW for x, f in zip(*rows[n])))
    un_top = {x for x, l in zip(read(s / "ft_topk.en"), read(s / "ft_topk.source")) if l == "un"}
    un_rnd = {x for x, l in zip(read(s / "ft_random.en"), read(s / "ft_random.source")) if l == "un"}
    ctx.check("no row from another source enters a single-source set",
              ok and set(rows["ft_un_div"][0]) <= un_top and set(rows["ft_un_rep"][0]) <= un_top
              and set(rows["ft_un_random"][0]) <= un_rnd)
    ctx.check("the single-source halves are split on repetitiveness: every repetitive row carries the shared phrase, no diverse row does",
              ok and all(PHRASE in x for x in rows["ft_un_rep"][0]) and not any(PHRASE in x for x in rows["ft_un_div"][0]))
    man = json.loads((e / "manifest.json").read_text(encoding="utf-8")) if ok else {}
    w = man.get("written", [])
    ctx.check("the manifest lists all twelve set files once, the held-out pair, and itself last",
              len(w) == 15 and len(set(w)) == 15 and w[-1:] == ["manifest.json"]
              and all(f"{n}.{x}" in w for n in SIX for x in ("en", "fr")) and set(man.get("sha256", {})) == set(w) - {"manifest.json"}, str(w))
    e05 = man.get("e05", {})
    ctx.check("it marks the build complete, records the inputs' hashes and that they were NOT the pinned ones",
              man.get("e05_complete") is True and e05.get("src_pinned") is False and len(e05.get("src_sha256", {})) == 8
              and e05.get("source_held_fixed") == "un", str({k: e05.get(k) for k in ("src_pinned", "source_held_fixed")}))
    comp = {n: v.get("sources", {}) for n, v in e05.get("sets", {}).items()}
    ctx.check("and the source mix of every set (single-source sets are 100% that source)",
              all(set(comp.get(n, {})) == {"un"} for n in SIX[3:]) and set(comp.get("ft_topk_div", {})) >= {"un", "europarl"}, str(comp))
    spec = importlib.util.spec_from_file_location("cf_e05", ctx.ROOT / "phase0/controls_fingerprint.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    try:
        fp = cf.fingerprint(e)
        fine = all(f"{n}.{x}" in fp["files"] for n in SIX for x in ("en", "fr"))
    except SystemExit as ex:
        fine, fp = False, str(ex)
    ctx.check("all six sets are inside the controls fingerprint", fine, str(fp)[:200])

    before = {f.name: f.read_bytes() for f in e.iterdir()} if ok else {}
    rc2, _ = build(ctx, e, s, "--allow-unpinned-src")
    ctx.check("building again reproduces every byte (so the fingerprint is stable)",
              ok and rc2 == 0 and {f.name: f.read_bytes() for f in e.iterdir()} == before)
    rcv, outv = build(ctx, e, None, "--verify")
    ctx.check("--verify accepts a finished, untouched build", rcv == 0 and "intact" in outv, outv[-200:])
    if ok:
        with open(e / "ft_un_random.fr", "a", encoding="utf-8") as f:
            f.write("one extra line\n")
    rcv, outv = build(ctx, e, None, "--verify")
    ctx.check("--verify refuses a build in which one file changed afterwards",
              rcv == 1 and "CHANGED" in outv and "ft_un_random.fr" in outv, outv[-200:])

    rcv, outv = 0, ""
    if ok:
        marker = d / "marker/e05"
        marker.mkdir(parents=True)
        for nm, blob in before.items():
            (marker / nm).write_bytes(blob)
        m2 = json.loads(before["manifest.json"])
        del m2["e05_complete"]
        (marker / "manifest.json").write_text(json.dumps(m2), encoding="utf-8")
        rcv, outv = build(ctx, marker, None, "--verify")
    ctx.check("--verify refuses a manifest that lists every hash but does not say the build completed",
              ok and rcv == 1 and "NOT BUILT" in outv and "e05_complete" in outv, outv[-200:])

    half = d / "half/e05"
    make_src(d / "half/src")
    split(ctx, d / "half/src", half)
    rcv, outv = build(ctx, half, None, "--verify")
    ctx.check("--verify refuses a directory that only e04_split has written (an interrupted build looks like this)",
              rcv == 1 and "NOT BUILT" in outv, outv[-200:])

    rc, out = build(ctx, half, d / "half/src")
    ctx.check("source files other than the pinned E0.3 sets are refused unless the deviation is asked for",
              rc == 1 and "pinned" in out and not (half / "ft_random_small.en").exists(), f"rc={rc} {out[-300:]}")

    s2, e2 = d / "short/src", d / "short/e05"
    make_src(s2, un_random=3)                          # six UN rows are needed, three exist
    split(ctx, s2, e2)
    rc, out = build(ctx, e2, s2, "--allow-unpinned-src")
    left = sorted(f.name for f in e2.iterdir())
    ctx.check("a pool that cannot fill a bucket stops the build: no set published, no temporary file left, manifest untouched",
              rc == 1 and "short" in out and not any(n.startswith(NEW) or n.endswith(".tmp") for n in left)
              and "e05_complete" not in (e2 / "manifest.json").read_text(encoding="utf-8"), f"rc={rc} {left} {out[-300:]}")

    s3 = d / "other/src"
    make_src(s3)
    write(s3 / "ft_topk.en", [uniq("zz", i) for i in range(20)])        # nothing repeats here
    write(s3 / "ft_topk.fr", [fr_of(uniq("zz", i)) for i in range(20)])
    e3 = d / "other/e05"
    shutil.copytree(half, e3)
    rc, out = build(ctx, e3, s3, "--allow-unpinned-src")
    ctx.check("halves that were split from a different set than --src-dir are refused",
              rc == 1 and "other side of the median" in out and not (e3 / "ft_random_small.en").exists(), f"rc={rc} {out[-300:]}")

    s4 = d / "ragged/src"
    make_src(s4)
    write(s4 / "ft_random.source", ["un"])
    e4 = d / "ragged/e05"
    shutil.copytree(half, e4)
    rc, out = build(ctx, e4, s4, "--allow-unpinned-src")
    ctx.check("a set whose text and label files differ in length is refused", rc == 1 and "lines" in out, f"rc={rc} {out[-300:]}")

    # ---- report ----
    t = lambda m: [m, m + 0.1, m - 0.1, m + 0.05, m - 0.05]
    flat = [33.05, 32.95, 33.0, 33.02, 32.98]
    good = {"ft_topk_div": t(33.0), "ft_topk_rep": t(30.0), "ft_random_small": t(33.4),
            "ft_un_div": t(33.0), "ft_un_rep": t(30.5), "ft_un_random": flat}
    rc, out = report(ctx, tsv(d / "r/yes_a.tsv", good))
    ctx.check("Q1 YES, Q2 outcome A and a replication, all named in one report",
              rc == 0 and "Q1 — YES" in out and "Q2 — OUTCOME A" in out and "E0.4 REPLICATES" in out, f"rc={rc} {out[-600:]}")
    ctx.check("outcome A prints the gap it could not have detected and says it is not a demonstration of equality",
              "could not have been called" in out and "not a demonstration" in out, out[-500:])
    rc, out = report(ctx, tsv(d / "r/no.tsv", {**good, "ft_un_rep": flat}))
    ctx.check("Q1 NO when the halves do not separate inside one source, and the E0.4 reading is withdrawn",
              rc == 0 and "Q1 — NO" in out and "withdrawn" in out and "E0.4 REPLICATES" in out, f"rc={rc} {out[-600:]}")
    rc, out = report(ctx, tsv(d / "r/rev.tsv", {**good, "ft_un_div": t(30.0), "ft_un_rep": t(33.0), "ft_un_random": t(30.0)}))
    ctx.check("Q1 REVERSED when the diverse half is the worse one", rc == 0 and "Q1 — REVERSED" in out, f"rc={rc} {out[-400:]}")
    rc, out = report(ctx, tsv(d / "r/b.tsv", {**good, "ft_un_random": t(34.5)}))
    ctx.check("Q2 outcome B when the diverse half is below matched random", rc == 0 and "Q2 — OUTCOME B" in out, f"rc={rc} {out[-400:]}")
    rc, out = report(ctx, tsv(d / "r/c.tsv", {**good, "ft_un_random": t(31.5)}))
    ctx.check("Q2 outcome C when the diverse half is above matched random", rc == 0 and "Q2 — OUTCOME C" in out, f"rc={rc} {out[-400:]}")
    three = {k: v[:3] for k, v in good.items()}
    rc, out = report(ctx, tsv(d / "r/edge.tsv", {**three, "ft_un_div": [32.57, 32.77, 32.97], "ft_un_random": [33.05, 33.20, 33.35]}),
                     "--expect-seeds", "3")
    ctx.check("a p = 0.045 gap decides Q2 (outcome B); the stricter floor(df) table would have printed A",
              rc == 0 and "Q2 — OUTCOME B" in out and "p = 0.045" in out, f"rc={rc} {out[-600:]}")
    rc, out = report(ctx, tsv(d / "r/norep.tsv", {**good, "ft_topk_rep": flat}))
    ctx.check("a failed all-source replication is reported beside a Q1 that still holds",
              rc == 0 and "DOES NOT REPLICATE" in out and "Q1 — YES" in out, f"rc={rc} {out[-600:]}")
    rc, out = report(ctx, tsv(d / "r/noisy.tsv", {**good, "ft_un_div": [34.5, 31.0, 33.5, 32.0, 34.0], "ft_un_rep": [32.0, 30.0, 34.0, 31.0, 33.0]}))
    ctx.check("a gap inside the seed-noise floor cannot make Q1 a YES", rc == 0 and "Q1 — NO" in out, f"rc={rc} {out[-400:]}")
    rc, out = report(ctx, tsv(d / "r/five.tsv", {k: v for k, v in good.items() if k != "ft_un_random"}))
    ctx.check("five of six sets -> NO DECISION (exit 2)", rc == 2 and "NO DECISION" in out and "ft_un_random" in out, f"rc={rc} {out[-300:]}")
    rc, out = report(ctx, tsv(d / "r/seeds.tsv", {**good, "ft_un_rep": [30.5, 30.6, 30.4]}))
    ctx.check("3 of the 5 registered seeds on one set -> NO DECISION, not a verdict on what finished",
              rc == 2 and "NO DECISION" in out and "3 of 5" in out, f"rc={rc} {out[-300:]}")
    rc, out = report(ctx, tsv(d / "r/nobase.tsv", good, baseline=(("heldout_un", 46.0),)))
    ctx.check("no baseline on the news set -> NO DECISION", rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")
    rc, out = report(ctx, tsv(d / "r/zero.tsv", {**good, "ft_un_div": [33.0] * 5, "ft_un_rep": [30.0] * 5}))
    ctx.check("zero variance on both sides of Q1 -> NO DECISION (Welch undefined)", rc == 2 and "NO DECISION" in out, f"rc={rc} {out[-300:]}")

    p = tsv(d / "tok/e05.tsv", good)
    exact = {"ft_topk_div": 72_700_000, "ft_topk_rep": 71_400_000, "ft_random_small": 72_849_000,
             "ft_un_div": 70_000_000, "ft_un_rep": 68_600_000, "ft_un_random": 71_000_000}
    seeds = (42, 1, 2, 3, 4)
    p.with_suffix(".meta.json").write_text(json.dumps(
        {"budget": {f"{c} {sd}": {"applied_target_tokens": v} for c, v in exact.items() for sd in seeds}}), encoding="utf-8")
    ld = d / "tok/logs"
    ld.mkdir(parents=True, exist_ok=True)

    def logs(drop_per_1000):
        # ten dev evaluations, 106,000 .. 115,000; BLEU falls by drop_per_1000 at each
        for cnd, v in exact.items():
            for sd in seeds:
                trace = "".join(f"Step  {106000 + 1000 * k} | Valid BLEU: {30.0 - drop_per_1000 * k:.2f} (next eval)\n" for k in range(10))
                (ld / f"{cnd}_lr1_s{sd}_st2.log").write_text(trace + f"Step 115000 | Tok(applied) {v / 1e9:.4f}B\n", encoding="utf-8")

    logs(0.01)
    rc, out = report(ctx, p, "--logdir", str(ld))
    ctx.check("each deciding pair gets its own token line, from exact counts, saying which set got more",
              "tokens applied, Q1 (exact counts)" in out and "2.02% more for ft_un_div" in out and "ABOVE the 2%" in out
              and "tokens applied, Q2 (exact counts)" in out and "1.42% more for ft_un_random" in out
              and "tokens applied, replication (exact counts)" in out and "1.80% more for ft_topk_div" in out, out[-1500:])
    ctx.check("a flat dev trace makes the imbalance immaterial: 2.02% of 10,000 steps at 0.01 BLEU per 1,000 is 0.002 BLEU",
              "worth at most 0.002 BLEU" in out and "immaterial" in out and "QUALIFIED" not in out, out[-1500:])
    logs(1.0)
    rc, out = report(ctx, p, "--logdir", str(ld))
    ctx.check("a steep dev trace makes the same imbalance QUALIFY the outcome, on the verdict line itself",
              rc == 0 and "QUALIFIES the outcome" in out and "Repetitiveness is not reducible to source mix. [QUALIFIED" in out, out[-1500:])
    rc, out = report(ctx, p)
    ctx.check("without the logs the effect is reported as NOT BOUNDED, and Q1's verdict carries that",
              "NOT BOUNDED" in out and "[token imbalance above 2% and not bounded]" in out, out[-1200:])
    p.with_suffix(".meta.json").unlink()
    logs(0.01)
    rc, out = report(ctx, p, "--logdir", str(ld))
    ctx.check("without exact counts it falls back to the log lines and says how coarse they are",
              "training logs, 4 decimals" in out and "tokens applied, Q1" in out, out[-900:])
    rc, out = report(ctx, p)
    ctx.check("with neither it says the imbalance is not reported, rather than printing nothing",
              out.count("NOT REPORTED") == 3 and "[tokens applied not reported]" in out, out[-900:])

    # ---- runner wrappers (SFT points at a scratch dir: the wrappers write logs and results) ----
    scratch = d / "sft"
    scratch.mkdir()
    env0 = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(d)}
    run = lambda env, script: subprocess.run(["bash", str(ctx.ROOT / "phase0" / script)], capture_output=True,
                                             text=True, timeout=120, env={**env0, **env})
    q = run({}, "probe_run.sh")
    ctx.check("probe_run.sh refuses to run without PROBE", q.returncode != 0 and "PROBE" in q.stdout + q.stderr)
    empty = d / "empty"
    empty.mkdir()
    q = run({"PROBE": "zz", "CONDITIONS": "ft_a,ft_b", "DATA": str(empty), "SFT": str(ctx.ROOT), "PY": ctx.PY}, "probe_run.sh")
    ctx.check("probe_run.sh names the first missing set and stops before touching anything",
              q.returncode == 1 and "MISSING" in q.stdout and "ft_a.en" in q.stdout, (q.stdout + q.stderr)[-300:])
    stray = d / "stray"
    shutil.copytree(half, stray)                       # e04_split output: only the two halves are in the manifest
    write(stray / "ft_extra.en", ["a b c"])
    write(stray / "ft_extra.fr", ["a b c"])
    q = run({"PROBE": "zz", "CONDITIONS": "ft_topk_div,ft_extra", "DATA": str(stray), "SFT": str(ctx.ROOT), "PY": ctx.PY}, "probe_run.sh")
    ctx.check("probe_run.sh refuses a set that is on disk but not in the manifest (so outside the fingerprint)",
              q.returncode == 1 and "NOT IN THE MANIFEST" in q.stdout and "ft_extra.en" in q.stdout, (q.stdout + q.stderr)[-300:])
    q = run({"DATA": str(empty), "SFT": str(scratch), "PY": ctx.PY}, "e05_run.sh")
    ctx.check("e05_run.sh asks for SRC when the sets are not built, and exits non-zero",
              q.returncode != 0 and "SRC" in q.stdout + q.stderr and "NOT BUILT" in q.stdout, (q.stdout + q.stderr)[-400:])
    ctx.check("and still leaves a console log behind for the record", (scratch / "logs/e05_console.log").is_file())
    q = run({"DATA": str(half), "SFT": str(scratch), "PY": ctx.PY}, "e05_run.sh")
    ctx.check("e05_run.sh does not accept a half-built directory as built (it asks for SRC again)",
              q.returncode != 0 and "NOT BUILT" in q.stdout and "SRC" in q.stdout + q.stderr, (q.stdout + q.stderr)[-400:])
    mx = d / "matrix"
    (mx / "data").mkdir(parents=True)
    for nm in ("ft_a", "ft_b"):
        for ext in ("en", "fr"):
            (mx / "data" / f"{nm}.{ext}").write_text("line\n", encoding="utf-8")
    (mx / "data/manifest.json").write_text('{"written": [], "heldout_sets": {}}', encoding="utf-8")
    (mx / "base.yaml").write_text(json.dumps({
        "model": {"d_model": 512, "max_seq_len": 256},
        "training": {"batch_size": 4096, "accumulate_steps": 1, "max_steps": 115000, "lr_scale": 1.0,
                     "min_lr": 1e-5, "seed": 42, "eval_interval": 2000},
        "data": {"train_src": "x.en", "train_tgt": "x.fr", "valid_src": "v.en"},
        "checkpoint": {"dir": "ck", "keep_last": 5}, "logging": {}}), encoding="utf-8")
    shim = mx / "shim"
    shim.mkdir()
    (shim / "yaml.py").write_text("import json\ndef safe_load(f): return json.load(f)\n"
                                  "def safe_dump(o, f, **k): json.dump(o, f, indent=1)\n", encoding="utf-8")
    margs = [ctx.ROOT / "phase0/e03_run_matrix.py", "--base-config", mx / "base.yaml", "--data-dir", mx / "data",
             "--out-dir", mx / "out", "--no-controls-fingerprint", "--conditions", "ft_a,ft_b"]
    rc, o = ctx.run(margs + ["--seeds", "42,1,2,3,4"], env={"PYTHONPATH": str(shim)})
    runner = (mx / "out/run_stage2.sh").read_text(encoding="utf-8") if (mx / "out/run_stage2.sh").exists() else ""
    ctx.check("--seeds 42,1,2,3,4 generates five runs per set, each with its own suffix",
              rc == 0 and runner.count("train.py") == 10 and all(f"--seed {sd} --suffix _s{sd}_st2" in runner for sd in (42, 1, 2, 3, 4)),
              f"rc={rc} {o[-200:]}")
    rc, o = ctx.run(margs, env={"PYTHONPATH": str(shim)})
    runner = (mx / "out/run_stage2.sh").read_text(encoding="utf-8") if rc == 0 else ""
    ctx.check("without --seeds the gate's three seeds are unchanged", rc == 0 and runner.count("train.py") == 6 and "--seed 3 " not in runner)
    for bad in ("42,42", "7", "1,x"):
        rc, o = ctx.run(margs + ["--seeds", bad], env={"PYTHONPATH": str(shim)})
        ctx.check(f"--seeds {bad} is refused", rc != 0 and "--seeds" in o, f"rc={rc} {o[-200:]}")
    ctx.check("e05_run.sh fixes the five registered seeds and passes them down",
              "export SEEDS=42,1,2,3,4" in src("e05_run.sh") and '${SEEDS:+--seeds "$SEEDS"}' in src("probe_run.sh")
              and "--expect-seeds 5" in src("e05_run.sh"))

    q = run({"DATA": str(empty), "SFT": str(ctx.ROOT), "PY": ctx.PY}, "e04_run.sh")
    ctx.check("e04_run.sh still runs E0.4's two conditions through the shared body",
              q.returncode == 1 and "ft_topk_div.en" in q.stdout, (q.stdout + q.stderr)[-300:])
