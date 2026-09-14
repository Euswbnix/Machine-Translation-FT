"""Data tooling guards: rebuild_corpus.py output verification and blocked-set check,
line-separator safety of the corpus readers, and e01 ambiguous-row accounting.

Every fixture is built here from scratch; rebuild_corpus.py is driven through its real
main(argv) with hf_rows replaced by in-memory rows (no pyarrow needed), because the
corruption test must reach between the write and the verification."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations


def _load_rebuild(ctx, tag):
    spec = importlib.util.spec_from_file_location(f"rebuild_{tag}", ctx.ROOT / "phase0/rebuild_corpus.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    return rb


def _run_main(rb, rows, argv):
    """Call rebuild_corpus.main(argv) with hf_rows serving `rows`. Returns (rc, output)."""
    rb.hf_rows = lambda parquet_dir, src, tgt, max_rows=0: iter(list(rows))
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        rc = rb.main([str(a) for a in argv])
    return rc, out.getvalue()


def _expected_output(rows, thr, mode="fixed"):
    """Independent emulation of the cleaner for fixtures WITHOUT any CR (every field is one
    line in both modes): strip/\\n->space/skip empty, token filters, >thr duplicate target
    lines. Returns the exact bytes each side must contain."""
    pairs = []
    for s, g in rows:
        s = s.strip().replace("\n", " "); g = g.strip().replace("\n", " ")
        if s and g:
            assert "\r" not in s + g
            pairs.append((s, g))
    counts = {}
    for _, g in pairs:
        counts[g] = counts.get(g, 0) + 1
    src, tgt = [], []
    for s, g in pairs:
        sl, tl = len(s.split()), len(g.split())
        if sl < 3 or tl < 3 or sl > 200 or tl > 200 or not (0.5 <= sl / tl <= 2.0):
            continue
        lat = lambda x: sum(1 for c in x if c.isascii() or 0xC0 <= ord(c) <= 0x17F) / len(x)
        if lat(s) < 0.9 or lat(g) < 0.9 or counts[g] > thr:
            continue
        src.append(s + "\n"); tgt.append(g + "\n")
    return "".join(src).encode(), "".join(tgt).encode()


def suite(ctx):
    d, check, ROOT = ctx.d, ctx.check, ctx.ROOT
    rows = [(f"sentence number {i} in english here", f"phrase numero {i} en francais ici") for i in range(30)]
    exp_en, exp_fr = _expected_output(rows, 50)
    sha_en, sha_fr = hashlib.sha256(exp_en).hexdigest(), hashlib.sha256(exp_fr).hexdigest()

    # ---- 1. accepted run: verified from disk, moved into place ------------------------
    rb = _load_rebuild(ctx, "ok")
    ok = d / "ok"
    rc, out = _run_main(rb, rows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed",
                                   "--out-dir", ok, "--expect-sha-src", sha_en, "--expect-sha-tgt", sha_fr])
    st = json.loads((ok / "rebuild_stats.json").read_text()) if (ok / "rebuild_stats.json").exists() else {}
    check("rebuild: verified run exits 0, moves outputs into place, leaves no .tmp/.rejected",
          rc == 0 and (ok / "train.clean.en").read_bytes() == exp_en and (ok / "train.clean.fr").read_bytes() == exp_fr
          and not list(ok.glob("*.tmp")) and not list(ok.glob("*.rejected")) and st.get("outcome") == "accepted",
          f"rc={rc} {out[-300:]}")
    check("rebuild: stats and printout report the on-disk sha256 next to the in-memory one",
          st.get("sha256_src_disk") == sha_en and st.get("sha256_tgt_disk") == sha_fr
          and f"on-disk   {sha_en}" in out and f"in-memory {sha_en}" in out, out[-300:])

    # ---- 2. file corrupted between write and verification -> caught from disk ----------
    # The in-memory hash of what was written still equals --expect-sha, so a verifier that
    # hashes only in-memory strings would accept; only re-reading the file catches it.
    rb = _load_rebuild(ctx, "corrupt")
    real_rebuild = rb.rebuild

    def corrupting_rebuild(rows_factory, mode, out_src, out_tgt, *a, **k):
        stats = real_rebuild(rows_factory, mode, out_src, out_tgt, *a, **k)
        b = bytearray(open(out_tgt, "rb").read())
        b[5] ^= 0x20                                                # same length, one flipped byte
        open(out_tgt, "wb").write(bytes(b))
        return stats

    rb.rebuild = corrupting_rebuild
    bad = d / "corrupt"
    (bad).mkdir()
    (bad / "train.clean.fr").write_bytes(b"EARLIER VERIFIED RUN\n")
    rc, out = _run_main(rb, rows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed",
                                   "--out-dir", bad, "--expect-sha-src", sha_en, "--expect-sha-tgt", sha_fr])
    check("rebuild: a file corrupted on disk after writing is REJECTED (exit 1) although in-memory hashes match",
          rc == 1 and (bad / "train.clean.fr.rejected").exists() and (bad / "train.clean.en.rejected").exists()
          and not (bad / "train.clean.en").exists(), f"rc={rc} {out[-400:]}")
    check("rebuild: a rejected run never overwrites an earlier train.clean.<lang> and leaves no .tmp",
          (bad / "train.clean.fr").read_bytes() == b"EARLIER VERIFIED RUN\n" and not list(bad.glob("*.tmp")))
    rb = _load_rebuild(ctx, "corrupt_noexpect")
    real_rebuild = rb.rebuild
    rb.rebuild = corrupting_rebuild
    rc, out = _run_main(rb, rows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed",
                                   "--out-dir", d / "corrupt2"])
    check("rebuild: on-disk != in-memory is rejected even without --expect-sha",
          rc == 1 and (d / "corrupt2/train.clean.fr.rejected").exists(), f"rc={rc} {out[-300:]}")
    rb = _load_rebuild(ctx, "wrongsha")
    rc, out = _run_main(rb, rows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed",
                                   "--out-dir", d / "wrongsha", "--expect-sha-src", sha_en, "--expect-sha-tgt", "0" * 64])
    check("rebuild: wrong --expect-sha-tgt -> exit 1, both sides kept as .rejected",
          rc == 1 and (d / "wrongsha/train.clean.en.rejected").read_bytes() == exp_en
          and not (d / "wrongsha/train.clean.en").exists(), f"rc={rc} {out[-300:]}")

    # ---- 3. blocked-duplicate set differs between modes --------------------------------
    # threshold 50: the boilerplate target occurs 50 times as a whole line (not blocked in
    # fixed mode), plus once as a CR fragment -> 51 lines in legacy mode (blocked there only).
    boiler = "texte standard repete partout ici"
    brows = [(f"english text number {i} here", f"texte francais numero {i} ici") for i in range(20)]
    brows += [(f"boilerplate english line {i} ok", boiler) for i in range(50)]
    brows += [("header part then the english body text", f"entete coupe\r{boiler}")]
    brows += [(f"more english text {i} here", f"encore du texte {i} ici") for i in range(10)]
    rb = _load_rebuild(ctx, "blocked")
    bd = d / "blocked"
    rc, out = _run_main(rb, brows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed", "--out-dir", bd])
    st = json.loads((bd / "rebuild_stats.json").read_text()) if (bd / "rebuild_stats.json").exists() else {}
    bc = st.get("blocked_check", {})
    check("rebuild: CR fragment pushing a line over the threshold in legacy only -> exit 3, nothing written",
          rc == 3 and not list(bd.glob("train.clean.*")) and "--allow-blocked-diff" in out,
          f"rc={rc} files={[p.name for p in bd.glob('*')]} {out[-300:]}")
    check("rebuild: blocked_check records both counts and the symmetric difference (legacy 1, fixed 0, diff 1)",
          bc.get("legacy") == 1 and bc.get("fixed") == 0 and bc.get("symmetric_difference") == 1
          and bc.get("only_legacy") == 1 and any(e.get("line") == boiler for e in bc.get("examples", [])), str(bc))
    rb = _load_rebuild(ctx, "blocked_allow")
    rc, out = _run_main(rb, brows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed",
                                    "--out-dir", d / "blocked_allow", "--allow-blocked-diff"])
    st = json.loads((d / "blocked_allow/rebuild_stats.json").read_text()) if (d / "blocked_allow/rebuild_stats.json").exists() else {}
    fr_lines = (d / "blocked_allow/train.clean.fr").read_bytes().split(b"\n") if (d / "blocked_allow/train.clean.fr").exists() else []
    check("rebuild: --allow-blocked-diff proceeds (exit 0), fixed mode still keeps the 50 boilerplate rows, diff recorded",
          rc == 0 and st.get("blocked_check", {}).get("symmetric_difference") == 1
          and fr_lines.count(boiler.encode()) == 50, f"rc={rc} {out[-300:]}")
    rb = _load_rebuild(ctx, "blocked_same")
    rc, out = _run_main(rb, rows + [("a line with a carriage\rreturn inside it", "une ligne avec un retour\rchariot dedans")],
                        ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "legacy", "--out-dir", d / "blocked_same"])
    st = json.loads((d / "blocked_same/rebuild_stats.json").read_text()) if (d / "blocked_same/rebuild_stats.json").exists() else {}
    check("rebuild: CRs that do not change the blocked set -> exit 0 with symmetric_difference 0",
          rc == 0 and st.get("blocked_check", {}).get("symmetric_difference") == 0, f"rc={rc} {out[-300:]}")

    # ---- 4. Unicode line separators inside lines ----------------------------------------
    seps = {"U+2028": "\u2028", "U+2029": "\u2029", "NEL": "\x85", "VT": "\x0b", "FF": "\x0c",
            "FS": "\x1c", "GS": "\x1d", "RS": "\x1e"}
    urows = [(f"english words before {c} and after {i}", f"mots francais avant {c} et apres {i}")
             for i, c in enumerate(seps.values())]
    urows += [(f"plain english line {i} here", f"ligne francaise simple {i} ici") for i in range(5)]
    rb = _load_rebuild(ctx, "useps")
    ud = d / "useps"
    rc, out = _run_main(rb, urows, ["--parquet-dir", "-", "--src", "en", "--tgt", "fr", "--mode", "fixed", "--out-dir", ud])
    en_b = (ud / "train.clean.en").read_bytes() if (ud / "train.clean.en").exists() else b""
    fr_b = (ud / "train.clean.fr").read_bytes() if (ud / "train.clean.fr").exists() else b""
    en_rows, fr_rows = en_b.split(b"\n")[:-1], fr_b.split(b"\n")[:-1]
    want_en = [(s + "").encode() for s, _ in urows]
    check("rebuild fixed mode: U+2028/U+2029/NEL/VT/FF/FS/GS/RS stay inside single lines (13 rows, bytes intact)",
          rc == 0 and en_rows == want_en and len(fr_rows) == len(urows)
          and all(c.encode() in en_rows[i] and c.encode() in fr_rows[i] for i, c in enumerate(seps.values())),
          f"rc={rc} en_rows={len(en_rows)} {out[-200:]}")

    # e01 reads the same bytes as single lines (constituent == the rebuilt corpus)
    rc, out = ctx.run([ROOT / "phase0/e01_provenance.py", "--corpus", f"src:{ud / 'train.clean.en'}:{ud / 'train.clean.fr'}",
                       "--clean-src", ud / "train.clean.en", "--clean-tgt", ud / "train.clean.fr", "--out", d / "useps_prov"])
    lab = ctx.np.load(d / "useps_prov_labels.npy") if (d / "useps_prov_labels.npy").exists() else []
    rep = json.loads((d / "useps_prov_report.json").read_text()) if (d / "useps_prov_report.json").exists() else {}
    check("e01 reads lines containing U+2028/NEL/VT/FS as ONE row each (13 labels, all matched)",
          rc == 0 and len(lab) == len(urows) and rep.get("match_rate") == 1.0, f"rc={rc} n={len(lab)} {out[-300:]}")
    # a reference whose U+2028 lines are split would shift everything; make the constituent differ
    # from the corpus by splitting ONE line at U+2028 (as splitlines() would) and require that
    # e01 does NOT treat the two halves as the corpus line.
    split_en = en_b.replace("\u2028".encode(), b"\n", 1)
    (d / "split.en").write_bytes(split_en)
    rc, out = ctx.run([ROOT / "phase0/e01_provenance.py", "--corpus", f"src:{d / 'split.en'}:{ud / 'train.clean.fr'}",
                       "--clean-src", ud / "train.clean.en", "--clean-tgt", ud / "train.clean.fr", "--out", d / "split_prov"])
    lab2 = ctx.np.load(d / "split_prov_labels.npy").tolist() if (d / "split_prov_labels.npy").exists() else []
    check("e01: a U+2028 line that was split on the reference side no longer matches (only '\\n' ends a line)",
          rc == 0 and len(lab2) == len(urows) and lab2[0] == 255, f"rc={rc} {lab2}")

    # ---- 5. static: no splitlines() on corpus text in phase0/, phase1/, scripts/ --------
    allow = {  # (relative path, exact call source): non-corpus text only
        ("phase0/e03_collect.py", "runner.read_text().splitlines()"),         # a runner shell script
        ("phase0/run_parallel.py", "text.splitlines()"),                       # a job-list text
        ("phase0/inventory.py", "status.splitlines()"),                        # git status output
    }
    allow_prefix = {("phase0/inventory.py", "r.stderr.strip().splitlines()")}  # git stderr
    offenders, seen_allowed = [], set()
    for sub in ("phase0", "phase1", "scripts"):
        base = ROOT / sub
        if not base.exists():
            continue
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(ROOT).as_posix()
            src = py.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                offenders.append(f"{rel}: unparsable ({e})")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "splitlines":
                    seg = ast.get_source_segment(src, node) or ""
                    if (rel, seg) in allow or (rel, seg) in allow_prefix:
                        seen_allowed.add((rel, seg))
                    else:
                        offenders.append(f"{rel}:{node.lineno}: {seg}")
    check("static: no splitlines() call in phase0/, phase1/, scripts/ outside the non-corpus allowlist",
          not offenders, "; ".join(offenders)[:600])
    check("static: the splitlines() scanner really sees calls (the allowlisted e03_collect/run_parallel uses are found)",
          ("phase0/e03_collect.py", "runner.read_text().splitlines()") in seen_allowed
          and ("phase0/run_parallel.py", "text.splitlines()") in seen_allowed, str(sorted(seen_allowed)))
    probe = "x = open('c.en').read().splitlines()\n"
    hits = [n for n in ast.walk(ast.parse(probe)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "splitlines"]
    check("static: the scanner flags a corpus-style read().splitlines() probe", len(hits) == 1)

    # ---- 6. e01 ambiguous ROW counts per source pair ------------------------------------
    e = d / "amb"
    e.mkdir()
    A = [f"alpha pair {i}" for i in range(6)]
    B = [f"beta pair {i}" for i in range(6)]
    shared_ab = ["shared ab one", "shared ab two"]
    shared_bc = ["shared bc"]
    shared_abc = ["shared abc"]
    corp = {
        "a": A + shared_ab + shared_abc,
        "b": B + shared_ab + shared_bc + shared_abc + ["beta pair 0"],   # within-source duplicate
        "c": [f"gamma {i}" for i in range(4)] + shared_bc + shared_abc,
    }
    specs = []
    for k, lines in corp.items():
        (e / f"{k}.en").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        (e / f"{k}.fr").write_text("".join(l.upper() + "\n" for l in lines), encoding="utf-8")
        specs += ["--corpus", f"{k}:{e / (k + '.en')}:{e / (k + '.fr')}"]
    # the cleaned corpus: repeated ambiguous rows count once per ROW, not once per key
    clean = (A[:3] + shared_ab * 3 + shared_bc * 2 + shared_abc + B[:2] + ["gamma 0", "not anywhere"])
    (e / "clean.en").write_text("".join(l + "\n" for l in clean), encoding="utf-8")
    (e / "clean.fr").write_text("".join(l.upper() + "\n" for l in clean), encoding="utf-8")
    rc, out = ctx.run([ROOT / "phase0/e01_provenance.py", *specs, "--clean-src", e / "clean.en",
                       "--clean-tgt", e / "clean.fr", "--out", e / "prov"])
    rep = json.loads((e / "prov_report.json").read_text()) if (e / "prov_report.json").exists() else {}
    labs = ctx.np.load(e / "prov_labels.npy").tolist() if (e / "prov_labels.npy").exists() else []
    amb = rep.get("ambiguity") or {}
    # independent recomputation from the fixture definition
    order = list(corp)
    members = {l: sorted({order.index(k) for k in order if l in corp[k]}) for l in set(sum(corp.values(), []))}
    exp_pair, exp_rows, exp_label = {}, 0, {k: 0 for k in order}
    for l in clean:
        m = members.get(l, [])
        if len(m) > 1:
            exp_rows += 1
            exp_label[order[m[0]]] += 1
            for x, y in combinations(m, 2):
                key = f"{order[x]}+{order[y]}"
                exp_pair[key] = exp_pair.get(key, 0) + 1
    exp_labels = [members[l][0] if l in members else 255 for l in clean]
    check("e01 ambiguity: ambiguous ROW count and per-source-pair row counts match an independent recomputation",
          rc == 0 and amb.get("ambiguous_rows") == exp_rows == 9 and amb.get("ambiguous_rows_by_source_pair") == exp_pair
          and exp_pair == {"a+b": 7, "a+c": 1, "b+c": 3} and amb.get("cross_source_keys") == 4,
          f"rc={rc} got={amb} want rows={exp_rows} pairs={exp_pair} {out[-300:]}")
    check("e01 ambiguity: ambiguous rows go to the FIRST-LISTED --corpus; the report and printout say so with the order",
          labs == exp_labels and amb.get("ambiguous_rows_by_assigned_label") == exp_label
          and "first-listed" in amb.get("assignment_rule", "").lower() and "a < b < c" in amb.get("assignment_rule", "")
          and "first-listed" in out.lower() and "a < b < c" in out, f"labels={labs} want={exp_labels} {amb}")
    check("e01: without --qe-scores top_k is null with an explicit note (not an empty table)",
          "top_k" in rep and rep["top_k"] is None and "not computed" in (rep.get("top_k_note") or "").lower(),
          f"top_k={rep.get('top_k')!r} note={rep.get('top_k_note')!r}")
    # reversing the order moves the ambiguous rows to the new first-listed source
    rev = []
    for k in reversed(order):
        rev += ["--corpus", f"{k}:{e / (k + '.en')}:{e / (k + '.fr')}"]
    rc, out = ctx.run([ROOT / "phase0/e01_provenance.py", *rev, "--clean-src", e / "clean.en",
                       "--clean-tgt", e / "clean.fr", "--out", e / "prov_rev"])
    rrep = json.loads((e / "prov_rev_report.json").read_text()) if (e / "prov_rev_report.json").exists() else {}
    ramb = (rrep.get("ambiguity") or {}).get("ambiguous_rows_by_assigned_label", {})
    check("e01 ambiguity: reversing --corpus order hands the same 9 ambiguous rows to the new first-listed sources",
          rc == 0 and ramb == {"c": 3, "b": 6, "a": 0}, f"{ramb}")
