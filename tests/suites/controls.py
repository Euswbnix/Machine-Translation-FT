"""E0.3 controls lane: e03_build_controls fail-closed exits, pool mask, pretraining
exclusion, normalised held-out reservation, manifest, byte-identity with the
baseline one-pass script; e03_decide fail-closed; e03_collect manifest-driven sets.

Every fixture is generated here; expected results are recomputed from the fixture
text, never read back from the script's own messages."""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import unicodedata

SOURCES = ["europarl", "commoncrawl", "un", "news-commentary", "giga-fren"]
BUILD = "phase0/e03_build_controls.py"
BASELINE_COMMIT = "0f146e4"


def nrm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def read_pairs(stem):
    with open(f"{stem}.en", encoding="utf-8", newline="\n") as a, \
         open(f"{stem}.fr", encoding="utf-8", newline="\n") as b:
        sa, tb = a.read().split("\n"), b.read().split("\n")
    assert sa[-1] == "" and tb[-1] == "", "file does not end with a newline"
    return list(zip(sa[:-1], tb[:-1]))


def make_rows(seed, sizes, dup=0.03, special=True):
    rng = random.Random(seed)
    rows = []                                     # (score, src, tgt, label)
    for lab, n in sizes.items():
        for i in range(n):
            s = f"{lab} source {i} w{rng.randint(0, 10**9)}"
            t = f"{lab} cible {i} m{rng.randint(0, 10**9)}"
            if special and i % 40 == 7:          # characters str.splitlines() would split on
                s = s.replace(" ", "\u2028", 1)
                t = t.replace(" ", "\x85", 1) + "\x0b end"
            rows.append((round(rng.uniform(0.5, 1.0), 4), s, t, lab))
    rows += rng.sample(rows, int(len(rows) * dup))
    rng.shuffle(rows)
    return rows


def write_fixture(d, rows, stem="prov", score_fmt=None):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "scores.tsv", "w", encoding="utf-8", newline="\n") as f:
        for k, (sc, s, t, _) in enumerate(rows):
            f.write(f"{score_fmt(k, sc) if score_fmt else sc}\t{s}\t{t}\n")
    import numpy as np
    np.save(d / f"{stem}_labels.npy", np.array([SOURCES.index(r[3]) for r in rows], dtype=np.uint8))
    json.dump({"sources": SOURCES, "match_rate": 1.0, "normalizer": "exact"},
              open(d / f"{stem}_report.json", "w"))
    return d / "scores.tsv", d / f"{stem}_labels.npy"


def baseline_script_text(root):
    """The pre-two-pass builder from BASELINE_COMMIT. Tries plain git and, because an
    x86_64 python on Apple silicon cannot exec the arm64-only xcrun git shim, the
    CommandLineTools binary and `arch -arm64 git`. A repo copy without .git (mutation
    runs) may point CONTROLS_BASELINE_SCRIPT at an extracted copy instead."""
    import os
    errs = []
    for git in (["git"], ["/Library/Developer/CommandLineTools/usr/bin/git"], ["arch", "-arm64", "git"]):
        try:
            g = subprocess.run([*git, "-C", str(root), "show", f"{BASELINE_COMMIT}:{BUILD}"],
                               capture_output=True, text=True)
        except OSError as exc:
            errs.append(f"{git[0]}: {exc}")
            continue
        if g.returncode == 0 and g.stdout:
            return g.stdout, ""
        errs.append(f"{' '.join(git)}: {g.stderr.strip()[-80:]}")
    env = os.environ.get("CONTROLS_BASELINE_SCRIPT")
    if env and os.path.isfile(env):
        return open(env, encoding="utf-8").read(), ""
    return None, "cannot obtain baseline script: " + " | ".join(errs)[-300:]


PLAIN = {"europarl": 900, "commoncrawl": 900, "un": 1200, "news-commentary": 150, "giga-fren": 1500}
SETS = ("heldout_un", "heldout_europarl", "ft_topk", "ft_bottom", "ft_random")
FT = ("ft_topk", "ft_bottom", "ft_random")


def suite(ctx):
    np, check, run, d, ROOT = ctx.np, ctx.check, ctx.run, ctx.d, ctx.ROOT

    def build(qe, out, *extra, prov=None):
        a = [ROOT / BUILD, "--qe-scores", qe, "--out-dir", out, "--n-ft", "600", "--n-heldout", "100"]
        if prov is not None:
            a += ["--provenance", prov]
        return run([*a, *extra])

    def held_files(out):
        return sorted(p.name for p in out.glob("heldout_*"))

    # ------------------------------------------------------------ plain run + manifest
    rows = make_rows(1, PLAIN)
    fx = d / "plain"
    qe, prov = write_fixture(fx, rows)
    t2l = {(s, t): lab for _, s, t, lab in rows}
    out = fx / "new"
    rc, o = build(qe, out, prov=prov)
    check("controls: default run succeeds on a clean fixture", rc == 0, o[-600:])
    man = json.load(open(out / "manifest.json")) if (out / "manifest.json").exists() else {}
    ok_sc, detail = True, ""
    for name in SETS:
        if not (out / f"{name}.en").exists():
            ok_sc, detail = False, f"{name} not written"
            break
        pairs = read_pairs(out / name)
        want = {s: 0 for s in SOURCES} | {"unmatched": 0}
        for p in pairs:
            want[t2l[p]] += 1
        entry = man.get("heldout_sets", {}).get(name) if name.startswith("heldout") else man.get(name)
        got = (entry or {}).get("source_counts")
        if got != want or (entry or {}).get("n") != len(pairs):
            ok_sc, detail = False, f"{name}: manifest {got} n={(entry or {}).get('n')} vs recomputed {want} n={len(pairs)}"
            break
    check("manifest source_counts and n per set equal a recount from the written text", ok_sc, detail)
    with open(qe, "rb") as f:
        head = hashlib.blake2b(f.read(16 * 1024 * 1024)).hexdigest()
    inp = man.get("inputs", {})
    check("manifest inputs block: size + blake2b of --qe-scores, provenance, report; report match_rate",
          (inp.get("qe_scores") or {}).get("size") == qe.stat().st_size
          and (inp.get("qe_scores") or {}).get("blake2b_first_16MiB") == head
          and (inp.get("provenance") or {}).get("size") == prov.stat().st_size
          and (inp.get("provenance_report") or {}).get("size") == (fx / "prov_report.json").stat().st_size
          and inp.get("report_match_rate") == 1.0 and inp.get("pool_mask") is None, str(inp)[:300])
    check("manifest records heldout_sets, reservation counts, written files, and keeps heldout_<d> ints",
          list(man.get("heldout_sets", {})) == ["heldout_un", "heldout_europarl"]
          and man.get("heldout_un") == 100 and man.get("heldout_europarl") == 100
          and {"exact_pair", "extra_by_source", "extra_by_target"} <= set(man.get("reservation", {}))
          and set(man.get("written", [])) == {f"{s}.{e}" for s in SETS for e in ("en", "fr")} | {"manifest.json"}
          and man.get("pool_mask") is None and man.get("pretrain") is None, str(man)[:300])
    check("controls: every held-out set comes from its exact-named source",
          all(t2l[p] == "un" for p in read_pairs(out / "heldout_un"))
          and all(t2l[p] == "europarl" for p in read_pairs(out / "heldout_europarl")))
    check("controls: characters splitlines() would break on survive (line counts equal manifest n)",
          all(len(read_pairs(out / s)) == 100 for s in ("heldout_un", "heldout_europarl"))
          and any("\u2028" in p[0] or "\x85" in p[1] for s in FT for p in read_pairs(out / s)))

    # ------------------------------------------------------------ byte identity vs baseline
    base_script = d / "baseline_build_controls.py"
    baseline_src, why = baseline_script_text(ROOT)
    if baseline_src is None:
        ctx.skip("byte-identity vs baseline e03_build_controls", why)
    else:
        base_script.write_text(baseline_src, encoding="utf-8")
        rc_b, ob = run([base_script, "--qe-scores", qe, "--provenance", prov, "--out-dir", fx / "old",
                        "--n-ft", "600", "--n-heldout", "100"])
        res = man.get("reservation", {})
        check("byte-identity precondition: source/target reservation removed nothing extra here",
              res.get("extra_by_source") == 0 and res.get("extra_by_target") == 0, str(res))
        same = rc_b == 0 and all((fx / "old" / f"{s}.{e}").read_bytes() == (out / f"{s}.{e}").read_bytes()
                                 for s in SETS for e in ("en", "fr") if (out / f"{s}.{e}").exists()) \
            and all((out / f"{s}.{e}").exists() for s in SETS for e in ("en", "fr"))
        check("two-pass output is byte-identical to the baseline one-pass script (all 10 files)",
              same, f"baseline rc={rc_b} {ob[-200:]}")

    # ------------------------------------------------------------ hard exits + escapes
    np.save(fx / "mm_labels.npy", np.concatenate([np.load(prov), np.array([0], dtype=np.uint8)]))
    shutil.copy(fx / "prov_report.json", fx / "mm_report.json")
    rc, o = build(qe, fx / "o_mm", prov=fx / "mm_labels.npy")
    check("hard exit: provenance/scored row-count mismatch", rc == 1 and "provenance rows" in o, f"rc={rc}")
    rc, o = build(qe, fx / "o_mm2", "--allow-no-heldout", prov=fx / "mm_labels.npy")
    m2 = json.load(open(fx / "o_mm2/manifest.json")) if (fx / "o_mm2/manifest.json").exists() else {}
    check("--allow-no-heldout: mismatch -> FT sets only, manifest says why, message says no held-out",
          rc == 0 and held_files(fx / "o_mm2") == [] and m2.get("heldout_sets") == {}
          and len(m2.get("heldout_skipped", [])) == 1 and "NO held-out sets" in o
          and (fx / "o_mm2/ft_topk.en").exists(), f"rc={rc} {o[-200:]}")

    shutil.copy(prov, fx / "norep_labels.npy")
    rc, o = build(qe, fx / "o_nr", prov=fx / "norep_labels.npy")
    check("hard exit: no e01 report and no --sources (its own message, not only the catch-all)",
          rc == 1 and "no e01 report" in o, f"rc={rc} {o[-150:]}")
    rc, _ = build(qe, fx / "o_nr2", "--allow-no-heldout", prov=fx / "norep_labels.npy")
    check("--allow-no-heldout: no report -> exit 0 without held-out sets",
          rc == 0 and held_files(fx / "o_nr2") == [], f"rc={rc}")
    rc, _ = build(qe, fx / "o_nr3", "--sources", ",".join(SOURCES), prov=fx / "norep_labels.npy")
    check("no report but --sources given -> falls back and builds held-out sets",
          rc == 0 and len(held_files(fx / "o_nr3")) == 4, f"rc={rc}")

    rc, o = build(qe, fx / "o_dom", "--heldout-domains", "un,giga", prov=prov)
    check("hard exit: --heldout-domains 'giga' is not an exact report name (no substring match)",
          rc == 1 and "giga" in o and not (fx / "o_dom/manifest.json").exists(), f"rc={rc}")
    rc, o = build(qe, fx / "o_dom2", "--heldout-domains", "un,giga", "--allow-no-heldout", prov=prov)
    check("--allow-no-heldout: unknown domain skipped, the others still built",
          rc == 0 and held_files(fx / "o_dom2") == ["heldout_un.en", "heldout_un.fr"], f"rc={rc} {o[-150:]}")
    rc, o = build(qe, fx / "o_gf", "--heldout-domains", "un,europarl,giga-fren", prov=prov)
    gf = read_pairs(fx / "o_gf/heldout_giga-fren") if (fx / "o_gf/heldout_giga-fren.en").exists() else []
    mg = json.load(open(fx / "o_gf/manifest.json")) if rc == 0 else {}
    check("--heldout-domains giga-fren (exact name) builds heldout_giga-fren from giga-fren rows only",
          rc == 0 and len(gf) == 100 and all(t2l[p] == "giga-fren" for p in gf)
          and list(mg.get("heldout_sets", {})) == ["heldout_un", "heldout_europarl", "heldout_giga-fren"],
          f"rc={rc} {o[-200:]}")
    ftp = set().union(*(read_pairs(fx / "o_gf" / s) for s in FT)) if rc == 0 else set()
    check("no heldout_giga-fren pair in any FT set", rc == 0 and not (set(gf) & ftp))

    rc, o = build(qe, fx / "o_small", "--heldout-domains", "news-commentary", "--n-heldout", "400", prov=prov)
    check("hard exit: held-out pool smaller than --n-heldout", rc == 1 and "400" in o, f"rc={rc}")
    rc, _ = build(qe, fx / "o_small2", "--heldout-domains", "news-commentary", "--n-heldout", "400",
                  "--allow-no-heldout", prov=prov)
    check("--allow-no-heldout: small pool skipped", rc == 0 and held_files(fx / "o_small2") == [], f"rc={rc}")

    rc, o = build(qe, fx / "o_noprov")
    check("hard exit: missing --provenance", rc == 1 and "--provenance" in o, f"rc={rc}")
    rc, _ = build(qe, fx / "o_noprov2", "--allow-no-heldout")
    check("--allow-no-heldout: no --provenance -> FT sets only",
          rc == 0 and held_files(fx / "o_noprov2") == [] and (fx / "o_noprov2/ft_random.fr").exists(), f"rc={rc}")

    bad = fx / "short.tsv"
    lines = qe.read_text(encoding="utf-8").split("\n")
    lines[10] = lines[10].split("\t")[0] + "\t" + lines[10].split("\t")[1]
    bad.write_text("\n".join(lines), encoding="utf-8")
    rc, o = build(bad, fx / "o_short", prov=prov)
    check("hard exit: a scored line with < 3 fields (was silently skipped, shifting labels)",
          rc == 1 and "line 11" in o, f"rc={rc} {o[-150:]}")
    lab7 = np.load(prov).copy(); lab7[5] = 7
    np.save(fx / "l7_labels.npy", lab7); shutil.copy(fx / "prov_report.json", fx / "l7_report.json")
    rc, o = build(qe, fx / "o_l7", prov=fx / "l7_labels.npy")
    check("hard exit: provenance labels outside the report's source codes", rc == 1 and "outside" in o, f"rc={rc}")

    # pass-2 guard: the scored file changes between the two passes
    alt = fx / "alt.tsv"
    alt.write_text("".join(f"{sc}\t{s}X\t{t}\n" for sc, s, t, _ in rows), encoding="utf-8")
    code = ("import builtins, runpy, sys\n"
            "real = builtins.open; script, qe, alt = sys.argv[1:4]; n = [0]\n"
            "def op(f, *a, **k):\n"
            "    if str(f) == qe:\n"
            "        n[0] += 1\n"
            "        if n[0] == 2: f = alt\n"
            "    return real(f, *a, **k)\n"
            "builtins.open = op\n"
            "sys.argv = [script] + sys.argv[4:]\n"
            "runpy.run_path(script, run_name='__main__')\n")
    rc, o = run(["-c", code, ROOT / BUILD, qe, alt, "--qe-scores", qe, "--provenance", prov,
                 "--out-dir", fx / "o_chg", "--n-ft", "600", "--n-heldout", "100"])
    check("hard exit: scored file changed between pass 1 and pass 2",
          rc == 1 and "changed between pass 1 and pass 2" in o, f"rc={rc} {o[-200:]}")

    # ------------------------------------------------------------ pool mask + NaN
    mrows = make_rows(2, PLAIN, dup=0.0)
    mfx = d / "mask"
    rng = np.random.RandomState(5)
    mask = rng.rand(len(mrows)) < 0.55
    nan_rows = set(np.flatnonzero(~mask & (rng.rand(len(mrows)) < 0.3)).tolist())
    mqe, mprov = write_fixture(mfx, mrows, score_fmt=lambda k, sc: "nan" if k in nan_rows else sc)
    np.save(mfx / "mask.npy", mask)
    mout = mfx / "out"
    rc, o = build(mqe, mout, "--pool-mask", mfx / "mask.npy", prov=mprov)
    check("pool mask: NaN scores outside the mask are accepted", rc == 0 and len(nan_rows) > 0, o[-300:])
    idx_of = {(s, t): k for k, (_, s, t, _) in enumerate(mrows)}
    if rc == 0:
        written = {s: read_pairs(mout / s) for s in SETS}
        outside = {s: sum(1 for p in v if not mask[idx_of[p]]) for s, v in written.items()}
        check("pool mask: every held-out and FT row is a mask row", not any(outside.values()), str(outside))
        held = written["heldout_un"] + written["heldout_europarl"]
        hp, hs, ht = set(held), {nrm(p[0]) for p in held}, {nrm(p[1]) for p in held}
        avail = [k for k, (_, s, t, _) in enumerate(mrows)
                 if mask[k] and (s, t) not in hp and nrm(s) not in hs and nrm(t) not in ht]
        order = sorted(avail, key=lambda k: -np.float32(mrows[k][0]))
        want_top = [mrows[k][1:3] for k in order[:600]]
        want_bot = [mrows[k][1:3] for k in order[-600:]]
        check("pool mask: ft_topk and ft_bottom recomputed from mask rows only (order included)",
              want_top == written["ft_topk"] and want_bot == written["ft_bottom"],
              f"top {len(want_top)} vs {len(written['ft_topk'])}")
        mm = json.load(open(mout / "manifest.json"))
        pm = mm.get("pool_mask") or {}
        check("manifest records pool mask path, sha256, size and true count",
              pm.get("sha256") == hashlib.sha256((mfx / "mask.npy").read_bytes()).hexdigest()
              and pm.get("size") == len(mask) and pm.get("n_true") == int(mask.sum())
              and pm.get("path", "").endswith("mask.npy"), str(pm))
    in_mask_row = int(np.flatnonzero(mask)[3])
    write_fixture(mfx / "nan_in", mrows, score_fmt=lambda k, sc: "nan" if k in nan_rows or k == in_mask_row else sc)
    rc, o = build(mfx / "nan_in/scores.tsv", mfx / "o_nanin", "--pool-mask", mfx / "mask.npy",
                  prov=mfx / "nan_in/prov_labels.npy")
    check("pool mask: a NaN score inside the mask is a hard error", rc == 1 and "NaN" in o, f"rc={rc}")
    rc, o = build(mqe, mfx / "o_nomask", prov=mprov)
    check("without a pool mask any NaN score is a hard error", rc == 1 and "NaN" in o, f"rc={rc}")
    np.save(mfx / "mask_short.npy", mask[:-1])
    np.save(mfx / "mask_int.npy", mask.astype(np.int64))
    rc1, _ = build(mqe, mfx / "o_ms", "--pool-mask", mfx / "mask_short.npy", prov=mprov)
    rc2, _ = build(mqe, mfx / "o_mi", "--pool-mask", mfx / "mask_int.npy", prov=mprov)
    check("pool mask of the wrong length or dtype is a hard error", rc1 == 1 and rc2 == 1, f"{rc1} {rc2}")

    # ------------------------------------------------------------ pretraining exclusion
    pfx = d / "pre"
    prows = make_rows(3, PLAIN)
    pqe, pprov = write_fixture(pfx, prows)
    ptl = {(s, t): lab for _, s, t, lab in prows}
    un_pairs = sorted({(s, t) for _, s, t, lab in prows if lab == "un"})
    ep_pairs = sorted({(s, t) for _, s, t, lab in prows if lab == "europarl"})
    random.Random(9).shuffle(un_pairs)
    clean_un = set(un_pairs[:200])
    pre_lines = []
    for i, (s, t) in enumerate(un_pairs[200:]):
        if i % 3 == 0:
            pre_lines.append((s, t))                                     # exact pair
        elif i % 3 == 1:
            pre_lines.append(("  " + s.upper() + " ", f"autre traduction {i}"))  # same source, other target
        else:
            pre_lines.append((f"another source {i}", t.replace(" ", "\u00a0")))  # same target (NBSP), other source
    pre_lines += ep_pairs[150:]                                          # europarl mostly pretrained
    with open(pfx / "pre.en", "w", encoding="utf-8", newline="\n") as a, \
         open(pfx / "pre.fr", "w", encoding="utf-8", newline="\n") as b:
        for s, t in pre_lines:
            a.write(s + "\n"); b.write(t + "\n")
    pout = pfx / "out"
    rc, o = build(pqe, pout, "--pretrain-src", pfx / "pre.en", "--pretrain-tgt", pfx / "pre.fr", prov=pprov)
    check("pretrain exclusion run succeeds", rc == 0, o[-400:])
    if rc == 0:
        pp = set(pre_lines); ps = {nrm(s) for s, _ in pre_lines}; pt = {nrm(t) for _, t in pre_lines}
        hu, he = read_pairs(pout / "heldout_un"), read_pairs(pout / "heldout_europarl")
        hit = [p for p in hu + he if p in pp or nrm(p[0]) in ps or nrm(p[1]) in pt]
        check("no held-out pair, normalised source or normalised target occurs in the pretraining file",
              not hit and set(hu) <= clean_un, f"{len(hit)} hits; {len(set(hu) - clean_un)} outside clean UN")
        pm = json.load(open(pout / "manifest.json"))
        un_rows = [(s, t) for _, s, t, lab in prows if lab == "un"]
        want_excl = sum(1 for p in un_rows if p in pp or nrm(p[0]) in ps or nrm(p[1]) in pt)
        fr = pm["ft_random"]["pretrain_overlap_fraction"]["any"]
        rnd = read_pairs(pout / "ft_random")
        want_fr = sum(1 for p in rnd if p in pp or nrm(p[0]) in ps or nrm(p[1]) in pt) / len(rnd)
        check("manifest: pretrain rows, held-out pool exclusions and FT overlap fraction match recounts",
              pm["pretrain"]["rows"] == len(pre_lines)
              and pm["heldout_sets"]["heldout_un"]["pool_rows_excluded_pretrain"] == want_excl
              and abs(fr - want_fr) < 1e-9 and "heldout_post_write_check" in pm["pretrain"],
              f"excl {pm['heldout_sets']['heldout_un']['pool_rows_excluded_pretrain']} vs {want_excl}; fr {fr} vs {want_fr}")
        check("FT sets are NOT filtered by pretraining (ft_random still contains pretraining rows)",
              want_fr > 0, str(want_fr))
    with open(pfx / "pre_all.en", "w", encoding="utf-8", newline="\n") as a, \
         open(pfx / "pre_all.fr", "w", encoding="utf-8", newline="\n") as b:
        for s, t in ep_pairs:
            a.write(s + "\n"); b.write(t + "\n")
    rc, o = build(pqe, pfx / "o_all", "--pretrain-src", pfx / "pre_all.en", "--pretrain-tgt", pfx / "pre_all.fr",
                  prov=pprov)
    check("hard exit: pretraining covers every Europarl row -> heldout_europarl pool too small",
          rc == 1 and "europarl" in o, f"rc={rc}")
    (pfx / "pre_short.fr").write_text("".join(t + "\n" for _, t in pre_lines[:-1]), encoding="utf-8")
    rc, o = build(pqe, pfx / "o_ps", "--pretrain-src", pfx / "pre.en", "--pretrain-tgt", pfx / "pre_short.fr",
                  prov=pprov)
    check("hard exit: pretraining sides differ in length", rc == 1 and "differ" in o, f"rc={rc}")

    # ------------------------------------------------------------ normalised reservation
    tfx = d / "twins"
    base = make_rows(4, PLAIN, dup=0.0, special=False)
    twins = []
    for _, s, t, lab in base:
        if lab == "un":
            twins.append((0.9999, "  " + s.upper().replace(" ", "   "), f"une autre traduction de {s}", "giga-fren"))
            twins.append((0.9998, f"a different source for {t}", t.replace(" ", "\u00a0"), "giga-fren"))
    trows = base + twins
    random.Random(11).shuffle(trows)
    tqe, tprov = write_fixture(tfx, trows)
    tout = tfx / "out"
    rc, o = run([ROOT / BUILD, "--qe-scores", tqe, "--provenance", tprov, "--out-dir", tout,
                 "--n-ft", "2600", "--n-heldout", "100"])
    check("twin fixture run succeeds", rc == 0, o[-400:])

    def norm_leaks(outdir):
        held = read_pairs(outdir / "heldout_un") + read_pairs(outdir / "heldout_europarl")
        hs, ht, hp = {nrm(s) for s, _ in held}, {nrm(t) for _, t in held}, set(held)
        return {s: sum(1 for p in read_pairs(outdir / s) if p in hp or nrm(p[0]) in hs or nrm(p[1]) in ht)
                for s in FT}

    if rc == 0:
        lk = norm_leaks(tout)
        tm = json.load(open(tout / "manifest.json"))["reservation"]
        check("a held-out source/reference with a different partner never enters any FT set",
              not any(lk.values()), str(lk))
        check("manifest counts the extra source- and target-matched reservations",
              tm["extra_by_source"] >= 100 and tm["extra_by_target"] >= 100, str(tm))
    if base_script.exists():
        rc_b, _ = run([base_script, "--qe-scores", tqe, "--provenance", tprov, "--out-dir", tfx / "old",
                       "--n-ft", "2600", "--n-heldout", "100"])
        check("twin fixture has power: the baseline exact-pair reservation DOES leak here",
              rc_b == 0 and any(norm_leaks(tfx / "old").values()), f"rc={rc_b}")

    suite_decide_failclosed(ctx)
    suite_collect_manifest(ctx)


# ---------------------------------------------------------------- e03_decide
GO_ROWS = """condition seed testset bleu
baseline - newstest2014 38.21
baseline - heldout_un 35.00
baseline - heldout_europarl 33.00
ft_topk 42 newstest2014 36.60
ft_topk 1 newstest2014 36.70
ft_topk 2 newstest2014 36.50
ft_topk 42 heldout_un 36.50
ft_topk 1 heldout_un 36.60
ft_topk 2 heldout_un 36.40
ft_random 42 newstest2014 37.90
ft_random 1 newstest2014 38.00
ft_random 2 newstest2014 37.80
"""
EP_UP = "ft_topk 42 heldout_europarl 33.80\nft_topk 1 heldout_europarl 33.90\nft_topk 2 heldout_europarl 33.70\n"
EP_DOWN = "ft_topk 42 heldout_europarl 32.80\nft_topk 1 heldout_europarl 32.90\nft_topk 2 heldout_europarl 32.70\n"


def suite_decide_failclosed(ctx):
    d, run, check, ROOT = ctx.d / "decide", ctx.run, ctx.check, ctx.ROOT
    d.mkdir()

    def dec(text, *extra):
        p = d / f"r{abs(hash((text, extra)))}.tsv"
        p.write_text(text)
        return run([ROOT / "phase0/e03_decide.py", "--results", p, *extra])

    # The frozen gate (2026-09-18, D5) is UN-only, so these pass the two-set list
    # explicitly: what is under test is fail-closed behaviour for whatever is listed.
    rc, o = dec(GO_ROWS, "--indomain", "heldout_un,heldout_europarl")
    check("decide: a listed in-domain set with no ft_topk rows -> exit 2 (explicit two-set list)",
          rc == 2 and "NO DECISION" in o and "heldout_europarl" in o, f"rc={rc}")
    no_base = GO_ROWS.replace("baseline - heldout_europarl 33.00\n", "") + EP_UP
    rc, o = dec(no_base, "--indomain", "heldout_un,heldout_europarl")
    check("decide: baseline lacks a listed in-domain set -> exit 2", rc == 2 and "baseline" in o, f"rc={rc}")
    rc, o = dec(GO_ROWS, "--indomain", "heldout_un")
    check("decide: --indomain heldout_un on the same TSV decides (exit 0)", rc == 0 and "VERDICT: GO" in o, f"rc={rc}")
    rc, o = dec(GO_ROWS + EP_UP)
    check("decide: all listed sets present and improved -> GO", rc == 0, f"rc={rc}")
    rc, o = dec(GO_ROWS + EP_DOWN, "--indomain", "heldout_un,heldout_europarl")
    check("decide: UN up, Europarl down -> NO-GO when both sets are listed",
          rc == 1 and "criterion 2: FAIL" in o, f"rc={rc}")
    rc, o = dec(GO_ROWS + EP_UP, "--indomain", " , ")
    check("decide: empty --indomain -> exit 2", rc == 2, f"rc={rc}")
    zv = GO_ROWS + EP_UP
    for sd, v in (("42", "36.60"), ("1", "36.70"), ("2", "36.50")):
        zv = zv.replace(f"ft_topk {sd} newstest2014 {v}", f"ft_topk {sd} newstest2014 36.60")
    for sd, v in (("42", "37.90"), ("1", "38.00"), ("2", "37.80")):
        zv = zv.replace(f"ft_random {sd} newstest2014 {v}", f"ft_random {sd} newstest2014 37.90")
    rc, o = dec(zv)
    check("decide: zero seed variance in BOTH news cells -> exit 2 'NO DECISION ... zero seed variance' (was NO-GO)",
          rc == 2 and "NO DECISION" in o and "zero seed variance" in o and "VERDICT: NO-GO" not in o, f"rc={rc} {o[-300:]}")
    one = GO_ROWS + EP_UP
    for sd, v in (("42", "37.90"), ("1", "38.00"), ("2", "37.80")):
        one = one.replace(f"ft_random {sd} newstest2014 {v}", f"ft_random {sd} newstest2014 37.90")
    rc, o = dec(one)
    check("decide: zero variance in only one news cell still decides (exit 0 or 1)", rc in (0, 1) and "VERDICT" in o,
          f"rc={rc} {o[-300:]}")
    aux = GO_ROWS + EP_UP + "baseline - heldout_giga-fren 30.0\nft_topk 42 heldout_giga-fren 20.0\n"
    rc, o = dec(aux)
    check("decide: a set outside --indomain is shown as [aux] and does not change the verdict",
          rc == 0 and "[aux] heldout_giga-fren" in o, f"rc={rc}")


# ---------------------------------------------------------------- e03_collect
def suite_collect_manifest(ctx):
    d, run, check, ROOT = ctx.d / "collect", ctx.run, ctx.check, ctx.ROOT
    mt, sf = d / "mt", d / "sf"
    for p in (mt / "scripts", mt / "data_enfr_v1", mt / "ckpt_hf", mt / "src", sf / "configs/phase0", sf / "ctrl"):
        p.mkdir(parents=True, exist_ok=True)
    for x in ("test.en", "test.fr"):
        (mt / "data_enfr_v1" / x).write_text("line\n")
    (mt / "src/__init__.py").write_text("")
    ctx.write_ckpt(mt / "ckpt_hf/base.pt")
    (sf / "configs/base.yaml").write_text('{"model": {}}')
    (mt / "scripts/fake_eval.py").write_text(
        "import src  # noqa: F401\nimport argparse\nap = argparse.ArgumentParser()\n"
        "for f in ('--ckpt','--config','--src','--ref','--beam','--length-penalty'): ap.add_argument(f)\n"
        "a = ap.parse_args()\nprint('BLEU (fr, sacrebleu 13a): 30.00')\n")
    (sf / "configs/phase0/ft_topk_lr0.15.yaml").write_text(json.dumps(
        {"checkpoint": {"dir": "checkpoints/ft_topk_lr0.15"}, "training": {"max_steps": 10}, "model": {}}))
    (sf / "configs/phase0/run_stage2.sh").write_text(
        f"python train.py --config {sf}/configs/phase0/ft_topk_lr${{LR}}.yaml --seed 42 --suffix _s42\n")
    (mt / "checkpoints/ft_topk_lr0.15_s42").mkdir(parents=True)
    ctx.write_ckpt(mt / "checkpoints/ft_topk_lr0.15_s42/final.pt", global_step=10)
    for x in ("heldout_un", "heldout_giga-fren"):
        for e in ("en", "fr"):
            (sf / "ctrl" / f"{x}.{e}").write_text("line\n")
    out = sf / "results/bleu.tsv"
    args = [ROOT / "phase0/e03_collect.py", "--mt-root", mt, "--runner", sf / "configs/phase0/run_stage2.sh",
            "--lr-scale", "0.15", "--baseline-ckpt", "ckpt_hf/base.pt", "--baseline-config", sf / "configs/base.yaml",
            "--controls-dir", sf / "ctrl", "--out", out, "--eval-script", "scripts/fake_eval.py"]
    # checkpoints are JSON fixtures: e03_collect must read them through the shared stub,
    # or real torch (present on any training box) tries to unpickle text
    cenv = {"PYTHONPATH": str(ctx.make_faketorch(ctx.d))}

    def manifest(sets):
        json.dump({"heldout_sets": {s: {"n": 1} for s in sets}}, open(sf / "ctrl/manifest.json", "w"))

    manifest(["heldout_un", "heldout_giga-fren"])
    rc, o = run(args, env=cenv)
    got = sorted({l.split("\t")[2] for l in out.read_text().splitlines()[1:]}) if out.exists() else []
    check("collect: test sets come from manifest.json heldout_sets (heldout_europarl not required when unlisted)",
          rc == 0 and got == ["heldout_giga-fren", "heldout_un", "newstest2014"], f"rc={rc} {got} {o[-200:]}")
    out.unlink(missing_ok=True)
    manifest(["heldout_un", "heldout_europarl"])
    rc, o = run(args, env=cenv)
    check("collect: a manifest-listed held-out set with missing files is a hard failure",
          rc == 1 and "heldout_europarl" in o and "missing" in o and not out.exists(), f"rc={rc}")
    # from here the default sets' files all exist, so only the explicit guard can fail
    for e in ("en", "fr"):
        (sf / "ctrl" / f"heldout_europarl.{e}").write_text("line\n")
    manifest([])
    rc, o = run(args, env=cenv)
    check("collect: a manifest listing no held-out sets is a hard failure (no silent fallback)",
          rc == 1 and "NO held-out sets" in o and not out.exists(), f"rc={rc} {o[-150:]}")
    json.dump({"n_ft": 1}, open(sf / "ctrl/manifest.json", "w"))
    rc, o = run(args, env=cenv)
    got = sorted({l.split("\t")[2] for l in out.read_text().splitlines()[1:]}) if out.exists() else []
    check("collect: a manifest without heldout_sets falls back to heldout_un + heldout_europarl",
          rc == 0 and got == ["heldout_europarl", "heldout_un", "newstest2014"], f"rc={rc} {got}")
