"""Hand-computed test of forgetgenuse.py (stdlib unittest). Run:  python test_forgetgenuse.py

Fine-tuning set (target side):  "le chat noir" / "un chien" / "un chat"
  -> words vocabulary {le:1, chat:2, noir:1, un:2, chien:1}

Test set, two segments (lowercased \\w+ tokens):
  seg 1  src  the black cat sleeps in Paris
         ref  le chat noir dort à paris          (a U+2028 sits inside the line: must not split it)
         base le chat noir dort à paris          B: le chat noir dort à paris = 6
         ft   le chat noir mange à lyon          F: le chat noir à = 4      lost: dort, paris
  seg 2  src  a dog and a dog
         ref  un chien et un chien               (un 2, chien 2, et 1)
         base un chien et un chat                B: un 2, chien 1, et 1 = 4
         ft   un chien un chien chien            F: un 2, chien min(3,2)=2 = 4  lost: et; gained: chien 1

  ForgetGenUse (Eq 1) = L / reference occurrences; the variant _per_base_correct = L / B.
  All : B = 10, L = 3, G = 1, F = 8 (= 10 - 3 + 1), ref occurrences 11   -> 3/11  (variant 3/10 = 0.3)
  ID  (le chat noir un chien): B = 3 + 3 = 6, L = 0, ref occ 7           -> 0/7   (variant 0/6)
  OOD (dort à paris et):       B = 3 + 1 = 4, L = 3, ref occ 4           -> 3/4   (variant 3/4)
  macro: per-type L/B over the 9 types: dort 1, paris 1, et 1, the rest 0 -> All 3/9, ID 0, OOD 3/4
  copy split: only "paris" is also in its source segment -> OOD_in_src 1/1, OOD_not_in_src 2/3,
              ID_in_src no reference occurrences (None), ID_not_in_src 0/7
  --tok words_cased: "Le" (ref, base, ft) is not in the cased vocabulary (training has "le"):
              ID ref occ 6, B = 5, L = 0; OOD ref occ 5, B = 5, L = 3 -> 3/5 (variant 0.6); All still 3/11
              (variant 0.3); OOD macro (Le 0, dort 1, à 0, Paris 1, et 1) = 0.6
  --id-min-count 2: ID = {chat, un}: ref occ 3, B = 1 + 2 = 3, L = 0 -> 0.0; OOD ref occ 8, B = 7, L = 3
              -> 3/8 (variant 3/7)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forgetgenuse as fg  # noqa: E402

SRC = ["the black cat sleeps in Paris", "a dog and a dog"]
REF = ["Le chat noir dort  à Paris.", "un chien et un chien"]
BASE = ["Le chat noir dort à Paris.", "un chien et un chat"]
FT = ["Le chat noir mange à Lyon.", "un chien un chien chien"]
TRAIN_FR = ["le chat noir", "un chien", "un chat"]
TRAIN_EN = ["the black cat", "a dog", "a cat"]


def write(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(x + "\n" for x in lines))
    return str(path)


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        self.src, self.ref = write(d / "test.en", SRC), write(d / "test.fr", REF)
        self.base, self.ft = write(d / "base.hyp", BASE), write(d / "ft.hyp", FT)
        write(d / "ft.fr", TRAIN_FR)
        write(d / "ft.en", TRAIN_EN)
        self.prefix = str(d / "ft")

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, *extra, ft=None, ref=None):
        out = str(self.d / "out.json")
        res = fg.run(["--src", self.src, "--ref", ref or self.ref, "--base-hyp", self.base,
                      "--ft-hyp", ft or self.ft, "--ft-train", self.prefix, "--out", out, *extra])
        self.assertEqual(json.loads(Path(out).read_text(encoding="utf-8")), json.loads(json.dumps(res)))
        return res

    def test_read_lines_splits_on_newline_only(self):
        p = write(self.d / "x.txt", ["a b\rc\x85d", "e"])
        self.assertEqual(len(fg.read_lines(p)), 2)

    def test_words_hand_computed(self):
        r = self.go()["runs"][0]["results"]
        self.assertEqual({k: r["All"][k] for k in ("ref_occ", "base_correct", "ft_correct", "lost", "gained")},
                         {"ref_occ": 11, "base_correct": 10, "ft_correct": 8, "lost": 3, "gained": 1})
        self.assertAlmostEqual(r["All"]["forgetgenuse"], 3 / 11)                 # Eq 1: lost / reference occurrences
        self.assertAlmostEqual(r["All"]["forgetgenuse_per_base_correct"], 0.3)   # variant: lost / base-correct
        self.assertEqual((r["ID"]["base_correct"], r["ID"]["lost"], r["ID"]["ref_occ"]), (6, 0, 7))
        self.assertAlmostEqual(r["ID"]["forgetgenuse"], 0 / 7)
        self.assertEqual((r["OOD"]["base_correct"], r["OOD"]["lost"], r["OOD"]["ref_occ"]), (4, 3, 4))
        self.assertAlmostEqual(r["OOD"]["forgetgenuse"], 3 / 4)
        self.assertAlmostEqual(r["All"]["forgetgenuse_macro"], 3 / 9)
        self.assertAlmostEqual(r["ID"]["forgetgenuse_macro"], 0.0)
        self.assertAlmostEqual(r["OOD"]["forgetgenuse_macro"], 0.75)
        self.assertEqual((r["All"]["ref_types"], r["ID"]["ref_types"], r["OOD"]["ref_types"]), (9, 5, 4))
        self.assertAlmostEqual(r["ref_occ_share_ID"], 7 / 11)
        self.assertAlmostEqual(r["OOD_in_src"]["forgetgenuse"], 1.0)
        self.assertAlmostEqual(r["OOD_not_in_src"]["forgetgenuse"], 2 / 3)
        self.assertIsNone(r["ID_in_src"]["forgetgenuse"])
        self.assertAlmostEqual(r["ID_not_in_src"]["forgetgenuse"], 0.0)

    def test_top_lists(self):
        run = self.go()["runs"][0]
        self.assertEqual([x["type"] for x in run["top_lost_OOD"]], ["dort", "et", "paris"])
        self.assertEqual(run["top_lost_ID"], [])
        self.assertEqual([(x["type"], x["gained"], x["ft_train_count"]) for x in run["top_gained"]], [("chien", 1, 1)])

    def test_words_cased(self):
        r = self.go("--tok", "words_cased")["runs"][0]["results"]
        self.assertEqual((r["ID"]["base_correct"], r["ID"]["lost"]), (5, 0))
        self.assertEqual((r["OOD"]["base_correct"], r["OOD"]["lost"]), (5, 3))
        self.assertEqual((r["ID"]["ref_occ"], r["OOD"]["ref_occ"]), (6, 5))
        self.assertAlmostEqual(r["OOD"]["forgetgenuse_per_base_correct"], 0.6)
        self.assertAlmostEqual(r["All"]["forgetgenuse_per_base_correct"], 0.3)
        for k in ("All", "ID", "OOD"):
            self.assertAlmostEqual(r[k]["forgetgenuse"], r[k]["lost"] / r[k]["ref_occ"])
        self.assertAlmostEqual(r["All"]["forgetgenuse"], 3 / 11)
        self.assertAlmostEqual(r["OOD"]["forgetgenuse_macro"], 0.6)

    def test_id_min_count(self):
        out = self.go("--id-min-count", "2")
        r = out["runs"][0]["results"]
        self.assertEqual(out["ft_train"]["id_types"], 2)
        self.assertEqual((r["ID"]["base_correct"], r["ID"]["lost"]), (3, 0))
        self.assertAlmostEqual(r["OOD"]["forgetgenuse_per_base_correct"], 3 / 7)
        self.assertEqual((r["ID"]["ref_occ"], r["OOD"]["ref_occ"]), (3, 8))
        self.assertAlmostEqual(r["OOD"]["forgetgenuse"], 3 / 8)

    def test_invariants_and_summary(self):
        same_as_ref = write(self.d / "ft_ref.hyp", REF)
        same_as_base = write(self.d / "ft_base.hyp", BASE)
        out = fg.run(["--src", self.src, "--ref", self.ref, "--base-hyp", self.base, "--ft-hyp", same_as_ref,
                      same_as_base, self.ft, "--ft-train", self.prefix, "--out", str(self.d / "o.json")])
        a, b, c = (x["results"] for x in out["runs"])
        for k in fg.SPLITS + fg.COPY_SPLITS:
            self.assertEqual(a[k]["lost"], 0)
            self.assertEqual((b[k]["lost"], b[k]["gained"]), (0, 0))
        s = out["summary_over_runs"]["All.forgetgenuse"]
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 1 / 11)
        self.assertAlmostEqual(s["max"], 3 / 11)
        sv = out["summary_over_runs"]["All.forgetgenuse_per_base_correct"]
        self.assertAlmostEqual(sv["mean"], 0.1)
        self.assertAlmostEqual(sv["max"], 0.3)

    def test_clipping(self):
        # seg 1: ref "oui", base "oui oui oui" -> B = min(3,1) = 1; ft "non" -> F = 0, L = 1
        # seg 2: ref "oui oui", base "oui" -> B = 1; ft "oui oui oui" -> F = min(3,2) = 2, G = 1
        # B = 2, F = 2, L = 1, G = 1, ref occ 3 -> ForgetGenUse (Eq 1) 1/3; per base-correct 0.5 (all OOD)
        d = self.d
        src = write(d / "c.en", ["yes", "yes yes"])
        ref = write(d / "c.fr", ["oui", "oui oui"])
        base = write(d / "c_base.hyp", ["oui oui oui", "oui"])
        ft = write(d / "c_ft.hyp", ["non", "oui oui oui"])
        write(d / "c_train.fr", ["non"])
        write(d / "c_train.en", ["no"])
        r = fg.run(["--src", src, "--ref", ref, "--base-hyp", base, "--ft-hyp", ft, "--ft-train",
                    str(d / "c_train"), "--out", str(d / "c.json")])["runs"][0]["results"]
        self.assertEqual({k: r["All"][k] for k in ("ref_occ", "base_correct", "ft_correct", "lost", "gained")},
                         {"ref_occ": 3, "base_correct": 2, "ft_correct": 2, "lost": 1, "gained": 1})
        self.assertAlmostEqual(r["OOD"]["forgetgenuse"], 1 / 3)
        self.assertAlmostEqual(r["OOD"]["forgetgenuse_per_base_correct"], 0.5)
        self.assertEqual(r["ID"]["base_correct"], 0)

    def test_refuses(self):
        with self.assertRaises(SystemExit):
            self.go(ft=write(self.d / "short.hyp", FT[:1]))
        with self.assertRaises(SystemExit):
            self.go("--expect-ref-sha", "newstest2014")
        write(self.d / "ft.en", TRAIN_EN[:2])
        with self.assertRaises(SystemExit):
            self.go()


if __name__ == "__main__":
    unittest.main(verbosity=2)
