"""Diagnostic: the throughput statistic E of 04d by step range at K = 208, x = 0 (why P1 of session 1 differs)."""
import json, os, re, sys
import numpy as np
sys.argv = ["x"]; B = os.path.expanduser("~/replay_check/build")
src = open(f"{B}/04d_toks_test.py").read().split("import multiprocessing as mp")[0]
exec(src)
W1 = windows_s1(perm_seq(208, 0, 0)); e = eff(W1)
out = []
for a, b in ((100, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 49174), (49174, 60000),
             (60000, 70000), (70000, 80000), (80000, 90000), (90000, 98351), (98351, 100000)):
    m = ok1 & (ends1 > a) & (ends1 <= b)
    out.append({"steps": [a, b], "n": int(m.sum()), "r_E": corr(e[m], y1[m]),
                "toks_median": float(np.median(np.exp(y1[m]))), "toks_cv": float(np.exp(y1[m]).std() / np.exp(y1[m]).mean()),
                "eff_sd": float(e[m].std())})
W2 = windows_s2(perm_seq(208, 0, 0)); e2 = eff(W2)
out.append({"steps": "session2", "n": int(ok2.sum()), "r_E": corr(e2[ok2], y2[ok2]),
            "toks_median": float(np.median(np.exp(y2[ok2]))), "toks_cv": float(np.exp(y2[ok2]).std() / np.exp(y2[ok2]).mean())})
json.dump(out, open(f"{B}/04e_toks_by_range.json", "w"), indent=1)
for d in out: print(d)
