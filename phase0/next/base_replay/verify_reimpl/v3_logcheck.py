"""v3: check the reconstructed batch ORDER against the April training records.

Signal: the trainer logs Tok/s every 100 steps = (sum of non-pad target labels) / wall time.
Wall time per batch grows with padded size, so a window's Tok/s should track
  eff_w = sum(ntok) / sum(rows*max_src + rows*(max_tgt-1))  over the window's 100 batches.
That holds only if we know which batches ran in which window, i.e. only for the right order.
We score the emulated offset (208 words) against every other offset K in 0..KMAX, for
session 1 / P1, session 2 / P1 (the replay), and session 1 / P2, P3 (offset D between epochs).
Excluded windows: first window of a session, windows that start right after a step that is a
multiple of 500 (checkpoint / eval time lands there), windows that straddle an epoch end.
Also records: console epoch-end lines, spike-guard steps, the interrupt step.
"""
import json, os, random, re, sys
import numpy as np

OUT = os.path.expanduser("~/replay_check/verify_reimpl")
BASE = os.path.expanduser("~/workspace/Machine_translation/swanlog/")
S1 = "run-20260424_000243-n00ar93oggh8om85j3o5f"
S2 = "run-20260424_021306-2fo9021j4lyiq27t2jghh"
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
K0 = 208


def read_backup(run):
    raw = open(BASE + run + "/backup.swanlab", "rb").read()
    msgs, sc = [], {}
    for line in raw.split(b"\n"):
        i = line.find(b'{"model_type": ')
        if i < 0:
            continue
        try:
            d = json.loads(line[i:])
        except Exception:
            continue
        if d["model_type"] == "Log":
            msgs.append(d["data"]["message"])
        elif d["model_type"] == "Scalar":
            k = d["data"]["key"]
            if k in ("train/tok_per_sec", "train/loss"):
                sc.setdefault(k, {})[int(d["data"]["step"])] = float(d["data"]["metric"]["data"])
    return msgs, sc


def parse(run):
    msgs, sc = read_backup(run)
    con = {}
    for m in msgs:
        r = re.match(r"\s*Step\s+(\d+) \| Loss ([\d.]+) \| LR \S+ \| Tok/s (\d+)", m)
        if r:
            con[int(r.group(1))] = (float(r.group(2)), float(r.group(3)))
    info = {"console_step_lines": len(con), "scalar_tokps": len(sc.get("train/tok_per_sec", {})),
            "epoch_lines": [m.strip() for m in msgs if "End of epoch" in m],
            "spike_steps": [int(x) for m in msgs for x in re.findall(r"\[spike guard\] step (\d+)", m)],
            "save_lines": [m.strip() for m in msgs if re.search(r"Saving checkpoint at step|Resumed from|batches/epoch", m)]}
    tok = dict(sc.get("train/tok_per_sec", {}))
    agree = [abs(tok[s] - con[s][1]) <= 1 for s in con if s in tok]
    info["console_vs_scalar_tokps_agree"] = f"{sum(agree)}/{len(agree)}"
    for s, (l, t) in con.items():
        tok.setdefault(s, t)
    return info, tok


C = np.load(f"{OUT}/v2_comp_np226.npz")
B = {e: int(C[f"starts{e}"].size) for e in (1, 2, 3)}
cost = {e: C[f"rows{e}"] * C[f"mxs{e}"].astype(np.int64) + C[f"rows{e}"] * (C[f"mxt{e}"].astype(np.int64) - 1) for e in (1, 2, 3)}
ntok = {e: C[f"ntok{e}"] for e in (1, 2, 3)}


def windows(tok, first_step, pos_of_step):
    """returns list of (observed tok/s, list of (epoch, position) for the 100 batches)"""
    out = []
    for s in sorted(tok):
        a = s - 99
        if a <= first_step + 1 or (s - 100) % 500 == 0:
            continue
        ep = [pos_of_step(g) for g in range(a, s + 1)]
        if len({e for e, _ in ep}) != 1:
            continue
        out.append((tok[s], ep[0][0], np.array([p for _, p in ep])))
    return out


def s1_pos(g):            # session 1, global step g (1-based) -> (epoch, position)
    if g <= B[1]:
        return 1, g - 1
    if g <= B[1] + B[2]:
        return 2, g - B[1] - 1
    return 3, g - B[1] - B[2] - 1


def s2_pos(g):            # session 2 restarted at 100,000 with a fresh epoch
    return 1, g - 100_001


def score(ws, perm_by_epoch, epoch):
    obs, pred = [], []
    for t, e, pos in ws:
        if e != epoch:
            continue
        b = perm_by_epoch[pos]
        obs.append(t); pred.append(ntok[e][b].sum() / cost[e][b].sum())
    obs, pred = np.array(obs), np.array(pred)
    return float(np.corrcoef(obs, pred)[0, 1]), len(obs)


def perms(K, D=0, upto=1):
    random.seed(42)
    for _ in range(K):
        random.getrandbits(32)
    res = {}
    for e in (1, 2, 3)[:upto]:
        if e > 1:
            for _ in range(D):
                random.getrandbits(32)
        p = list(range(B[e])); random.shuffle(p); res[e] = np.array(p)
    return res


def main():
    rep = {}
    i1, tok1 = parse(S1); i2, tok2 = parse(S2)
    rep["session1"] = i1; rep["session2"] = i2
    w1 = windows(tok1, 0, s1_pos); w2 = windows(tok2, 100_000, s2_pos)
    rep["eligible_windows"] = {"s1_P1": sum(e == 1 for _, e, _ in w1), "s1_P2": sum(e == 2 for _, e, _ in w1),
                               "s1_P3": sum(e == 3 for _, e, _ in w1), "s2_P1": len(w2)}
    # sanity: the saved perms equal perms(208)
    p208 = perms(K0, 0, 3)
    rep["v2_perm_equals_K208"] = all(np.array_equal(p208[e], C[f"perm{e}"]) for e in (1, 2, 3))

    # scan K for epoch 1 (both sessions)
    r1, r2, same = [], [], []
    for K in range(KMAX + 1):
        p = perms(K, 0, 1)[1]
        same.append(bool(np.array_equal(p, p208[1])))
        r1.append(score(w1, p, 1)[0]); r2.append(score(w2, p, 1)[0])
    r1, r2 = np.array(r1), np.array(r2)
    distinct = ~np.array(same)
    rep["K_scan"] = {
        "K_range": [0, KMAX],
        "K_giving_identical_epoch1_order": [int(k) for k in np.flatnonzero(same)],
        "s1_P1_r_at_208": float(r1[K0]), "s2_P1_r_at_208": float(r2[K0]),
        "s1_P1_rank_of_208": int(1 + (r1 > r1[K0]).sum()), "s2_P1_rank_of_208": int(1 + (r2 > r2[K0]).sum()),
        "s1_P1_other_K_max": float(r1[distinct].max()), "s1_P1_other_K_argmax": int(np.flatnonzero(distinct)[r1[distinct].argmax()]),
        "s1_P1_other_K_mean_sd": [float(r1[distinct].mean()), float(r1[distinct].std())],
        "s2_P1_other_K_max": float(r2[distinct].max()), "s2_P1_other_K_argmax": int(np.flatnonzero(distinct)[r2[distinct].argmax()]),
        "s2_P1_other_K_mean_sd": [float(r2[distinct].mean()), float(r2[distinct].std())],
        "top5_s1": [(int(k), round(float(r1[k]), 4)) for k in np.argsort(-r1)[:5]],
        "top5_s2": [(int(k), round(float(r2[k]), 4)) for k in np.argsort(-r2)[:5]],
    }
    # extra draws between epochs (D) for P2 and P3, K fixed at 208
    rD2, rD3 = [], []
    for D in range(0, 501):
        p = perms(K0, D, 3)
        rD2.append(score(w1, p[2], 2)[0]); rD3.append(score(w1, p[3], 3)[0])
    rD2, rD3 = np.array(rD2), np.array(rD3)
    rep["D_scan"] = {"D_range": [0, 500], "P2_r_at_D0": float(rD2[0]), "P2_rank_D0": int(1 + (rD2 > rD2[0]).sum()),
                     "P2_others_max": float(rD2[1:].max()),
                     "P3_r_at_D0": float(rD3[0]), "P3_rank_D0": int(1 + (rD3 > rD3[0]).sum()),
                     "P3_others_max": float(rD3[1:].max()), "P3_windows": rep["eligible_windows"]["s1_P3"]}
    # spike-guard batches: where do they sit?
    sp = []
    eff_all = ntok[1] / cost[1]
    for g in i1["spike_steps"]:
        e, pos = s1_pos(g)
        b = p208[e][pos]
        sp.append({"step": g, "epoch": e, "position": int(pos), "batch": int(b), "rows": int(C[f"rows{e}"][b]),
                   "max_src": int(C[f"mxs{e}"][b]), "max_tgt": int(C[f"mxt{e}"][b])})
    rep["spike_batches"] = sp
    json.dump(rep, open(f"{OUT}/v3_logcheck.json", "w"), indent=1)
    np.savez(f"{OUT}/v3_scan.npz", r1=r1, r2=r2, rD2=rD2, rD3=rD3, same=np.array(same))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
