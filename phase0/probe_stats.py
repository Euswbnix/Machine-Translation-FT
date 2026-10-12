#!/usr/bin/env python3
"""Two-sample statistics for the explanatory probes, with EXACT Welch critical values.

e03_decide.py looks its critical value up at floor(df) in an integer table. That is stricter
than the 0.05 it names, which was the safe direction for a gate whose hypothesis needed a
difference. E0.5 has contrasts where "no difference detected" is the outcome that favours
the hypothesis, so a rule that is quietly too strict would lean towards the convenient
answer. This module computes the two-sided Student-t tail and its inverse at the fractional
Welch-Satterthwaite df, with no dependency beyond math.

e03_decide.py is frozen with the E0.3 gate and is deliberately not changed.
"""
from __future__ import annotations

import math


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """P(|T| >= |t|) for Student's t with (possibly fractional) df."""
    if not (df > 0) or math.isnan(t):
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_crit(df: float, alpha: float = 0.05) -> float:
    """Two-sided critical value: the t with P(|T| >= t) = alpha. inf when df is unusable."""
    if not (df > 0) or math.isnan(df):
        return float("inf")
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e9:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_sd(xs: list[float]):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def contrast(a: list[float], b: list[float], alpha: float = 0.05):
    """Compare two sets of per-seed scores. Returns a dict, or None when Welch is undefined
    (fewer than two values on a side, or zero variance on both).

      gap    mean(a) - mean(b)
      noise  the larger of the two seed sds (the seed-noise floor)
      t, df, crit, p   Welch's test with the exact critical value at the fractional df
      bound  the smallest |gap| this rule could have called with these seeds:
             max(noise, crit * standard error)
      clear  |gap| > noise AND |t| > crit -- the only condition under which a gap counts
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    if len(set(a)) == 1 and len(set(b)) == 1:
        # no variance on either side. Tested on the values, not on se2 == 0: the mean of
        # five copies of 7.27 is not exactly 7.27, so the computed variance can be 1e-31
        # and a 'significant' t in the billions would come out of pure rounding.
        return None
    (ma, sa), (mb, sb) = mean_sd(a), mean_sd(b)
    se2 = sa * sa / na + sb * sb / nb
    if se2 == 0:
        return None
    se = math.sqrt(se2)
    t = (ma - mb) / se
    df = se2 ** 2 / ((sa * sa / na) ** 2 / (na - 1) + (sb * sb / nb) ** 2 / (nb - 1))
    crit = t_crit(df, alpha)
    noise = max(sa, sb)
    gap = ma - mb
    return {"gap": gap, "noise": noise, "t": t, "df": df, "crit": crit,
            "p": t_two_sided_p(t, df), "se": se, "bound": max(noise, crit * se),
            "clear": abs(gap) > noise and abs(t) > crit}
