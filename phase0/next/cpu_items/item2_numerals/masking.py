"""Item 2 (WMT2027_PLAN.md section 3.2): the masking rules for numbers, dates and UN document
symbols, applied on top of E0.4's tokenisation (lowercased \\w+, phase0/e04_split.py line 13).

The token sequence is never changed: every \\w+ token of the source keeps its position, and each
one is given a class. A token is

  SYM   if its character span overlaps a UN document symbol (SYMBOL below), else
  DATE  if its span overlaps a date expression (DATE below), else
  NUM   if the lowercased token is a numeral: NUMTOK below (digits, optionally followed by an
        ordinal suffix or a plural s: 1997, 22, 1st, 58th, 1990s), else
  WORD.

Every SYMBOL and DATE pattern needs at least one digit, so a line without a Unicode digit has
no masked token at all (the fast path in classify()).
"""
from __future__ import annotations

import re

WORD = re.compile(r"\w+", re.UNICODE)      # identical to phase0/e04_split.py and e05_build.py
DIGIT = re.compile(r"\d")                  # any Unicode decimal digit, as e06_build.py's A17 rule

# UN document symbols: an upper-case series code, then one or more "/"-separated components
# (A/RES/68/1, A/C.3/58/L.12/Add.1, E/CN.4/2004/L.10, S/2005/86, CEDAW/C/USA/CO/5,
# CCPR/C/79/Add.1, ST/SGB/2003/13, UNEP/GC.22/INF/1, ICCD/COP(6)/3, A/58/12-S/2003/1234).
# The whole match must contain a digit, which keeps out HIV/AIDS, UNDP/UNFPA, and/or.
SYMBOL = re.compile(
    r"(?<![\w/])[A-Z][A-Za-z]{0,11}(?:\.\d+)?(?:/[A-Za-z0-9][A-Za-z0-9.()\-]*)+")

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|"
          r"December)")
_WDAY = r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_JOIN = r"\s*(?:-|–|—|to|and|or)\s*"
_YEAR = r"\d{4}"
# Date expressions, in this order of alternatives:
#   26 January 2004 | 26 and 27 January | 4-6 May 2004 | 1st of July      (day [range] month [year])
#   January 26, 2004 | January 26-28                                       (month day [range] [year])
#   January 2004 | January-March 2004                                     (month [range] year)
#   26/01/2004 | 26.01.04 | 2004-01-26                                     (all-numeric dates)
# Month names are case-sensitive (capitalised), so the verb "may" is never a month.
DATE = re.compile(
    rf"(?<!\w){_WDAY}{_DAY}(?:{_JOIN}{_DAY})?\s+(?:of\s+)?{_MONTH}(?:\s*,?\s*{_YEAR})?(?!\w)"
    rf"|(?<!\w){_WDAY}{_MONTH}\s+{_DAY}(?:{_JOIN}{_DAY})?(?:\s*,\s*{_YEAR}|\s+{_YEAR})?(?!\w)"
    rf"|(?<!\w){_MONTH}(?:{_JOIN}{_MONTH})?\s+{_YEAR}(?!\w)"
    r"|(?<![\w./-])\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})(?![\w/-])"
    r"|(?<![\w./-])\d{4}-\d{1,2}-\d{1,2}(?![\w/-])")

NUMTOK = re.compile(r"\d+(?:st|nd|rd|th|s)?")   # fullmatch on the lowercased token

W, NUM, DATE_C, SYM = 0, 1, 2, 3
CLASS_NAME = {W: "word", NUM: "num", DATE_C: "date", SYM: "sym"}
PLACEHOLDER = {NUM: "#num", DATE_C: "#date", SYM: "#sym"}  # "#" is not \w: no real token collides


def symbol_spans(s: str) -> list[tuple[int, int]]:
    return [m.span() for m in SYMBOL.finditer(s) if DIGIT.search(m.group())]


def date_spans(s: str) -> list[tuple[int, int]]:
    return [m.span() for m in DATE.finditer(s)]


def _overlaps(a: int, b: int, spans: list[tuple[int, int]]) -> bool:
    for x, y in spans:
        if a < y and x < b:
            return True
    return False


def classify(s: str) -> tuple[list[str], list[int], bool]:
    """(tokens, classes, any_digit_in_line). tokens == [w.lower() for w in WORD.findall(s)]."""
    if not DIGIT.search(s):
        t = [w.lower() for w in WORD.findall(s)]
        return t, [W] * len(t), False
    ss, ds = symbol_spans(s), date_spans(s)
    toks, cls = [], []
    for m in WORD.finditer(s):
        a, b = m.span()
        w = m.group().lower()
        if ss and _overlaps(a, b, ss):
            c = SYM
        elif ds and _overlaps(a, b, ds):
            c = DATE_C
        elif NUMTOK.fullmatch(w):
            c = NUM
        else:
            c = W
        toks.append(w)
        cls.append(c)
    return toks, cls, True


def render(s: str) -> str:
    """The source with every masked token shown as [class:token], for the example rows."""
    toks, cls, _ = classify(s)
    return " ".join(w if c == W else f"[{CLASS_NAME[c]}:{w}]" for w, c in zip(toks, cls))
