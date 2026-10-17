"""Word frequencies (lowercased \\w+ tokens, as e04_split.py) of a text file, plus a hashed set of
normalised lines (NFKC, casefold, whitespace collapsed) for exposure lookups.
usage: freq.py <in> <out.pkl> [--lines-hash]"""
import collections, hashlib, pickle, re, sys, unicodedata
from multiprocessing import Pool
WORD = re.compile(r"\w+", re.UNICODE)
WS = re.compile(r"\s+")
def norm(s): return WS.sub(" ", unicodedata.normalize("NFKC", s).casefold()).strip()
def work(lines):
    c = collections.Counter(); hs = set()
    for s in lines:
        c.update(w.lower() for w in WORD.findall(s))
        if DO_HASH: hs.add(hashlib.blake2b(norm(s).encode(), digest_size=8).digest())
    return c, hs
def chunks(p, n=200000):
    buf = []
    with open(p, encoding="utf-8", newline="\n") as f:
        for ln in f:
            buf.append(ln.rstrip("\n"))
            if len(buf) == n: yield buf; buf = []
    if buf: yield buf
DO_HASH = "--lines-hash" in sys.argv
if __name__ == "__main__":
    tot = collections.Counter(); H = set(); n = 0
    with Pool(14) as pool:
        for c, hs in pool.imap(work, chunks(sys.argv[1])):
            tot.update(c); H |= hs; n += 1
    pickle.dump({"counts": tot, "line_hashes": H}, open(sys.argv[2], "wb"))
    print(sys.argv[1], "types", len(tot), "tokens", sum(tot.values()), "hashed lines", len(H))
