#!/usr/bin/env python3
"""Pull Phase-0 artifacts from the training machine, driven by its inventory.

Two steps, run from this Mac:

    # 1. run inventory.py ON the remote without copying it there first, and
    #    bring its JSON back
    python3 phase0/fetch.py inventory --host mtbox

    # 2. show what would be pulled (DRY RUN by default), then pull
    python3 phase0/fetch.py pull --host mtbox --prio P0,P1
    python3 phase0/fetch.py pull --host mtbox --prio P0,P1 --go

`--host` is anything `ssh` accepts: an alias from ~/.ssh/config or user@address.
`--host local` copies from this machine's own filesystem without ssh; it exists so
the pull path (file list, path mirroring, manifest) can be tested end to end.

rsync flags are restricted to what macOS's bundled openrsync (protocol 29)
accepts: -a --partial --progress --files-from -e. In particular NOT
--info=progress2, which the first version used and which openrsync rejects.

Authentication is KEY-BASED ONLY. Every ssh/rsync call uses BatchMode=yes, so a
host that would prompt for a password fails immediately instead of prompting.
That is deliberate: this script must never be the thing that handles a password.
Set up a key first:  ssh-copy-id <host>

Paths are mirrored under --dest with their full remote path, so nothing from two
different directories can collide, and provenance of every file stays obvious.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
# rotating step checkpoints are large and superseded by averaged/best
NEVER = ("step_*.pt",)


def human(n: float) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}P"


def parse_size(s: str) -> int:
    mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    s = s.strip().upper()
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def check_ssh(host: str) -> None:
    r = subprocess.run(["ssh", *SSH_OPTS, host, "echo ok"], capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip() != "ok":
        sys.exit(f"cannot reach {host} with key auth:\n  {r.stderr.strip()}\n"
                 "Set up a key (ssh-copy-id) — this script will not use a password.")


def cmd_inventory(a) -> int:
    check_ssh(a.host)
    remote_json = "/tmp/mt_inventory.json"
    extra = " ".join(a.roots)
    flags = f"--max-depth {a.max_depth} --json-out {remote_json}"
    if a.inspect_ckpt:
        flags += " --inspect-ckpt"
    print(f"running inventory.py on {a.host} (streamed over ssh, nothing copied there)…")
    with open(HERE / "inventory.py", "rb") as script:
        r = subprocess.run(["ssh", *SSH_OPTS, a.host, f"python3 - {extra} {flags}"],
                           stdin=script)
    if r.returncode != 0:
        return r.returncode
    subprocess.run(["scp", *SSH_OPTS, f"{a.host}:{remote_json}", a.out], check=True)
    print(f"\ninventory saved to {a.out}")
    return 0


def cmd_pull(a) -> int:
    inv = json.load(open(a.inventory))
    prios = set(a.prio.split(","))
    cap = parse_size(a.max_file)
    import fnmatch
    chosen, skipped = [], []
    for f in inv["files"]:
        name = os.path.basename(f["path"])
        if f["prio"] not in prios:
            continue
        if any(fnmatch.fnmatch(name, p) for p in NEVER):
            skipped.append((f, "rotating step checkpoint"))
        elif f["size"] > cap:
            skipped.append((f, f"larger than --max-file {a.max_file}"))
        else:
            chosen.append(f)

    total = sum(f["size"] for f in chosen)
    print(f"plan: {len(chosen)} files, {human(total)} -> {a.dest}")
    for f in sorted(chosen, key=lambda f: (f["prio"], f["kind"], f["path"])):
        print(f"  {f['prio']}  {human(f['size']):>8}  {f['path']}")
    if skipped:
        print(f"\nNOT pulled ({len(skipped)} files, "
              f"{human(sum(f['size'] for f, _ in skipped))}) — listed so nothing is dropped silently:")
        for f, why in skipped:
            print(f"  {f['prio']}  {human(f['size']):>8}  {f['path']}   [{why}]")

    free = os.statvfs(os.path.expanduser(a.dest) if os.path.exists(os.path.expanduser(a.dest))
                      else os.path.expanduser("~")).f_bavail * os.statvfs(os.path.expanduser("~")).f_frsize
    print(f"\nlocal free space: {human(free)}")
    if total > free * 0.8:
        print("⚠️  plan exceeds 80% of free space — narrow --prio or lower --max-file")
        return 1
    if not a.go:
        print("\nDRY RUN. Re-run with --go to pull.")
        return 0

    if a.host != "local":
        check_ssh(a.host)
    dest = os.path.expanduser(a.dest)
    os.makedirs(dest, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as lst:
        for f in chosen:
            lst.write(f["path"].lstrip("/") + "\n")
    if a.host == "local":
        rsync = ["rsync", "-a", "--partial", "--progress", "--files-from", lst.name, "/", dest]
    else:
        rsync = ["rsync", "-a", "--partial", "--progress", "--files-from", lst.name,
                 "-e", "ssh " + " ".join(SSH_OPTS), f"{a.host}:/", dest]
    print("\n" + " ".join(rsync))
    r = subprocess.run(rsync)
    os.unlink(lst.name)
    if r.returncode == 0:
        with open(os.path.join(dest, "FETCH_MANIFEST.json"), "w") as m:
            json.dump({"host": a.host, "inventory": os.path.abspath(a.inventory),
                       "files": chosen, "not_pulled": [f for f, _ in skipped]}, m, indent=1)
        print(f"done; manifest at {dest}/FETCH_MANIFEST.json")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--host", required=True)
    i.add_argument("--roots", nargs="*", default=["~"])
    i.add_argument("--max-depth", type=int, default=7)
    i.add_argument("--inspect-ckpt", action="store_true")
    i.add_argument("--out", default="inventory.json")
    p = sub.add_parser("pull")
    p.add_argument("--host", required=True)
    p.add_argument("--inventory", default="inventory.json")
    p.add_argument("--prio", default="P0")
    p.add_argument("--max-file", default="20G")
    p.add_argument("--dest", default="~/mt_fetch")
    p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return cmd_inventory(a) if a.cmd == "inventory" else cmd_pull(a)


if __name__ == "__main__":
    sys.exit(main())
