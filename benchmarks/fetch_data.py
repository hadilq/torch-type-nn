"""Download the pinned benchmark datasets without Nix.

Reads ``benchmarks/datasets.json`` (the manifest ``flake.nix`` also reads),
downloads each file, checks its sha256 against the pinned SRI hash, and writes
it to ``benchmarks/data`` (or ``--dir``). A file already present with the right
hash is kept; a wrong hash is an error and nothing is written for that file.

    python benchmarks/fetch_data.py [--dir DIR] [--verify-only]

Standard library only. With Nix, ``nix develop`` does the same from the store.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "datasets.json"


def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())["datasets"]


def sri(data: bytes) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()


def fetch(entry: dict, out: Path, verify_only: bool) -> bool:
    path = out / entry["file"]
    if path.exists() and sri(path.read_bytes()) == entry["hash"]:
        print(f"ok       {entry['file']}")
        return True
    if verify_only:
        print(f"MISSING  {entry['file']}" if not path.exists() else f"BAD HASH {entry['file']}")
        return False
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "torch-type-nn"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    got = sri(data)
    if got != entry["hash"]:
        print(f"BAD HASH {entry['file']}: expected {entry['hash']}, got {got}", file=sys.stderr)
        return False
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    print(f"fetched  {entry['file']} ({len(data)} bytes)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=HERE / "data")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)
    ok = [fetch(e, a.dir, a.verify_only) for e in manifest()]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
