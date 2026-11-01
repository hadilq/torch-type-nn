#!/usr/bin/env python3
"""Cut a release: bump the version, commit, and tag.

Usage
-----
    nix run .#release -- 0.2.0
    nix run .#release -- 0.2.0 --dry-run
    nix run .#release -- 0.2.0 --push

What it does
------------
1. Validates the version (PEP 440 / simple semver, no ``.dev``).
2. Rewrites the single source of truth and the places that quote it:
   - ``pyproject.toml`` (``project.version``)
   - ``README.md`` status line (``alpha (X)``)
   - ``VERSIONS.md`` package row
3. Commits those files with message ``Release 0.2.0``.
4. Creates an annotated tag ``v0.2.0``.
5. Prints the push commands (or pushes with ``--push``).

The tag matches what ``.github/workflows/publish.yml`` expects: a ``v*``
tag whose name equals ``v`` + the version in ``pyproject.toml``. That
workflow builds, tests, and publishes to PyPI via trusted publishing.

Requires a clean git working tree and ``git`` on ``PATH``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# 0.2.0, 1.0.0, 0.2.0rc1, 0.2.0a1 — no .dev (that is a development version).
VERSION_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?P<pre>(?:a|b|rc)\d+)?$"
)


def die(msg: str) -> None:
    print(f"release: {msg}", file=sys.stderr)
    sys.exit(1)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        cwd=cwd,
    )


def find_root() -> Path:
    here = Path.cwd()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "flake.nix").is_file():
            return parent
    die("run from the torch-type-nn checkout (pyproject.toml + flake.nix not found)")
    raise AssertionError  # unreachable


def read_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        die("project.version not found in pyproject.toml")
    return match.group(1)


def write_version_files(root: Path, old: str, new: str) -> None:
    # pyproject.toml
    path = root / "pyproject.toml"
    text = path.read_text()
    replaced, n = re.subn(
        r'(?m)^(version\s*=\s*")' + re.escape(old) + r'("\s*)$',
        rf"\g<1>{new}\2",
        text,
        count=1,
    )
    if n != 1:
        die("refused to rewrite pyproject.toml (version line not unique)")
    path.write_text(replaced)

    # README.md status line: "alpha (0.1.1.dev0)"
    readme = root / "README.md"
    if readme.is_file():
        body = readme.read_text()
        body = body.replace(f"alpha ({old})", f"alpha ({new})", 1)
        readme.write_text(body)

    # VERSIONS.md package row
    versions = root / "VERSIONS.md"
    if versions.is_file():
        body = versions.read_text()
        body = body.replace(f"| `{old}` |", f"| `{new}` |", 1)
        versions.write_text(body)


def ensure_clean(root: Path) -> None:
    proc = run(["git", "status", "--porcelain"], capture=True, cwd=root)
    if proc.stdout.strip():
        die("working tree is not clean; commit or stash first\n" + proc.stdout)


def is_git_repo(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def tag_exists(root: Path, tag: str) -> bool:
    proc = run(["git", "tag", "-l", tag], capture=True, cwd=root)
    return tag in proc.stdout.split()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a release: update version files, commit, and tag.",
    )
    parser.add_argument("version", help="Release version, e.g. 0.2.0 (no .dev)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change; do not write or commit",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="After tagging, push the current branch and the tag",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Commit / tag message (default: 'Release VERSION')",
    )
    args = parser.parse_args()

    version = args.version.strip()
    if not VERSION_RE.fullmatch(version):
        die(
            f"invalid version {version!r}. "
            "Expected something like 0.2.0 or 0.2.0rc1 (no .dev)."
        )

    root = find_root()
    old = read_version(root)
    tag = f"v{version}"
    message = args.message or f"Release {version}"

    print(f"release: {old} → {version}  (tag {tag})")

    if not is_git_repo(root):
        die("not a git checkout (git rev-parse failed)")

    if tag_exists(root, tag):
        die(f"tag {tag} already exists")

    if args.dry_run:
        print("release: dry-run, not writing files or committing")
        print("  would update pyproject.toml, README.md, VERSIONS.md")
        print(f"  would commit: {message}")
        print(f"  would tag:    {tag}")
        return

    ensure_clean(root)
    write_version_files(root, old, new=version)

    # Stage only the version files we touched.
    files = ["pyproject.toml", "README.md", "VERSIONS.md"]
    run(["git", "add", "--"] + files, cwd=root)
    run(["git", "commit", "-m", message], cwd=root)
    run(["git", "tag", "-a", tag, "-m", message], cwd=root)

    print()
    print(f"release: committed and tagged {tag}")
    print()
    if args.push:
        branch = run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True, cwd=root
        ).stdout.strip()
        run(["git", "push", "origin", branch], cwd=root)
        run(["git", "push", "origin", tag], cwd=root)
        print(f"release: pushed {branch} and {tag}")
        print("release: publish.yml will build, test, and upload to PyPI")
    else:
        print("Next (when you are ready):")
        print(f"  git push origin HEAD")
        print(f"  git push origin {tag}")
        print()
        print("The tag triggers .github/workflows/publish.yml")
        print("(trusted publishing to PyPI). See RELEASE.md.")


if __name__ == "__main__":
    main()
