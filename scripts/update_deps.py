#!/usr/bin/env python3
"""Check upstream pins and, if any moved, apply the smallest possible update.

What this touches
-----------------
- ``flake.lock`` — ``nix flake update`` (nixpkgs, flake-utils, type-nn).
- ``src/torch_type_nn/native/c`` — only when the type-nn rev moved, and only
  the vendored snapshot (``type_nn*``, ``common.h``). ``tnn_py.c`` / ``tnn_py.h``
  are this repo's ABI and are never overwritten. The Python port is not edited;
  the PR exists so ``nix flake check`` can run against the new snapshot.
- ``pyproject.toml`` — patch bump (``0.1.0.dev0`` → ``0.1.1.dev0``). The flake
  reads the version from there.
- ``README.md`` — the status line, if it quotes the old version.
- ``VERSIONS.md`` — one table of this package and every pin.

A pin that did not move is left byte-for-byte alone. Dataset URLs in
``benchmarks/datasets.json`` are content-addressed; they are listed, not
rewritten (a new hash is not a drop-in).

    nix run .#update-deps                 # write the tree, no commit
    nix run .#update-deps -- --pr         # also push deps/update and open a PR
    nix run .#update-deps -- --dry-run    # report only; do not call flake update

``--pr`` needs ``git``, ``gh`` and ``GH_TOKEN``. The daily workflow is
``.github/workflows/update-deps.yml``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Vendored from hadilq/type-nn. tnn_py.c / tnn_py.h stay; they are the ABI.
VENDORED = (
    "common.h",
    "type_nn.c",
    "type_nn.h",
    "type_nn_overfit.c",
    "type_nn_overfit.h",
    "type_nn_overfit_scale.c",
    "type_nn_overfit_scale.h",
    "type_nn_scale.c",
    "type_nn_scale.h",
)
ORIGIN_NAME = "ORIGIN"
BRANCH = "deps/update"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(\.dev\d+)?$")

# Resolved from the locked nixpkgs. Unfree torch-bin is best-effort.
RESOLVED = (
    ("python", "python312.version", "interpreter (flake uses python312)"),
    ("torch", "python312Packages.torch.version", "declared torch>=2.2"),
    ("numpy", "python312Packages.numpy.version", "dev extra"),
    ("pytest", "python312Packages.pytest.version", "declared pytest>=8"),
    ("hatchling", "python312Packages.hatchling.version", "build-system >=1.24"),
    ("ruff", "ruff.version", "lint, nix flake check"),
    ("torch-bin", "python312Packages.torch-bin.version", "CUDA shell / checks (unfree)"),
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        if (parent / "flake.nix").is_file() and (parent / "pyproject.toml").is_file():
            return parent
    sys.exit("update-deps: run from a torch-type-nn checkout (no flake.nix found)")


def run(cmd: list[str], *, cwd: Path, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True, env=merged)


def bump_patch(version: str) -> str:
    """0.1.0 → 0.1.1; 0.1.0.dev0 → 0.1.1.dev0. Dev stays a dev release."""
    match = VERSION_RE.fullmatch(version.strip())
    if not match:
        sys.exit(f"update-deps: cannot bump patch of {version!r}")
    major, minor, patch, dev = match.groups()
    bumped = f"{major}.{minor}.{int(patch) + 1}"
    if dev:
        bumped += ".dev0"
    return bumped


def read_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        sys.exit("update-deps: project.version not found in pyproject.toml")
    return match.group(1)


def write_version(root: Path, old: str, new: str) -> None:
    path = root / "pyproject.toml"
    text = path.read_text()
    replaced, n = re.subn(
        r'(?m)^(version\s*=\s*")' + re.escape(old) + r'("\s*)$',
        rf"\g<1>{new}\2",
        text,
        count=1,
    )
    if n != 1:
        sys.exit("update-deps: refused to rewrite pyproject.toml (version line not unique)")
    path.write_text(replaced)
    readme = root / "README.md"
    if readme.is_file():
        body = readme.read_text()
        body = body.replace(f"alpha ({old})", f"alpha ({new})", 1)
        readme.write_text(body)


def lock_nodes(root: Path) -> dict:
    lock = json.loads((root / "flake.lock").read_text())
    return lock["nodes"]


def node_rev(nodes: dict, name: str) -> str | None:
    locked = nodes.get(name, {}).get("locked") or {}
    return locked.get("rev")


def node_when(nodes: dict, name: str) -> str:
    locked = nodes.get(name, {}).get("locked") or {}
    stamp = locked.get("lastModified")
    if not stamp:
        return ""
    return datetime.fromtimestamp(int(stamp), timezone.utc).strftime("%Y-%m-%d")


def short(rev: str | None) -> str:
    if not rev:
        return "—"
    return rev[:12]


def github_tip(owner: str, repo: str, ref: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}"
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "torch-type-nn-update-deps",
        **({"Authorization": f"Bearer {os.environ['GH_TOKEN']}"} if os.environ.get("GH_TOKEN") else {}),
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # network is best-effort; the lock update is the source of truth
        print(f"update-deps: tip of {owner}/{repo}@{ref} unavailable ({exc})", file=sys.stderr)
        return None
    return payload.get("sha")


def nix_version(root: Path, nixpkgs_rev: str, attr: str) -> str | None:
    cmd = [
        "nix", "eval", "--raw", "--accept-flake-config",
        f"github:NixOS/nixpkgs/{nixpkgs_rev}#{attr}",
    ]
    env = {"NIXPKGS_ALLOW_UNFREE": "1"}
    proc = run(cmd, cwd=root, check=False, env=env)
    if proc.returncode != 0:
        proc = run(cmd + ["--impure"], cwd=root, check=False, env=env)
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def fetch_type_nn(root: Path, rev: str) -> Path | None:
    """Store path of the type-nn tree at rev (flake metadata downloads it)."""
    proc = run(
        ["nix", "flake", "metadata", "--json", f"github:hadilq/type-nn/{rev}"],
        cwd=root, check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return None
    path = json.loads(proc.stdout).get("path")
    return Path(path) if path else None


def vendor(root: Path, source: Path, rev: str) -> list[str]:
    dest = root / "src" / "torch_type_nn" / "native" / "c"
    changed: list[str] = []
    for name in VENDORED:
        src = source / name
        if not src.is_file():
            sys.exit(f"update-deps: {name} missing from type-nn@{rev}")
        target = dest / name
        if target.read_bytes() != src.read_bytes():
            shutil.copyfile(src, target)
            changed.append(name)
    origin = (
        "Vendored snapshot of https://github.com/hadilq/type-nn at commit\n"
        f"{rev}, plus tnn_py.c/h (the stable\n"
        "Python ABI). Lives at src/torch_type_nn/native/c so the wheel ships the\n"
        "sources as package data once — do not also force-include them from csrc/.\n"
    )
    origin_path = dest / ORIGIN_NAME
    if origin_path.read_text() != origin:
        origin_path.write_text(origin)
        changed.append(ORIGIN_NAME)
    return changed


def dataset_pin(url: str) -> str:
    if "scikit-learn/" in url and "/sklearn/" in url:
        return url.split("scikit-learn/")[-1].split("/", 1)[0]
    if "/pmlb/" in url:
        return url.split("/pmlb/", 1)[1].split("/", 1)[0][:12]
    return url.split("/")[2]


def dataset_rows(root: Path) -> list[tuple[str, str, str]]:
    manifest = json.loads((root / "benchmarks" / "datasets.json").read_text())
    return [(item["file"], dataset_pin(item["url"]), item["hash"]) for item in manifest["datasets"]]


def render_versions(
    root: Path,
    version: str,
    nodes: dict,
    tips: dict[str, str | None],
    resolved: dict[str, str | None],
) -> str:
    def tip_cell(name: str, rev: str | None) -> str:
        upstream = tips.get(name)
        if not upstream:
            return "—"
        if rev and upstream == rev:
            return f"`{short(upstream)}` (matches pin)"
        return f"`{short(upstream)}`"

    inputs = (
        ("nixpkgs", "flake input", "nixos-unstable", "NixOS/nixpkgs"),
        ("flake-utils", "flake input", "github:numtide/flake-utils", "numtide/flake-utils"),
        ("type-nn-c", "flake input, vendored C", "github:hadilq/type-nn", "hadilq/type-nn"),
        ("systems", "transitive (flake-utils)", "github:nix-systems/default", "nix-systems/default"),
    )
    lines = [
        "# Versions",
        "",
        "This package and everything it is pinned against, in one table.",
        "Regenerated by `nix run .#update-deps`. The daily",
        "[update-deps](.github/workflows/update-deps.yml) cron opens a PR when a",
        "pin moves: it updates `flake.lock`, copies the type-nn C snapshot if that",
        "rev moved (`tnn_py.c` / `tnn_py.h` are not part of the snapshot), bumps",
        "the patch version in `pyproject.toml`, and leaves the Python port alone",
        "so the check workflow can run against the new pins.",
        "",
        "The package version has one source: `project.version` in `pyproject.toml`.",
        "The flake reads it. Dataset rows are content-addressed and are not",
        "rewritten by the updater.",
        "",
        "| Component | Kind | Pin | Locked | Upstream |",
        "| --- | --- | --- | --- | --- |",
        f"| torch-type-nn | package | `pyproject.toml` | `{version}` | — |",
    ]
    for name, kind, pin, _slug in inputs:
        rev = node_rev(nodes, name)
        when = node_when(nodes, name)
        locked = f"`{short(rev)}`" + (f" ({when})" if when else "")
        lines.append(f"| {name} | {kind} | `{pin}` | {locked} | {tip_cell(name, rev)} |")
    lines += [
        "",
        "Resolved from the locked nixpkgs (the versions `nix develop` and",
        "`nix flake check` actually build). A missing cell means the eval did",
        "not run (no Nix, or an unfree package refused).",
        "",
        "| Component | Kind | Declared | Resolved |",
        "| --- | --- | --- | --- |",
    ]
    for name, _attr, declared in RESOLVED:
        got = resolved.get(name) or "—"
        lines.append(f"| {name} | nixpkgs | {declared} | `{got}` |")
    lines += [
        "",
        "Benchmark datasets (`benchmarks/datasets.json`). Listed, not bumped:",
        "a new upstream file is a different hash and not a drop-in.",
        "",
        "| File | Upstream pin in the URL | SRI hash |",
        "| --- | --- | --- |",
    ]
    for filename, pinned, digest in dataset_rows(root):
        lines.append(f"| `{filename}` | `{pinned}` | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def git(cmd: list[str], root: Path, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *cmd], cwd=root, check=check)


def ensure_identity(root: Path) -> None:
    email = git(["config", "user.email"], root, check=False)
    if email.stdout.strip():
        return
    git(["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], root)
    git(["config", "user.name", "github-actions[bot]"], root)


def open_pr(root: Path, title: str, body: str) -> None:
    ensure_identity(root)
    default = "main"
    view = run(["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"],
               cwd=root, check=False)
    if view.returncode == 0 and view.stdout.strip():
        default = view.stdout.strip()
    # Changes are already in the worktree. Reset the branch onto the default
    # tip and keep those edits (CI checks out the default branch first).
    git(["fetch", "origin", default], root, check=False)
    git(["checkout", "-B", BRANCH, f"origin/{default}"], root)
    git(["add", "--",
         "flake.lock", "pyproject.toml", "README.md", "VERSIONS.md",
         "src/torch_type_nn/native/c"], root)
    staged = git(["diff", "--cached", "--quiet"], root, check=False)
    if staged.returncode == 0:
        print("update-deps: nothing staged; not opening a PR")
        return
    git(["commit", "-m", title], root)
    git(["push", "--force-with-lease", "-u", "origin", BRANCH], root)
    existing = run(
        ["gh", "pr", "list", "--head", BRANCH, "--state", "open", "--json", "number", "--jq", ".[0].number"],
        cwd=root, check=False,
    )
    number = existing.stdout.strip()
    if number:
        run(["gh", "pr", "edit", number, "--title", title, "--body", body], cwd=root)
        print(f"update-deps: updated PR #{number}")
    else:
        created = run(
            ["gh", "pr", "create", "--base", default, "--head", BRANCH,
             "--title", title, "--body", body],
            cwd=root,
        )
        print(created.stdout.strip())


def write_output(updated: bool, version: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"updated={'true' if updated else 'false'}\n")
        handle.write(f"version={version}\n")
        handle.write(f"branch={BRANCH}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", action="store_true", help="commit, push deps/update, open or update the PR")
    parser.add_argument("--dry-run", action="store_true", help="report upstream tips; do not update the lock")
    parser.add_argument("--self-test", action="store_true", help="check the patch bumper and exit")
    parser.add_argument("--github-output", action="store_true", help="append updated/version/branch to $GITHUB_OUTPUT")
    args = parser.parse_args()

    if args.self_test:
        assert bump_patch("0.1.0") == "0.1.1"
        assert bump_patch("0.1.0.dev0") == "0.1.1.dev0"
        assert bump_patch("1.2.9.dev4") == "1.2.10.dev0"
        print("update-deps: self-test ok")
        return

    root = repo_root()
    before = lock_nodes(root)
    old_version = read_version(root)
    tips = {
        "nixpkgs": github_tip("NixOS", "nixpkgs", "nixos-unstable"),
        "flake-utils": github_tip("numtide", "flake-utils", "main"),
        "type-nn-c": github_tip("hadilq", "type-nn", "main"),
        "systems": github_tip("nix-systems", "default", "main"),
    }
    moved = [name for name, tip in tips.items() if tip and tip != node_rev(before, name)]
    if args.dry_run:
        print(f"update-deps: {old_version}; upstream ahead: {', '.join(moved) or 'nothing'}")
        for name, tip in tips.items():
            print(f"  {name}: locked {short(node_rev(before, name))} upstream {short(tip)}")
        return

    print("update-deps: nix flake update")
    proc = run(["nix", "flake", "update"], cwd=root, check=False)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    after = lock_nodes(root)
    pin_changes = [
        name for name in sorted(set(before) | set(after))
        if node_rev(before, name) != node_rev(after, name)
    ]
    vendored: list[str] = []
    type_nn_rev = node_rev(after, "type-nn-c")
    if type_nn_rev and type_nn_rev != node_rev(before, "type-nn-c"):
        source = fetch_type_nn(root, type_nn_rev)
        if source is None:
            sys.exit("update-deps: could not fetch the new type-nn tree")
        vendored = vendor(root, source, type_nn_rev)
        print(f"update-deps: vendored {vendored or 'no file bytes changed'} from {short(type_nn_rev)}")

    new_version = old_version
    if pin_changes:
        new_version = bump_patch(old_version)
        write_version(root, old_version, new_version)
        print(f"update-deps: {old_version} → {new_version} ({', '.join(pin_changes)})")
    else:
        print("update-deps: pins unchanged")

    nixpkgs_rev = node_rev(after, "nixpkgs") or ""
    resolved = {
        name: nix_version(root, nixpkgs_rev, attr) if nixpkgs_rev else None
        for name, attr, _declared in RESOLVED
    }
    versions = render_versions(root, new_version, after, tips, resolved)
    (root / "VERSIONS.md").write_text(versions)

    if not pin_changes:
        # A resolved-version fill-in may still dirty VERSIONS.md. That is a
        # docs refresh, not a release: leave the patch number alone.
        print("update-deps: wrote VERSIONS.md")
        dirty = git(["status", "--porcelain", "--", "VERSIONS.md"], root).stdout.strip()
        if args.github_output:
            write_output(bool(dirty and args.pr), new_version)
        if args.pr and dirty:
            print("update-deps: VERSIONS.md changed but no pin moved; PR without a version bump")
            open_pr(root, "docs: refresh VERSIONS.md",
                    "No upstream pin moved. Regenerated the versions table "
                    "(resolved nixpkgs versions and upstream tips).\n")
        return

    title = f"deps: {old_version} → {new_version}"
    body_lines = [
        f"Automated dependency bump ({old_version} → {new_version}).",
        "",
        "Pins that moved:",
        "",
    ]
    for name in pin_changes:
        body_lines.append(
            f"- `{name}`: `{short(node_rev(before, name))}` → `{short(node_rev(after, name))}`"
        )
    if vendored:
        body_lines += [
            "",
            "Vendored C snapshot (Python sources not edited; `tnn_py.c` / `tnn_py.h` kept):",
            "",
            *(f"- `{name}`" for name in vendored),
        ]
    body_lines += [
        "",
        "The check workflow should run on this PR. A type-nn rev change is a new",
        "C snapshot only — the cross-check is the gate, not a silent port.",
        "",
        "Versions: [VERSIONS.md](VERSIONS.md).",
    ]
    body = "\n".join(body_lines) + "\n"
    print(body)
    if args.github_output:
        write_output(True, new_version)
    if args.pr:
        open_pr(root, title, body)


if __name__ == "__main__":
    main()
