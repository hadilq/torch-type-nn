# Releasing

One command updates the version, commits, and tags. Pushing the tag
triggers the publish workflow, which builds, tests, and uploads to PyPI
via trusted publishing.

## Cut a release

From a clean checkout on the branch you want to release from:

```sh
nix run .#release -- 0.2.0
```

That:

1. Rewrites `project.version` in `pyproject.toml` (the single source of truth).
2. Updates the status line in `README.md` (`alpha (…)`).
3. Updates the package row in `VERSIONS.md`.
4. Commits those three files with message `Release 0.2.0`.
5. Creates an annotated tag `v0.2.0`.

Then push (the command prints these if you did not pass `--push`):

```sh
git push origin HEAD
git push origin v0.2.0
```

Or do it in one step:

```sh
nix run .#release -- 0.2.0 --push
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--dry-run` | Show the plan; write nothing, commit nothing. |
| `--push` | After tagging, push the current branch and the tag. |
| `--message TEXT` | Commit and tag message (default: `Release VERSION`). |

The version must look like `0.2.0` or `0.2.0rc1`. Development versions
(`.dev0`) are rejected — those are for the daily `update-deps` flow.

## What the tag does

`.github/workflows/publish.yml` runs on any `v*` tag. It:

1. Checks that the tag name equals `v` + `project.version` in
   `pyproject.toml`.
2. Builds the sdist and wheel.
3. Runs the test suite against the wheel.
4. Publishes to PyPI with trusted publishing (no token in the repo).

One-time setup (already done for this repo): on pypi.org, add a
pending publisher for owner `hadilq`, repository `torch-type-nn`,
workflow `publish.yml`, environment `pypi`.

## After the release

Development continues on a `.dev0` version. Bump it yourself, or let
the next `update-deps` run do a patch bump when a pin moves:

```sh
# optional: open the next development version by hand
# edit pyproject.toml to 0.2.1.dev0, then commit
```

## Checklist

- [ ] `nix flake check` is green on the branch
- [ ] Working tree is clean
- [ ] `nix run .#release -- 0.2.0` (or with `--dry-run` first)
- [ ] `git push origin HEAD && git push origin v0.2.0`
- [ ] Watch the `publish` workflow
- [ ] Confirm the release on PyPI and the GitHub releases page
