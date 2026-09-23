{
  description = "torch-type-nn: type-nn (dynamic And-Or partition-function networks) for PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # the reference C implementation, for the number-for-number cross-check
    type-nn-c = {
      url = "github:hadilq/type-nn";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, type-nn-c }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        py = python.pkgs;

        # single source of truth for the version: pyproject.toml
        version = (pkgs.lib.importTOML ./pyproject.toml).project.version;

        # The library itself. `nix build` produces it, `nix flake check`
        # builds it and runs the whole pytest suite against it.
        torch-type-nn = py.buildPythonPackage {
          pname = "torch-type-nn";
          inherit version;
          pyproject = true;
          src = pkgs.lib.cleanSource ./.;

          build-system = [ py.hatchling ];
          dependencies = [ py.torch ];

          # gcc + TYPE_NN_SRC enable tests/test_crosscheck_c.py, which compiles
          # the reference C code and compares forward, backward and Adam
          nativeCheckInputs = [ py.pytestCheckHook py.numpy pkgs.stdenv.cc ];
          preCheck = ''
            export TYPE_NN_SRC=${type-nn-c}
          '';
          pythonImportsCheck = [ "torch_type_nn" ];

          meta = {
            description = "type-nn: networks of partition functions that grow and prune their own structure";
            homepage = "https://github.com/hadilq/type-nn";
            license = pkgs.lib.licenses.mit;
          };
        };

        # Interpreter with the library and every developer tool.
        devPython = python.withPackages (ps: [
          ps.torch
          ps.numpy
          ps.pytest
          ps.hatchling
          ps.build
          ps.twine
        ]);

        # `nix run .#<name>` helpers. Each one runs from the checkout.
        script = name: text: {
          type = "app";
          program = "${pkgs.writeShellScript name ''
            set -euo pipefail
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            ${text}
          ''}";
        };
      in
      {
        packages = {
          default = torch-type-nn;
          torch-type-nn = torch-type-nn;
        };

        checks = {
          # builds the wheel and runs pytest (pytestCheckHook)
          package = torch-type-nn;
          lint = pkgs.runCommand "ruff" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            cd ${./.}
            ruff check --no-cache src tests benchmarks
            touch $out
          '';
        };

        apps = {
          test = script "tnn-test" "${devPython}/bin/python -m pytest -q \"$@\"";
          # sdist + wheel in ./dist, then `twine check`
          dist = script "tnn-dist" ''
            rm -rf dist
            ${devPython}/bin/python -m build --no-isolation
            ${devPython}/bin/python -m twine check dist/*
          '';
          # upload to PyPI (or TestPyPI with: nix run .#publish -- --repository testpypi)
          publish = script "tnn-publish" "${devPython}/bin/python -m twine upload \"$@\" dist/*";
          # the board: nix run .#bench -- iris type-nn --seeds 5
          bench = script "tnn-bench" "${devPython}/bin/python benchmarks/bench.py \"$@\"";
          # markdown board vs the C reference: nix run .#board -- benchmarks/out/*.jsonl
          board = script "tnn-board" "${devPython}/bin/python benchmarks/board.py \"$@\"";
        };

        devShells.default = pkgs.mkShell {
          packages = [ devPython pkgs.ruff pkgs.gcc pkgs.gnumake ];
          TYPE_NN_SRC = "${type-nn-c}";
          shellHook = ''
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            echo "torch-type-nn ${version} dev shell: pytest | nix run .#dist | nix flake check"
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
