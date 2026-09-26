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
    let
      # --- GPU gate -----------------------------------------------------------
      # A pure evaluation (plain `nix flake check`) cannot look at the host, so
      # it never adds the CUDA check. `nix flake check --impure` adds it when
      # the GPU *and* CUDA are usable by Nix builds: a GPU is visible
      # (/dev/nvidiactl) and the daemon's configuration enables the `cuda`
      # system feature (the check requires it; on NixOS
      # programs.nix-required-mounts.presets.nvidia-gpu.enable sets it). A GPU
      # without the feature skips the check with a warning instead of failing
      # it; `nix run .#test-cuda` runs the same suite on the host meanwhile.
      # TORCH_TYPE_NN_CUDA=1 / =0 forces the check on / off.
      lib = nixpkgs.lib;
      impure = builtins ? currentSystem;
      cudaFlag = if impure then builtins.getEnv "TORCH_TYPE_NN_CUDA" else "";
      gpuVisible = impure && builtins.pathExists "/dev/nvidiactl";
      readIf = p: if builtins.pathExists p then builtins.readFile p else "";
      # the daemon's settings (NIX_CONFIG only reaches single-user builds)
      nixConf = lib.concatStringsSep "\n" [
        (readIf "/etc/nix/nix.conf")
        (readIf "/etc/nix/nix.custom.conf")   # Determinate Nix
        (if impure then builtins.getEnv "NIX_CONFIG" else "")
      ];
      enablesCuda = line: builtins.match
        "[[:space:]]*(extra-)?system-features[[:space:]]*=(.*[[:space:]])?cuda([[:space:]].*)?"
        line != null;
      cudaFeature = impure && builtins.any enablesCuda (lib.splitString "\n" nixConf);
      wantCuda =
        if cudaFlag == "1" then true
        else if cudaFlag == "0" then false
        else lib.warnIf (gpuVisible && !cudaFeature)
          ("torch-type-nn: a GPU is visible but the Nix daemon does not enable the "
            + "`cuda` system feature, so checks.cuda is skipped. Run the GPU suite on the "
            + "host with `nix run .#test-cuda`, or enable the feature (README, \"On a GPU\").")
          (gpuVisible && cudaFeature);
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        py = python.pkgs;

        # single source of truth for the version: pyproject.toml
        version = (lib.importTOML ./pyproject.toml).project.version;

        # Pinned benchmark datasets, as in the type-nn flake. The manifest
        # (url + SRI hash per file) is shared with benchmarks/fetch_data.py.
        # They are never committed: .gitignore excludes benchmarks/data/*.
        manifest = (builtins.fromJSON (builtins.readFile ./benchmarks/datasets.json)).datasets;
        datasets = pkgs.runCommand "torch-type-nn-datasets" { } (''
          mkdir -p $out/share/torch-type-nn
        '' + lib.concatMapStrings (d: ''
          cp ${pkgs.fetchurl { inherit (d) url hash; name = d.file; }} $out/share/torch-type-nn/${d.file}
        '') manifest);
        dataDir = "${datasets}/share/torch-type-nn";
        dataNames = lib.concatMapStringsSep " " (d: d.file) manifest;

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
          # TORCH_TYPE_NN_DATA + TNN_REQUIRE_DATA: tests/test_datasets.py checks
          # every pinned file (hash, shape, Friedman #1 by its formula) and the
          # benchmark loops run on real data; a missing file is a failure.
          nativeCheckInputs = [ py.pytestCheckHook py.numpy pkgs.stdenv.cc ];
          preCheck = ''
            export TYPE_NN_SRC=${type-nn-c}
            export TORCH_TYPE_NN_DATA=${dataDir} TNN_REQUIRE_DATA=1
            # NativeTypeNN compiles src/torch_type_nn/native/c into libtnn_py.so
            # here (sandbox $HOME is not writable)
            export XDG_CACHE_HOME="$TMPDIR"
            export TNN_PY_LIB="$TMPDIR/libtnn_py.so"
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

        # `nix run .#<name>` helpers. Each one runs from the checkout, with
        # the pinned datasets unless TORCH_TYPE_NN_DATA is already set.
        script = name: description: text: {
          type = "app";
          meta.description = description;
          program = "${pkgs.writeShellScript name ''
            set -euo pipefail
            export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            export TORCH_TYPE_NN_DATA="''${TORCH_TYPE_NN_DATA:-${dataDir}}"
            ${text}
          ''}";
        };

        # copies the pinned datasets to benchmarks/data (as the type-nn shell does)
        copyData = ''
          mkdir -p benchmarks/data
          for f in ${dataNames}; do
            cp -f --no-preserve=mode ${dataDir}/$f benchmarks/data/
          done
        '';

        # --- CUDA ---------------------------------------------------------------
        # The official PyTorch wheel (torch-bin 2.13, unfree) is built for CUDA
        # 13.0 (cu130), so the CUDA set is pinned to exactly 13.0: its
        # cuda-bindings (13.0.3) satisfy torch-bin's >= 13.0.3 guard. Not
        # `cudaPackages_13`: in the locked nixpkgs that is 13.3, whose cccl
        # applies a patch (fix-invalid-cpp-syntax, for 13.2 <= cuda < 13.4) that
        # 13.3.3 already contains, so cccl -> cuda_cudart -> torch fail to build.
        cudaSet = "cudaPackages_13_0";
        # NCCL is built from source. NVIDIA re-pointed the v2.32.3-1 tag
        # (2026-09-17) to a merge commit, so the tarball nixpkgs pinned
        # (sha256-ytAJ...) is no longer what GitHub serves and the fetch fails
        # with a hash mismatch; nixpkgs master still has the old hash. Pin the
        # tag's current commit by rev (immutable), with its hash computed
        # independently. Applies only while nixpkgs is at that version with that
        # hash, so it switches itself off once nixpkgs is fixed or bumped.
        ncclStaleHash = "sha256-ytAJn8F0QEHhUadiOmVKTUiL7lsUnasoP4MOv/t60xk=";
        fixNccl = cfinal: cprev: {
          nccl =
            if cprev.nccl.version == "2.32.3-1" && cprev.nccl.src.outputHash == ncclStaleHash
            then cprev.nccl.overrideAttrs (old: {
              src = pkgs.fetchFromGitHub {
                owner = "NVIDIA";
                repo = "nccl";
                rev = "12df1a11afad322be5a204a2db890161cbf8131d"; # tag v2.32.3-1 since 2026-09-17
                hash = "sha256-xUllfdWAL0Ee9P9T9CZC2ddkPRnSXZXgwApgO398i6g=";
              };
            })
            else cprev.nccl;
        };
        cudaPkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; cudaSupport = true; };
          overlays = [
            (final: prev: {
              # nixpkgs' hook for extending *every* CUDA package set: overriding
              # one set (or the `cudaPackages` alias) leaves the stale NCCL
              # reachable through others (libnvshmem -> openmpi -> ucc).
              _cuda = prev._cuda.extend (_: prevAttrs: {
                extensions = prevAttrs.extensions ++ [ fixNccl ];
              });
              cudaPackages = final.${cudaSet};
            })
          ];
        };
        cudaPython = cudaPkgs.python312.withPackages (ps: [ ps.torch-bin ps.numpy ps.pytest ]);

        # The whole test-suite on the GPU: every `device` test runs on cuda too,
        # TNN_REQUIRE_CUDA=1 makes a missing GPU a failure (never a silent CPU
        # pass), and the board's own training loop runs once on the device.
        # One script, two runners: the sandboxed check below, and
        # `nix run .#test-cuda` on the host (no `cuda` system feature needed).
        cudaSuite = pkgs.writeShellScript "tnn-cuda-suite" ''
          set -euo pipefail          # a failing pytest must fail the run, tee or not
          out="$1"
          mkdir -p "$out"
          export PATH=${cudaPython}/bin:${pkgs.stdenv.cc}/bin:$PATH
          export TYPE_NN_SRC="''${TYPE_NN_SRC:-${type-nn-c}}"
          export TORCH_TYPE_NN_DATA="''${TORCH_TYPE_NN_DATA:-${dataDir}}"
          export TNN_REQUIRE_CUDA=1 TNN_REQUIRE_DATA=1
          python - <<'PY' | tee "$out/device.txt"
          import torch
          assert torch.cuda.is_available(), (
              "no usable CUDA device: torch-bin loads the driver's libcuda.so from "
              "/run/opengl-driver/lib (NixOS: hardware.nvidia + hardware.graphics); "
              "inside a Nix build it also needs the `cuda` system feature and its "
              "mounts (programs.nix-required-mounts.presets.nvidia-gpu.enable)")
          print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
          PY
          python -m pytest -q -p no:cacheprovider tests | tee "$out/pytest.txt"
          python benchmarks/bench.py xor all --seeds 1 --device cuda | tee "$out/bench-xor-cuda.jsonl"
        '';
        cuda-tests = pkgs.runCommand "torch-type-nn-cuda-tests-${version}"
          { requiredSystemFeatures = [ "cuda" ]; } ''
          cp -r ${lib.cleanSource ./.} tree
          chmod -R u+w tree
          cd tree
          export HOME=$TMPDIR PYTHONPATH=$PWD/src
          ${cudaSuite} $out
        '';
        onLinux = lib.elem system [ "x86_64-linux" "aarch64-linux" ];
      in
      {
        packages = {
          default = torch-type-nn;
          torch-type-nn = torch-type-nn;
          inherit datasets;
        } // lib.optionalAttrs onLinux {
          # explicit GPU run without --impure: nix build .#cuda-tests
          inherit cuda-tests;
        };

        checks = {
          # builds the wheel and runs pytest (pytestCheckHook)
          package = torch-type-nn;
          lint = pkgs.runCommand "ruff" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            cd ${./.}
            ruff check --no-cache src tests benchmarks
            touch $out
          '';
        } // lib.optionalAttrs (wantCuda && onLinux) {
          # only under `nix flake check --impure` with a visible GPU (see the gate)
          cuda = cuda-tests;
        };

        apps = {
          test = script "tnn-test" "Run the test-suite from the checkout" "${devPython}/bin/python -m pytest -q \"$@\"";
          # sdist + wheel in ./dist, then `twine check`
          dist = script "tnn-dist" "Build the sdist and wheel into ./dist and check them with twine" ''
            rm -rf dist
            ${devPython}/bin/python -m build --no-isolation
            ${devPython}/bin/python -m twine check dist/*
          '';
          # upload to PyPI (or TestPyPI with: nix run .#publish -- --repository testpypi)
          publish = script "tnn-publish" "Upload ./dist with twine" "${devPython}/bin/python -m twine upload \"$@\" dist/*";
          # the board: nix run .#bench -- iris type-nn --seeds 5
          bench = script "tnn-bench" "Run the benchmark board (benchmarks/bench.py)" "${devPython}/bin/python benchmarks/bench.py \"$@\"";
          # markdown board vs the C reference: nix run .#board -- benchmarks/out/*.jsonl
          # larger data (digits, Friedman #1): nix run .#scale -- all --seeds 5
          scale = script "tnn-scale" "Run the scale benchmark on digits and Friedman #1 (benchmarks/scale.py)" "${devPython}/bin/python benchmarks/scale.py \"$@\"";
          board = script "tnn-board" "Render BOARD tables from benchmark results (benchmarks/board.py)" "${devPython}/bin/python benchmarks/board.py \"$@\"";
          # every benchmark result and the tables of BOARD.md, on all cores
          board-all = script "tnn-board-all" "Regenerate all benchmark results and BOARD.md" ''
            jobs="''${TNN_JOBS:-0}"
            mkdir -p benchmarks/results
            ${devPython}/bin/python benchmarks/bench.py all all --seeds 5 --jobs "$jobs" -v \
              > benchmarks/results/board-per-sample.jsonl.tmp
            mv benchmarks/results/board-per-sample.jsonl.tmp benchmarks/results/board-per-sample.jsonl
            ${devPython}/bin/python benchmarks/scale.py all --seeds 5 --jobs "$jobs" \
              > benchmarks/results/scale-batch32.jsonl.tmp
            mv benchmarks/results/scale-batch32.jsonl.tmp benchmarks/results/scale-batch32.jsonl
            ${devPython}/bin/python benchmarks/board.py --write BOARD.md
            echo "wrote benchmarks/results/*.jsonl and BOARD.md"
          '';
          # copy the pinned datasets to benchmarks/data
          data = script "tnn-data" "Copy the pinned datasets to benchmarks/data" copyData;
        } // lib.optionalAttrs onLinux {
          # the GPU suite on the host, outside the sandbox (results: $TNN_CUDA_OUT)
          test-cuda = script "tnn-test-cuda" "Run the GPU suite on the host (CUDA required)" ''
            out="''${TNN_CUDA_OUT:-$(mktemp -d -t tnn-cuda.XXXXXX)}"
            ${cudaSuite} "$out"
            echo "results in $out"
          '';
          # the board on the GPU: nix run .#bench-cuda -- all all --seeds 5 --device cuda
          bench-cuda = script "tnn-bench-cuda" "Run the benchmark board with CUDA-enabled torch" "${cudaPython}/bin/python benchmarks/bench.py \"$@\"";
          scale-cuda = script "tnn-scale-cuda" "Run the scale benchmark with CUDA-enabled torch" "${cudaPython}/bin/python benchmarks/scale.py \"$@\"";
        };

        devShells = {
          default = pkgs.mkShell {
            packages = [ devPython pkgs.ruff pkgs.gcc pkgs.gnumake ];
            TYPE_NN_SRC = "${type-nn-c}";
            TORCH_TYPE_NN_DATA = dataDir;
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
              ${copyData}
              echo "TORCH_TYPE_NN_DATA=$TORCH_TYPE_NN_DATA (copied to benchmarks/data)"
              echo "datasets: ${dataNames}"
              echo "torch-type-nn ${version} dev shell: pytest | nix run .#dist | nix flake check"
            '';
          };
        } // lib.optionalAttrs onLinux {
          # torch-bin with CUDA 13.0: nix develop .#cuda
          cuda = pkgs.mkShell {
            packages = [ cudaPython pkgs.ruff pkgs.gcc ];
            TYPE_NN_SRC = "${type-nn-c}";
            TORCH_TYPE_NN_DATA = dataDir;
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
              ${copyData}
              echo "torch-type-nn ${version} CUDA shell: TNN_REQUIRE_CUDA=1 pytest"
            '';
          };
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
