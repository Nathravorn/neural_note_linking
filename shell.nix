let
  moz_overlay = import (builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz);
  pkgs = import <nixpkgs> { overlays = [ moz_overlay ]; };

  # From https://github.com/edolstra/import-cargo
  import_cargo = import
    (builtins.fetchurl https://raw.githubusercontent.com/edolstra/import-cargo/8abf7b3a8cbe1c8a885391f826357a74d382a422/flake.nix);
  importCargo = lockFile: (import_cargo.outputs { self = {}; }).builders.importCargo { inherit pkgs lockFile; };


  setuptools-rust = pkgs.python3Packages.buildPythonPackage rec {
    pname = "setuptools-rust";
    version = "0.10.6";

    src = pkgs.fetchFromGitHub {
      owner = "PyO3";
      repo = pname;
      rev = "v${version}";
      sha256 = "19q0b17n604ngcv8lq5imb91i37frr1gppi8rrg6s4f5aajsm5fm";
    };

    nativeBuildInputs = [
      pkgs.rustc
    ];

    propagatedBuildInputs = with pkgs.python3Packages; [
      semantic-version toml
    ];

    doCheck = false;
  };

  tokenizers = let
    pname = "tokenizers";
    version = "0.7.0";

    root = pkgs.fetchFromGitHub {
      owner = "huggingface";
      repo = pname;
      rev = "python-v${version}";
      sha256 = "1f13rmqa0zy9x6hilk8j3pdly287ill6qs0xah6q7d0cj8pf5p8f";
    };

    # Make the python bindings the root and the rust lib a subdirectory
    src = pkgs.runCommand "tokenizers-python-${version}" {} ''
      cp --no-preserve=mode -r "${root}/bindings/python" $out
      cp --no-preserve=mode -r "${root}/tokenizers" $out/tokenizers-rust
      sed -i 's#path = "../../tokenizers"#path = "./tokenizers-rust"#' $out/Cargo.toml
    '';

    toolchain-txt = builtins.readFile "${root}/bindings/python/rust-toolchain";
    toolchain-match = builtins.match ''nightly-([0-9-]*).*'' toolchain-txt;
    toolchain-date = if toolchain-match != null
      then builtins.elemAt toolchain-match 0
      else "2020-05-14";
    rust = (pkgs.rustChannelOf { date = toolchain-date; channel = "nightly"; }).rust;

    vendorDir = (importCargo "${root}/bindings/python/Cargo.lock").vendorDir;
    cargoHome = pkgs.makeSetupHook {} (pkgs.writeScript "make-cargo-home" ''
      export CARGO_HOME=$TMPDIR/vendor
      cp -prd ${vendorDir}/vendor $CARGO_HOME
      chmod -R u+w $CARGO_HOME
    '');

  in pkgs.python3Packages.buildPythonPackage rec {
    inherit pname version src;

    nativeBuildInputs = [
      rust cargoHome
    ];

    buildInputs = with pkgs.python3Packages; [
      setuptools-rust
    ];

    doCheck = false;
  };

  transformers-2-11 = pkgs.python3Packages.buildPythonPackage rec {
    pname = "transformers";
    version = "2.11.0";

    src = pkgs.fetchFromGitHub {
      owner = "huggingface";
      repo = pname;
      rev = "v${version}";
      sha256 = "1caqz5kp8mfywhiq8018c2jf14v15blj02fywh9xgvpq2dns9sc1";
    };

    propagatedBuildInputs = with pkgs.python3Packages; [
      filelock numpy packaging regex requests sacremoses sentencepiece tokenizers tqdm
    ];

    doCheck = false;

    meta = with pkgs.stdenv.lib; {
      homepage = "https://github.com/huggingface/transformers";
      description = "State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch";
      license = licenses.asl20;
      platforms = [ "x86_64-linux" ];
    };
  };

  sentence-transformers = pkgs.python3Packages.buildPythonPackage rec {
    pname = "sentence-transformers";
    version = "0.2.6.1";

    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "0a6ixs9lp448bq4q0h7ciixszfqhlzn3sqwshwy03mra4wg0w9b8";
    };

    propagatedBuildInputs = with pkgs.python3Packages; [
      transformers-2-11 tqdm pytorch numpy scikitlearn scipy nltk
    ];

    doCheck = false;
  };

in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      pandas numpy matplotlib tqdm sentence-transformers scikitlearn
    ]))
  ];
}
