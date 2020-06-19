let
  pkgs = import <nixpkgs> {};

  importCargo = lockFile:
    let lockFile' = builtins.fromTOML (builtins.readFile lockFile); in
    rec {
      # Fetch and unpack the crates specified in the lock file.
      unpackedCrates = map
        (pkg:

          let
            isGit = builtins.match ''git\+(.*)\?rev=([0-9a-f]+)(#.*)?'' pkg.source;
            registry = "registry+https://github.com/rust-lang/crates.io-index";
          in

          if pkg.source == registry then
            let
              sha256 = lockFile'.metadata."checksum ${pkg.name} ${pkg.version} (${registry})";
              tarball = import <nix/fetchurl.nix> {
                url = "https://crates.io/api/v1/crates/${pkg.name}/${pkg.version}/download";
                inherit sha256;
              };
            in pkgs.runCommand "${pkg.name}-${pkg.version}" {}
              ''
                mkdir $out
                tar xvf ${tarball} -C $out --strip-components=1
                # Add just enough metadata to keep Cargo happy.
                printf '{"files":{},"package":"${sha256}"}' > "$out/.cargo-checksum.json"
              ''

          else if isGit != null then
            let
              rev = builtins.elemAt isGit 1;
              url = builtins.elemAt isGit 0;
              tree = builtins.fetchGit { inherit url rev; };
            in pkgs.runCommand "${pkg.name}-${pkg.version}" {}
              ''
                tree=${tree}
                if grep --quiet '\[workspace\]' $tree/Cargo.toml; then
                  if [[ -e $tree/${pkg.name} ]]; then
                    tree=$tree/${pkg.name}
                  fi
                fi
                cp -prvd $tree/ $out
                chmod u+w $out
                # Add just enough metadata to keep Cargo happy.
                printf '{"files":{},"package":null}' > "$out/.cargo-checksum.json"
                cat > $out/.cargo-config <<EOF
                [source."${url}"]
                git = "${url}"
                rev = "${rev}"
                replace-with = "vendored-sources"
                EOF
              ''

          else throw "Unsupported crate source '${pkg.source}' in dependency '${pkg.name}-${pkg.version}'.")

        (builtins.filter (pkg: pkg.source or "" != "") lockFile'.package);

      # Create a directory that symlinks all the crate sources and
      # contains a cargo configuration file that redirects to those
      # sources.
      vendorDir = pkgs.runCommand "cargo-vendor-dir" {}
        ''
          mkdir -p $out/vendor
          cat > $out/vendor/config <<EOF
          [source.crates-io]
          replace-with = "vendored-sources"
          [source.vendored-sources]
          directory = "vendor"
          EOF
          declare -A keysSeen
          for i in ${toString unpackedCrates}; do
            ln -s $i $out/vendor/$(basename "$i" | cut -c 34-)
            if [[ -e "$i/.cargo-config" ]]; then
              # Ensure we emit TOML keys only once.
              key=$(sed 's/\[source\."\(.*\)"\]/\1/; t; d' < "$i/.cargo-config")
              if [[ -z ''${keysSeen[$key]} ]]; then
                keysSeen[$key]=1
                cat "$i/.cargo-config" >> $out/vendor/config
              fi
            fi
          done
        '';

      # Create a setup hook that will initialize CARGO_HOME. Note:
      # we don't point CARGO_HOME at the vendor tree directly
      # because then we end up with a runtime dependency on it.
      cargoHome = pkgs.makeSetupHook {}
        (pkgs.writeScript "make-cargo-home" ''
          if [[ -z $CARGO_HOME || $CARGO_HOME = /build ]]; then
            export CARGO_HOME=$TMPDIR/vendor
            # FIXME: work around Rust 1.36 wanting a $CARGO_HOME/.package-cache file.
            #ln -s ${vendorDir}/vendor $CARGO_HOME
            cp -prd ${vendorDir}/vendor $CARGO_HOME
            chmod -R u+w $CARGO_HOME
          fi
        '');
    };


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
    rust-version = "0.10.1";
    rust-checksum = "3632a2635700a4b46f875eba3d6bf83c48c3c4289aa6330442a9ed8e3dcf587c";

    root = pkgs.fetchFromGitHub {
      owner = "huggingface";
      repo = pname;
      rev = "python-v${version}";
      sha256 = "1f13rmqa0zy9x6hilk8j3pdly287ill6qs0xah6q7d0cj8pf5p8f";
    };

    src = pkgs.runCommand "tokenizers-python-${version}" {} ''
      cp --no-preserve=mode -r "${root}/bindings/python" $out
      sed -i 's#path = "../../tokenizers"##' $out/Cargo.toml
      sed -i 's#version = "\*"#version = "${rust-version}"#' $out/Cargo.toml
    '';

    cargoLock = pkgs.runCommand "Cargo.lock" {} ''
      cp --no-preserve=mode ${src}/Cargo.lock $out
      sed -i 's#name = "tokenizers"#name = "tokenizers"'"\n"'source = "registry+https://github.com/rust-lang/crates.io-index"#' $out
      echo '"checksum tokenizers ${rust-version} (registry+https://github.com/rust-lang/crates.io-index)" = "${rust-checksum}"' >> $out
    '';
    cargoHome = (importCargo "${cargoLock}").cargoHome;

  in pkgs.python3Packages.buildPythonPackage rec {
    inherit pname version src;

    nativeBuildInputs = [
      pkgs.rustc pkgs.cargo cargoHome
    ];

    buildInputs = with pkgs.python3Packages; [
      setuptools-rust
    ];
  };

  transformers-2-11 = pkgs.python3Packages.transformers.overrideDerivation (_: rec {
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
  });

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
