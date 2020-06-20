let
  pkgs = import <nixpkgs> {};

  # nixpkgs doesn't have sentence-transformers
  nadrieril = import (builtins.fetchTarball "https://github.com/Nadrieril/nur-packages/archive/master.tar.gz") { inherit pkgs; };

in
pkgs.mkShell {
  buildInputs = [
    (pkgs.python3.withPackages (ps: with ps; [
      pandas numpy matplotlib tqdm nadrieril.sentence-transformers scikitlearn
    ]))
  ];
}
