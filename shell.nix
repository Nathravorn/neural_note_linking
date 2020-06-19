let
  pkgs = import <nixpkgs> {};

  sentence-transformers = pkgs.python3Packages.buildPythonPackage rec {
    pname = "sentence-transformers";
    version = "0.2.4";

    src = pkgs.python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "0yx976alq4gmr18cwy3z975q618zl1wl1pd37b0jb5v9xx64zxs6";
    };

    propagatedBuildInputs = with pkgs.python3Packages; [
      transformers tqdm pytorch numpy scikitlearn scipy nltk
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
