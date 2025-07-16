{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.matplotlib
    python311Packages.numpy
    python311Packages.pygame
    python311Packages.scipy
    (python311Packages.pytorch.override {
      cudaSupport = false;
    })
  ];
}

