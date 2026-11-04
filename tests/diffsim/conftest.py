"""quantem.diffsim lives in packages/quantem.diffsim; test it from the source tree, installed or not."""

import pathlib

import quantem

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "packages" / "quantem.diffsim" / "src" / "quantem"
if str(SOURCE) not in list(quantem.__path__):
    quantem.__path__.append(str(SOURCE))
