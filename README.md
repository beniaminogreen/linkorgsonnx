
# linkorgsonnx

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

`linkorgsonnx` is an R package that allows you to join datasets using
neural networks in R.

# Installation

## Prerequisites:

To use the development version of the package, you must have the
following installed:

1.  [Cargo](https://www.rust-lang.org/tools/install), the package
    manager for Rust
2.  The [ONNX Runtime](https://www.onnxruntime.ai), needed for running
    the neural networks.

Installation for both should require copying a few commands in your
terminal. If you are looking to run the models on a GPU, please make
sure that you install the relevant GPU version.

## Installing from GitHub

At the moment, the only way to install this package is from this github
repository, although we hope to have pre-built binaries available on
CRAN / R-Universe soon.

``` r
devtools::install_github("beniaminogreen/linkorgsonnx")
```
