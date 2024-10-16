
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
sure that you install the relevant GPU version. Please see the following
note reguarding a common issue compiling packages using Rust on Windows.

#### Installing Rust on Windows:

To install Rust on windows, you can use the Rust installation wizard,
`rustup-init.exe`, found [at this
site](https://forge.rust-lang.org/infra/other-installation-methods.html).
Depending on your version of Windows, you are likely to see an error
that looks something like the following when you try and compile the
package:

    error: toolchain 'stable-x86_64-pc-windows-gnu' is not installed

In this case, you should run
`rustup install stable-x86_64-pc_windows-gnu` to install the missing
toolchain. If you’re missing another toolchain, simply type this in the
place of `stable-x86_64-pc_windows-gnu` in the command above.

## Installing from GitHub

At the moment, the only way to install this package is from this github
repository, although we hope to have pre-built binaries available on
CRAN / R-Universe soon.

``` r
devtools::install_github("beniaminogreen/linkorgsonnx")
```

# Obtaining Models From HuggingFace

We have designed `linkedorgsonnx` to be compatible with most of the
sentence-similarity models available on HuggingFace, so browsing the
[sentence
similarity](https://huggingface.co/models?pipeline_tag=sentence-similarity)
task is a great place to start.

Huggingface distributes a tool called
[Optimum](https://github.com/huggingface/optimum/tree/main) that can be
used to download models from the huggingface hub and convert them into
the ONNX format. The code below can be used to download the
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model (the most popular model trained for this task), convert it to
ONNX, and save it to `./mini_lm_l6/`.

``` sh
# pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]

# If using CUDA GPU
optimum-cli export onnx --model "sentence-transformers/all-MiniLM-L6-v2" mini_lm_l6/ --optimize O4 --device cuda --task sentence-similarity


# If using CPU
optimum-cli export onnx --model "sentence-transformers/all-MiniLM-L6-v2" mini_lm_l6/ --optimize O3 --device cpu --task sentence-similarity
```

# Generating Embeddings

Now that you have your hands on a model, it should be easy to generate
embeddings for text inputs. Simply point `linkorgsonnx` to where the
ONNX Runtime is installed, load the model into memory, and get to
generating embeddings!

``` r
# Point Package to where the ONNX Runtime is installed
Sys.setenv(ORT_DYLIB_PATH = "path/to/libonnxruntime.so")

# Loads model from a given directory
model <- new_inference_session("mini_lm_l6/")

inputs <- c(
            "The quick brown fox jumped over the lazy dog",
            "Sphynx of black quartx, judge my vow",
            "The five boxing wizards jump quickly"
)
generate_embeddings(model, inputs)
```

# Neural Joins

The flagship feature of `linkorgsonnx` is it’s capacity to join datasets
on the basis of neural network embeddings.

To (inner) join two dataframes on the basis of a single column, you can
run the following code, which generates embeddings for the `name_1`
column in dataset 1 and `name_2` in dataset, and returns pairs that have
a cosine distance less than .1 in the embedding space.

``` r
neural_inner_join(dataframe_1, dataframe_2, model, by = c("name_1", "name_2"), radius = .1)
```
