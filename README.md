# ifcb-analysis

## Running locally

### Set up environment

Make sure you have `fftw3-dev` installed for computing Fourier transforms. On Debian-based OSes:

```sh
sudo apt install fftw3-dev
```

Build `conda` environment:

```sh
conda env create
```

Install testing dependencies with:

```sh
conda env update -f test-environment.yml
```

Set environment variable so Tensorflow knows where to look for things:

```sh
'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > .env
```

### Run tests

To run tests you'll need a CNN model to use. `ifcb-analysis` is configured to work with Keras models, which, for testing purposes, you can copy into `src/python/tests/data`.

With that in place, run:

```sh
pytest
```