tyssue is a pure python package, installed with `pip`. No compiler toolchain is required.

Conda packaging is no longer maintained; the conda-forge `tyssue` package is frozen
at the last upstream release and does not track this project.

## Install tyssue using pip

```sh
python -m pip install --upgrade tyssue
```

Optional extras:

```sh
python -m pip install "tyssue[viz]"   # ipyvolume, pythreejs, vispy, jupyter
python -m pip install "tyssue[zarr]"  # zarr + xarray, for tyssue.io.zarr
```

## ImageMagick (only for gif export)

`create_gif` and `create_gif_3d` shell out to [ImageMagick](https://imagemagick.org),
which is a system binary, not a Python package, and so is not installed by pip:

| Platform | Command |
| --- | --- |
| macOS | `brew install imagemagick` |
| Debian/Ubuntu | `sudo apt install imagemagick` |
| Windows | [installer](https://imagemagick.org/script/download.php#windows) |

## Installing from source

### Download and install `tyssue` from source

Clone the repository:

```bash
git clone https://github.com/vivarium-collective/vivatyssue.git
cd vivatyssue
```

Create a virtual environment and install in editable mode with the dev extras:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

All runtime dependencies are declared in `pyproject.toml`, so pip resolves them
for you.



If all went well, you have successfully installed tyssue.

### Install testing utilities

```sh
pip install pytest pytest-cov nbval
```

A `Makefile` provides some utility function. Try :

```sh
make tests  # Run tests with nose
make coverage  # Run tests with coverage
make flake8  # Check PEP8 on the code
make nbtest # Tests all  the demo notebooks - requires nbval
```


### Building the documentation

The documentation uses
[nbsphinx](http://nbsphinx.readthedocs.io/en/0.2.9/index.html) to
convert the jupyter notebooks in doc/notebooks to html with sphinx.


```sh
pip install sphinx nbsphinx sphinx-autobuild
cd tyssue/doc
make html
```
