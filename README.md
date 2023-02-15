# cev

## Installation

```sh
pip install cev
```

## Development

Create a virtual environment with all the required dependencies with `conda`:

```sh
conda env create -n cev python=3.10
conda activate cev
pip install -e ".[notebooks,dev]"
```

or automatically if you use [`hatch`](https://github.com/pypa/hatch):

```sh
hatch shell
# syncs and installs deps from `pyproject.toml`
```

You can now run the notebooks with:

```sh
jupyterlab
```

## Release

releases are triggered via tagged commits

```
git tag -a vX.X.X -m "vX.X.X"
git push --follow-tags
```

