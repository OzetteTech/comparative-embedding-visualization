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

### Commands Cheatsheet

If using `hatch` CLI, the following commands are available in the default environment:

| Command                | Action                                                              |
| :--------------------- | :------------------------------------------------------------------ |
| `hatch run fix`        | Format project with `black .` and apply linting with `ruff --fix .` |
| `hatch run check`      | Check formatting and linting with `black --check .` and `ruff .`.   |
| `hatch run test`       | Run unittests with `pytest` in base environment.                    |
| `hatch run test:test`  | Run unittests with `pytest` in all supported environments.          |

Alternatively, you can devlop **cev** by manually creating a virtual environment and managing
dependencies with `pip`.

Our CI linting/formatting checks are configured with [`pre-commit`](https://pre-commit.com/).
We recommend installing the git hook scripts to allow `pre-commit` to run automatically on `git commit`.

```sh
pre-commit install # run this once to install the git hooks
```

This will ensure that code pushed to CI meets our linting and formatting criteria. Code that does
not comply will fail in CI.

## Release

releases are triggered via tagged commits

```
git tag -a vX.X.X -m "vX.X.X"
git push --follow-tags
```

