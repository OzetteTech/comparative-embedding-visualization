# Comparative Embedding Visualization with `cev`

`cev` is an interactive Jupyter widget for comparing a pair of 2D embeddings with shared labels by label confusion, neighborhood composition, and label size.

## Installation

```sh
pip install cev
```

## Development

First, create a virtual environment with all the required dependencies. We highly recommend to use [`hatch`](https://github.com/pypa/hatch), which installs and sync all dependencies from `pyproject.toml` automatically/

```sh
hatch shell
```

Alternatively, you can also use [`conda`](https://docs.conda.io/en/latest/).

```sh
conda env create -n cev python=3.11
conda activate cev
```

Next, install `cev` with all development assets.

```sh
pip install -e ".[notebooks,dev]"
```

Finally, you can now run the notebooks with:

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

## License

`cev` is distributed under the terms of the [Apache License 2.0](LICENSE).
