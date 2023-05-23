<h1 align="center">
  Comparative Embedding Visualization with <code>cev</code>
</h1>


<div align="center">
  
  [![pypi version](https://img.shields.io/badge/ozette-technologies-ozette.svg?color=0072E1&labelColor=0B1117&style=flat-square)](https://ozette.com/)
  [![pypi version](https://img.shields.io/pypi/v/cev.svg?color=0072E1&labelColor=0B1117&style=flat-square)](https://pypi.org/project/cev/)
  [![build status](https://img.shields.io/github/actions/workflow/status/OzetteTech/comparative-embedding-visualization/ci.yml?branch=main&color=0072E1&labelColor=0B1117&style=flat-square)](https://github.com/OzetteTech/comparative-embedding-visualization/actions?query=workflow%3ARelease)
  [![notebook examples](https://img.shields.io/badge/notebook-examples-0072E1.svg?labelColor=0B1117&style=flat-square)](notebooks)
  
</div>

<div align="center">
  
  <strong><code>cev</code> is an interactive Jupyter widget for comparing a pair of 2D embeddings with shared labels.</strong><br />Its novel metric allows to surface differences in label confusion, neighborhood composition, and label size.
  
</div>

<br/>

<div align="center">
  
  ![Teaser](https://github.com/OzetteTech/comparative-embedding-visualization/assets/84813279/297cbdb9-b6a2-4102-bde9-b14f0ca24a09)
  
  <sub>The figure shows data from [Mair et al. (2022)](https://doi.org/10.1038/s41586-022-04718-w) that were analyzed with [Greene et al.'s (2021) FAUST method](https://doi.org/10.1016/j.patter.2021.100372).<br />The embeddings were generated with [Greene et al.'s (2021) annotation transformation](https://github.com/flekschas-ozette/ismb-biovis-2022) and [UMAP](https://github.com/lmcinnes/umap).</sub>
  
  <br/>
  
  `cev` is implemented with [anywidget](https://anywidget.dev) and builds upon [jupyter-scatter](https://github.com/flekschas/jupyter-scatter/).
  
</div>

## Installation

> **Warning**: `cev` is new and under active development. It is not yet ready for production and APIs are subject to change.

```sh
pip install cev
```

## Getting Started

```py
import pandas as pd
from cev.widgets import Embedding, EmbeddingComparisonWidget

umap_embedding = Embedding.from_ozette(df=pd.read_parquet("../data/mair-2022-tissue-138-umap.pq"))
ozette_embedding = Embedding.from_ozette(df=pd.read_parquet("../data/mair-2022-tissue-138-ozette.pq"))

umap_vs_ozette = EmbeddingComparisonWidget(
    umap_embedding,
    ozette_embedding,
    titles=["Standard UMAP", "Annotation-Transformed UMAP"],
    metric="confusion",
    selection="synced",
    auto_zoom=True,
    row_height=320,
)
umap_vs_ozette
```

See [notebooks/getting-started.ipynb](notebooks/getting-started.ipynb) for the complete example.

## Development

First, create a virtual environment with all the required dependencies. We highly recommend to use [`hatch`](https://github.com/pypa/hatch), which installs and sync all dependencies from `pyproject.toml` automatically.

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
