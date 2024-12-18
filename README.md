<h1 align="center">
  Comparative Embedding Visualization with <code>cev</code>
</h1>


<div align="center">
  
  [![pypi version](https://img.shields.io/badge/ozette-technologies-ozette.svg?color=0072E1&labelColor=0B1117&style=flat-square)](https://ozette.com/)
  [![pypi version](https://img.shields.io/pypi/v/cev.svg?color=0072E1&labelColor=0B1117&style=flat-square)](https://pypi.org/project/cev/)
  [![build status](https://img.shields.io/github/actions/workflow/status/OzetteTech/comparative-embedding-visualization/ci.yml?branch=main&color=0072E1&labelColor=0B1117&style=flat-square)](https://github.com/OzetteTech/comparative-embedding-visualization/actions?query=workflow%3ARelease)
  [![notebook examples](https://img.shields.io/badge/notebook-examples-0072E1.svg?labelColor=0B1117&style=flat-square)](notebooks)
  [![ISMB BioVis 2023 Poster](https://img.shields.io/badge/ISMB_BioVis_'23-poster-0072E1.svg?labelColor=0B1117&style=flat-square)](ismb-biovis-2023-poster.jpg)
  
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

## Quick Start

The **cev** package has a cli to quickly try out a demo of comparison widget in JupyterLab. It requires [uv](https://astral.sh/uv) to be installed.

```sh
uvx --python 3.11 cev demo # Downloads datasets and launches Jupyter Lab
```

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

<img width="1269" alt="User interface of cev's comparison widget" src="https://github.com/OzetteTech/comparative-embedding-visualization/assets/84813279/db28944b-fa36-475c-b3b9-efd07272e1b9">


See [notebooks/getting-started.ipynb](notebooks/getting-started.ipynb) for the complete example.

## Development

We use [`uv`](https://astral.sh/uv) for development.

```sh
uv run jupyter lab
```

### Commands Cheatsheet

| Command                | Action                                                              |
| :--------------------- | :------------------------------------------------------------------ |
| `uv run ruff format`   | Format the source code.                                             |
| `uv run ruff check`    | Check the source code for formatting issues.                        |
| `uv run pytest`        | Run unit tests with `pytest` in base environment.                   |


## Release

releases are triggered via tagged commits

```
git tag -a vX.X.X -m "vX.X.X"
git push --follow-tags
```

## License

`cev` is distributed under the terms of the [Apache License 2.0](LICENSE).

## Citation

If you use `cev` in your research, please cite the following preprint:

```bibtex
@article{manz2024general,
  title = {A General Framework for Comparing Embedding Visualizations Across Class-Label Hierarchies},
  author = {Trevor Manz and Fritz Lekschas and Evan Greene and Greg Finak and Nils Gehlenborg},
  url = {https://doi.org/10.1109/TVCG.2024.3456370},
  doi = {10.1109/TVCG.2024.3456370},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  series = {VIS ’24},
  publisher = {IEEE},
  year = {2024},
  month = {9},
  pages = {1-11}
}
```
