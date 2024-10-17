import argparse
import json
import os
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path

import pooch

from cev._version import __version__

_DEV = False


def download_data() -> tuple[Path, Path]:
    archive = pooch.retrieve(
        url="https://figshare.com/ndownloader/articles/23063615/versions/1",
        path=pooch.os_cache("cev"),
        fname="data.zip",
        known_hash=None,
    )
    archive = Path(archive)
    files = [
        "mair-2022-tissue-138-umap.pq",
        "mair-2022-tissue-138-ozette.pq",
    ]
    with zipfile.ZipFile(archive, "r") as zip_ref:
        for file in files:
            zip_ref.extract(file, path=archive.parent)
    return (
        archive.parent / "mair-2022-tissue-138-umap.pq",
        archive.parent / "mair-2022-tissue-138-ozette.pq",
    )


def write_notebook(output: Path):
    umap_path, ozette_path = download_data()
    source = textwrap.dedent(
        f"""
        import pandas as pd
        from cev.widgets import Embedding, EmbeddingComparisonWidget

        umap_embedding = pd.read_parquet("{umap_path}").pipe(Embedding.from_ozette)
        ozette_embedding = pd.read_parquet("{ozette_path}").pipe(Embedding.from_ozette)

        EmbeddingComparisonWidget(
            umap_embedding,
            ozette_embedding,
            titles=("Standard UMAP", "Annotation-Transformed UMAP"),
            metric="confusion",
            selection="synced",
            auto_zoom=True,
            row_height=320,
        )
    """
    ).strip()

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with output.open("w") as f:
        json.dump(nb, f, indent=2)


def check_uv_available():
    if shutil.which("uv") is None:
        print("Error: 'uv' command not found.", file=sys.stderr)
        print("Please install 'uv' to run `cev demo` entrypoint.", file=sys.stderr)
        print(
            "For more information, visit: https://github.com/astral-sh/uv",
            file=sys.stderr,
        )
        sys.exit(1)


def run_notebook(notebook_path: Path):
    check_uv_available()
    command = [
        "uvx",
        "--python",
        "3.11",
        "--with",
        "." if _DEV else f"cev=={__version__}",
        "--with",
        "jupyterlab",
        "jupyter",
        "lab",
        str(notebook_path),
    ]
    try:
        os.execvp(command[0], command)
    except OSError as e:
        print(f"Error executing {command[0]}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(prog="cev")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("download", help="Download the demo notebook (and data)")
    subparsers.add_parser("demo", help="Run the demo notebook in JupyterLab")
    args = parser.parse_args()

    notebook_path = Path("cev-demo.ipynb")
    if args.command == "download":
        write_notebook(notebook_path)
    elif args.command == "demo":
        write_notebook(notebook_path)
        run_notebook(notebook_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
