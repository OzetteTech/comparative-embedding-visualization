import pathlib
from argparse import ArgumentParser
from time import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from umap import UMAP


def prepare(
    df: pd.DataFrame, num_markers: int = -1, spread_factor: float = 1000
) -> tuple[list[str], dict[str, float], npt.NDArray[np.double]]:
    suffix = "_faust_annotation"
    all_markers: list[str] = [
        c.rstrip(suffix) for c in list(df.columns) if c.endswith(suffix)
    ]
    markers = all_markers[:num_markers] if num_markers > 0 else all_markers

    df["complete_faust_label"] = ""
    for marker in markers:
        df["complete_faust_label"] += marker + df[f"{marker}_faust_annotation"]

    expression_levels: dict[str, float] = {
        label: i * spread_factor
        for i, label in enumerate(df[f"{markers[0]}_faust_annotation"].unique())
    }

    df.sort_values(by=["faustLabels"], ignore_index=True, inplace=True)

    return markers, expression_levels, df[markers].values


def transform(
    df: pd.DataFrame, markers: list[str], expression_levels, log: bool = False
):
    faust_labels = df.complete_faust_label.unique()

    marker_annotation_cols = [f"{m}_faust_annotation" for m in markers]

    embedding_expression: npt.NDArray[np.double] = df[markers].values.copy()

    t = 0

    # For each cluster (i.e., cell phenotype defined by the FAUST label)
    for i, faust_label in enumerate(faust_labels):
        if log and i % 1000 == 0:
            t = time()
            print(
                f"Transform {i}-{i + 999} of {len(faust_labels)} clusters... ", end=""
            )

        # First, we get the indices of all data points belonging to
        # the cluster (i.e., cell phenotype)
        idxs = df.query(f'complete_faust_label == "{faust_label}"').index

        # 1. We winsorize the expression values to [0.01, 99.9]
        embedding_expression[idxs] = winsorize(
            embedding_expression[idxs],
            limits=[0.01, 0.01],
            axis=0,
        )

        # 2. Then we standardize the expression values
        # to have zero mean and unit standard deviation
        mean = embedding_expression[idxs].mean(axis=0)
        sd = np.nan_to_num(embedding_expression[idxs].std(axis=0))
        sd[sd == 0] = 1

        embedding_expression[idxs] -= mean
        embedding_expression[idxs] /= sd

        # 3. Next, we translate the expressions values based on their expression levels
        embedding_expression[idxs] += (
            df.iloc[idxs[0]][marker_annotation_cols].map(expression_levels).values
        )

        if log and (i % 1000 == 999 or i == len(faust_labels) - 1):
            print(f"done! ({round(time() - t)}s)")

    return embedding_expression


def to_df(df: pd.DataFrame, xy: npt.NDArray[np.double], save_as=None):
    df_embedding = pd.concat(
        [
            pd.DataFrame(xy, columns=["x", "y"]),
            pd.DataFrame(df.complete_faust_label.values, columns=["cellType"]),
            df,
        ],
        axis=1,
    )
    df_embedding.cellType = df_embedding.cellType.where(
        df.faustLabels != "0_0_0_0_0", "0_0_0_0_0"
    ).astype("category")

    if save_as is not None:
        df_embedding.to_parquet(f"data/{save_as}.pq", compression="gzip")

    return df_embedding


def embed(df: pd.DataFrame, data: npt.NDArray[np.double], embeddor: UMAP, save_as=None):
    return to_df(df, embeddor.fit_transform(data), save_as=save_as)  # type:ignore


def parse_args():
    parser = ArgumentParser("Transform and embed FAUST-annotated data.")
    parser.add_argument("input", type=pathlib.Path, help="Input parquet file")
    parser.add_argument("output", type=pathlib.Path, help="Output parquet file")
    parser.add_argument("--seed", type=int, help="Random seed for UMAP")
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Whether to apply the transformtion to the expressions prior to embedding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)
    markers, expression_levels, expressions = prepare(df)

    print(f'Markers: {", ".join(markers)}')
    print(f'Expression Levels: {" and ".join(expression_levels.keys())}')

    pca = PCA(n_components=2).fit_transform(
        df[[f"{m}_Windsorized" for m in markers]].values
    )

    if args.transform:
        expressions = transform(df, markers, expression_levels, log=True)

    umap_instance = UMAP(init=pca, random_state=args.seed or 42)
    embedding = embed(df, expressions, umap_instance)
    embedding.to_parquet(args.output, compression="gzip")


if __name__ == "__main__":
    main()
