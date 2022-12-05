from typing import Literal, Callable, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from embcomp.metrics import count_neighbor_labels


def _validate_df(df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"

    if "x" not in df.columns or "y" not in df.columns or "label" not in df.columns:
        raise AttributeError("Must have 'x' and 'y' and 'label' columns.")


def process_bags(
    labels: pd.Series,
    indices_bags: dict[str, np.ndarray],
    kind: Literal["set", "sum"]
):
    if kind == "set":
        indices_bags = {k: np.unique(v) for k, v in indices_bags.items()}

    index = pd.Series(indices_bags.keys(), name="label", dtype="category")
    df = pd.DataFrame(
        np.stack([labels.values[v].value_counts() for v in indices_bags.values()]),
        index=index,
    )
    df.columns = index.values
    return df

def process_bags(
    labels: pd.Series,
    indices_bags: dict[str, np.ndarray],
    kind: Literal["set", "sum"]
):
    if kind == "set":
        indices_bags = {k: np.unique(v) for k, v in indices_bags.items()}

    index = pd.Series(indices_bags.keys(), name="label", dtype="category")
    df = pd.DataFrame(
        np.stack([labels.values[v].value_counts() for v in indices_bags.values()]),
        index=index,
    )
    df.columns = index.values
    return df


def kneighbors(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    return nn.kneighbors(return_distance=False)


def fixed_k(df: pd.DataFrame, k: int):
    _validate_df(df)

    knn_indices = kneighbors(df[["x", "y"]].values, k=k)
    counts = count_neighbor_labels(knn_indices, df.label)
    index = pd.Series(df.label, name="label", dtype="category")

    result = pd.DataFrame(counts, index=index).groupby("label").sum()
    result.columns = result.index
    return result


def dynamic_k(
    df: pd.DataFrame,
    compute_k: Callable[[int], int] = lambda size: int(np.ceil(np.log10(size))),
    kind: Literal["set", "sum"] = "set",
    knn_indices: Union[np.ndarray, None] = None,
):
    _validate_df(df)

    sizes = df.label.value_counts(sort=True)

    if knn_indices is None:
        knn_indices = kneighbors(
            X=df[["x", "y"]].values,
            k=max(map(compute_k, sizes)),
        )

    indices_bags = {}
    for label in df.label.cat.categories:
        k = compute_k(sizes.loc[label])
        label_knn_indices = knn_indices[df.label.values == label, 0:k]
        indices_bags[label] = label_knn_indices.ravel()

    return process_bags(labels=df.label, indices_bags=indices_bags, kind=kind)


def count_first(
    df: pd.DataFrame,
    n: int = 0,
    type: Literal["incoming", "outgoing", "both"],
    agg: Literal["set", "sum"] = "set",
):
    _validate_df(df)

    # clamp computing the neighborhood graph up to some point
    largest_category_size = min(
        500,
        df.label.value_counts(sort=True)[0],
    )

    # only compute the complete neighborhood graph up until the largest class size
    knn_indices = kneighbors(
        X=df[["x", "y"]].values,
        k=largest_category_size + n + 1,
    )
    knn_identities = df.label.values[knn_indices]

    indices_bags = {}
    for label in df.label.cat.categories:
        # TODO: dynamically set the start position for k
        confusion_k_start = 0
        label_mask = df.label.values == label
        first_non_self_indices = np.argmax(
            knn_identities[label_mask, confusion_k_start:] != label,
            axis=1,
        )
        bag = []
        for _ in range(n + 1):
            bag.append(
                knn_indices[label_mask, first_non_self_indices + confusion_k_start]
            )
            first_non_self_indices += 1
        indices_bags[label] = np.concatenate(bag)

    return process_bags(labels=df.label, indices_bags=indices_bags, kind=agg)
