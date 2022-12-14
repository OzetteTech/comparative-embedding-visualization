from typing import Callable, Literal, Union

import numpy as np
import pandas as pd
import scipy.stats.mstats
from sklearn.neighbors import NearestNeighbors

from embcomp.metrics import count_neighbor_labels


def _validate_df(df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"

    if "x" not in df.columns or "y" not in df.columns or "label" not in df.columns:
        raise AttributeError("Must have 'x' and 'y' and 'label' columns.")


def process_bags(
    labels: pd.Series,
    indices_bags: dict[str, np.ndarray],
    type: Literal["incoming", "outgoing", "both"],
    agg: Literal["set", "sum"],
):

    # flatten
    outgoing = {k: v.ravel() for k, v in indices_bags.items()}

    if type == "outgoing":
        bags = outgoing
    else:
        # incoming or both
        incoming = {}
        for label in indices_bags:
            bag = []
            for other, out_inds in indices_bags.items():
                in_inds = np.where(labels == other)[0]

                if label == other:
                    continue

                mask = np.any(labels.to_numpy()[out_inds] == label, axis=1)
                bag.append(in_inds[mask])
            incoming[label] = np.concatenate(bag)

        if type == "incoming":
            bags = incoming
        else:
            bags = {}
            for k in outgoing:
                bags[k] = np.concatenate((outgoing[k], incoming[k]))

    if agg == "set":
        bags = {k: np.unique(v) for k, v in bags.items()}

    index = pd.Series(bags.keys(), name="label", dtype="category")
    df = pd.DataFrame(
        np.stack([labels.values[v].value_counts() for v in bags.values()]),
        index=index,
    )
    df.columns = index.values
    return df


def kneighbors(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    return nn.kneighbors(return_distance=False)


def fixed_k(df: pd.DataFrame, k: int, knn_indices: Union[None, np.ndarray] = None):
    _validate_df(df)

    if knn_indices is None:
        knn_indices = kneighbors(df[["x", "y"]].values, k=k)

    counts = count_neighbor_labels(knn_indices, df.label)
    index = pd.Series(df.label, name="label", dtype="category")

    result = pd.DataFrame(counts, index=index).groupby("label").sum()
    result.columns = result.index
    return result


def dynamic_k(
    df: pd.DataFrame,
    compute_k: Callable[[int], int] = lambda size: int(np.ceil(np.log2(size))),
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
        indices_bags[label] = knn_indices[df.label.values == label, 0:k]

    return process_bags(
        labels=df.label, indices_bags=indices_bags, type="outgoing", agg=kind
    )


def count_first(
    df: pd.DataFrame,
    n: int = 0,
    agg: Literal["set", "sum"] = "set",
    type: Literal["incoming", "outgoing", "both"] = "incoming",
    knn_indices: Union[np.ndarray, None] = None,
):
    _validate_df(df)

    if knn_indices is None:
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
        bag = np.zeros((len(first_non_self_indices), n + 1), dtype="int64")
        for i in range(n + 1):
            bag[:, i] = knn_indices[
                label_mask, first_non_self_indices + confusion_k_start
            ]
            first_non_self_indices += 1
        indices_bags[label] = bag

    return process_bags(labels=df.label, indices_bags=indices_bags, agg=agg, type=type)


def transform_abundance(
    label_representation: pd.DataFrame,
    abundances: dict[str, int],
    force_include_self: bool = True,
):
    """
    Creates an abundance-based representation.

    This function transforms a label-level neighborhood representation
    into an abundance-based representation by replacing the non-zero
    elements with the abundances.

    Parameters
    ----------
    label_representation : label-level neighborhood representation (e.g., result of `count_first`)
    abundances : label abundances
    force_include_self: force include self abundance even if missing from lable-level representaion.
    """
    assert (
        label_representation.index.to_list() == label_representation.columns.to_list()
    ), "must be a symmetric DataFrame with shared rows/cols"

    mask = label_representation.to_numpy() > 0
    if force_include_self:
        np.fill_diagonal(mask, True)
    return pd.DataFrame(
        mask * np.array([abundances[col] for col in label_representation.columns]),
        columns=label_representation.columns,
        index=label_representation.index,
    )


def merge_abundances_left(left: pd.DataFrame, right: pd.DataFrame):
    """Create single label-mask using all labels from left and right.
    If a label in `right` is missing in `left`, the neighbors from `right`
    are copied into `left` for that label. The label itself is set to False.
    """
    index = pd.CategoricalIndex(left.index.union(right.index).sort_values())
    merged = pd.DataFrame(
        np.full((len(index),) * 2, 0),
        columns=index,
        index=index,
    )
    # copy left values in to unified matrix
    merged.loc[left.index, left.columns] = left
    # find missing labels for left and populate with right
    missing = list(set(index).difference(left.index))
    merged.loc[missing, right.columns] = right.loc[missing, right.columns]
    # make sure to zero out diagonal for right-copied rows
    merged.loc[missing, missing] = 1
    return merged


def relative_abundance(abundance_representation: pd.DataFrame):

    return np.diagonal(abundance_representation) / abundance_representation.sum(axis=1)


def centered_logratio(abundance_representation: pd.DataFrame):
    copy = abundance_representation.to_numpy().copy()
    diag = np.diagonal(copy)
    np.fill_diagonal(copy, np.where(diag > 0, diag, 1))

    def _compute(row, i):
        value = row[i]
        values = row[np.nonzero(row)]
        gmean = scipy.stats.mstats.gmean(values)
        ratio = np.log(value / gmean)
        return ratio

    return pd.Series(
        [_compute(row, i) for i, row in enumerate(copy)],
        index=abundance_representation.index,
    )
