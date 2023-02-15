from typing import Callable, Literal, Union

import numpy as np
import pandas as pd
import scipy.stats.mstats
from sklearn.neighbors import NearestNeighbors

from cev.metrics import count_neighbor_labels


def _validate_df(df: pd.DataFrame):
    assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"

    if "x" not in df.columns or "y" not in df.columns or "label" not in df.columns:
        raise AttributeError("Must have 'x' and 'y' and 'label' columns.")


def process_bags(
    labels: pd.Series,
    outgoing: np.ndarray,
    type: Literal["incoming", "outgoing", "both"],
    agg: Literal["set", "sum"],
):
    """Process an outgoing "bag" of indices.

    `outgoing` is a 2D array containing positive indices into
    the labels series. Negative indices (e.g., -1) are filtered
    and ignored from final outcoming/incoming.
    """
    categories = labels.cat.categories
    outgoing_bags = {}
    for label in categories:
        indices = outgoing[labels.values == label].ravel()
        outgoing_bags[label] = indices[indices > 0]

    if type == "outgoing":
        bags = outgoing_bags
    else:
        incoming_bags = {}  # dict[str, 1D array of indices]
        outgoing_identities = np.where(
            outgoing < 0, "__ignored_label", labels.values[outgoing]
        )
        for label in categories:
            # find all points (rows) where one of the outgoing neighbors is `label`.
            row_matches = np.any(outgoing_identities == label, axis=1)
            # ignore outgoing for this label
            row_matches[labels.values == label] = False
            # grab the row indices which correspond to the original label ilocs
            incoming_bags[label] = np.where(row_matches)[0]
        if type == "incoming":
            bags = incoming_bags
        else:
            bags = {
                label: np.concatenate((outgoing_bags[label], incoming_bags[label]))
                for label in categories
            }

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
    type: Literal["incoming", "outgoing", "both"] = "outgoing",
):
    _validate_df(df)

    categories = df.label.cat.categories
    sizes = df.label.value_counts(sort=True)

    if knn_indices is None:
        knn_indices = kneighbors(
            X=df[["x", "y"]].values,
            k=max(map(compute_k, sizes)),
        )

    ks = [compute_k(sizes.loc[label]) for label in categories]

    outgoing = np.full((len(df), max(ks)), fill_value=-1, dtype="int64")
    for label, k in zip(categories, ks):
        label_mask = df.label.values == label
        outgoing[label_mask, 0:k] = knn_indices[df.label.values == label, 0:k]

    return process_bags(labels=df.label, outgoing=outgoing, type=type, agg=kind)


def count_first(
    df: pd.DataFrame,
    n: int = 0,
    agg: Literal["set", "sum"] = "set",
    type: Literal["incoming", "outgoing", "both"] = "both",
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

    outgoing = np.full((len(df.label), n + 1), fill_value=-1, dtype="int64")

    for label in df.label.cat.categories:
        # TODO: dynamically set the start position for k
        confusion_k_start = 0
        label_mask = df.label.values == label
        first_non_self_indices = np.argmax(
            knn_identities[label_mask, confusion_k_start:] != label,
            axis=1,
        )
        for i in range(n + 1):
            outgoing[label_mask, i] = knn_indices[
                label_mask, first_non_self_indices + confusion_k_start
            ]
            first_non_self_indices += 1

    return process_bags(labels=df.label, outgoing=outgoing, agg=agg, type=type)


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
    merged.loc[missing, missing] = 0
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
