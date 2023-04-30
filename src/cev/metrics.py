from __future__ import annotations

import typing

import numpy as np
import numpy.linalg as nplg

if typing.TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

__all__ = [
    "rowise_cosine_similarity",
    "confusion",
    "transform_abundance",
    "merge_abundances_left",
    "relative_abundance",
    "centered_logratio",
]


def rowise_cosine_similarity(X0: npt.ArrayLike, X1: npt.ArrayLike):
    """Computes the cosine similary per row of two equally shaped 2D matrices."""
    return np.sum(X0 * X1, axis=1) / (nplg.norm(X0, axis=1) * nplg.norm(X1, axis=1))


def confusion(df: pd.DataFrame, py: bool = False):
    if py:
        import cev_metrics.py

        mat = cev_metrics.py.confusion(df, counts=True)
    else:
        import cev_metrics

        cats = df["label"].cat.categories
        mat = pd.DataFrame(cev_metrics.confusion(df), index=cats, columns=cats)

    normed = (mat / mat.sum(axis=1)).to_numpy()
    return pd.Series(1 - normed.diagonal(), index=mat.index, name="confusion")


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
    import scipy

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
