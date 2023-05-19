from __future__ import annotations

import typing

import cev_metrics
import numpy as np
import numpy.linalg as nplg
import pandas as pd

if typing.TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "centered_logratio",
    "compare_neighborhoods",
    "confusion",
    "merge_abundances_left",
    "neighborhood",
    "relative_abundance",
    "rowise_cosine_similarity",
    "transform_abundance",
]


def confusion(df: pd.DataFrame) -> pd.Series:
    confusion_matrix = cev_metrics.confusion(df)
    normed = confusion_matrix / confusion_matrix.sum(axis=1)
    data = pd.Series(1 - normed.diagonal(), index=df["label"].cat.categories)
    # TODO: move to cev-metrics
    # Replace any label with 2 or less count with 0.0 confusion.
    counts = df["label"].value_counts()
    data.loc[counts[counts <= 2].index] = 0
    return data


def neighborhood(df: pd.DataFrame, max_depth: int = 1) -> pd.DataFrame:
    categories = df["label"].cat.categories
    neighborhood_scores = cev_metrics.neighborhood(df, max_depth)
    np.fill_diagonal(neighborhood_scores, 1)
    return pd.DataFrame(neighborhood_scores, index=categories, columns=categories)


def compare_neighborhoods(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Computes the cosine similarity between two neighborhood matrices.

    Parameters
    ----------
    a : pd.DataFrame
        A symmetric DataFrame with shared rows/cols.
    b : pd.DataFrame
        A symmetric DataFrame with shared rows/cols.

    Returns
    -------
    dict[str, float]
        A dictionary mapping labels to cosine similarity.
    """
    assert len(a) == len(a.columns)
    assert len(b) == len(b.columns)
    overlap = a.index.intersection(b.index)
    dist = {label: 0.0 for label in typing.cast(pd.Series, a.index.union(b.index))}
    sim = 1 - rowise_cosine_similarity(a.loc[overlap, overlap], b.loc[overlap, overlap])
    dist.update(sim)
    return dist


def rowise_cosine_similarity(X0: npt.ArrayLike, X1: npt.ArrayLike):
    """Computes the cosine similary per row of two equally shaped 2D matrices."""
    return np.sum(X0 * X1, axis=1) / (nplg.norm(X0, axis=1) * nplg.norm(X1, axis=1))


def transform_abundance(
    frequencies: pd.DataFrame,
    abundances: dict[str, int],
    force_include_self: bool = True,
    bit_mask: bool = False,
    clr: bool = False,
):
    """Creates an abundance-based representation.

    This function transforms a label-level neighborhood representation
    into an abundance-based representation by multiplying the frequencies
    with the abundances. Alternatively, a bitmask can be used to treat
    all non-zero frequencies as 1.

    Parameters
    ----------
    frequencies : pd.DataFrame
        A symmetric DataFrame with shared rows/cols.
    abundances : dict[str, int]
        A dictionary mapping labels to abundances.
    force_include_self : bool, optional
        Whether to include the label itself in the neighborhood, by default True.
    bit_mask : bool, optional
        Whether to use a bit mask instead of the frequencies when expanding
        abundances, by default False.
    clr : bool, optional
        Whether to normalize the count values by transforming them to centered
        log ratios, by default False.
    """
    assert (
        frequencies.index.to_list() == frequencies.columns.to_list()
    ), "must be a symmetric DataFrame with shared rows/cols"

    if bit_mask:
        mask = frequencies.to_numpy() > 0
        if force_include_self:
            np.fill_diagonal(mask, True)
    else:
        mask = frequencies.to_numpy()
        if force_include_self:
            np.fill_diagonal(mask, 1.0)

    if clr:
        inflated_counts = np.fromiter(abundances.values(), dtype=int) + 1
        gmean = _gmean(inflated_counts)
        values = dict(zip(abundances.keys(), np.log10(inflated_counts / gmean)))
    else:
        values = abundances

    return pd.DataFrame(
        mask * np.array([values[col] for col in frequencies.columns]),
        columns=frequencies.columns,
        index=frequencies.index,
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
        gmean = _gmean(row + 1)
        ratio = np.log10((row[i] + 1) / gmean)
        return ratio

    return pd.Series(
        [_compute(row, i) for i, row in enumerate(copy)],
        index=abundance_representation.index,
    )


# from scipy.stats.mstats.gmean
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def _gmean(a, axis=0, dtype=None, weights=None):
    r"""Compute the weighted geometric mean along the specified axis.

    The weighted geometric mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \exp \left( \frac{ \sum_{i=1}^n w_i \ln a_i }{ \sum_{i=1}^n w_i }
                   \right) \, ,

    and, with equal weights, it gives:

    .. math::

        \sqrt[n]{ \prod_{i=1}^n a_i } \, .

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the geometric mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type to which the input arrays are cast before the calculation is
        performed.
    weights : array_like, optional
        The `weights` array must be broadcastable to the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    gmean : ndarray
        See `dtype` parameter above.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    hmean : Harmonic mean

    References
    ----------
    .. [1] "Weighted Geometric Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Weighted_geometric_mean.

    Examples
    --------
    >>> from scipy.stats import gmean
    >>> gmean([1, 4])
    2.0
    >>> gmean([1, 2, 3, 4, 5, 6, 7])
    3.3800151591412964
    >>> gmean([1, 4, 7], weights=[3, 1, 3])
    2.80668351922014

    """

    a = np.asarray(a, dtype=dtype)

    if weights is not None:
        weights = np.asarray(weights, dtype=dtype)

    with np.errstate(divide="ignore"):
        log_a = np.log(a)

    return np.exp(np.average(log_a, axis=axis, weights=weights))
