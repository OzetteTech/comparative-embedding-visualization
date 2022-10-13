import numpy as np
import numpy.typing as npt
import pandas as pd

import numba as nb
from sklearn.neighbors import NearestNeighbors


@nb.njit
def jaccard_pointwise(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray
) -> npt.NDArray:
    dist = np.zeros(len(knn_indices0))
    for i in range(len(dist)):
        A = set(knn_indices0[i])
        B = set(knn_indices1[i])
        intersect = A.intersection(B)
        dist[i] = len(intersect) / (len(A) + len(B) - len(intersect))
    return dist


def kneighbors(X: npt.ArrayLike, k: int) -> npt.NDArray:
    # first neighbor is always self
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    ind = nbrs.kneighbors(X, return_distance=False)
    return ind[:, 1:]


@nb.njit
def _jaccard_group(
    grp_knn_indices0: npt.NDArray,
    grp_knn_indices1: npt.NDArray,
):
    num: set[int] = set()
    denom: set[int] = set()
    for i in range(len(grp_knn_indices0)):
        A = set(grp_knn_indices0[i])
        B = set(grp_knn_indices1[i])
        num.update(A.intersection(B))
        denom.update(A.union(B))
    return num, denom


def jaccard_groupwise(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray, labels: npt.NDArray
):
    groups = np.unique(labels)
    distances = []
    for group in groups:
        mask = labels == group
        num, denom = _jaccard_group(knn_indices0[mask], knn_indices1[mask])
        distances.append(len(num) / len(denom))
    return pd.Series(distances, index=groups)


def jaccard_pointwise_average(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray, labels: npt.NDArray
) -> pd.Series:
    scores = jaccard_pointwise(knn_indices0, knn_indices1)
    index = pd.Series(labels, name="label")
    return pd.Series(scores, index).groupby("label").mean()  # type: ignore

@nb.njit
def _count_labels(knn_indices: npt.NDArray, codes: npt.NDArray) -> npt.NDArray:
    dist = np.zeros((len(knn_indices), len(np.unique(codes))))
    for i in range(len(knn_indices)):
        for code in codes[knn_indices[i]]:
            dist[i, code] += 1
    return dist


def count_labels(knn_indices: npt.NDArray, labels: pd.Series) -> npt.NDArray:
    return _count_labels(knn_indices, np.array(labels.cat.codes))
