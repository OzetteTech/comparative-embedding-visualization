import numba as nb
import numpy as np
import numpy.linalg as nplg
import numpy.typing as npt
import pandas as pd
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


def kneighbors(X: npt.NDArray, k: int) -> npt.NDArray:
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    return nbrs.kneighbors(X, return_distance=False)


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
def _count_neighbor_labels(knn_indices: npt.NDArray, codes: npt.NDArray) -> npt.NDArray:
    dist = np.zeros((len(knn_indices), len(np.unique(codes))))
    for i in range(len(knn_indices)):
        for code in codes[knn_indices[i]]:
            dist[i, code] += 1
    return dist


def count_neighbor_labels(
    knn_indices: npt.NDArray[np.int_], labels: pd.Series
) -> npt.NDArray:
    return _count_neighbor_labels(knn_indices, np.array(labels.cat.codes))


def label_label_sets(knn_indices: npt.NDArray[np.int_], labels: npt.NDArray):
    return {label: knn_indices[labels == label].unique() for label in np.unique(labels)}


def rowise_cosine_similarity(X0: npt.ArrayLike, X1: npt.ArrayLike):
    """Computes the cosine similary per row of two equally shaped 2D matrices."""
    return np.sum(X0 * X1, axis=1) / (nplg.norm(X0, axis=1) * nplg.norm(X1, axis=1))
