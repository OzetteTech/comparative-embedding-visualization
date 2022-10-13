import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


@njit
def jaccard_pointwise(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray
) -> npt.NDArray:
    dist = np.zeros(len(knn_indices0))
    for i, (A, B) in enumerate(zip(knn_indices0, knn_indices1)):
        a = set(A)
        b = set(B)
        intersect = a.intersection(b)
        dist[i] = len(intersect) / (len(a) + len(b) - len(intersect))
    return dist


def kneighbors(X: npt.ArrayLike, k: int) -> npt.NDArray:
    # first neighbor is always self
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    ind = nbrs.kneighbors(X, return_distance=False)
    return ind[:, 1:]


def jaccard_groupwise(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray, labels: npt.NDArray
) -> pd.DataFrame:

    groups = {label: (set(), set()) for label in np.unique(labels)}

    for (A, B, label) in zip(knn_indices0, knn_indices1, labels):
        A = set(A)
        B = set(B)
        groups[label][0].update(A.intersection(B))
        groups[label][1].update(A.union(B))

    return pd.DataFrame.from_records(
        (
            (label, len(intersection) / len(union))
            for label, (intersection, union) in groups.items()
        ),
        columns=["label", "score"],
    )


def jaccard_pointwise_average(
    knn_indices0: npt.NDArray, knn_indices1: npt.NDArray, labels: npt.ArrayLike
) -> pd.DataFrame:
    score = jaccard_pointwise(knn_indices0, knn_indices1)
    return (
        pd.DataFrame({"labels": labels, "score": score})
        .groupby("labels")
        .mean()
        .reset_index()
    )


def count_labels(knn_indices: npt.NDArray, labels: pd.Series) -> npt.NDArray:
    dist = np.zeros((len(knn_indices), len(np.unique(labels))))
    for i, ind in enumerate(tqdm(knn_indices)):
        dist[i] = labels[ind].value_counts(normalize=True, sort=False)  # type: ignore
    return dist
