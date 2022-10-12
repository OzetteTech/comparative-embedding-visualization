import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
from numba import njit
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


@njit
def _jaccard(A: npt.NDArray, B: npt.NDArray) -> npt.NDArray:
    dist = np.zeros(len(A))
    for i, rows in enumerate(zip(A, B)):
        a = set(rows[0])
        b = set(rows[1])
        intersect = a.intersection(b)
        dist[i] = len(intersect) / (len(a) + len(b) - len(intersect))
    return dist


def kneighbors(X: npt.ArrayLike, k: int) -> npt.NDArray:
    # first neighbor is always self
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    ind = nbrs.kneighbors(X, return_distance=False)
    return ind[:, 1:]


def jaccard_pointwise(X0: npt.ArrayLike, X1: npt.ArrayLike, k: int) -> npt.NDArray:
    return _jaccard(kneighbors(X0, k=k), kneighbors(X1, k=k))


def jaccard_groupwise(
    X0: npt.ArrayLike, X1: npt.ArrayLike, labels: npt.NDArray, k: int
) -> pd.DataFrame:

    groups = {label: (set(), set()) for label in np.unique(labels)}

    for (A, B, label) in zip(
        kneighbors(X0, k=k),
        kneighbors(X1, k=k),
        labels,
    ):
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
    X0: npt.ArrayLike, X1: npt.ArrayLike, labels: npt.ArrayLike, k: int
) -> pd.DataFrame:
    score = _jaccard(kneighbors(X0, k=k), kneighbors(X1, k=k))
    return (
        pd.DataFrame({"labels": labels, "score": score})
        .groupby("labels")
        .mean()
        .reset_index()
    )

def count_labels(X: npt.NDArray, labels: pd.Series, k: int) -> npt.NDArray:
    dist = np.zeros((len(X), len(np.unique(labels))))
    for i, ind in enumerate(tqdm(kneighbors(X, k=k))):
        dist[i] = labels[ind].value_counts(normalize=True, sort=False)  # type: ignore
    return dist
