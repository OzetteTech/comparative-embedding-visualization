from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _validate_df(obj: pd.DataFrame):
    if "x" not in obj.columns or "y" not in obj.columns or "label" not in obj.columns:
        raise AttributeError("Must have 'x' and 'y' and 'label' columns.")


def kneighbors(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    return nn.kneighbors(return_distance=False)


def count_first(df: pd.DataFrame, n: int = 0, kind: Literal["set", "sum"] = "set"):
    _validate_df(df)

    # only compute the complete neighborhood graph up until the largest class size
    largest_category_size = df.label.value_counts().sort_values()[-1]
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

    if kind == "set":
        # don't double count points
        indices_bags = {k: np.unique(v) for k, v in indices_bags.items()}

    index = pd.Series(indices_bags.keys(), name="label", dtype="category")
    out = pd.DataFrame(
        np.stack([df.label.values[v].value_counts() for v in indices_bags.values()]),
        index=index,
    )
    out.columns = index.values
    return out
