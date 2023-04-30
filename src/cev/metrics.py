from __future__ import annotations

import cev_metrics
import cev_metrics.py
import numpy as np
import numpy.linalg as nplg
import numpy.typing as npt
import pandas as pd


def rowise_cosine_similarity(X0: npt.ArrayLike, X1: npt.ArrayLike):
    """Computes the cosine similary per row of two equally shaped 2D matrices."""
    return np.sum(X0 * X1, axis=1) / (nplg.norm(X0, axis=1) * nplg.norm(X1, axis=1))

def confusion(df: pd.DataFrame, counts: bool = False, py: bool = False):
    if py:
        return cev_metrics.py.confusion(df, counts)
    cats = df["label"].cat.categories
    mat = pd.DataFrame(cev_metrics.confusion(df), index=cats, columns=cats)
    if counts:
        return mat
    normed = (mat / mat.sum(axis=1)).to_numpy()
    return pd.Series(1 - normed.diagonal(), index=mat.index, name="confusion")
