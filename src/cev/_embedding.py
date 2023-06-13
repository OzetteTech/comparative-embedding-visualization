from __future__ import annotations

import dataclasses
import typing

import pandas as pd

from cev._widget_utils import parse_label

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

NON_ROBUST_LABEL = "0_0_0_0_0"


@dataclasses.dataclass
class Embedding:
    coords: npt.ArrayLike
    labels: pd.Series
    robust: npt.NDArray[np.bool_] | None = None

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(
            coords=df[["x", "y"]].values,
            labels=df["label"],
            robust=df["robust"] if "robust" in df else None,
        )

    @classmethod
    def from_ozette(cls, df: pd.DataFrame, **kwargs):
        coords, labels, robust = _prepare_ozette(df, **kwargs)
        return cls(coords=coords, labels=labels, robust=robust)

    def widgets(self, **kwargs):
        from ._embedding_widget import EmbeddingWidgetCollection

        return EmbeddingWidgetCollection.from_embedding(self, **kwargs)


def _prepare_ozette(df: pd.DataFrame, robust_only: bool = True):
    # ISMB data
    if "cellType" in df.columns:
        robust = (df["cellType"] != NON_ROBUST_LABEL).to_numpy()
        if robust_only:
            df = df[robust].reset_index(drop=True)
            robust = None

        coords = df[["x", "y"]].to_numpy()
        labels = df["complete_faust_label"].to_numpy()

    else:
        robust = (df["faustLabels"] != NON_ROBUST_LABEL).to_numpy()
        representative_label = df["faustLabels"][robust].iloc[0]

        if robust_only:
            df = df[robust].reset_index(drop=True)
            labels = df["faustLabels"].to_numpy()
            robust = None
        else:
            labels = pd.Series("", index=df.index)
            for marker in parse_label(representative_label):
                marker_annotation = marker.name + df[f"{marker.name}_faust_annotation"]
                labels += marker_annotation

    coords = df[["umapX", "umapY"]].to_numpy()
    labels = pd.Series(labels, dtype="category")

    return coords, labels, robust
