from __future__ import annotations

import typing

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import traitlets

from uuid import uuid4

from cev._embedding import Embedding
from cev._widget_utils import create_colormaps, link_widgets, robust_labels
from cev.components import MarkerCompositionLogo

_LABEL_COLUMN = "label"
_ROBUST_LABEL_COLUMN = "robust_label"
_DISTANCE_COLUMN = "distance"


class EmbeddingWidgetCollection(traitlets.HasTraits):
    inverted = traitlets.Bool(False)

    def __init__(
        self,
        labels: pd.Series,
        categorial_scatter: jscatter.Scatter,
        metric_scatter: jscatter.Scatter,
        logo: MarkerCompositionLogo,
        labeler: typing.Callable[[npt.ArrayLike], pd.Series],
    ):
        self.categorial_scatter = categorial_scatter
        self.metric_scatter = metric_scatter
        self.logo = logo
        self._labeler = labeler
        self.metric_color_options: tuple[str, str, list[int], tuple] = (
            "viridis",
            "viridis_r",
            [0, 1],
            ("min", "max", "value"),
        )

        self.labels = labels
        self.distances = 0  # type: ignore
        self.colormap = create_colormaps(self.robust_labels.cat.categories)

        ipywidgets.dlink(
            source=(self.categorial_scatter.widget, "selection"),
            target=(self.logo, "counts"),
            transform=self.label_counts,
        )

    def label_counts(self, ilocs: None | np.ndarray = None) -> dict:
        labels = self.labels if ilocs is None else self.labels.iloc[ilocs]
        return {k: int(v) for k, v in labels.value_counts().items()}

    @property
    def _data(self) -> pd.DataFrame:
        assert self.categorial_scatter._data is self.metric_scatter._data
        assert self.categorial_scatter._data is not None
        return self.categorial_scatter._data

    @classmethod
    def from_embedding(
        cls,
        emb: Embedding,
        background_color: str = "black",
        axes: bool = False,
        opacity_unselected: float = 0.05,
        **kwargs,
    ):
        X = np.array(emb.coords)
        data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})

        categorial_scatter, metric_scatter = (
            jscatter.Scatter(
                data=data,
                x="x",
                y="y",
                background_color=background_color,
                axes=axes,
                opacity_by='density',
                lasso_initiator=False,
                **kwargs,
            )
            for _ in range(2)
        )
        # link the plots together with js
        link_widgets(
            (categorial_scatter.widget, "selection"),
            (metric_scatter.widget, "selection"),
        )

        return cls(
            labels=emb.labels,
            categorial_scatter=categorial_scatter,
            metric_scatter=metric_scatter,
            logo=MarkerCompositionLogo(),
            labeler=lambda labels: robust_labels(labels, emb.robust),
        )

    @property
    def labels(self) -> pd.Series:
        return self._data[_LABEL_COLUMN]

    @property
    def robust_labels(self) -> pd.Series:
        return self._data[_ROBUST_LABEL_COLUMN]

    @labels.setter
    def labels(self, labels: npt.ArrayLike):
        self._data[_LABEL_COLUMN] = pd.Series(np.asarray(labels), dtype="category")
        self._data[_ROBUST_LABEL_COLUMN] = pd.Series(
            np.asarray(self._labeler(labels)), dtype="category"
        )
        self.logo.counts = self.label_counts(self.categorial_scatter.widget.selection)

    @traitlets.observe("inverted")
    def _update_metric_scatter(self, *args, **kwargs):
        cmap, cmapr, norm, labeling = self.metric_color_options
        self.metric_scatter.color(
            by=_DISTANCE_COLUMN,
            map=cmapr if self.inverted else cmap,
            norm=norm,
            labeling=labeling,
        )
        self.metric_scatter.legend(True)

    def _update_categorial_scatter(self, *args, **kwargs):
        self.categorial_scatter.legend(False)
        self.categorial_scatter.color(by=_ROBUST_LABEL_COLUMN, map=self._colormap)

    @property
    def distances(self) -> pd.Series:
        return self._data[_DISTANCE_COLUMN]

    @distances.setter
    def distances(self, distances: npt.NDArray[np.float_]):
        self._data[_DISTANCE_COLUMN] = distances
        self._update_metric_scatter()

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, cmap: dict):
        self._colormap = cmap
        self._update_categorial_scatter()

    @property
    def scatters(self):
        yield self.categorial_scatter
        yield self.metric_scatter

    def show(self, row_height: int | None = None, **kwargs):
        widgets = []
        
        uuid = uuid4().hex

        for scatter in self.scatters:
            if row_height is not None:
                scatter.height(row_height)
            widget = scatter.show()
            widget.layout = {"margin": "0 0 2px 0"}
            widgets.append(widget)
            scatter.widget.view_sync = uuid

        widgets.append(self.logo)

        return ipywidgets.VBox(widgets, **kwargs)

    def zoom(self, to: None | npt.NDArray = None):
        if to is not None:
            to = to if len(to) > 0 else None
        for s in self.scatters:
            s.zoom(to=to)

    def zoom(self, to: None | npt.NDArray = None):
        if to is not None:
            to = to if len(to) > 0 else None
        for s in self.scatters:
            s.zoom(to=to)
