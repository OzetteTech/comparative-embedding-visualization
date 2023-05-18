from __future__ import annotations

import typing
from uuid import uuid4

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import traitlets

from cev._embedding import Embedding
from cev._widget_utils import create_colormaps, link_widgets, robust_labels
from cev.components import MarkerCompositionLogo

_LABEL_COLUMN = "label"
_ROBUST_LABEL_COLUMN = "robust_label"
_DISTANCE_COLUMN = "distance"


class EmbeddingWidgetCollection(traitlets.HasTraits):
    inverted = traitlets.Bool(default_value=False)
    labels = traitlets.Any()
    distances = traitlets.Any()
    colormap = traitlets.Dict()

    def __init__(
        self,
        labels: pd.Series,
        categorical_scatter: jscatter.Scatter,
        metric_scatter: jscatter.Scatter,
        logo: MarkerCompositionLogo,
        labeler: typing.Callable[[npt.ArrayLike], pd.Series],
    ):
        super().__init__()
        self.categorical_scatter = categorical_scatter
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
        self.distances = pd.Series(0.0, index=self._data.index, dtype="float64")
        self.colormap = create_colormaps(self.robust_labels.cat.categories)

        ipywidgets.dlink(
            source=(self.categorical_scatter.widget, "selection"),
            target=(self.logo, "counts"),
            transform=self.label_counts,
        )

    def label_counts(self, ilocs: None | np.ndarray = None) -> dict:
        labels = self.labels if ilocs is None else self.labels.iloc[ilocs]
        return {k: int(v) for k, v in labels.value_counts().items()}

    @traitlets.validate("labels")
    def _validate_labels(self, proposal: object):
        assert isinstance(proposal.value, pd.Series)
        # convert to category if not already
        return (
            proposal.value
            if proposal.value.dtype.name == "category"
            else proposal.value.astype("category")
        )

    @property
    def _data(self) -> pd.DataFrame:
        assert self.categorical_scatter._data is self.metric_scatter._data
        assert self.categorical_scatter._data is not None
        return self.categorical_scatter._data

    @traitlets.observe("labels")
    def _on_labels_change(self, change):
        labels = change.new
        self._data[_LABEL_COLUMN] = pd.Series(np.asarray(labels), dtype="category")
        self._data[_ROBUST_LABEL_COLUMN] = pd.Series(
            np.asarray(self._labeler(labels)), dtype="category"
        )
        self.logo.counts = self.label_counts(self.categorical_scatter.widget.selection)
        self.has_markers = "+" in self._data[_LABEL_COLUMN][0]

    @traitlets.validate("distances")
    def _validate_distances(self, proposal: object):
        assert isinstance(proposal.value, pd.Series)
        assert proposal.value.dtype == "float64"
        return proposal.value

    @traitlets.observe("distances")
    def _on_distances_change(self, change):
        self._data[_DISTANCE_COLUMN] = change.new.values
        self._update_metric_scatter()

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

    @traitlets.observe("colormap")
    def _update_categorical_scatter(self, *args, **kwargs):
        self.categorical_scatter.legend(False)
        self.categorical_scatter.color(by=_ROBUST_LABEL_COLUMN, map=self.colormap)

    @classmethod
    def from_embedding(
        cls,
        emb: Embedding,
        background_color: str = "black",
        axes: bool = False,
        **kwargs,
    ):
        X = np.array(emb.coords)
        data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})

        categorical_scatter, metric_scatter = (
            jscatter.Scatter(
                data=data,
                x="x",
                y="y",
                background_color=background_color,
                axes=axes,
                opacity_by="density",
                lasso_initiator=False,
                **kwargs,
            )
            for _ in range(2)
        )
        # link the plots together with js
        link_widgets(
            (categorical_scatter.widget, "selection"),
            (metric_scatter.widget, "selection"),
        )

        return cls(
            labels=emb.labels,
            categorical_scatter=categorical_scatter,
            metric_scatter=metric_scatter,
            logo=MarkerCompositionLogo(),
            labeler=lambda labels: robust_labels(labels, emb.robust),
        )

    @property
    def robust_labels(self) -> pd.Series:
        return self._data[_ROBUST_LABEL_COLUMN]

    @property
    def scatters(self):
        yield self.categorical_scatter
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

        if self.has_markers:
            widgets.append(self.logo)

        return ipywidgets.VBox(widgets, **kwargs)

    def zoom(self, to: None | npt.NDArray = None):
        if to is not None:
            to = to if len(to) > 0 else None
        for s in self.scatters:
            s.zoom(to=to)

    def __hash__(self):
        # Warning: this is a hack! You should probably not rely on this hash
        # unless you know what you're doing.
        #
        # Creates a unique hash for the current "state" of this object
        # to make sure that functools caching works correctly.
        # See the usage in cev._compare_metrics_dropdown
        obj_id = str(id(self))
        categories = ",".join(self.labels.cat.categories.to_list())
        return hash(obj_id + categories)
