import dataclasses
import itertools
from collections.abc import Callable
from typing import Union

import ipywidgets
import jscatter
import numpy as np
import numpy.linalg as nplg
import numpy.typing as npt
import pandas as pd
import traitlets

import embcomp.metrics as metrics
from embcomp.logo import AnnotationLogo, Labeler, label_parts

Coordinates = npt.ArrayLike
KnnIndices = npt.NDArray[np.int_]
Labels = pd.Series
Distances = npt.NDArray[np.float_]

DistanceMetric = Callable[[KnnIndices, KnnIndices], Distances]

NON_ROBUST_LABEL = "0_0_0_0_0"


def rowise_cosine_similarity(X0: npt.ArrayLike, X1: npt.ArrayLike):
    """Computes the cosine similary per row of two equally shaped 2D matrices."""
    return np.sum(X0 * X1, axis=1) / (nplg.norm(X0, axis=1) * nplg.norm(X1, axis=1))


def robust_labels(
    labels: npt.ArrayLike, robust: Union[npt.NDArray[np.bool_], None] = None
):
    if robust is not None:
        labels = np.where(
            robust,
            labels,
            NON_ROBUST_LABEL,
        )
    return pd.Series(labels, dtype="category")


@dataclasses.dataclass
class Embedding:
    coords: Coordinates
    knn_indices: KnnIndices
    labels: Labels
    robust: Union[npt.NDArray[np.bool_], None] = None


LABEL_COLUMN = "_label"
DISTANCE_COLUMN = "_distance"


class PairwiseComponent(traitlets.HasTraits):
    inverted = traitlets.Bool(False)

    def __init__(
        self, scatter: jscatter.Scatter, logo: AnnotationLogo, embedding: Embedding
    ):
        self.scatter = scatter
        self.logo = logo
        self.embedding = embedding

        self.link = ipywidgets.link(
            (self.scatter.widget, "selection"), (self.logo, "selection")
        )
        self._by = LABEL_COLUMN
        self.labels = self.embedding.labels
        self.color_by_labels()

    def show(self):
        return ipywidgets.VBox([self.scatter.show(), self.logo])

    def color_by_labels(self):
        self._by = LABEL_COLUMN
        self.scatter.legend(False)
        self.scatter.color(by=LABEL_COLUMN, map=self._colormap)

    def color_by_distances(self):
        self._by = DISTANCE_COLUMN
        cmap = "viridis_r" if self.inverted else "viridis"
        self.scatter.color(by=DISTANCE_COLUMN, map=cmap)
        self.scatter.legend(True)

    @traitlets.observe("inverted")
    def _update_colormap(self, _change):
        if self._by == DISTANCE_COLUMN:
            self.color_by_distances()

    @property
    def _data(self):
        assert self.scatter._data is not None
        return self.scatter._data

    @property
    def distances(self):
        return self._data[DISTANCE_COLUMN]

    @distances.setter
    def distances(self, distances: npt.NDArray[np.float_]):
        self._data[DISTANCE_COLUMN] = distances
        if self._by == DISTANCE_COLUMN:
            self.color_by_distances()

    @property
    def labels(self) -> pd.Series:
        return self._data[LABEL_COLUMN]

    @labels.setter
    def labels(self, labels: npt.ArrayLike):
        self.logo.labels = labels
        rlabels = robust_labels(labels, self.embedding.robust)
        self._colormap = dict(
            zip(
                rlabels.cat.categories,
                itertools.cycle(jscatter.glasbey_dark),
            )
        )
        if self.embedding.robust is not None:
            self._colormap.update({NON_ROBUST_LABEL: "#333333"})
        self._data[LABEL_COLUMN] = rlabels
        if self._by == LABEL_COLUMN:
            self.color_by_labels()


def assert_pointwise_correspondence(a: Embedding, b: Embedding):
    assert np.array_equal(a.labels, b.labels) and (
        (a.robust is None and b.robust is None)
        or (
            a.robust is not None
            and b.robust is not None
            and np.array_equal(a.robust, b.robust)
        )
    ), "label-only correspondence not currently supported."


def pairwise(a: Embedding, b: Embedding, row_height: int = 600):
    assert_pointwise_correspondence(a, b)

    # can use one set of labels since they share correspondence
    labeler = Labeler(a.labels)

    label_slider = ipywidgets.IntSlider(
        description="label level:",
        min=0,
        max=labeler.levels,
    )

    ipywidgets.link((label_slider, "value"), (labeler, "level"))

    left, right = (
        PairwiseComponent(
            scatter=jscatter.Scatter(
                data=pd.DataFrame(
                    {"x": np.array(emb.coords)[:, 0], "y": np.array(emb.coords)[:, 1]}
                ),
                x="x",
                y="y",
                legend=True,
                background_color="black",
                axes=False,
                opacity_unselected=0.05,
                height=row_height,
            ),
            logo=AnnotationLogo(emb.labels),
            embedding=emb,
        )
        for emb in (a, b)
    )

    def jaccard_pointwise():
        distances = metrics.jaccard_pointwise(
            left.embedding.knn_indices,
            right.embedding.knn_indices,
        )
        return distances, distances

    def jaccard_groupwise():
        grp_distances = metrics.jaccard_groupwise(
            left.embedding.knn_indices,
            right.embedding.knn_indices,
            np.array(left.labels),
        )
        distances = left.labels.map(grp_distances)
        return distances, distances

    def counts():
        acounts = metrics.count_neighbor_labels(
            left.embedding.knn_indices,
            left.labels,
        )
        bcounts = metrics.count_neighbor_labels(
            right.embedding.knn_indices,
            right.labels,
        )
        return acounts, bcounts

    def point_label():
        acounts, bcounts = counts()
        distances = rowise_cosine_similarity(acounts, bcounts)
        return distances, distances

    def label_label():
        acounts, bcounts = counts()

        aindex = pd.Series(left.labels, dtype="category", name="label")
        a = pd.DataFrame(acounts, index=aindex).groupby("label").sum()

        bindex = pd.Series(right.labels, dtype="category", name="label")
        b = pd.DataFrame(bcounts, index=bindex).groupby("label").sum()

        grp_distances = rowise_cosine_similarity(a, b)

        return (
            left.labels.map(grp_distances).astype(float),
            right.labels.map(grp_distances).astype(float),
        )

    # METRIC START
    metric = ipywidgets.Dropdown(
        options=[
            ("Label-Label distance", label_label),
            ("Point-Label distance", point_label),
            ("Jaccard groupwise", jaccard_groupwise),
            ("Jaccard", jaccard_pointwise),
        ],
        value=label_label,
        description="metric: ",
    )

    left.distances, right.distances = metric.value()

    def on_metric_change(change):
        compute_metric = change.new
        left.distances, right.distances = compute_metric()

    metric.observe(on_metric_change, names="value")
    # METRIC END

    # COLOR START
    inverted = ipywidgets.Checkbox(False, description="invert colormap")
    ipywidgets.link((left, "inverted"), (inverted, "value"))
    ipywidgets.link((right, "inverted"), (inverted, "value"))

    color_by = ipywidgets.RadioButtons(
        options=["label", "metric"],
        value="label",
        description="color by:",
    )

    def on_color_by_change(change):
        if change.new == "metric":
            left.color_by_distances()
            right.color_by_distances()
        else:
            left.color_by_labels()
            right.color_by_labels()

    color_by.observe(on_color_by_change, names="value")
    # COLOR END

    # SELECTION START
    unlink: Callable[[], None] = lambda: None

    def sync():
        nonlocal unlink
        unlink()
        selection_link = ipywidgets.link(
            source=(left.logo, "selection"), target=(right.logo, "selection")
        )
        hovering_link = ipywidgets.link(
            source=(left.scatter.widget, "hovering"),
            target=(right.scatter.widget, "hovering"),
        )

        def unlink_all():
            selection_link.unlink()
            hovering_link.unlink()

        unlink = unlink_all

    def expand_neighbors():
        nonlocal unlink
        if unlink:
            unlink()

        def transform(selection):
            return left.embedding.knn_indices[selection].ravel()

        link = ipywidgets.link(
            source=(left.logo, "selection"),
            target=(right.logo, "selection"),
            transform=(transform, transform),
        )

        unlink = link.unlink

    selection_type = ipywidgets.RadioButtons(
        options=[("synced", sync), ("neighbors", expand_neighbors)],
        value=sync,
        description="selection",
    )

    selection_type.observe(lambda change: change.new(), names="value")  # type: ignore
    sync()
    # SELECTION END

    # LABELS START
    active_labels = ipywidgets.Label("markers: ")

    def on_labels_change(change):
        left.labels = change.new
        right.labels = change.new
        active_labels.value = "markers: " + " ".join(
            l[:-1] for l in label_parts(change.new[0])
        )

    labeler.observe(on_labels_change, names="labels")
    # LABELS END

    header = ipywidgets.HBox(
        [
            ipywidgets.VBox([label_slider, inverted]),
            selection_type,
            ipywidgets.VBox([color_by, metric]),
        ],
        layout=ipywidgets.Layout(width="80%"),
    )

    header = ipywidgets.VBox([header, active_labels])

    main = ipywidgets.GridBox(
        children=[left.show(), right.show()],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr",
            grid_template_rows=f"{row_height * 2}px",
        ),
    )

    label_slider.value = labeler.levels
    return ipywidgets.VBox([header, main])
