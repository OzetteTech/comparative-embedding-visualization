import dataclasses
import itertools
from collections.abc import Callable
from typing import Union

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd

from embcomp.logo import AnnotationLogo, Labeler
from embcomp.metrics import jaccard_groupwise, jaccard_pointwise

Coordinates = npt.ArrayLike
KnnIndices = npt.NDArray[np.int_]
Labels = pd.Series
Distances = npt.NDArray[np.float_]

DistanceMetric = Callable[[KnnIndices, KnnIndices], Distances]

NON_ROBUST_LABEL = "0_0_0_0_0"


@dataclasses.dataclass
class Embedding:
    coords: Coordinates
    knn_indices: KnnIndices
    labels: Labels
    robust: Union[npt.NDArray[np.bool_], None] = None


@dataclasses.dataclass
class PairwiseComponent:
    scatter: jscatter.Scatter
    logo: AnnotationLogo
    embedding: Embedding

    def __post_init__(self):
        self.link = ipywidgets.link(
            (self.scatter.widget, "selection"), (self.logo, "selection")
        )
        self._by = "labels"
        self.labels = self.embedding.labels
        self.color_by_labels()

    def show(self):
        return ipywidgets.VBox([self.scatter.show(), self.logo])

    def color_by_labels(self):
        self._by = "labels"
        self.scatter.color(by="label", map=self._colormap)

    def color_by_distances(self):
        self._by = "distances"
        self.scatter.color(by="dist", map="plasma")

    @property
    def _data(self):
        assert self.scatter._data is not None
        return self.scatter._data

    @property
    def distances(self):
        return self._data["dist"]

    @distances.setter
    def distances(self, distances: npt.NDArray[np.float_]):
        self._data["dist"] = distances
        if self._by == "distances":
            self.color_by_distances()

    @property
    def labels(self):
        return self._data["labels"]

    @labels.setter
    def labels(self, labels: npt.ArrayLike):
        self.logo.labels = labels
        if self.embedding.robust is not None:
            labels = np.where(
                self.embedding.robust,
                labels,
                NON_ROBUST_LABEL,
            )
        robust_labels = pd.Series(labels, dtype="category")
        self._colormap = dict(
            zip(robust_labels.cat.categories, itertools.cycle(jscatter.glasbey_dark))
        )
        self._colormap.update({NON_ROBUST_LABEL: "#333333"})
        self._data["label"] = robust_labels
        if self._by == "labels":
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

    color_by = ipywidgets.RadioButtons(
        options=["label", "metric"],
        value="label",
        description="color by:",
    )

    # TODO: dynamic
    left.distances = right.distances = jaccard_pointwise(a.knn_indices, b.knn_indices)

    def on_color_by_change(change):
        if change.new == "metric":
            left.color_by_distances()
            right.color_by_distances()
        else:
            left.color_by_labels()
            right.color_by_labels()

    color_by.observe(on_color_by_change, names="value")

    link = ipywidgets.link((left.logo, "selection"), (right.logo, "selection"))

    selection_type = ipywidgets.RadioButtons(
        options=["synced", "neighbors"],
        value="synced",
        description="selection",
    )

    def handle_selection_type_change(change):
        nonlocal link
        if change.new == "synced":
            link.link()
            link.link()
        else:
            link.unlink()
            link.unlink()

    selection_type.observe(handle_selection_type_change, names="value")  # type: ignore

    def on_labels_change(change):
        left.labels = change.new
        right.labels = change.new

    labeler.observe(on_labels_change, names="labels")

    main = ipywidgets.GridBox(
        children=[left.show(), right.show()],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr",
            grid_template_rows=f"{row_height * 2}px",
        ),
    )

    header = ipywidgets.HBox([label_slider, selection_type, color_by])
    return ipywidgets.VBox([header, main]), left.scatter
