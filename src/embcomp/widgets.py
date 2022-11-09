import dataclasses
import itertools
from typing import Callable, Union

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import traitlets
from scipy.spatial import Delaunay

import embcomp.metrics as metrics
from embcomp.logo import AnnotationLogo, parse_label, trim_label_series

Coordinates = npt.ArrayLike
KnnIndices = npt.NDArray[np.int_]
Labels = pd.Series
Distances = npt.NDArray[np.float_]

NON_ROBUST_LABEL = "0_0_0_0_0"


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
        # TODO: fix me. hack to initialize the colormap
        self._colormap, _ = create_shared_colormaps(self, self)
        self.color_by_labels()

    def show(self):
        return ipywidgets.VBox([self.scatter.show(), self.logo])

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, cmap):
        self._colormap = cmap
        if self._by == LABEL_COLUMN:
            self.color_by_labels()

    def color_by_labels(self):
        self._by = LABEL_COLUMN
        self.scatter.legend(False)
        self.scatter.color(by=LABEL_COLUMN, map=self._colormap)

    def color_by_distances(self):
        self._by = DISTANCE_COLUMN
        cmap = "viridis_r" if self.inverted else "viridis"
        self.scatter.color(by=DISTANCE_COLUMN, map=cmap, norm=[0, 1])
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
        self._data[LABEL_COLUMN] = robust_labels(labels, self.embedding.robust)
        if self._by == LABEL_COLUMN:
            # self.color_by_labels()
            ...


def create_shared_colormaps(
    a: PairwiseComponent, b: PairwiseComponent
) -> tuple[dict, dict]:
    # create unified colormap
    lookup = dict(
        zip(
            set(a.labels.cat.categories).union(b.labels.cat.categories),
            itertools.cycle(jscatter.glasbey_dark),
        )
    )

    # force non-robust to be grey
    lookup[NON_ROBUST_LABEL] = "#333333"

    # create separate colormaps for each component
    return tuple(
        {label: lookup[label] for label in cmp.labels.cat.categories} for cmp in (a, b)
    )


def has_pointwise_correspondence(a: Embedding, b: Embedding) -> bool:
    return np.array_equal(a.labels, b.labels) and (
        (a.robust is None and b.robust is None)
        or (
            a.robust is not None
            and b.robust is not None
            and np.array_equal(a.robust, b.robust)
        )
    )


def compare(a: Embedding, b: Embedding, row_height: int = 600):
    pointwise_correspondence = has_pointwise_correspondence(a, b)

    # representative label
    max_label_level = len(parse_label(a.labels.values[0])) - 1

    label_slider = ipywidgets.IntSlider(
        description="label level:",
        value=max_label_level,
        min=0,
        max=max_label_level,
    )

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
        alabels = pd.Series(left.logo.labels, dtype="category", name="label")
        acounts = metrics.count_neighbor_labels(
            left.embedding.knn_indices,
            alabels,
        )
        blabels = pd.Series(right.logo.labels, dtype="category", name="label")
        bcounts = metrics.count_neighbor_labels(
            right.embedding.knn_indices,
            blabels,
        )
        return (
            pd.DataFrame(acounts, index=alabels),
            pd.DataFrame(bcounts, index=blabels),
        )

    def point_label():
        acounts, bcounts = counts()
        distances = metrics.rowise_cosine_similarity(acounts.values, bcounts.values)
        return distances, distances

    def label_counts():
        acounts, bcounts = counts()
        a = acounts.groupby("label").sum()
        a.columns = a.index
        b = bcounts.groupby("label").sum()
        b.columns = b.index
        return (a, acounts.index), (b, bcounts.index)

    def label_label():
        a, b = label_counts()

        # np.fill_diagonal(a.values, 0)
        # np.fill_diagonal(b.values, 0)

        # np.fill_diagonal(a.values, a.values.max(axis=0))
        # np.fill_diagonal(b.values, b.values.max(axis=0))

        # subset labels only by those represented in both sets
        overlap = a[0].index.intersection(b[0].index)
        asubset = a[0].loc[overlap, overlap]
        bsubset = b[0].loc[overlap, overlap]

        # dict of { <label>: <similarity> }
        grp_distances = pd.Series(
            metrics.rowise_cosine_similarity(asubset.values, bsubset.values),
            index=overlap,
        )

        return (
            a[1].map(grp_distances).astype(float),
            b[1].map(grp_distances).astype(float),
        )

    # METRIC START

    metric_options: list[tuple[str, Callable]] = [
        ("Label-Label similarity", label_label)
    ]

    if pointwise_correspondence:
        metric_options.extend(
            [
                ("Point-Label similarity", point_label),
                ("Groupwise Jaccard index", jaccard_groupwise),
                ("Jaccard index", jaccard_pointwise),
            ]
        )

    metric = ipywidgets.Dropdown(
        options=metric_options,
        value=label_label,
        description="metric: ",
    )

    def on_metric_change(change):
        left.distances, right.distances = change.new()

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

    def independent():
        nonlocal unlink
        unlink()

    # requires point-point correspondence
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

    # requires point-point correspondence
    def expand_neighbors():
        nonlocal unlink
        unlink()

        def transform(cmp: PairwiseComponent):
            def _expand_neighbors(selection):
                return cmp.embedding.knn_indices[selection].ravel()

            return _expand_neighbors

        link = ipywidgets.link(
            source=(left.logo, "selection"),
            target=(right.logo, "selection"),
            transform=(transform(left), transform(right)),
        )

        unlink = link.unlink

    # requires label-label correspondence
    def expand_phenotype():
        nonlocal unlink
        unlink()

        def transform(from_: PairwiseComponent, to_: PairwiseComponent):
            def _expand_phenotype(base_selection):
                # use the phenotype from logo, PairwiseComponent.labels have robust/non-robust
                from_labels = set(from_.logo.labels[base_selection].unique())
                return np.where(to_.logo.labels.isin(from_labels))[0]

            return _expand_phenotype

        link = ipywidgets.link(
            source=(left.logo, "selection"),
            target=(right.logo, "selection"),
            transform=(transform(left, right), transform(right, left)),
        )

        unlink = link.unlink

    # requires point-point correspondence
    def expand_convex_hull():
        nonlocal unlink
        unlink()

        def transform(to: PairwiseComponent):
            coords = np.array(to.embedding.coords)

            def _expand_convex_hull(from_selection):
                if from_selection is None or len(from_selection) == 0:
                    return []
                hull = Delaunay(coords[from_selection])
                in_hull = hull.find_simplex(coords) >= 0
                return np.where(in_hull)[0]

            return _expand_convex_hull

        link = ipywidgets.link(
            source=(left.logo, "selection"),
            target=(right.logo, "selection"),
            transform=(transform(to=right), transform(to=left)),
        )

        unlink = link.unlink

    if pointwise_correspondence:
        initial_selection = sync
        selection_type_options = [
            ("synced", sync),
            ("independent", independent),
            ("neighbors", expand_neighbors),
            ("phenotype", expand_phenotype),
            ("neighbors convex hull", expand_convex_hull),
        ]
    else:
        initial_selection = independent
        selection_type_options = [
            ("independent", independent),
            ("phenotype", expand_phenotype),
        ]

    selection_type = ipywidgets.RadioButtons(
        options=selection_type_options,
        value=initial_selection,
        description="selection",
    )

    selection_type.observe(lambda change: change.new(), names="value")  # type: ignore
    initial_selection()
    # SELECTION END

    # LABELS START
    active_labels = ipywidgets.Label("markers: ")

    def on_label_level_change(change):
        left_new = trim_label_series(a.labels, max_label_level - change.new)
        right_new = trim_label_series(b.labels, max_label_level - change.new)
        left.labels = left_new
        right.labels = right_new
        left.colormap, right.colormap = create_shared_colormaps(left, right)
        left.distances, right.distances = metric.value()
        active_labels.value = "markers: " + " ".join(
            l.name for l in parse_label(left_new[0])
        )

    label_slider.observe(on_label_level_change, names="value")
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

    # initialize
    label_slider.value = 0
    left.distances, right.distances = metric.value()

    widget = ipywidgets.VBox([header, main])
    widget.left = left
    widget.right = right
    widget.label_counts = label_counts
    widget.label_label = label_label

    return widget
