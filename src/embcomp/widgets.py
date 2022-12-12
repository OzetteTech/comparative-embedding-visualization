import dataclasses
import itertools
from typing import Callable, Iterable, Union, overload

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import traitlets
import traittypes

import embcomp.metrics as metrics
from embcomp._widget_utils import link_widgets
from embcomp.logo import AnnotationLogo, marker_slider, parse_label, trim_label_series

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


@overload
def create_colormaps(cats: Iterable[str]) -> dict:
    ...


@overload
def create_colormaps(cats: Iterable[str], *other: Iterable[str]) -> tuple[dict, ...]:
    ...


def create_colormaps(
    cats: Iterable[str], *others: Iterable[str]
) -> Union[dict, tuple[dict, ...]]:
    all_categories = set(cats)
    for other in others:
        all_categories.update(other)

    # create unified colormap
    lookup = dict(
        zip(
            all_categories,
            itertools.cycle(jscatter.glasbey_dark[1:]),
        )
    )

    # force non-robust to be grey
    lookup[NON_ROBUST_LABEL] = "#333333"

    # create separate colormaps for each component
    cmaps = tuple({c: lookup[c] for c in cmp} for cmp in (cats, *others))
    if len(cmaps) == 1:
        return cmaps[0]
    return cmaps


@dataclasses.dataclass
class Embedding:
    coords: Coordinates
    knn_indices: KnnIndices
    labels: Labels
    robust: Union[npt.NDArray[np.bool_], None] = None

    @classmethod
    def from_df(cls, df: pd.DataFrame, knn_indices: KnnIndices):
        return cls(
            coords=df[["x", "y"]].values,
            knn_indices=knn_indices,
            labels=df["label"],
            robust=df["robust"] if "robust" in df else None,
        )

    def widgets(self, **kwargs):
        return EmbeddingWidgetCollection(self, **kwargs)


_LABEL_COLUMN = "_label"
_ROBUST_LABEL_COLUMN = "_robust_label"
_DISTANCE_COLUMN = "_distance"


class EmbeddingWidgetCollection(traitlets.HasTraits):
    inverted = traitlets.Bool(False)

    def __init__(
        self,
        embedding: Embedding,
        background_color: str = "black",
        axes: bool = False,
        opacity_unselected: float = 0.05,
        **kwargs,
    ):
        X = np.array(embedding.coords)
        self._data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1]})

        categorial_scatter, metric_scatter = (
            jscatter.Scatter(
                data=self._data,
                x="x",
                y="y",
                background_color=background_color,
                axes=axes,
                opacity_unselected=opacity_unselected,
                **kwargs,
            )
            for _ in range(2)
        )
        # link the plots together with js
        link_widgets(
            (categorial_scatter.widget, "selection"),
            (metric_scatter.widget, "selection"),
        )

        self.categorial_scatter = categorial_scatter
        self.metric_scatter = metric_scatter
        # TODO: show?
        self.logo = AnnotationLogo(embedding.labels)

        self._labeler = lambda labels: robust_labels(labels, embedding.robust)
        self.labels = embedding.labels
        self.distances = 0  # type: ignore
        self.colormap = create_colormaps(self.robust_labels.cat.categories)

    @property
    def labels(self) -> pd.Series:
        return self._data[_LABEL_COLUMN]

    @property
    def robust_labels(self) -> pd.Series:
        return self._data[_ROBUST_LABEL_COLUMN]

    @labels.setter
    def labels(self, labels: npt.ArrayLike):
        self.logo.labels = labels
        self._data[_LABEL_COLUMN] = labels
        self._data[_ROBUST_LABEL_COLUMN] = self._labeler(labels)

    @traitlets.observe("inverted")
    def _update_metric_scatter(self, *args, **kwargs):
        cmap = "viridis_r" if self.inverted else "viridis"
        self.metric_scatter.color(by=_DISTANCE_COLUMN, map=cmap, norm=[0, 1])
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

    def show(self):
        return ipywidgets.VBox(
            [
                self.categorial_scatter.show(),
                self.metric_scatter.show(),
            ]
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


def compare(
    a: Union[tuple[pd.DataFrame, KnnIndices], Embedding],
    b: Union[tuple[pd.DataFrame, KnnIndices], Embedding],
    row_height: int = 250,
):
    a = Embedding.from_df(a[0], a[1]) if isinstance(a, tuple) else a
    b = Embedding.from_df(b[0], b[1]) if isinstance(b, tuple) else b

    pointwise_correspondence = has_pointwise_correspondence(a, b)

    # representative label
    markers = [m.name for m in parse_label(a.labels.iloc[0])]
    label_slider, marker_indicator = marker_slider(markers)

    left, right = a.widgets(), b.widgets()

    def counts():
        alabels = pd.Series(left.logo.labels, dtype="category", name="label")
        acounts = metrics.count_neighbor_labels(a.knn_indices, alabels)
        blabels = pd.Series(right.logo.labels, dtype="category", name="label")
        bcounts = metrics.count_neighbor_labels(b.knn_indices, blabels)
        return (
            pd.DataFrame(acounts, index=alabels),
            pd.DataFrame(bcounts, index=blabels),
        )

    # def point_label():
    #     acounts, bcounts = counts()
    #     distances = metrics.rowise_cosine_similarity(acounts.values, bcounts.values)
    #     return distances, distances

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
    link_widgets((left, "inverted"), (inverted, "value"))
    link_widgets((right, "inverted"), (inverted, "value"))
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

        unlink = link_widgets(
            (left.categorial_scatter.widget, "selection"),
            (right.categorial_scatter.widget, "selection"),
        ).unlink

    # requires label-label correspondence
    def phenotype():
        nonlocal unlink
        unlink()

        def expand_phenotype(src: EmbeddingWidgetCollection):
            def handler(change):
                phenotypes = set(src.labels.iloc[change.new].unique())

                def ilocs(labels):
                    return np.where(labels.isin(phenotypes))[0]

                left.categorial_scatter.widget.selection = ilocs(left.robust_labels)
                right.categorial_scatter.widget.selection = ilocs(right.robust_labels)

            return handler

        transform_left = expand_phenotype(left)
        left.categorial_scatter.widget.observe(transform_left, names="selection")
        transform_right = expand_phenotype(right)
        right.categorial_scatter.widget.observe(transform_right, names="selection")

        def unlink_all():
            left.categorial_scatter.unobserve(transform_left, names="selection")
            right.categorial_scatter.unobserve(transform_right, names="selection")

        unlink = unlink_all

    if pointwise_correspondence:
        initial_selection = sync
        selection_type_options = [
            ("synced", sync),
            ("independent", independent),
            ("phenotype", phenotype),
        ]
    else:
        initial_selection = independent
        selection_type_options = [
            ("independent", independent),
            ("phenotype", phenotype),
        ]

    selection_type = ipywidgets.Dropdown(
        options=selection_type_options,
        value=initial_selection,
        description="selection",
    )

    selection_type.observe(lambda change: change.new(), names="value")  # type: ignore
    initial_selection()
    # SELECTION END

    def on_label_level_change(change):
        left.labels = trim_label_series(a.labels, len(markers) - change.new)
        right.labels = trim_label_series(b.labels, len(markers) - change.new)
        left.colormap, right.colormap = create_colormaps(
            left.robust_labels.cat.categories,
            right.robust_labels.cat.categories,
        )
        left.distances, right.distances = metric.value()

    label_slider.observe(on_label_level_change, names="value")
    # LABELS END

    header = ipywidgets.VBox(
        [
            marker_indicator,
            ipywidgets.HBox(
                [label_slider, selection_type, metric, inverted],
            ),
        ]
    )

    for s in [
        left.categorial_scatter,
        left.metric_scatter,
        right.categorial_scatter,
        right.metric_scatter,
    ]:
        s.height(row_height)

    main = ipywidgets.GridBox(
        children=[left.show(), right.show()],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr",
            grid_template_rows=f"{row_height * 2}px",
        ),
    )

    # initialize
    label_slider.value = 1
    left.distances, right.distances = metric.value()
    widget = ipywidgets.VBox([header, main])

    return widget
