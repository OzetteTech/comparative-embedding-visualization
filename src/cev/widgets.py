from __future__ import annotations

import dataclasses
import itertools
from typing import Callable, Iterable, overload

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd
import traitlets

import cev.metrics as metrics
from cev._widget_utils import diverging_cmap, link_widgets
from cev.logo import (
    Logo,
    MarkerIndicator,
    parse_label,
    trim_label_series,
)
from cev.test_cases.metrics import (
    centered_logratio,
    count_first,
    dynamic_k,
    kneighbors,
    merge_abundances_left,
    transform_abundance,
)

Coordinates = npt.ArrayLike
KnnIndices = npt.NDArray[np.int_]
Labels = pd.Series
Distances = npt.NDArray[np.float_]

NON_ROBUST_LABEL = "0_0_0_0_0"


def robust_labels(labels: npt.ArrayLike, robust: npt.NDArray[np.bool_] | None = None):
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
) -> dict | tuple[dict, ...]:
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
    robust: npt.NDArray[np.bool_] | None = None

    @classmethod
    def from_df(cls, df: pd.DataFrame, knn_indices: None | KnnIndices = None):
        if knn_indices is None:
            largest_category_size = min(
                500,
                df.label.value_counts(sort=True)[0],
            )
            knn_indices = kneighbors(df[["x", "y"]].to_numpy(), k=largest_category_size)
        return cls(
            coords=df[["x", "y"]].values,
            knn_indices=knn_indices,
            labels=df["label"],
            robust=df["robust"] if "robust" in df else None,
        )

    @classmethod
    def from_ozette(
        cls,
        df: pd.DataFrame,
        robust_only: bool = True,
        knn_indices: None | KnnIndices = None,
    ):
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
                coords = df[["umapX", "umapY"]].to_numpy()
                labels = df["faustLabels"].to_numpy()
                robust = None
            else:
                coords = df[["umapX", "umapY"]].to_numpy()
                df = df[["faustLabels"]]
                df["label"] = ""
                for marker in parse_label(representative_label):
                    marker_annoation = (
                        marker.name + df[f"{marker.name}_faust_annotation"]
                    )
                    df["label"] += marker_annoation
                labels = df["label"].to_numpy()

        labels = pd.Series(labels, dtype="category")
        if knn_indices is None:
            largest_category_size = min(
                500,
                labels.value_counts(sort=True)[0],
            )
            knn_indices = kneighbors(coords, k=largest_category_size)

        return cls(coords=coords, labels=labels, robust=robust, knn_indices=knn_indices)

    def widgets(self, **kwargs):
        return EmbeddingWidgetCollection.from_embedding(self, **kwargs)


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
        logo: Logo,
        labeler: Callable[[npt.ArrayLike], pd.Series],
    ):
        self.categorial_scatter = categorial_scatter
        self.metric_scatter = metric_scatter
        self.logo = logo
        self._labeler = labeler
        self.metric_color_options = (
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
                opacity_unselected=opacity_unselected,
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

        logo = Logo()

        return cls(
            labels=emb.labels,
            categorial_scatter=categorial_scatter,
            metric_scatter=metric_scatter,
            logo=logo,
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

        for scatter in self.scatters:
            if row_height is not None:
                scatter.height(row_height)
            widget = scatter.show()
            widget.layout = {"margin": "0 0 2px 0"}
            widgets.append(widget)

        widgets.append(self.logo)

        return ipywidgets.VBox(widgets, **kwargs)

    def zoom(self, to: None | npt.NDArray = None):
        if to is not None:
            to = to if len(to) > 0 else None
        for s in self.scatters:
            s.zoom(to=to)


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
    a: tuple[pd.DataFrame, KnnIndices] | Embedding,
    b: tuple[pd.DataFrame, KnnIndices] | Embedding,
    row_height: int = 250,
    **kwargs,
):
    a = Embedding.from_df(a[0], a[1]) if isinstance(a, tuple) else a
    b = Embedding.from_df(b[0], b[1]) if isinstance(b, tuple) else b

    pointwise_correspondence = has_pointwise_correspondence(a, b)

    # representative label
    markers = [m.name for m in parse_label(a.labels.iloc[0])]
    marker_level = MarkerIndicator(markers=markers, value=len(markers))

    left, right = a.widgets(**kwargs), b.widgets(**kwargs)

    def confusion():
        def _confusion(emb: EmbeddingWidgetCollection, knn_indices: KnnIndices):
            res = dynamic_k(emb._data, knn_indices=knn_indices)
            label_confusion = 1 - metrics.rowise_cosine_similarity(
                res, np.eye(len(res))
            )
            return emb.labels.map(label_confusion).astype(float)

        return _confusion(left, a.knn_indices), _confusion(right, b.knn_indices)

    def _count_first():
        ma = count_first(left._data, type="both", agg="set", knn_indices=a.knn_indices)
        mb = count_first(right._data, type="both", agg="set", knn_indices=b.knn_indices)
        return ma, mb

    def neighborhood():
        ma, mb = _count_first()
        overlap = ma.index.intersection(mb.index)
        dist = {label: 0 for label in ma.index.union(mb.index)}
        sim = metrics.rowise_cosine_similarity(
            ma.loc[overlap, overlap], mb.loc[overlap, overlap]
        )
        dist.update(sim)
        return left.labels.map(dist).astype(float), right.labels.map(dist).astype(float)

    def abundance():
        counts = _count_first()
        abundances = [
            transform_abundance(rep, abundances=emb.labels.value_counts().to_dict())
            for rep, emb in zip(counts, (left, right))
        ]
        merged = [
            merge_abundances_left(abundances[0], abundances[1]),
            merge_abundances_left(abundances[1], abundances[0]),
        ]
        label_dista, label_distb = (centered_logratio(ab) for ab in merged)
        return (
            left.labels.map(label_dista - label_distb).astype(float),
            right.labels.map(label_distb - label_dista).astype(float),
        )

    # METRIC START

    metric_options: list[tuple[str, Callable]] = [
        ("confusion", confusion),
        ("neighborhood", neighborhood),
        ("abundance", abundance),
    ]

    metric = ipywidgets.Dropdown(
        options=metric_options,
        value=confusion,
        description="metric: ",
    )

    def update_distances():
        distances = metric.value()
        for dist, emb in zip(distances, (left, right)):
            if metric.value == abundance:
                vmax = max(abs(dist.min()), abs(dist.max()), 3)
                emb.metric_color_options = (
                    diverging_cmap[::-1],
                    diverging_cmap,
                    [-vmax, vmax],
                    ("low", "high", "abundance"),
                )
            elif metric.value == confusion:
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("least", "most", "confusion"),
                )

            elif metric.value == neighborhood:
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("least", "most", "similarity"),
                )
            else:
                raise ValueError(
                    f"color options unspecified for metric, {metric.value.__name__}"
                )

            emb.distances = dist

    metric.observe(lambda _change: update_distances(), names="value")
    # METRIC END

    # COLOR START
    inverted = ipywidgets.Checkbox(False, description="invert colormap")
    link_widgets((left, "inverted"), (inverted, "value"))
    link_widgets((right, "inverted"), (inverted, "value"))
    # COLOR END

    # ZOOM START
    zoom = ipywidgets.Checkbox(False, description="auto-zoom")

    def handle_selection_change_zoom(emb: EmbeddingWidgetCollection):
        def on_change(change):
            if zoom.value is False:
                return
            emb.zoom(to=change.new)

        return on_change

    left.categorial_scatter.widget.observe(
        handle_selection_change_zoom(left), names="selection"
    )
    right.categorial_scatter.widget.observe(
        handle_selection_change_zoom(right), names="selection"
    )

    def handle_zoom_change(change):
        if change.new is False:
            left.zoom(to=None)
            right.zoom(to=None)
        else:
            left.zoom(to=left.categorial_scatter.selection())
            right.zoom(to=right.categorial_scatter.selection())

    zoom.observe(handle_zoom_change, names="value")

    # ZOOM END

    # SELECTION START
    def unlink():
        return None

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

                for emb in (left, right):
                    ilocs = np.where(emb.robust_labels.isin(phenotypes))[0]
                    emb.categorial_scatter.widget.selection = ilocs
                    emb.metric_scatter.widget.selection = ilocs

            return handler

        transform_left = expand_phenotype(left)
        left.categorial_scatter.widget.observe(transform_left, names="selection")
        transform_right = expand_phenotype(right)
        right.categorial_scatter.widget.observe(transform_right, names="selection")

        def unlink_all():
            left.categorial_scatter.widget.unobserve(transform_left, names="selection")
            right.categorial_scatter.widget.unobserve(
                transform_right, names="selection"
            )

        unlink = unlink_all

    if pointwise_correspondence:
        initial_selection = sync
        selection_type_options = [
            ("synced", sync),
            ("independent", independent),
            ("phenotype", phenotype),
        ]
    else:
        initial_selection = phenotype
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
        update_distances()

    marker_level.observe(on_label_level_change, names="value")
    # LABELS END

    header = ipywidgets.VBox(
        [
            marker_level,
            ipywidgets.HBox([selection_type, metric, inverted, zoom]),
        ]
    )

    main = ipywidgets.HBox(
        [
            cmp.show(row_height=row_height, layout=ipywidgets.Layout(width="50%"))
            for cmp in (left, right)
        ]
    )

    # initialize
    widget = ipywidgets.VBox([header, main])
    add_ilocs_trait(widget, left, right)

    widget.left = left
    widget.right = right
    widget.metric = metric
    widget.count_first = _count_first

    marker_level.value = 1  # set the lowest marker_level
    return widget


def add_ilocs_trait(
    widget: traitlets.HasTraits,
    right: EmbeddingWidgetCollection,
    left: EmbeddingWidgetCollection,
):
    """Adds a `.ilocs` tuple trait to the final widget.

    Containts the (left, right) selections.
    """
    initial = (
        left.categorial_scatter.selection(),
        right.categorial_scatter.selection(),
    )
    widget.add_traits(ilocs=traitlets.Tuple(initial))

    ipywidgets.dlink(
        source=(left.categorial_scatter.widget, "selection"),
        target=(widget, "ilocs"),
        transform=lambda iloc: (iloc, widget.ilocs[1]),  # type: ignore
    )

    ipywidgets.dlink(
        source=(right.categorial_scatter.widget, "selection"),
        target=(widget, "ilocs"),
        transform=lambda iloc: (widget.ilocs[0], iloc),  # type: ignore
    )
