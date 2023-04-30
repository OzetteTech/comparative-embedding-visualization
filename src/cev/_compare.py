import typing

import ipywidgets
import numpy as np

import cev.metrics as metrics
from cev._embedding_widget import EmbeddingWidgetCollection
from cev._label_utils import parse_label, trim_label_series
from cev._widget_utils import (
    add_ilocs_trait,
    create_colormaps,
    diverging_cmap,
    link_widgets,
)
from cev.components import MarkerSelectionIndicator

if typing.TYPE_CHECKING:
    from ._embedding import Embedding


def compare(a: Embedding, b: Embedding, row_height: int = 250, **kwargs):
    pointwise_correspondence = has_pointwise_correspondence(a, b)

    # representative label
    markers = [m.name for m in parse_label(a.labels.iloc[0])]
    marker_level = MarkerSelectionIndicator(markers=markers, value=len(markers))

    left, right = (
        EmbeddingWidgetCollection.from_embedding(**kwargs),
        EmbeddingWidgetCollection.from_embedding(**kwargs),
    )

    def confusion():
        def _confusion(emb: EmbeddingWidgetCollection):
            label_confusion = metrics.confusion(emb._data)
            return emb.labels.map(label_confusion).astype(float)

        return _confusion(left), _confusion(right)

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
            metrics.transform_abundance(
                rep, abundances=emb.labels.value_counts().to_dict()
            )
            for rep, emb in zip(counts, (left, right))
        ]
        merged = [
            metrics.merge_abundances_left(abundances[0], abundances[1]),
            metrics.merge_abundances_left(abundances[1], abundances[0]),
        ]
        label_dista, label_distb = (metrics.centered_logratio(ab) for ab in merged)
        return (
            left.labels.map(label_dista - label_distb).astype(float),
            right.labels.map(label_distb - label_dista).astype(float),
        )

    # METRIC START

    metric_options: list[tuple[str, typing.Callable]] = [
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

    typing.cast(typing.Any, widget).left = left
    typing.cast(typing.Any, widget).right = right

    marker_level.value = 1  # set the lowest marker_level
    return widget


def has_pointwise_correspondence(a: Embedding, b: Embedding) -> bool:
    return np.array_equal(a.labels, b.labels) and (
        (a.robust is None and b.robust is None)
        or (
            a.robust is not None
            and b.robust is not None
            and np.array_equal(a.robust, b.robust)
        )
    )
