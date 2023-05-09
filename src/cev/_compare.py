from __future__ import annotations

import typing

import ipywidgets
import numpy as np

from cev._compare_metric_dropdown import (
    create_metric_dropdown,
    create_update_distance_callback,
)
from cev._compare_selection_type_dropdown import create_selection_type_dropdown
from cev._compare_zoom_toggle import create_zoom_toggle
from cev._widget_utils import (
    add_ilocs_trait,
    create_colormaps,
    link_widgets,
    parse_label,
    trim_label_series,
)
from cev.components import MarkerSelectionIndicator

if typing.TYPE_CHECKING:
    from cev._embedding import Embedding
    from cev._embedding_widget import EmbeddingWidgetCollection


def compare(a: Embedding, b: Embedding, row_height: int = 250, **kwargs):
    pointwise_correspondence = has_pointwise_correspondence(a, b)
    left, right = a.widgets(**kwargs), b.widgets(**kwargs)

    # representative label
    markers = [m.name for m in parse_label(a.labels.iloc[0])]
    marker_selection = MarkerSelectionIndicator(
        markers=markers, active=[True] + [False for x in range(len(markers) - 1)]
    )

    metric_dropdown = create_metric_dropdown(left, right)
    update_distances = create_update_distance_callback(metric_dropdown, left, right)
    zoom = create_zoom_toggle(left, right)
    inverted = create_invert_color_checkbox(left, right)
    selection_type = create_selection_type_dropdown(
        left, right, pointwise_correspondence
    )
    connect_marker_selection(marker_selection, (a, left), (b, right), update_distances)
    header = ipywidgets.VBox(
        [
            marker_selection,
            ipywidgets.HBox([selection_type, metric_dropdown, inverted, zoom]),
        ]
    )
    main = ipywidgets.HBox(
        [
            cmp.show(row_height=row_height, layout=ipywidgets.Layout(width="50%"))
            for cmp in (left, right)
        ]
    )
    widget = ipywidgets.VBox([header, main])

    add_ilocs_trait(widget, left, right)
    typing.cast(typing.Any, widget).left = left
    typing.cast(typing.Any, widget).right = right
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


def create_invert_color_checkbox(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    default: bool = False,
):
    inverted = ipywidgets.Checkbox(default, description="Invert Colormap")
    link_widgets((left, "inverted"), (inverted, "value"))
    link_widgets((right, "inverted"), (inverted, "value"))
    return inverted


def connect_marker_selection(
    marker_selection: MarkerSelectionIndicator,
    left_pair: tuple[Embedding, EmbeddingWidgetCollection],
    right_pair: tuple[Embedding, EmbeddingWidgetCollection],
    update_distances: typing.Callable,
):
    markers = marker_selection.markers
    a, left = left_pair
    b, right = right_pair

    def update_labels(active):
        active_markers = set([marker for i, marker in enumerate(markers) if active[i]])

        left.labels = trim_label_series(a.labels, active_markers)
        right.labels = trim_label_series(b.labels, active_markers)

        left.colormap, right.colormap = create_colormaps(
            left.robust_labels.cat.categories,
            right.robust_labels.cat.categories,
        )

        update_distances()

    def on_active_marker_selection_change(change):
        update_labels(change.new)

    update_labels(marker_selection.active)

    marker_selection.observe(on_active_marker_selection_change, names="active")
