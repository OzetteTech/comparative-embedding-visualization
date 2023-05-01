import typing

import ipywidgets
import numpy as np

from cev._compare_metric_dropdown import (
    create_metric_dropdown,
    create_update_distance_callback,
)
from cev._compare_selection_type_dropdown import create_selection_type_dropdown
from cev._compare_zoom_toggle import create_zoom_toggle
from cev._label_utils import parse_label, trim_label_series
from cev._widget_utils import (
    add_ilocs_trait,
    create_colormaps,
    link_widgets,
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
    marker_level = MarkerSelectionIndicator(markers=markers, value=len(markers))

    metric_dropdown = create_metric_dropdown(left, right)
    update_distances = create_update_distance_callback(metric_dropdown, left, right)
    zoom = create_zoom_toggle(left, right)
    inverted = create_invert_color_checkbox(left, right)
    selection_type = create_selection_type_dropdown(
        left, right, pointwise_correspondence
    )
    connect_marker_level(marker_level, (a, left), (b, right), update_distances)
    header = ipywidgets.VBox(
        [
            marker_level,
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


def create_invert_color_checkbox(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    inverted = ipywidgets.Checkbox(False, description="invert colormap")
    link_widgets((left, "inverted"), (inverted, "value"))
    link_widgets((right, "inverted"), (inverted, "value"))
    return inverted


def connect_marker_level(
    marker_level: MarkerSelectionIndicator,
    left_pair: tuple[Embedding, EmbeddingWidgetCollection],
    right_pair: tuple[Embedding, EmbeddingWidgetCollection],
    update_distances: typing.Callable,
):
    markers = marker_level.markers
    a, left = left_pair
    b, right = right_pair

    def on_label_level_change(change):
        left.labels = trim_label_series(a.labels, len(markers) - change.new)
        right.labels = trim_label_series(b.labels, len(markers) - change.new)
        left.colormap, right.colormap = create_colormaps(
            left.robust_labels.cat.categories,
            right.robust_labels.cat.categories,
        )
        update_distances()

    marker_level.observe(on_label_level_change, names="value")