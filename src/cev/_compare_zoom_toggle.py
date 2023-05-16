from __future__ import annotations

import typing

import ipywidgets

if typing.TYPE_CHECKING:
    from ._embedding_widget import EmbeddingWidgetCollection


def create_zoom_toggle(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    default: bool = False,
):
    zoom = ipywidgets.Checkbox(default, description="Auto Zoom")

    def handle_selection_change_zoom(emb: EmbeddingWidgetCollection):
        def on_change(change):
            if zoom.value is False:
                return
            emb.zoom(to=change.new)

        return on_change

    left.categorical_scatter.widget.observe(
        handle_selection_change_zoom(left), names="selection"
    )
    right.categorical_scatter.widget.observe(
        handle_selection_change_zoom(right), names="selection"
    )

    def handle_zoom_change(change):
        if change.new is False:
            left.zoom(to=None)
            right.zoom(to=None)
        else:
            left.zoom(to=left.categorical_scatter.selection())
            right.zoom(to=right.categorical_scatter.selection())

    zoom.observe(handle_zoom_change, names="value")
    return zoom
