import typing

import ipywidgets

if typing.TYPE_CHECKING:
    from ._embedding_widget import EmbeddingWidgetCollection

def create_zoom_toggle(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
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
    return zoom
