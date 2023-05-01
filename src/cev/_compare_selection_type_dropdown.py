import typing

import ipywidgets
import numpy as np

from ._widget_utils import link_widgets

if typing.TYPE_CHECKING:
    from ._embedding_widget import EmbeddingWidgetCollection


def create_selection_type_dropdown(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    pointwise_correspondence: bool,
):
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
    selection_type.value()
    return selection_type
