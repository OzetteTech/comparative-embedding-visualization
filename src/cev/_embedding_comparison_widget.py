from __future__ import annotations

import ipywidgets

from cev._compare import (
    connect_marker_selection,
    create_invert_color_checkbox,
    has_pointwise_correspondence,
)
from cev._compare_metric_dropdown import (
    create_max_depth_dropdown,
    create_metric_dropdown,
    create_update_distance_callback,
)
from cev._compare_selection_type_dropdown import create_selection_type_dropdown
from cev._compare_zoom_toggle import create_zoom_toggle
from cev._embedding import Embedding
from cev._widget_utils import add_ilocs_trait, parse_label
from cev.components import MarkerSelectionIndicator


class EmbeddingComparisonWidget(ipywidgets.VBox):
    def __init__(
        self,
        left_embedding: Embedding,
        right_embedding: Embedding,
        row_height: int = 250,
        metric: None | str = None,
        inverted_colormap: bool = False,
        auto_zoom: bool = False,
        phenotype_selection: bool = False,
        max_depth: int = 1,
        titles: tuple[str, str] | None = None,
        **kwargs,
    ):
        pointwise_correspondence = has_pointwise_correspondence(
            left_embedding, right_embedding
        )

        self.left_embedding = left_embedding
        self.right_embedding = right_embedding
        self.left = left_embedding.widgets(**kwargs)
        self.right = right_embedding.widgets(**kwargs)

        # representative label
        markers = [m.name for m in parse_label(left_embedding.labels.iloc[0])]
        self.marker_selection = MarkerSelectionIndicator(
            markers=markers, active=[True] * len(markers)
        )
        metric_dropdown = create_metric_dropdown(self.left, self.right, metric)
        max_depth_dropdown = create_max_depth_dropdown(metric_dropdown, max_depth)
        update_distances = create_update_distance_callback(
            metric_dropdown, max_depth_dropdown, self.left, self.right
        )

        zoom = create_zoom_toggle(
            self.left, self.right, auto_zoom if auto_zoom is None else auto_zoom
        )

        inverted = create_invert_color_checkbox(
            self.left,
            self.right,
            inverted_colormap if inverted_colormap is None else inverted_colormap,
        )

        selection_type = create_selection_type_dropdown(
            self.left,
            self.right,
            pointwise_correspondence,
            "phenotype"
            if phenotype_selection is True or phenotype_selection is True
            else "independent",
        )

        connect_marker_selection(
            self.marker_selection,
            (self.left_embedding, self.left),
            (self.right_embedding, self.right),
            update_distances,
        )

        # Header
        sections: list[ipywidgets.Widget] = [
            ipywidgets.VBox(
                [
                    self.marker_selection,
                    ipywidgets.HBox(
                        [
                            metric_dropdown,
                            inverted,
                            selection_type,
                            zoom,
                            max_depth_dropdown,
                        ]
                    ),
                ]
            )
        ]

        if titles is not None:
            left_title, right_title = titles
            sections.append(
                ipywidgets.HTML(
                    value='<div style="height: 1px; background: #efefef;" />',
                    layout=ipywidgets.Layout(width="100%"),
                )
            )
            sections.append(
                ipywidgets.HBox(
                    [
                        ipywidgets.HTML(
                            value=f'<h3 style="display: flex; justify-content: center; margin: 0 0 0 38px;">{left_title}</h3>',
                            layout=ipywidgets.Layout(width="50%"),
                        ),
                        ipywidgets.HTML(
                            value=f'<h3 style="display: flex; justify-content: center; margin: 0 0 0 38px;">{right_title}</h3>',
                            layout=ipywidgets.Layout(width="50%"),
                        ),
                    ]
                )
            )

        sections.append(
            ipywidgets.HBox(
                [
                    cmp.show(
                        row_height=row_height if row_height is None else row_height,
                        layout=ipywidgets.Layout(width="50%"),
                    )
                    for cmp in (self.left, self.right)
                ]
            )
        )

        super().__init__(sections)
        add_ilocs_trait(self, self.left, self.right)

    @property
    def embeddings(self):
        yield [self.left_embedding, self.left]
        yield [self.right_embedding, self.right]

    def select(self, label: str):
        for [embedding, embedding_widget] in self.embeddings:
            point_idxs = embedding.labels[embedding.labels.str.startswith(label)].index
            print(f"Found {len(point_idxs)} points")
            for scatter in embedding_widget.scatters:
                scatter.selection(point_idxs)
