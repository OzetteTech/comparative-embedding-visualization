from __future__ import annotations

import typing

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


class EmbeddingComparisonWidget:
    def __init__(
        self,
        left_embedding: Embedding,
        right_embedding: Embedding,
        row_height: int = 250,
        metric: str = "",
        inverted_colormap: bool = False,
        auto_zoom: bool = False,
        phenotype_selection: bool = False,
        max_depth: int = 1,
        titles: tuple[str, str] | None = None,
        **kwargs,
    ):
        self.pointwise_correspondence = has_pointwise_correspondence(
            left_embedding,
            right_embedding,
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

        self.row_height = row_height
        self.metric = metric
        self.auto_zoom = auto_zoom
        self.phenotype_selection = phenotype_selection
        self.inverted_colormap = inverted_colormap
        self.max_depth = max_depth
        self.titles = titles

    def show(
        self,
        row_height: int | None = None,
        metric: str | None = None,
        inverted_colormap: bool | None = None,
        auto_zoom: bool | None = None,
        phenotype_selection: bool | None = None,
        max_depth: int | None = None,
        **kwargs,
    ):
        metric_dropdown = create_metric_dropdown(
            self.left, self.right, self.metric if metric is None else metric
        )

        max_depth_dropdown = create_max_depth_dropdown(
            metric_dropdown, self.max_depth if max_depth is None else max_depth
        )

        update_distances = create_update_distance_callback(
            metric_dropdown, max_depth_dropdown, self.left, self.right
        )

        zoom = create_zoom_toggle(
            self.left, self.right, self.auto_zoom if auto_zoom is None else auto_zoom
        )

        inverted = create_invert_color_checkbox(
            self.left,
            self.right,
            self.inverted_colormap if inverted_colormap is None else inverted_colormap,
        )

        selection_type = create_selection_type_dropdown(
            self.left,
            self.right,
            self.pointwise_correspondence,
            "phenotype"
            if phenotype_selection is True or self.phenotype_selection is True
            else "independent",
        )

        connect_marker_selection(
            self.marker_selection,
            (self.left_embedding, self.left),
            (self.right_embedding, self.right),
            update_distances,
        )

        # Header
        sections = [
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

        if self.titles is not None:
            left_title, right_title = self.titles
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
                        row_height=self.row_height
                        if row_height is None
                        else row_height,
                        layout=ipywidgets.Layout(width="50%"),
                    )
                    for cmp in (self.left, self.right)
                ]
            )
        )

        widget = ipywidgets.VBox(sections)

        add_ilocs_trait(widget, self.left, self.right)
        typing.cast(typing.Any, widget).left = self.left
        typing.cast(typing.Any, widget).right = self.right
        return widget

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
