from __future__ import annotations

import re
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
    create_value_range_slider,
)
from cev._compare_selection_type_dropdown import create_selection_type_dropdown
from cev._compare_zoom_toggle import create_zoom_toggle
from cev._embedding import Embedding
from cev._widget_utils import add_ilocs_trait, parse_label
from cev.components import MarkerSelectionIndicator


def _create_titles(
    titles: tuple[str, str]
) -> tuple[ipywidgets.Widget, ipywidgets.Widget]:
    left_title, right_title = titles
    spacer = ipywidgets.HTML(
        value='<div style="height: 1px; background: #efefef;" />',
        layout=ipywidgets.Layout(width="100%"),
    )
    title_widget = ipywidgets.HBox(
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
    return spacer, title_widget


class EmbeddingComparisonWidget(ipywidgets.VBox):
    def __init__(
        self,
        left_embedding: Embedding,
        right_embedding: Embedding,
        row_height: int = 250,
        metric: typing.Literal[
            "confusion", "neigbhorhood", "abundance", "abundance_norm"
        ] = "confusion",
        inverted_colormap: bool = False,
        auto_zoom: bool = False,
        selection: typing.Literal["independent", "synced", "phenotype"] = "independent",
        max_depth: int = 1,
        titles: tuple[str, str] | None = None,
        active_markers: list[str] | typing.Literal["all"] = "all",
        **kwargs,
    ):
        pointwise_correspondence = has_pointwise_correspondence(
            left_embedding, right_embedding
        )

        self.left_embedding = left_embedding
        self.right_embedding = right_embedding
        self.left = left_embedding.widgets(**kwargs)
        self.right = right_embedding.widgets(**kwargs)

        metric_dropdown = create_metric_dropdown(self.left, self.right, metric)
        max_depth_dropdown = create_max_depth_dropdown(metric_dropdown, max_depth)
        value_range_slider = create_value_range_slider(metric_dropdown)
        update_distances = create_update_distance_callback(
            metric_dropdown,
            max_depth_dropdown,
            value_range_slider,
            self.left,
            self.right,
        )

        has_markers = "+" in left_embedding.labels.iloc[0]

        if has_markers:
            # representative label
            markers = [m.name for m in parse_label(left_embedding.labels.iloc[0])]
            _active_markers = (
                [True] * len(markers)
                if active_markers == "all"
                else [False] * len(markers)
            )
            for active_marker in active_markers:
                try:
                    _active_markers[markers.index(active_marker)] = True
                except ValueError:
                    pass
            marker_selection = MarkerSelectionIndicator(
                markers=markers, active=_active_markers
            )
            connect_marker_selection(
                marker_selection,
                (self.left_embedding, self.left),
                (self.right_embedding, self.right),
                update_distances,
            )

        zoom = create_zoom_toggle(self.left, self.right, auto_zoom)
        inverted = create_invert_color_checkbox(
            self.left, self.right, inverted_colormap
        )

        selection_type = create_selection_type_dropdown(
            self.left,
            self.right,
            pointwise_correspondence,
            selection,
        )

        metric_dropdown.observe(lambda _: update_distances(), names="value")
        max_depth_dropdown.observe(lambda _: update_distances(), names="value")
        value_range_slider.observe(lambda _: update_distances(), names="value")

        update_distances()

        # Header
        settings = ipywidgets.HBox(
            [
                metric_dropdown,
                inverted,
                value_range_slider,
                selection_type,
                zoom,
                max_depth_dropdown,
            ]
        )
        header = [marker_selection, settings] if has_markers else [settings]
        sections: list[ipywidgets.Widget] = [ipywidgets.VBox(header)]

        if titles is not None:
            sections.extend(_create_titles(titles))

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

    def select(self, labels: str | list[str]):
        if isinstance(labels, str):
            for [embedding, embedding_widget] in self.embeddings:
                point_idxs = embedding.labels[
                    embedding.labels.str.startswith(labels)
                ].index
                print(f"Found {len(point_idxs)} points")
                for scatter in embedding_widget.scatters:
                    scatter.selection(point_idxs)
            return

        regexs = []

        for [embedding, embedding_widget] in self.embeddings:
            markers = list(filter(None, re.split("[+-]", embedding.labels[0])))
            marker_set = set(markers)
            marker_order = {s: i for i, s in enumerate(markers)}

            valid_labels = list(filter(lambda label: label[:-1] in marker_set, labels))
            ordered_labels = sorted(
                valid_labels, key=lambda label: marker_order.get(label[:-1], 0)
            )

            regex = (
                ".*" + ".*".join([re.escape(label) for label in ordered_labels]) + ".*"
            )
            regexs.append(regex)

        for i, [embedding, embedding_widget] in enumerate(self.embeddings):
            regex = regexs[i]
            point_idxs = embedding.labels[
                embedding.labels.str.match(regex, flags=re.IGNORECASE)
            ].index
            print(f"Found {len(point_idxs)} points")
            for scatter in embedding_widget.scatters:
                scatter.selection(point_idxs)
