from __future__ import annotations

import typing

import ipywidgets

import cev.metrics as metrics
from cev._widget_utils import diverging_cmap

if typing.TYPE_CHECKING:
    from cev._embedding_widget import EmbeddingWidgetCollection


def create_metric_dropdown(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    default: str | None = 'confusion',
):
    def _confusion(emb: EmbeddingWidgetCollection):
        label_confusion = metrics.confusion(emb._data)
        return emb.labels.map(label_confusion).astype(float)

    def confusion():
        return _confusion(left), _confusion(right)

    def neighborhood():
        dist = metrics.compare_neighborhoods(left._data, right._data)
        return left.labels.map(dist).astype(float), right.labels.map(dist).astype(float)

    def abundance():
        frequencies = metrics.neighborhood(left._data), metrics.neighborhood(right._data)
        abundances = [
            metrics.transform_abundance(
                ## Fritz: Test if moving CLR here makes a difference
                rep, abundances=emb.labels.value_counts().to_dict()
            )
            for rep, emb in zip(frequencies, (left, right))
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
    
    default_value = confusion
    if default == 'neighborhood':
        default_value = neighborhood
    elif default == 'abundance':
        default_value = abundance

    return ipywidgets.Dropdown(
        options=[
            ("Confusion", confusion),
            ("Neighborhood", neighborhood),
            ("Abundance", abundance),
        ],
        value=default_value,
        description="Metric",
    )


def create_update_distance_callback(
    metric_dropdown: ipywidgets.Dropdown,
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    def callback():
        distances = metric_dropdown.value()
        for dist, emb in zip(distances, (left, right)):
            if metric_dropdown.label == "Abundance":
                vmax = max(abs(dist.min()), abs(dist.max()), 3)
                emb.metric_color_options = (
                    diverging_cmap,
                    diverging_cmap[::-1],
                    [-vmax, vmax],
                    ("Low", "High", "Abundance"),
                )
            elif metric_dropdown.label == "Confusion":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("Least", "Most", "Confusion"),
                )
            elif metric_dropdown.label == "Neighborhood":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("Least", "Most", "Similarity"),
                )
            else:
                raise ValueError(
                    f"color options unspecified for metric '{metric_dropdown.value.__name__}'"
                )

            emb.distances = dist

    metric_dropdown.observe(lambda _: callback(), names="value")
    callback()
    
    return callback
