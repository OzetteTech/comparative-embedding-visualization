from __future__ import annotations

import typing

import ipywidgets
import pandas as pd

import cev.metrics as metrics
from cev._widget_utils import diverging_cmap

if typing.TYPE_CHECKING:
    from cev._embedding_widget import EmbeddingWidgetCollection


def create_metric_dropdown(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    default: str | None = "confusion",
):
    def _confusion(emb: EmbeddingWidgetCollection):
        label_confusion = metrics.confusion(emb._data)
        return emb.labels.map(label_confusion).astype(float)

    def confusion():
        return _confusion(left), _confusion(right)

    def neighborhood(max_depth: int = 1):
        dist = metrics.compare_neighborhoods(left._data, right._data, max_depth)
        return left.labels.map(dist).astype(float), right.labels.map(dist).astype(float)

    def abundance(max_depth: int = 1, clr: bool = False):
        frequencies = (
            metrics.neighborhood(left._data, max_depth),
            metrics.neighborhood(right._data, max_depth),
        )
        abundances = [
            metrics.transform_abundance(
                freq,
                abundances=emb.labels.value_counts().to_dict(),
                clr=clr,
            )
            for freq, emb in zip(frequencies, (left, right))
        ]

        label_dist_a = metrics.merge_abundances_left(abundances[0], abundances[1])
        label_dist_a = pd.Series(
            label_dist_a.to_numpy().diagonal(), index=label_dist_a.index
        )

        label_dist_b = metrics.merge_abundances_left(abundances[1], abundances[0])
        label_dist_b = pd.Series(
            label_dist_b.to_numpy().diagonal(), index=label_dist_b.index
        )

        return (
            left.labels.map(label_dist_a - label_dist_b).astype(float),
            right.labels.map(label_dist_b - label_dist_a).astype(float),
        )

    def abundance_norm(max_depth: int = 1):
        return abundance(max_depth, clr=True)

    default_value = confusion
    if default == "neighborhood":
        default_value = neighborhood
    elif default == "abundance":
        default_value = abundance
    elif default == "abundance_norm":
        default_value = abundance_norm

    return ipywidgets.Dropdown(
        options=[
            ("Confusion", confusion),
            ("Neighborhood", neighborhood),
            ("Abundance (Absolute)", abundance),
            ("Abundance (Normalized)", abundance_norm),
        ],
        value=default_value,
        description="Metric",
    )


def has_max_depth(metric_dropdown: ipywidgets.Dropdown):
    return (
        metric_dropdown.label.lower().startswith("abundance")
        or metric_dropdown.label == "Neighborhood"
    )


def create_max_depth_dropdown(
    metric_dropdown: ipywidgets.Dropdown,
    default: int = 1,
):
    dropdown = ipywidgets.Dropdown(
        options=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        value=default,
        description="Max Depth",
        disabled=True,
    )

    def callback():
        if has_max_depth(metric_dropdown):
            dropdown.disabled = False
        else:
            dropdown.disabled = True

    metric_dropdown.observe(lambda _: callback(), names="value")
    callback()

    return dropdown


def create_update_distance_callback(
    metric_dropdown: ipywidgets.Dropdown,
    max_depth_dropdown: ipywidgets.Dropdown,
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    def callback():
        kwargs = {}
        if has_max_depth(metric_dropdown):
            distances = metric_dropdown.value(max_depth=max_depth_dropdown.value)
        else:
            distances = metric_dropdown.value(**kwargs)

        for dist, emb in zip(distances, (left, right)):
            if metric_dropdown.label == "Abundance (Absolute)":
                vmax = max(abs(dist.min()), abs(dist.max()))
                emb.metric_color_options = (
                    diverging_cmap,
                    diverging_cmap[::-1],
                    [-vmax, vmax],
                    ("Lower", "Higher", "Abs. Abundance Difference"),
                )
            elif metric_dropdown.label == "Abundance (Normalized)":
                vmax = max(abs(dist.min()), abs(dist.max()))
                emb.metric_color_options = (
                    diverging_cmap,
                    diverging_cmap[::-1],
                    [-vmax, vmax],
                    ("Lower", "Higher", "Rel. Abundance Difference"),
                )
            elif metric_dropdown.label == "Confusion":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("Lower", "Higher", "Confusion Difference"),
                )
            elif metric_dropdown.label == "Neighborhood":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("Low", "High", "Neighborhood Difference"),
                )
            else:
                raise ValueError(
                    f"color options unspecified for metric '{metric_dropdown.value.__name__}'"
                )

            emb.distances = dist

    metric_dropdown.observe(lambda _: callback(), names="value")
    max_depth_dropdown.observe(lambda _: callback(), names="value")
    callback()

    return callback
