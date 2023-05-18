from __future__ import annotations

import functools
import typing

import ipywidgets
import pandas as pd

import cev.metrics as metrics
from cev._widget_utils import diverging_cmap

if typing.TYPE_CHECKING:
    from cev._embedding_widget import EmbeddingWidgetCollection


CACHE_SIZE = 5


def create_metric_dropdown(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
    default: typing.Literal[
        "confusion", "neigbhorhood", "abundance", "abundance_norm"
    ] = "confusion",
):
    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cached_confusion(emb: EmbeddingWidgetCollection):
        return metrics.confusion(emb._data)

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cached_neighborhood(emb: EmbeddingWidgetCollection, max_depth: int = 1):
        return metrics.neighborhood(emb._data, max_depth=max_depth)

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cached_abundance(
        left: EmbeddingWidgetCollection,
        right: EmbeddingWidgetCollection,
        max_depth: int = 1,
        clr: bool = False,
    ):
        frequencies = (
            cached_neighborhood(left, max_depth),
            cached_neighborhood(right, max_depth),
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

    def confusion(**kwargs):
        left_label_confusion = cached_confusion(left)
        right_label_confusion = cached_confusion(right)
        return (
            left.labels.map(left_label_confusion).astype(float),
            right.labels.map(right_label_confusion).astype(float),
        )

    def neighborhood(max_depth: int = 1):
        a = cached_neighborhood(left, max_depth)
        b = cached_neighborhood(right, max_depth)
        dist = metrics.compare_neighborhoods(a, b)
        return left.labels.map(dist).astype(float), right.labels.map(dist).astype(float)

    abundance = functools.partial(cached_abundance, left, right)
    abundance_norm = functools.partial(cached_abundance, left, right, clr=True)

    default_value = {
        "confusion": confusion,
        "neighborhood": neighborhood,
        "abundance": abundance,
        "abundance_norm": abundance_norm,
    }[default]

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


def create_value_range_slider(metric_dropdown: ipywidgets.Dropdown):
    slider = ipywidgets.FloatRangeSlider(
        value=[0, 1],
        min=0,
        max=1,
        step=0.05,
        description="Color Range:",
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".2f",
    )

    def callback():
        if metric_dropdown.label.lower().startswith("abundance"):
            slider.value = [0.05, 0.95]
        else:
            slider.value = [0, 1]

    metric_dropdown.observe(lambda _: callback(), names="value")
    callback()

    return slider


def create_update_distance_callback(
    metric_dropdown: ipywidgets.Dropdown,
    max_depth_dropdown: ipywidgets.Dropdown,
    value_range_slider: ipywidgets.FloatRangeSlider,
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    def callback():
        distances = metric_dropdown.value(max_depth=max_depth_dropdown.value)

        for dist, emb in zip(distances, (left, right)):
            if metric_dropdown.label == "Abundance (Absolute)":
                lower, upper = dist.quantile(value_range_slider.value)
                vmax = max(abs(lower), abs(upper))
                emb.metric_color_options = (
                    diverging_cmap,
                    diverging_cmap[::-1],
                    [-vmax, vmax],
                    ("Lower", "Higher", "Abs. Abundance Difference"),
                )
            elif metric_dropdown.label == "Abundance (Normalized)":
                lower, upper = dist.quantile(value_range_slider.value)
                vmax = max(abs(lower), abs(upper))
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
                    value_range_slider.value,
                    ("Low", "High", "Confusion"),
                )
            elif metric_dropdown.label == "Neighborhood":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    value_range_slider.value,
                    ("Low", "High", "Neighborhood Difference"),
                )
            else:
                raise ValueError(
                    f"color options unspecified for metric '{metric_dropdown.value.__name__}'"
                )

            emb.distances = dist

    return callback
