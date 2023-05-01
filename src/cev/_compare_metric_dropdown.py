import typing

import ipywidgets

import cev.metrics as metrics
from cev._widget_utils import diverging_cmap

if typing.TYPE_CHECKING:
    from cev._embedding_widget import EmbeddingWidgetCollection


def create_metric_dropdown(
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    def _confusion(emb: EmbeddingWidgetCollection):
        label_confusion = metrics.confusion(emb._data)
        return emb.labels.map(label_confusion).astype(float)

    def confusion():
        return _confusion(left), _confusion(right)

    def neighborhood():
        ma, mb = _count_first()
        overlap = ma.index.intersection(mb.index)
        dist = {label: 0 for label in ma.index.union(mb.index)}
        sim = metrics.rowise_cosine_similarity(
            ma.loc[overlap, overlap], mb.loc[overlap, overlap]
        )
        dist.update(sim)
        return left.labels.map(dist).astype(float), right.labels.map(dist).astype(float)

    def abundance():
        counts = _count_first()
        abundances = [
            metrics.transform_abundance(
                rep, abundances=emb.labels.value_counts().to_dict()
            )
            for rep, emb in zip(counts, (left, right))
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

    return ipywidgets.Dropdown(
        options=[
            ("confusion", confusion),
            ("neighborhood", neighborhood),
            ("abundance", abundance),
        ],
        value=confusion,
        description="metric: ",
    )


def create_update_distance_callback(
    metric_dropdown: ipywidgets.Dropdown,
    left: EmbeddingWidgetCollection,
    right: EmbeddingWidgetCollection,
):
    def callback():
        distances = metric_dropdown.value()
        for dist, emb in zip(distances, (left, right)):
            if metric_dropdown.label == "abundance":
                vmax = max(abs(dist.min()), abs(dist.max()), 3)
                emb.metric_color_options = (
                    diverging_cmap[::-1],
                    diverging_cmap,
                    [-vmax, vmax],
                    ("low", "high", "abundance"),
                )
            elif metric_dropdown.label == "confusion":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("least", "most", "confusion"),
                )
            elif metric_dropdown.label == "neighborhood":
                emb.metric_color_options = (
                    "viridis",
                    "viridis_r",
                    [0, 1],
                    ("least", "most", "similarity"),
                )
            else:
                raise ValueError(
                    f"color options unspecified for metric, {metric_dropdown.value.__name__}"
                )

            emb.distances = dist

    metric_dropdown.observe(lambda _: callback(), names="value")
    return callback