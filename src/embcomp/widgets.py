import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd

import embcomp.colors as colors
from embcomp.logo import AnnotationLogo


Embedding = tuple[npt.ArrayLike, npt.NDArray]  # x y  # knn_indices

CATEGORICAL_COLORMAP = (
    [colors.gray_dark]
    + jscatter.glasbey_light
    + jscatter.glasbey_light
    + jscatter.glasbey_light
)


def _init_df(
    X: npt.ArrayLike, labels: pd.Series, robust: npt.NDArray[np.bool_]
) -> pd.DataFrame:
    xy = np.array(X)
    return pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "label": labels,
            "robust": robust,
            "robust_label": pd.Series(
                np.where(robust, labels, "0_0_0_0_0"),
                dtype="category",
            ),
        }
    )

def pairwise(
    A: Embedding,
    B: Embedding,
    labels: pd.Series,
    robust: npt.NDArray[np.bool_],
):
    row_height = 600

    left, right = (
        jscatter.Scatter(
            data=_init_df(xy, labels, robust),
            x="x",
            y="y",
            color_by="robust_label",
            color_map=CATEGORICAL_COLORMAP,
            background_color="black",
            axes=False,
            opacity_unselected=0.05,
            height=row_height,
        )
        for xy in (A[0], B[0])
    )

    # sync scatters
    for prop in ("selection", "hovering"):
        ipywidgets.jslink((left.widget, prop), (right.widget, prop))

    EMPTY_DATA = [dict(label=str(labels[0]), count=0)]
    logo = AnnotationLogo(data=EMPTY_DATA)

    label_slider = ipywidgets.IntSlider(
        description="label level",
        min=0,
        max=logo.levels,
    )

    threshold = ipywidgets.IntSlider(
        description="threshold",
        value=100,
        min=0,
        max=1000,
    )

    robust_labels = set(labels[robust].unique())  # type: ignore

    def selection_change(change):
        if change.new is None:
            return

        if len(change.new) == 0:
            logo.data = EMPTY_DATA
            return

        counts = labels[change.new].value_counts(sort=False)  # type: ignore

        logo.data = [
            dict(label=k, count=v, robust=(k in robust_labels))  # type: ignore
            for k, v in counts[counts > 0].items()  # type: ignore
        ]

        threshold.max = max(l["count"] for l in logo.data)
        logo.threshold = threshold.value = threshold.max

    left.widget.observe(selection_change, names="selection")  # type: ignore

    ipywidgets.link((logo, "threshold"), (threshold, "value"))
    ipywidgets.link((logo, "label_level"), (label_slider, "value"))

    controls = ipywidgets.VBox([label_slider, threshold])

    widget = ipywidgets.GridBox(
        children=[left.show(), right.show(), controls, logo],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr",
            grid_template_rows=f"{row_height}px " * 2,
        ),
    )

    return widget
