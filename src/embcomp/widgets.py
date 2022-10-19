from typing import Any
import IPython.display
import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd

import embcomp.colors as colors
import embcomp.metrics as metrics
from embcomp.logo import label_comparer

Embedding = tuple[npt.ArrayLike, npt.NDArray]  # x y  # knn_indices

CATEGORICAL_COLORMAP = (
    [colors.gray_dark]
    + colors.glasbey_light
    + colors.glasbey_light
    + colors.glasbey_light
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

    slider = ipywidgets.SelectionSlider(
        options=range(10),
        orientation="vertical",
    )

    logo = ipywidgets.Output()
    robust_labels = set(labels[robust].unique()) # type: ignore
    with logo:
        IPython.display.display(
            label_comparer(labels, empty=True)
        )

    @logo.capture()
    def handle_change(change):
        if change.new is None:
            return
        subset: pd.Series = labels[change.new] # type: ignore
        if len(subset) == 0:
            return
        logo.clear_output()
        IPython.display.display(label_comparer(subset, robust_labels))

    left.widget.observe(handle_change, names="selection")  # type: ignore

    return ipywidgets.GridBox(
        children=[left.show(), right.show(), slider, logo],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr 0.1fr",
            grid_template_rows=f"{row_height}px " * 2,
        ),
    )
