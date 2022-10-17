
import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd


import embcomp.colors as colors
import embcomp.metrics as metrics

Embedding = tuple[
    npt.ArrayLike, # x y
    npt.NDArray    # knn_indices
]

def run(A: Embedding, B: Embedding, labels: pd.Series):

    color_map = (
        [colors.gray_dark]
        + colors.glasbey_light
        + colors.glasbey_light
        + colors.glasbey_light
    )

    def init_df(X: npt.ArrayLike) -> pd.DataFrame:
        xy = np.array(X)
        return pd.DataFrame(
            {
                "x": xy[:, 0],
                "y": xy[:, 1],
                "label": labels,
            }
        )

    left, right = [
        jscatter.Scatter(
            data=init_df(xy),
            x="x",
            y="y",
            color_by="label",
            color_map=color_map,
            background_color="black",
            axes=False,
            opacity_unselected=0.05,
        )
        for xy in (A[0], B[0])
    ]

    return left.show()
