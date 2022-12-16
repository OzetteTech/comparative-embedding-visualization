import functools
from typing import Union

import jscatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cev.metrics import rowise_cosine_similarity
from cev.test_cases.utils import Covariance2D, dataframe


@dataframe
def generate_data(
    neighborhood_offset: Union[float, None] = 13,
    x: float = 2.5,
    cov: Covariance2D = ((0.2, 0), (0, 0.2)),
    size: int = 500,
):
    means = [
        (0, 0),
        (-x, -x),
        (-x, x),
        (x, x),
        (x, -x),
    ]
    if neighborhood_offset is not None:
        means.extend(
            [
                (neighborhood_offset - x, -x),
                (neighborhood_offset - x, x),
                (neighborhood_offset + x, x),
                (neighborhood_offset + x, -x),
            ]
        )
    yield from map(
        functools.partial(np.random.multivariate_normal, cov=cov, size=size), means
    )


@dataframe
def case1(x: float = 2.5, cov: Covariance2D = ((0.2, 0), (0, 0.2)), size: int = 300):
    """
    - Case: 5 equal sized groups, well separated
    - Expected: no confusion
    """
    yield np.random.multivariate_normal((-x, -x), cov, size)
    yield np.random.multivariate_normal((-x, x), cov, size)
    yield np.random.multivariate_normal((x, -x), cov, size)
    yield np.random.multivariate_normal((x, x), cov, size)
    yield np.random.multivariate_normal((0, 0), cov, size)


@dataframe
def case2(x: float = 2.5, cov: Covariance2D = ((0.2, 0), (0, 0.2)), size: int = 300):
    """
    - Case: 5 equal sized groups, 2 mixed
    - Expected: confusion with last two groups
    """
    yield np.random.multivariate_normal((-x, -x), cov, size)
    yield np.random.multivariate_normal((-x, x), cov, size)
    yield np.random.multivariate_normal((x, -x), cov, size)
    yield np.random.multivariate_normal((0, 0), cov, size)
    yield np.random.multivariate_normal((0, 0), cov, size)


@dataframe
def case3(x: float = 4.5, cov: Covariance2D = ((0.2, 0), (0, 0.2)), size: int = 300):
    """
    - Case: 5 groups of various sizes, small group intermixed with larger
    - Expected: smaller group should be confused with larger mixed group
    """
    cov_factor = 6
    cov_arr = np.array(cov)

    yield np.random.multivariate_normal((-x, -x), cov_arr, size)
    yield np.random.multivariate_normal((-x, x), cov_arr, size)
    yield np.random.multivariate_normal((x, -x), cov_arr, size)
    yield np.random.multivariate_normal((0, 0), cov_arr * cov_factor, size)
    yield np.random.multivariate_normal((0, 0), cov_arr, size)


@dataframe
def case4(x: float = 4.5, cov: Covariance2D = ((0.2, 0), (0, 0.2)), size: int = 300):
    """
    - Case: 5 groups of various sizes, small group intermixed with larger
    - Expected: smaller group should be confused with larger mixed group
    """
    size_factor = 2
    cov_factor = 6
    cov_arr = np.array(cov)

    yield np.random.multivariate_normal((-x, -x), cov_arr, size)
    yield np.random.multivariate_normal((-x, x), cov_arr, size)
    yield np.random.multivariate_normal((x, -x), cov_arr, size)
    yield np.random.multivariate_normal(
        (0, 0), cov_arr * cov_factor, size * size_factor
    )
    yield np.random.multivariate_normal((0, 0), cov_arr * cov_factor, size)


@dataframe
def case5(x: float = 3, cov: Covariance2D = ((0.5, 0), (0, 0.5))):
    """
    - Case: 4 groups of various sizes
    """
    yield np.random.multivariate_normal((-x, -x), cov, 10)
    yield np.random.multivariate_normal((-x, x), cov, 50)
    yield np.random.multivariate_normal((x, -x), cov, 100)
    yield np.random.multivariate_normal((x, x), cov, 1000)


def plot_confusion(df: pd.DataFrame, metrics):
    fig, (ax0, *axs) = plt.subplots(nrows=1, ncols=len(metrics) + 1, figsize=(12, 2))

    ax0.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    for (_, data), color in zip(df.groupby("label"), jscatter.glasbey_dark):
        ax0.scatter("x", "y", data=data, color=color, s=1, alpha=0.5)

    sideax = make_axes_locatable(ax0).append_axes("left", size="20%", pad=0.1)
    sideax.set_xlabel("size")
    sideax.barh(
        df.label.cat.categories.values,
        df.label.value_counts(sort=False),
        color=jscatter.glasbey_dark[: len(df.label.cat.categories)],
    )

    for ax, maybe_func in zip(axs, metrics):
        if callable(maybe_func):
            func = maybe_func
        else:
            title, func = maybe_func
            ax.set_title(title)
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        confusion = 1 - rowise_cosine_similarity(
            func(df), np.eye(len(df.label.cat.categories))
        )

        sideax = make_axes_locatable(ax).append_axes("left", size="20%", pad=0.1)
        sideax.set_xlim([0, 1])
        sideax.set_xlabel("conf.")
        sideax.barh(
            confusion.index,
            confusion.values,
            color=jscatter.glasbey_dark[: len(confusion)],
        )

        norm, cmap = Normalize(0, 1), "viridis"
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
        )
        ax.scatter(
            df["x"],
            df["y"],
            c=df["label"].map(confusion),
            s=1,
            cmap=cmap,
            alpha=0.5,
            norm=norm,
        )
