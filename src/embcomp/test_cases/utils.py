from typing import Iterable, Union, Callable, ParamSpec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jscatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

Covariance2D = tuple[tuple[float, float], tuple[float, float]]

P = ParamSpec("P")


def dataframe(
    generate_pts: Callable[P, Iterable[Union[tuple[str, np.ndarray], np.ndarray]]],
) -> Callable[P, pd.DataFrame]:
    """Collects sequence of X into a labeled pd.DataFrame"""
    labels: Iterable[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def wrapper(*args: P.args, **kwargs: P.kwargs):
        dfs = []
        for label, result in zip(labels, generate_pts(*args, **kwargs)):
            if isinstance(result, tuple):
                label, X = result
            else:
                X = result

            dfs.append(pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=label)))

        df = pd.concat(dfs)
        df.label = df.label.astype("category")
        return df

    return wrapper


def plot(
    a: pd.DataFrame,
    b: Union[None, pd.DataFrame] = None,
    ax: Union[None, plt.Axes] = None,  # type: ignore
):
    ax: plt.Axes = ax or plt.gca()

    def _plot(data):
        ax.set_aspect("equal")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        for (_, grp), color in zip(data.groupby("label"), jscatter.glasbey_dark):
            ax.scatter(grp.x, grp.y, color=color, s=1)

    _plot(a)
    if b is None:
        return

    ax = make_axes_locatable(ax).append_axes(
        "right", size="100%", pad=0.1, sharex=ax, sharey=ax
    )
    _plot(b)
