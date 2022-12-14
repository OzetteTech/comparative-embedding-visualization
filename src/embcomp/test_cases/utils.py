from typing import Callable, Iterable, ParamSpec, Union

import jscatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return df.reset_index(drop=True)

    return wrapper


def plot(
    a: pd.DataFrame,
    b: Union[None, pd.DataFrame] = None,
    ax: Union[None, plt.Axes] = None,  # type: ignore
):
    root: plt.Axes = ax or plt.gca()

    categories = set(a.label.cat.categories)
    if b is not None:
        categories.union(b.label.cat.categories)
    colors = dict(zip(categories, jscatter.glasbey_dark))

    def _plot(data: pd.DataFrame, ax: plt.Axes):
        ax.set_aspect("equal")
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        for label, grp in data.groupby("label"):
            color = colors[label]  # type: ignore
            ax.scatter(grp.x, grp.y, color=color, label=label, s=1)

    _plot(data=a, ax=root)
    if b is not None:
        _plot(
            data=b,
            ax=make_axes_locatable(root).append_axes(
                "right", size="100%", pad=0.1, sharex=root, sharey=root
            ),
        )
