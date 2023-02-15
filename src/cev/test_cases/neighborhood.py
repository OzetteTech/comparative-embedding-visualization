import functools
from typing import Callable, Union

import jscatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import cev.test_cases.confusion as confusion
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
def generate_data_unbalanced(
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


def swap_labels(df, swap: tuple[str, str]):
    df = df.copy()
    src_mask, target_mask = map(lambda label: df.label == label, swap)
    df.label[src_mask] = swap[1]
    df.label[target_mask] = swap[0]
    return df


def translate(data, labels: list[str], offset: tuple[float, float] = (13, 0)):
    copy = data.copy()
    copy.loc[copy["label"].isin(labels), ["x", "y"]] += offset
    return copy


def rotate(data, labels: list[str], theta: float = np.radians(-45)):
    copy = data.copy()
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    mask = copy["label"].isin(labels)
    copy.loc[mask, ["x", "y"]] = copy.loc[mask, ["x", "y"]].values @ R.T

    return copy


def downsample(data: pd.DataFrame, labels: list[str], frac: float = 0.5):
    dfs = []
    for label, df in data.groupby("label"):
        if label in labels:
            df = df.sample(frac=frac)
        dfs.append(df)
    return pd.concat(dfs).reset_index()


def case1():
    """Displacement"""
    data = generate_data()
    return data, translate(data, labels=["A"], offset=(6.5, 0))


def case2():
    """Local rotation"""
    data = generate_data()
    return data, rotate(data, labels=["B", "C", "D", "E"])


def case3():
    """Global rotation"""
    data = generate_data()
    return data, rotate(data, labels=["B", "C", "D", "E", "F", "G", "H", "I"])


def case4a(frac: float = 0.1):
    """Composition change: relative decrease"""
    a = generate_data(neighborhood_offset=None)
    b = downsample(generate_data(neighborhood_offset=None), labels=["A"], frac=frac)
    return a, b


def case4b(frac: float = 0.1):
    """Composition change: relative increase"""
    a = generate_data(neighborhood_offset=None)
    b = downsample(
        generate_data(neighborhood_offset=None), labels=list("BCDE"), frac=frac
    )
    return a, b


def case5a(label: str = "D"):
    """Neighborhood change: remove neighbor"""
    a = generate_data()
    b = generate_data()
    return a, b[b.label != label].reset_index()


def case5b():
    """Neighborhood change: swap D <-> G"""
    a = generate_data()
    b = swap_labels(generate_data(), swap=("D", "G"))
    return a, b


def case5c():
    """Neighborhood change: swap D <-> F"""
    a = generate_data()
    b = swap_labels(generate_data(), swap=("D", "G"))
    return a, b


def case5d():
    """Neighborhood change: swap C <-> I"""
    a = generate_data()
    b = swap_labels(generate_data(), swap=("C", "I"))
    return a, b


def case5e():
    """Neighborhood change: dominant neighbor"""

    @dataframe
    def _generate_data(
        x: float = 2,
        cov: Covariance2D = ((0.2, 0), (0, 0.2)),
        size: int = 300,
        factor: float = 1.0,
    ):
        yield np.random.multivariate_normal((-x, -x), cov, size)
        yield np.random.multivariate_normal((-factor * x, factor * x), cov, size)
        yield np.random.multivariate_normal((factor * x, -factor * x), cov, size)
        yield np.random.multivariate_normal((factor * x, factor * x), cov, size)
        yield np.random.multivariate_normal((0, 0), cov, size)

    return _generate_data(), _generate_data(factor=1.7)


def case6a():
    """A confused with D"""
    return confusion.case1(), confusion.case2()


def case6b():
    """A confused with D (D spread)"""
    return confusion.case1(), confusion.case3()


def case6c():
    """A confused with D (A & D spread)"""
    return confusion.case1(), confusion.case4()


def case6d():
    """A missing; various size groups"""
    return confusion.case1(), confusion.case5()


MetricFn = Callable[[pd.DataFrame], pd.DataFrame]


def plot_neighborhood(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metrics: list[Union[MetricFn, tuple[str, MetricFn]]],
    name: Union[str, None] = None,
):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(metrics) + 1, figsize=(12, 3), sharex=True, sharey=True
    )
    if name:
        axs[0, 0].set_title(name, fontsize="medium", loc="left")
    for ax, df in zip((axs[0, 0], axs[1, 0]), (a, b)):
        ax.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )
        for (_, data), color in zip(df.groupby("label"), jscatter.glasbey_dark):
            ax.scatter("x", "y", data=data, color=color, s=1, alpha=0.5)

    for j, metric in enumerate(metrics, start=1):
        if isinstance(metric, tuple):
            title, metric = metric
            axs[0, j].set_title(title)

        ma = metric(a)
        mb = metric(b)

        overlap = ma.index.intersection(mb.index)

        dist = {label: 0 for label in ma.index.union(mb.index)}

        sim = rowise_cosine_similarity(
            ma.loc[overlap, overlap], mb.loc[overlap, overlap]
        )

        dist.update(sim)

        for i, df in enumerate((a, b)):
            ax = axs[i, j]
            ax.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
            norm, cmap = Normalize(0, 1), "viridis_r"
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
            )
            ax.scatter(
                df.x,
                df.y,
                c=np.nan_to_num(df.label.map(sim)),
                cmap=cmap,
                s=1,
                alpha=0.5,
                norm=norm,
            )
