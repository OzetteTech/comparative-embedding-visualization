import functools
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import embcomp.test_cases.confusion as confusion
from embcomp.test_cases.utils import Covariance2D, dataframe, plot


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
    return data, translate(data, labels=["A"])


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
    metrics: list[MetricFn],
    **kwargs
):
    ax = kwargs.get("ax", plt.gca())
    plot(a, b, ax=ax)

    for metric in metrics:
        inner = make_axes_locatable(ax).append_axes("left", size="50%", pad=0.1)
        print(inner)
        plot(a, b, ax=inner)
