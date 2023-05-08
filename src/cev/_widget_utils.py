from __future__ import annotations

import dataclasses
import itertools
import re
import typing

import ipywidgets
import numpy as np
import pandas as pd
import traitlets
from jscatter.color_maps import glasbey_dark

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from .widgets import EmbeddingWidgetCollection

NON_ROBUST_LABEL = "0_0_0_0_0"
_ERR_MESSAGE = (
    "The truth value of an array with more than one element is ambiguous. "
    + "Use a.any() or a.all()"
)


# patched version which allows for numpy comparison
# https://github.com/jupyter-widgets/traittypes/issues/45
class link_widgets(traitlets.link):
    def _update_target(self, change):
        try:
            super()._update_target(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e
        except traitlets.TraitError:
            pass

    def _update_source(self, change):
        try:
            super()._update_source(change)
        except ValueError as e:
            if e.args[0] != _ERR_MESSAGE:
                raise e
        except traitlets.TraitError:
            pass


@dataclasses.dataclass
class Marker:
    name: str
    annotation: typing.Literal["+", "-"]

    def __str__(self) -> str:
        return self.name + self.annotation


def parse_label(label: str) -> list[Marker]:
    return [
        Marker(inner_label[:-1], inner_label[-1])
        for inner_label in re.split("(\w+[\-|\+])", label)
        if inner_label
    ]


def trim_label_series(labels: pd.Series, active_markers: typing.Set(str)):
    splitted_labels = [marker for marker in labels.str.split("(\w+[\+|\-])", regex=True)]
    
    out = []
    for splitted_label in splitted_labels:
        out.append("".join([marker for marker in splitted_label if marker[0:-1] in active_markers]))
        
    return out


def add_ilocs_trait(
    widget: traitlets.HasTraits,
    right: EmbeddingWidgetCollection,
    left: EmbeddingWidgetCollection,
):
    """Adds a `.ilocs` tuple trait to the final widget.

    Containts the (left, right) selections.
    """
    initial = (
        left.categorial_scatter.selection(),
        right.categorial_scatter.selection(),
    )
    widget.add_traits(ilocs=traitlets.Tuple(initial))

    ipywidgets.dlink(
        source=(left.categorial_scatter.widget, "selection"),
        target=(widget, "ilocs"),
        transform=lambda iloc: (iloc, widget.ilocs[1]),  # type: ignore
    )

    ipywidgets.dlink(
        source=(right.categorial_scatter.widget, "selection"),
        target=(widget, "ilocs"),
        transform=lambda iloc: (widget.ilocs[0], iloc),  # type: ignore
    )


# Created with https://gka.github.io/palettes/#/256|d|19ffff,33bbff,444444|444444,ff5023,ffaa00|1|1
diverging_cmap = [
    '#19ffff',
    '#1cfdff',
    '#1efcff',
    '#20faff',
    '#22f8ff',
    '#24f6fe',
    '#26f5fe',
    '#27f3fe',
    '#29f1fd',
    '#2af0fd',
    '#2beefc',
    '#2decfc',
    '#2eebfb',
    '#2fe9fb',
    '#31e7fa',
    '#32e6f9',
    '#33e4f9',
    '#34e2f8',
    '#35e1f7',
    '#36dff6',
    '#37ddf5',
    '#38dcf4',
    '#39daf3',
    '#39d9f2',
    '#3ad7f1',
    '#3bd5f0',
    '#3cd4ef',
    '#3dd2ee',
    '#3dd1ed',
    '#3ecfec',
    '#3fcdeb',
    '#40ccea',
    '#40cae8',
    '#41c9e7',
    '#42c7e6',
    '#42c5e5',
    '#43c4e3',
    '#43c2e2',
    '#44c1e1',
    '#45bfdf',
    '#45bede',
    '#46bcdd',
    '#46bbdb',
    '#47b9da',
    '#47b8d8',
    '#48b6d7',
    '#48b5d6',
    '#49b3d4',
    '#49b1d3',
    '#49b0d1',
    '#4aaed0',
    '#4aadce',
    '#4babcd',
    '#4baacb',
    '#4ba8c9',
    '#4ca7c8',
    '#4ca6c6',
    '#4ca4c5',
    '#4ca3c3',
    '#4da1c1',
    '#4da0c0',
    '#4d9ebe',
    '#4e9dbc',
    '#4e9bbb',
    '#4e9ab9',
    '#4e98b7',
    '#4e97b6',
    '#4f95b4',
    '#4f94b2',
    '#4f93b1',
    '#4f91af',
    '#4f90ad',
    '#4f8eab',
    '#4f8daa',
    '#508ba8',
    '#508aa6',
    '#5089a4',
    '#5087a3',
    '#5086a1',
    '#50849f',
    '#50839d',
    '#50819b',
    '#50809a',
    '#507f98',
    '#507d96',
    '#507c94',
    '#507b92',
    '#507991',
    '#50788f',
    '#50768d',
    '#50758b',
    '#4f7489',
    '#4f7287',
    '#4f7186',
    '#4f7084',
    '#4f6e82',
    '#4f6d80',
    '#4f6c7e',
    '#4e6a7c',
    '#4e697a',
    '#4e6879',
    '#4e6677',
    '#4e6575',
    '#4d6473',
    '#4d6271',
    '#4d616f',
    '#4d606d',
    '#4c5e6b',
    '#4c5d6a',
    '#4c5c68',
    '#4b5b66',
    '#4b5964',
    '#4b5862',
    '#4a5760',
    '#4a555e',
    '#4a545c',
    '#49535a',
    '#495259',
    '#495057',
    '#484f55',
    '#484e53',
    '#474d51',
    '#474b4f',
    '#464a4d',
    '#46494b',
    '#45484a',
    '#454648',
    '#454546',
    '#474444',
    '#494543',
    '#4c4543',
    '#4e4643',
    '#514643',
    '#534642',
    '#554742',
    '#584742',
    '#5a4842',
    '#5c4841',
    '#5f4841',
    '#614941',
    '#634941',
    '#654a40',
    '#684a40',
    '#6a4a40',
    '#6c4b40',
    '#6e4b3f',
    '#704c3f',
    '#724c3f',
    '#744c3e',
    '#764d3e',
    '#794d3e',
    '#7b4e3e',
    '#7d4e3d',
    '#7f4e3d',
    '#814f3d',
    '#834f3c',
    '#85503c',
    '#87503c',
    '#89513c',
    '#8b513b',
    '#8c513b',
    '#8e523b',
    '#90523a',
    '#92533a',
    '#94533a',
    '#96543a',
    '#985439',
    '#9a5539',
    '#9c5539',
    '#9d5538',
    '#9f5638',
    '#a15638',
    '#a35737',
    '#a55737',
    '#a75837',
    '#a85836',
    '#aa5936',
    '#ac5936',
    '#ae5a36',
    '#af5b35',
    '#b15b35',
    '#b35c35',
    '#b55c34',
    '#b65d34',
    '#b85d34',
    '#ba5e33',
    '#bb5f33',
    '#bd5f32',
    '#be6032',
    '#c06032',
    '#c26131',
    '#c36231',
    '#c56231',
    '#c66330',
    '#c86430',
    '#ca652f',
    '#cb652f',
    '#cd662f',
    '#ce672e',
    '#cf672e',
    '#d1682d',
    '#d2692d',
    '#d46a2d',
    '#d56b2c',
    '#d76b2c',
    '#d86c2b',
    '#d96d2b',
    '#db6e2a',
    '#dc6f2a',
    '#dd702a',
    '#df7129',
    '#e07229',
    '#e17328',
    '#e37328',
    '#e47427',
    '#e57527',
    '#e67626',
    '#e77726',
    '#e97825',
    '#ea7924',
    '#eb7b24',
    '#ec7c23',
    '#ed7d23',
    '#ee7e22',
    '#ef7f22',
    '#f08021',
    '#f18120',
    '#f28220',
    '#f3841f',
    '#f4851e',
    '#f4861e',
    '#f5871d',
    '#f6881c',
    '#f78a1b',
    '#f88b1a',
    '#f88c1a',
    '#f98e19',
    '#fa8f18',
    '#fa9017',
    '#fb9216',
    '#fb9315',
    '#fc9414',
    '#fc9613',
    '#fd9712',
    '#fd9911',
    '#fe9a10',
    '#fe9c0e',
    '#fe9d0d',
    '#ff9f0c',
    '#ffa00a',
    '#ffa208',
    '#ffa307',
    '#ffa505',
    '#ffa703',
    '#ffa802',
    '#ffaa00'
]


def robust_labels(labels: npt.ArrayLike, robust: npt.NDArray[np.bool_] | None = None):
    if robust is not None:
        labels = np.where(
            robust,
            labels,
            NON_ROBUST_LABEL,
        )
    return pd.Series(labels, dtype="category")


@typing.overload
def create_colormaps(cats: typing.Iterable[str]) -> dict:
    ...


@typing.overload
def create_colormaps(
    cats: typing.Iterable[str], *other: typing.Iterable[str]
) -> tuple[dict, ...]:
    ...


def create_colormaps(
    cats: typing.Iterable[str], *others: typing.Iterable[str]
) -> dict | tuple[dict, ...]:
    all_categories = set(cats)
    for other in others:
        all_categories.update(other)

    # create unified colormap
    lookup = dict(
        zip(
            all_categories,
            itertools.cycle(glasbey_dark[1:]),
        )
    )

    # force non-robust to be grey
    lookup[NON_ROBUST_LABEL] = "#333333"

    # create separate colormaps for each component
    cmaps = tuple({c: lookup[c] for c in cmp} for cmp in (cats, *others))
    if len(cmaps) == 1:
        return cmaps[0]
    return cmaps
