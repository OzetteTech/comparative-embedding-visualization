from __future__ import annotations

import dataclasses
import re
import typing

if typing.TYPE_CHECKING:
    import pandas as pd


@dataclasses.dataclass
class Marker:
    name: str
    annotation: typing.Literal["+", "-"]

    def __str__(self) -> str:
        return self.name + self.annotation


def parse_label(label: str) -> list[Marker]:
    return [
        Marker(inner_label[:-1], inner_label[-1])  # type: ignore
        for inner_label in re.split("(\w+[\-|\+])", label)
        if inner_label
    ]


def trim_label_series(labels: pd.Series, level: int):
    if level == 0:
        return labels
    return (
        labels.str.split("(\w+[\+|\-])", regex=True)
        .str.slice(0, -level * 2)
        .str.join("")
    )
