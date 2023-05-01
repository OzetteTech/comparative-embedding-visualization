from __future__ import annotations

import dataclasses
import re
import typing

import numpy as np

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


def trim_label(label: str, level: int):
    annotation = parse_label(label)[:-level]
    return "".join(map(str, annotation))


def trim_labels(data, level: int):
    if level == 0:
        return data
    return [dict(label=trim_label(e["label"], level), count=e["count"]) for e in data]


def trim_label_series(labels: pd.Series, level: int):
    if level == 0:
        return labels
    return (
        labels.str.split("(\w+[\+|\-])", regex=True)
        .str.slice(0, -level * 2)
        .str.join("")
    )


def consensus_from_counts(counts: list[dict]):
    markers = None
    total = 0

    data = []
    for entry in counts:
        markers = parse_label(entry["label"])
        count = entry["count"]
        total += count
        vec = np.array(
            [-count if marker.annotation == "-" else count for marker in markers]
        )
        data.append(vec)

    if markers is None:
        return {"total": 0, "counts": []}

    consensus_counts = np.stack(data).sum(axis=0)
    names = map(lambda m: m.name, markers)

    return dict(
        total=total,
        counts=list(
            dict(name=name, value=int(value))
            for (name, value) in zip(names, consensus_counts)
        ),
    )
