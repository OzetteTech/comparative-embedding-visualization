import dataclasses
import pathlib
import re
import uuid
from typing import Literal

import IPython.display
import ipywidgets
import jinja2
import numpy as np
import pandas as pd
import traitlets
import traittypes

here = pathlib.Path(__file__).parent


@dataclasses.dataclass
class Marker:
    name: str
    annotation: Literal["+", "-"]

    def __str__(self) -> str:
        return self.name + self.annotation


Annotation = list[Marker]


def parse_label(label: str) -> Annotation:
    return [Marker(l[:-1], l[-1]) for l in re.split("(\w+[\-|\+])", label) if l]


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


class HTMLWidget(ipywidgets.Output):
    _template = jinja2.Template("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe(lambda _: self._render(), names=self.class_own_traits().keys())
        self._render()

    def _render(self):
        self.clear_output()
        state = {name: getattr(self, name) for name in self.class_own_traits()}
        html = self._template.render(id=uuid.uuid4().hex, **state)
        with self:
            IPython.display.display(IPython.display.HTML(html))


class LogoHTML(HTMLWidget):
    # fmt: off
    _template = jinja2.Template("""
    <style>
        .annotation-logo {
            // background-color: #e6ffec;
            padding-right: 0;
            height: unset;
        }
    </style>
    <div id="{{ id }}" class="annotation-logo"></div>
    <script type="module">
    """ + (here / "static" / "AnnotationLogo.js").read_text() + """
        let counts = JSON.parse(`{{ counts | tojson }}`);
        let options = JSON.parse(`{{ options | tojson }}`);
        if (counts.length > 0) {
            document.getElementById("{{ id }}")?.appendChild(
                AnnotationLogo(counts, { threshold: {{ threshold }}, ...options })
            );
        }
    </script>
    """)
    # fmt: on

    counts = traitlets.List(traitlets.Dict())
    options = traitlets.Dict()
    threshold = traitlets.Int()


class ConsensusLogo(HTMLWidget):
    # fmt: off
    _template = jinja2.Template("""
    <style>
        .consensus-logo {
            margin: 10px;
            position: relative;
        }
        .consensus-logo > button {
            position: absolute;
            top: 1px;
            right: 2px;
            background-color: rgb(255 255 255 / 80%);
            padding: 3px 4px 2px 4px;
            border-radius: 6px;
            visibility: hidden;
            border: 2px solid grey;
        }
        .consensus-logo > button:hover {
            background-color: rgb(255 255 255 / 100%);
        }
        .consensus-logo > button:active {
            border: 2px solid black;
        }
    </style>
    <div id="{{ id }}" class="consensus-logo">
        <div></div>
        <button title="Copy">
            <span hidden>Copy to clipboard</span>
            <svg height="15" width="15" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
            </svg>
        </button>
    </div>

    <script type="module">
        import * as d3 from "https://esm.sh/d3@7";

        function consensusStr({ name, value }) {
            return name + (value === 0 ? "" : value < 0 ? "-" : "+");
        }

        function ConsensusLogo(data, {
            range = d3.schemeRdBu[3],
            spacing = 2,
        } = {}) {
            let colorScale = d3.interpolateRgbBasis(range);

            let pre = d3.create("pre")
                .style("display", "flex")
                .style("flex-wrap", "wrap")
                .style("gap", `${spacing}px`)
                .style("font-size", "var(--jp-widgets-font-size, 11px)");

            let spans = pre
                .selectAll("span")
                .data(data.counts)
                .join("span")
                .style("padding", "0px 4px")
                .text(consensusStr)

            if (data.total > 0) {
                spans.style("background-color", (d) => colorScale((d.value / data.total + 1) / 2))
            } else {
                spans.style("background-color", "#f5f5f5");
            }

            return pre.node();
        }
        let data = JSON.parse(`{{ data | tojson }}`);
        let root = document.getElementById(`{{ id }}`);
        root.querySelector("div").appendChild(ConsensusLogo(data));
        let button = root.querySelector("button");
        root.addEventListener("mouseover", () => {
            button.style.visibility = "visible";
        });
        root.addEventListener("mouseout", () => {
            button.style.visibility = "hidden";
        });
        button.addEventListener("click", () => {
            let consensus = data.counts.map(consensusStr).join("");
            navigator.clipboard?.writeText?.(JSON.stringify(consensus));
        });
    </script>
    """)
    # fmt: on
    data = traitlets.Dict({})


class AnnotationLogo(ipywidgets.VBox):
    labels = traittypes.Series(default_value=pd.Series([], dtype="object"))
    selection = traittypes.Array(None, allow_none=True)

    def __init__(self, labels, **kwargs):
        counts = kwargs.pop("counts", [])
        threshold = kwargs.pop("threshold", 0)
        consensus_logo = ConsensusLogo()
        self._logo = LogoHTML(counts=counts, threshold=threshold, options=kwargs)
        self._threshold = ipywidgets.IntSlider(
            description="threshold",
            value=100,
            min=0,
            max=1000,
        )
        self.labels = labels
        ipywidgets.link((self._logo, "threshold"), (self._threshold, "value"))
        ipywidgets.dlink(
            (self._logo, "counts"),
            (consensus_logo, "data"),
            transform=consensus_from_counts,
        )
        header = ipywidgets.VBox([consensus_logo, self._threshold])
        super().__init__([header, self._logo])

    @traitlets.observe("labels", "selection")
    def _selection_change(self, change):
        labels = change.new if change.name == "labels" else self.labels
        selection = change.new if change.name == "selection" else self.selection

        try:
            representative_label = str(labels[0])  # type: ignore
        except KeyError:
            return

        empty = [dict(label=representative_label, count=0)]

        if selection is None:
            self._logo.counts = empty
            return

        counts = labels[selection].value_counts(sort=False)  # type: ignore
        if len(counts) == 0:
            self._logo.counts = empty
            return

        self._logo.counts = [
            dict(label=k, count=v)  # type: ignore
            for k, v in counts[counts > 0].items()  # type: ignore
        ]

        self._threshold.max = max(l["count"] for l in self._logo.counts)  # type: ignore
        self._threshold.value = self._threshold.max
