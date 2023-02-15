from __future__ import annotations

import dataclasses
import pathlib
import re
import uuid
from typing import Literal

import anywidget
import IPython.display
import ipywidgets
import jinja2
import numpy as np
import pandas as pd
import traitlets
import traittypes

from cev._widget_utils import link_widgets

here = pathlib.Path(__file__).parent


@dataclasses.dataclass
class Marker:
    name: str
    annotation: Literal["+", "-"]

    def __str__(self) -> str:
        return self.name + self.annotation


Annotation = list[Marker]


def parse_label(label: str) -> Annotation:
    return [
        Marker(inner_label[:-1], inner_label[-1])
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


class HTMLWidget(ipywidgets.Output):
    _template = jinja2.Template("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe(lambda _: self._render(), names=self.class_own_traits().keys())
        self._render()

    def _render(self):
        state = {name: getattr(self, name) for name in self.class_own_traits()}
        html = self._template.render(id=uuid.uuid4().hex, **state)
        self.clear_output()
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
                spans.style(
                  "background-color",
                  (d) => colorScale((d.value / data.total + 1) / 2),
                );
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
        link_widgets((self._logo, "threshold"), (self._threshold, "value"))
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

        self._threshold.max = max(count["count"] for count in self._logo.counts)
        self._threshold.value = self._threshold.max


_marker_indicator_esm = """
export async function render(view) {
  function rerender() {
    let el = view.el;
    Object.assign(el.style, { display: "flex", fontFamily: "monospace" });
    let markers = view.model.get("markers");
    let level = view.model.get("value") - 1;
    let diff = markers.length - el.childElementCount;
    if (diff > 0) {
      for (let i = 0; i < diff; i++) {
        let button = document.createElement("button");
        Object.assign(button.style, {
          // padding: "3px 3px",
          // borderRight: "1px solid",
          // borderTop: "1px solid",
          // borderBottom: "1px solid",
        });
        button.addEventListener("click", function () {
          view.model.set("value", +this.getAttribute("data-index") + 1);
          view.model.save_changes();
        });
        el.appendChild(button);
      }
    } else if (diff < 0) {
      for (let i = 0; i < -diff; i++) {
        el.removeChild(el.lastChild);
      }
    }
    for (let i = 0; i < markers.length; i++) {
      let child = el.childNodes[i];
      if (i <= level) {
        child.style.backgroundColor = "#d5d5d5";
      } else {
        child.style.backgroundColor = "#f5f5f5";
      }
      child.textContent = markers[i];
      child.setAttribute("data-index", i);
    }
  }
  view.model.on("change:markers", rerender);
  view.model.on("change:value", rerender);
  rerender();
}
"""


class MarkerIndicator(anywidget.AnyWidget):
    _esm = _marker_indicator_esm
    markers = traitlets.List().tag(sync=True)
    value = traitlets.Int().tag(sync=True)


class Logo(HTMLWidget):
    _template = jinja2.Template(
        """
    <div id="{{ id }}"></div>
    <script type="module">
        import * as d3 from "https://esm.sh/d3@7";
        function labelParts(label) {
          return label.split(/(\w+[\-|\+])/).filter(Boolean);
        }
        let root = d3.select(document.getElementById("{{ id }}"));
        let counts = Object.entries(JSON.parse(`{{ counts | tojson }}`));
        let totalCounts = counts.reduce((acc, [label, count]) => {
            let markers = labelParts(label);
            for (let i = 0; i < markers.length; i++) {
                if (markers[i].at(-1) == "-") {
                    acc[i][1] += count;
                } else {
                    acc[i][0] += count;
                }
            }
            return acc;
        }, Array.from({ length: labelParts(counts[0][0]).length }, d => [0, 0]));
        let total = counts.map(d => d[1]).reduce((a, b) => a + b, 0);
        let data = labelParts(counts[0][0]).map((marker, i) => {
            return [marker.slice(0, -1), totalCounts[i].map(d => d / total)]
        });

        root
            .style("display", "flex")
            .style("flex-wrap", "wrap")
            .style("font-size", "12px")
            .style("margin-left", "36px") // aligns to jscatter plot
            .selectAll("div")
            .data(data)
            .join("div")
            .call(function(g) {
                g.style("display", "flex")
                 .style("align-items", "center")
                 .style("border", "black solid 0.5px")
                 .style("background-color", d => d[1][0] > 0.5 ? "black" : "white")
                 .style("margin", "2px")

                g.append("div")
                    .style("margin", "0px 4px")
                    .style("color", d => d[1][0] > 0.5 ? "white" : "black")
                    .text(d => d[0])

                let [width, height] = [13, 20];
                let svg = g.append("svg")
                    .attr("width", width)
                    .attr("height", height)

                if (total == 0) return;

                svg.append("rect")
                    .attr("width", width)
                    .attr("height", d => d[1][0] * height)
                    .attr("fill", "black")
                svg.append("text")
                    .attr("alignment-baseline", "middle")
                    .attr("text-anchor", "middle")
                    .attr("fill", "white")
                    .attr("y", d => d[1][0] * height / 2)
                    .attr("x", d => width / 2)
                    .text(d => d[1][0] < 0.2 ? "" : "+");  
                svg.append("rect")
                    .attr("y", d => d[1][0] * height)
                    .attr("width", width)
                    .attr("height", d => d[1][1] * height)
                    .attr("fill", "white")
                    .attr("stroke", "white")
                svg.append("text")
                    .attr("alignment-baseline", "middle")
                    .attr("text-anchor", "middle")
                    .attr("x", d => width / 2)
                    .attr("y", d => d[1][0] * height + (d[1][1] * height / 2))
                    .text(d => d[1][1] < 0.2 ? "" : "-")
            })
    </script>
    """
    )

    counts = traitlets.Dict()
