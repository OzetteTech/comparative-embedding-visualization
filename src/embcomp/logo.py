import dataclasses
import json
import pathlib
import re
import uuid
from typing import Literal, Union

import IPython.display
import ipywidgets
import jinja2
import pandas as pd
import traitlets
import traittypes

here = pathlib.Path(__file__).parent

# fmt: off
HTML_TEMPLATE_DEV = jinja2.Template("""
<style>
iframe {
    border: none;
    width: {{ width }}px;
    height: {{ height + 10 }}px
}
</style>
<iframe id="{{ id }}" src="http://localhost:5173"></iframe>
<script type="module">
    let frame = document.getElementById('{{id}}');
    let message = {
        dataJson: `{{ data }}`,
        height: {{ height }},
        width: {{ width }},
    };
    frame.addEventListener("load", () => {
        frame.contentWindow.postMessage(message, "*");
    })
</script>
""")
# fmt: on

# fmt: off
HTML_TEMPLATE = jinja2.Template("""
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
    let data = JSON.parse(`{{ data }}`);
    let options = JSON.parse(`{{ options }}`);
    document.getElementById("{{ id }}")?.appendChild(
        AnnotationLogo(data, options)
    );
</script>
""")
# fmt: on


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


class LogoHTML(ipywidgets.Output):
    counts = traitlets.List(traitlets.Dict())
    threshold = traitlets.Int()

    def __init__(
        self, counts: Union[None, list[dict]] = None, threshold: int = 10, **options
    ):
        super().__init__()
        self._options = options
        self.counts = counts or []
        self.threshold = threshold

    @traitlets.observe("threshold", "counts")
    def _render(self, _change):
        self.clear_output()
        options = {"threshold": self.threshold, **self._options}
        if len(self.counts) == 0:
            html = ""
        else:
            html = HTML_TEMPLATE.render(
                id=uuid.uuid4().hex,
                data=json.dumps(self.counts),
                options=json.dumps(options),
            )
        with self:
            IPython.display.display(IPython.display.HTML(html))


class AnnotationLogo(ipywidgets.VBox):
    labels = traittypes.Series(default_value=pd.Series([], dtype="object"))
    selection = traittypes.Array(None, allow_none=True)

    def __init__(self, labels, **kwargs):
        self._logo = LogoHTML(**kwargs)
        self._threshold = ipywidgets.IntSlider(
            description="threshold",
            value=100,
            min=0,
            max=1000,
        )
        self.labels = labels
        ipywidgets.link((self._logo, "threshold"), (self._threshold, "value"))
        super().__init__([self._threshold, self._logo])

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
