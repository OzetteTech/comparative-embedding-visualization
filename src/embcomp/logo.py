import json
import pathlib
import re
import uuid
from typing import Union

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


def label_parts(label: str) -> list[str]:
    return [l for l in re.split("(\w+[\-|\+])", label) if l]


def trim_labels(data, level: int):
    if level == 0:
        return data
    return [
        dict(
            label="".join(label_parts(e["label"])[:-level]),
            count=e["count"],
        )
        for e in data
    ]

class Labeler(traitlets.HasTraits):
    labels = traittypes.Series(default_value=pd.Series([], dtype="object"))
    level = traitlets.Int()

    def __init__(self, labels: pd.Series, level: int = 0):
        self._labels = labels
        super().__init__(labels=labels, level=level)

    @traitlets.observe("level")
    def _trim_labels(self, change):
        assert change.name == "level"
        level = change.new
        if level == 0:
            self.labels = self._labels
            return
        self.labels = (
            self._labels.str.split("(\w+[\+|\-])", regex=True)
            .str.slice(0, -level * 2)
            .str.join("")
        )

    @property
    def levels(self):
        label = str(self._labels[0])
        return len(label_parts(label)) - 1


class LogoHTML(ipywidgets.Output):
    counts = traitlets.List(traitlets.Dict())
    threshold = traitlets.Int()

    def __init__(self, counts: Union[None, list[dict]] = None, threshold: int = 10, **options):
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
    selection = traittypes.Array(None, allow_none = True)

    def __init__(self, labels, **kwargs):
        self._labeler = Labeler(labels)
        self._logo = LogoHTML(**kwargs)
        self._label_slider = ipywidgets.IntSlider(
            description="label level",
            min=0,
            max=self._labeler.levels,
        )
        self._threshold = ipywidgets.IntSlider(
            description="threshold",
            value=100,
            min=0,
            max=1000,
        )

        self.labels = self._labeler.labels
        ipywidgets.link((self._logo, "threshold"), (self._threshold, "value"))
        ipywidgets.link((self._labeler, "level"), (self._label_slider, "value"))
        ipywidgets.link((self._labeler, "labels"), (self, "labels"))

        controls = ipywidgets.HBox([self._label_slider, self._threshold])

        super().__init__([controls, self._logo])

    @traitlets.observe("labels", "selection")
    def _selection_change(self, change):
        labels = change.new if change.name == "labels" else self.labels
        selection = change.new if change.name == "selection" else self.selection

        try:
            empty = [dict(label=str(labels[0]), count=0)]
        except KeyError:
            return

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
