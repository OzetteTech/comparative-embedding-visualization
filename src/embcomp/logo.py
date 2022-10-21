import json
import pathlib
import uuid
import re

import IPython.display
import ipywidgets
import jinja2
import traitlets

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


class AnnotationLogo(ipywidgets.Output):
    counts = traitlets.List(traitlets.Dict(), minlen=1)
    threshold = traitlets.Int()
    label_level = traitlets.Int()

    def __init__(self, counts, threshold: int = 10, label_level: int = 0, **options):
        super().__init__()
        self._options = options
        self.counts = counts
        self.threshold = threshold
        self.label_level = label_level

    @property
    def levels(self):
        return len(label_parts(self.counts[0]["label"])) - 1

    @traitlets.observe("threshold", "counts", "label_level")
    def _render(self, _change):
        self.clear_output()
        options = {"threshold": self.threshold, **self._options}
        html = HTML_TEMPLATE.render(
            id=uuid.uuid4().hex,
            data=json.dumps(trim_labels(self.counts, self.label_level)),
            options=json.dumps(options),
        )
        with self:
            IPython.display.display(IPython.display.HTML(html))
