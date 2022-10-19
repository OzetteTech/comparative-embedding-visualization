import json
import pathlib
import uuid
from typing import Union

import IPython.display
import jinja2
import pandas as pd

here = pathlib.Path(__file__).parent

DEV = False

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
<div id="{{ id }}"></div>
<script type="module">
""" + (here / "static" / "logo.js").read_text() + """
    render(document.getElementById('{{ id }}'), {
        dataJson: `{{ data }}`,
        height: {{ height }},
        width: {{ width }},
    });
</script>
""")
# fmt: on


def label_comparer(
    labels: pd.Series,
    robust: Union[set[str], None] = None,
    width: int = 300,
    height: int = 400,
    empty: bool = False,
    **kwargs
):
    if empty:
        data = [dict(label=labels[0], count=0)]
    else:
        counts = labels.value_counts(sort=False)
        robust = robust or set(counts.keys())
        data = [dict(label=k, count=v, robust=(k in robust)) for k, v in counts[counts > 0].items()]  # type: ignore

    html = (HTML_TEMPLATE_DEV if DEV else HTML_TEMPLATE).render(
        id=uuid.uuid4().hex,
        data=json.dumps(data),
        width=width,
        height=height,
        **kwargs,
    )

    return IPython.display.HTML(html)
