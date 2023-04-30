from __future__ import annotations

import uuid

import IPython.display
import ipywidgets
import jinja2

__all__ = ["HTMLWidget"]


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
