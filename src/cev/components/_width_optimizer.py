import anywidget

__all__ = ["WidthOptimizer"]


class WidthOptimizer(anywidget.AnyWidget):
    """This widget gets rid of unwanted whitespace in front of ipywidgets"""

    _esm = """
    export function render(view) {
      setTimeout(() => {
        view.el.parentNode.style.setProperty('--jp-widgets-inline-label-width', 'auto');
      }, 0);
    }
    """
