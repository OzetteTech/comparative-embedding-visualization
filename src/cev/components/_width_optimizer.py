import anywidget

__all__ = ["WidthOptimizer"]


class WidthOptimizer(anywidget.AnyWidget):
    _esm = """
    export function render(view) {
      setTimeout(() => {
        view.el.parentNode.style.setProperty('--jp-widgets-inline-label-width', 'auto');
      }, 0);
    }
    """
