import anywidget
import traitlets

__all__ = ["MarkerSelectionIndicator"]


class MarkerSelectionIndicator(anywidget.AnyWidget):
    _esm = """
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

    markers = traitlets.List().tag(sync=True)
    value = traitlets.Int().tag(sync=True)
