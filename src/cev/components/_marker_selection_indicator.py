import anywidget
import traitlets

__all__ = ["MarkerSelectionIndicator"]


class MarkerSelectionIndicator(anywidget.AnyWidget):
    _esm = """
    const FONT_COLOR = "var(--jp-ui-font-color0)";
    const FONT_COLOR_SECONDARY = "var(--jp-ui-font-color1)";
    const BUTTON_BG = "var(--jp-layout-color2)";
    const BUTTON_HOVER_BG = "var(--jp-layout-color2)";
    const BUTTON_ACTIVE_BG = "#1976d2";
    const BUTTON_ACTIVE_HOVER_BG = "#0069d3";
    const BUTTON_ACTIVE_SECONDARY_BG = "var(--jp-ui-font-color3)";
    const NATURAL_COMPARATOR = new Intl.Collator(undefined, { numeric: true }).compare;
    
    export async function render(view) {
      const container = document.createElement("div");
      view.el.appendChild(container);
        
      Object.assign(container.style, {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
      });
    
      const header = document.createElement("div");
      container.appendChild(header);
        
      Object.assign(header.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "2px",
      });
    
      const title = document.createElement("h4");
      header.appendChild(title);
        
      Object.assign(title.style, {
        padding: "0",
        margin: "0",
      });
      title.textContent = "Markers";
    
      const settings = document.createElement("div");
      header.appendChild(settings);
      Object.assign(settings.style, { display: "flex", alignItems: "center" });
    
      const sortLabel = document.createElement("div");
      sortLabel.textContent = "Sort by";  
      Object.assign(sortLabel.style, { fontSize: "0.875em", marginRight: "0.25rem" });
      
      const sortImportance = document.createElement("button");
      sortImportance.textContent = "Expression Discriminability";
      Object.assign(sortImportance.style, {
        background: view.model.get("sort_alphabetically") ? BUTTON_BG : BUTTON_ACTIVE_SECONDARY_BG,
        border: `1px solid ${view.model.get("sort_alphabetically") ? BUTTON_BG : BUTTON_ACTIVE_SECONDARY_BG}`,
        borderRadius: "4px 0 0 4px",
        userSelect: "none",
        cursor: "pointer",
      });
      sortImportance.addEventListener("click", function() {
        view.model.set("sort_alphabetically", false);
        view.model.save_changes();
      });
      
      const sortAlphabetically = document.createElement("button");
      sortAlphabetically.textContent = "Alphabetically";
      Object.assign(sortAlphabetically.style, {
        background: view.model.get("sort_alphabetically") ? BUTTON_ACTIVE_SECONDARY_BG : BUTTON_BG,
        border: `1px solid ${view.model.get("sort_alphabetically") ? BUTTON_ACTIVE_SECONDARY_BG : BUTTON_BG}`,
        borderRadius: "0 4px 4px 0",
        marginLeft: "-1px",
        userSelect: "none",
        cursor: "pointer",
      });
      sortAlphabetically.addEventListener("click", function() {
        view.model.set("sort_alphabetically", true);
        view.model.save_changes();
      });
      
      settings.appendChild(sortLabel);
      settings.appendChild(sortImportance);
      settings.appendChild(sortAlphabetically);
    
      const markersEl = document.createElement("div");
      container.appendChild(markersEl);
        
      Object.assign(markersEl.style, {
        display: "flex",
        flexWrap: "wrap",
        gap: "2px",
      });
      
      function getOrder() {
        const markers = view.model.get("markers");
        return view.model.get("sort_alphabetically")
          ? new Map(markers.map((marker, i) => [marker, i]).sort(([a], [b]) => NATURAL_COMPARATOR(a, b)).map(([marker, i], j) => [i, j]))
          : undefined;
      }
        
      function rerender() {        
        const markers = view.model.get("markers");
        const active = view.model.get("active");
        const diff = markers.length - markersEl.childElementCount;
        
        if (diff > 0) {
          for (let i = 0; i < diff; i++) {
            const button = document.createElement("button");
            
            Object.assign(button.style, {
              background: "var(--marker-selection-indicator-bg)",
              cursor: "pointer",
              padding: "4px 6px",
              border: "0",
              borderRadius: i === 0
                ? "2px 0 0 2px"
                : i === markers.length - 1
                  ? "0 2px 2px 0"
                  : "0",
              userSelect: "none",
            });
            
            button.addEventListener("click", function (event) {
              let newActive = [...view.model.get("active")];
              
              if (event.altKey) {
                newActive = Array.from({ length: markers.length }, (_, j) => j === i);
              } else if (event.shiftKey) {
                const order = getOrder();
                const _i = order ? order.get(i) : i;
                newActive = Array.from({ length: markers.length }, (_, j) => (order ? order.get(j) : j) <= _i);
              } else {
                const numActive = newActive.reduce((num, curr) => num + Number(curr), 0);
                if (!newActive[i] || numActive > 1) newActive[i] = !newActive[i];
              }

              view.model.set("active", newActive);
              view.model.save_changes();
            });
            
            button.addEventListener("mouseenter", function () {
              const active = view.model.get("active");
              button.style.setProperty("--marker-selection-indicator-bg", active[i] ? BUTTON_ACTIVE_HOVER_BG : BUTTON_HOVER_BG);
            });
            
            button.addEventListener("mouseleave", function () {
              const active = view.model.get("active");
              button.style.setProperty("--marker-selection-indicator-bg", active[i] ? BUTTON_ACTIVE_BG : BUTTON_BG);
            });
            
            markersEl.appendChild(button);
          }
        } else if (diff < 0) {
          for (let i = 0; i < -diff; i++) {
            markersEl.removeChild(markersEl.lastChild);
          }
        }
        
        const order = getOrder();
        
        for (let i = 0; i < markers.length; i++) {
          const child = markersEl.childNodes[i];
          
          if (active[i]) {
            child.style.color = "white";
            child.style.setProperty("--marker-selection-indicator-bg", BUTTON_ACTIVE_BG);
          } else {
            child.style.color = FONT_COLOR;
            child.style.setProperty("--marker-selection-indicator-bg", BUTTON_BG);
          }
          
          if (order?.has(i)) {
            child.style.order = order.get(i);
          } else {
            child.style.order = 0;
          }
          
          child.textContent = markers[i];
        }

        const isAlphabetically = view.model.get("sort_alphabetically");
        const isImportance = !isAlphabetically;

        const getButtonStyle = (active) => ({
          background: active ? BUTTON_ACTIVE_SECONDARY_BG : BUTTON_BG,
          border: 0,
          color: active ? FONT_COLOR : FONT_COLOR_SECONDARY,
        });

        Object.assign(sortImportance.style, getButtonStyle(isImportance));
        Object.assign(sortAlphabetically.style, getButtonStyle(isAlphabetically));
      }
      
      view.model.on("change:markers", rerender);
      view.model.on("change:active", rerender);
      view.model.on("change:sort_alphabetically", rerender);
      
      rerender();
    }
    """

    markers = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    active = traitlets.List(trait=traitlets.Bool()).tag(sync=True)
    sort_alphabetically = traitlets.Bool().tag(sync=True)
