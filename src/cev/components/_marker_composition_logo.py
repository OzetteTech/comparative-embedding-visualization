from __future__ import annotations

import jinja2
import traitlets

from ._html_widget import HTMLWidget

__all__ = ["MarkerCompositionLogo"]


class MarkerCompositionLogo(HTMLWidget):
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
                 .style("border", "var(--jp-ui-font-color0) solid 0.5px")
                 .style("background-color", d => d[1][0] > 0.5 ? "var(--jp-ui-font-color0)" : "var(--jp-ui-inverse-font-color0)")
                 .style("margin", "2px")

                g.append("div")
                    .style("margin", "0px 4px")
                    .style("color", d => d[1][0] > 0.5 ? "var(--jp-ui-inverse-font-color0)" : "var(--jp-ui-font-color0)")
                    .text(d => d[0])

                let [width, height] = [13, 20];
                let svg = g.append("svg")
                    .attr("width", width)
                    .attr("height", height)

                if (total == 0) return;

                svg.append("rect")
                    .attr("width", width)
                    .attr("height", d => d[1][0] * height)
                    .attr("fill", "currentColor")
                    .style("color", "var(--jp-ui-font-color0)")
                svg.append("text")
                    .attr("alignment-baseline", "middle")
                    .attr("text-anchor", "middle")
                    .attr("fill", "currentColor")
                    .attr("y", d => d[1][0] * height / 2)
                    .attr("x", d => width / 2)
                    .style("color", "var(--jp-ui-inverse-font-color0)")
                    .text(d => d[1][0] < 0.2 ? "" : "+");  
                svg.append("rect")
                    .attr("y", d => d[1][0] * height)
                    .attr("width", width)
                    .attr("height", d => d[1][1] * height)
                    .attr("fill", "currentColor")
                    .attr("stroke", "currentColor")
                    .style("color", "var(--jp-ui-inverse-font-color0)")
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
