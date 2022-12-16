import * as d3 from "https://esm.sh/d3";

function labelParts(label) {
	return label.split(/(\w+[\-|\+])/).filter(Boolean);
}

function parseEntry({ label, count }) {
	let markers = [];
	let counts = [];

	for (let marker of labelParts(label)) {
		markers.push(marker.slice(0, -1));
		counts.push(marker.at(-1) === "-" ? -count : count);
	}
	return { markers, counts };
}

export function AnnotationLogo(
	data,
	{
		threshold = 500,
		robustOnly = false,
		marginTop = 20, // the top margin, in pixels
		marginRight = 10, // the right margin, in pixels
		marginBottom = 10, // the bottom margin, in pixels
		marginLeft = 100, // the left margin, in pixels
		width = 640, // the outer width of the chart, in pixels
		height = 500, // the outer height of the chart, in pixels
		xPadding = 0.1, // amount of x-range to reserve to separate bars
		// color = "currentColor", // bar fill color
		color = "#cbd5e1",
		range = d3.schemeRdBu[3],
	} = {},
) {
	let xRange = [marginLeft, width - marginRight]; // [left, right]

	// helper to derive y-scale ranges
	let y = d3
		.scaleLinear()
		.domain([0, 1])
		.range([marginTop, height - marginBottom]);

	let yRangeBar = [y(0.25), y(0)];
	let yRangeCell = [y(1), y(0.25) + 10];

	let annotation = {
		markers: parseEntry(data[0]).markers,
		data: data.map((d) => ({
			label: d.label,
			robust: d.robust,
			total: d.count,
			counts: parseEntry(d).counts,
		})),
	};

	let grouped = d3.rollup(
		robustOnly
			? d3.sort(annotation.data, (d) => d.robust === true && -d.total)
			: d3.sort(annotation.data, (d) => -d.total),
		(g) =>
			g.reduce(
				(acc, d) => {
					acc.total += d.total;
					for (let i = 0; i < acc.counts.length; i++) {
						acc.counts[i] += d.counts[i];
					}
					return acc;
				},
				{ total: 0, counts: Array(g[0].counts.length).fill(0) },
			),
		robustOnly
			? (d) =>
				d.robust === false
					? "not-robust"
					: (d.total <= threshold ? "other" : d.label)
			: (d) => (d.total <= threshold ? "other" : d.label),
	);

	let counts = Array.from(grouped.values());

	// Compute values.
	let X = d3.map(counts, (_, i) => i);
	let Y = d3.map(counts, (d) => d.total);

	// Compute default domains, and unique the x-domain.
	let xDomain = new d3.InternSet(X);
	let yDomain = [0, d3.max(Y)];

	// Omit any data not present in the x-domain.
	let I = d3.range(X.length).filter((i) => xDomain.has(X[i]));

	// Construct scales, axes, and formats.
	let xScale = d3.scaleBand(xDomain, xRange).padding(xPadding);
	let yScale = d3.scaleLinear(yDomain, yRangeBar);
	let yAxis = d3.axisLeft(yScale).ticks(height / 100);

	let svg = d3.create("svg")
		.attr("width", width)
		.attr("height", height)
		.attr("viewBox", [0, 0, width, height])
		// .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

	svg
		.append("g")
		.attr("transform", `translate(${marginLeft},0)`)
		.call(yAxis)
		.call((g) => g.select(".domain").remove())
		.call((g) =>
			g
				.selectAll(".tick line")
				.clone()
				.attr("x2", width - marginLeft - marginRight)
				.attr("stroke-opacity", 0.1)
		);

	let bar = svg
		.append("g")
		.attr("fill", color)
		.selectAll("rect")
		.data(I)
		.join("rect")
		.attr("x", (i) => xScale(X[i]))
		.attr("y", (i) => yScale(Y[i]))
		.attr("height", (i) => yScale(0) - yScale(Y[i]))
		.attr("width", xScale.bandwidth());

	// cell
	yScale = d3
		.scaleBand([...annotation.markers].reverse(), yRangeCell)
		.padding(0.1);
	yAxis = d3.axisLeft(yScale).ticks(height / 40);

	svg
		.append("g")
		.attr("transform", `translate(${marginLeft},0)`)
		.call(yAxis)
		.call((g) => g.select(".domain").remove());

	let groups = svg
		.append("g")
		.selectAll("g")
		.data(I)
		.join("g")
		.attr("transform", (i) => `translate(${xScale(X[i])},0)`);

	let lastScaleX = d3
		.scaleLinear()
		.domain([0, 1])
		.range([0, xScale.bandwidth()]);

	let colorScale = d3.interpolateRgbBasis(range);

	groups
		.append("g")
		.selectAll("rect")
		.data((i) => {
			let normalized = counts[i].counts.map((count) => {
				return count / counts[i].total; // scale between (-1, 1)
			});
			return normalized;
		})
		.join("rect")
		.attr("fill", (normed) => colorScale((normed + 1) / 2)) // interpolate (-1, 1) => (0, 1)
		.attr("y", (_, i) => yScale(annotation.markers[i]))
		.attr("x", 0)
		.attr("height", yScale.bandwidth())
		.attr("width", (normed) => lastScaleX(Math.abs(normed)));

	return svg.node();
}
