import * as d3 from "https://esm.sh/d3";

export function labelParts(label) {
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

export function CompareAnnotation(
	data,
	{
		svgElement,
		threshold = 500,
		groupRobust = false,
		marginTop = 20, // the top margin, in pixels
		marginRight = 0, // the right margin, in pixels
		marginBottom = 10, // the bottom margin, in pixels
		marginLeft = 100, // the left margin, in pixels
		width = 640, // the outer width of the chart, in pixels
		height = 500, // the outer height of the chart, in pixels
		xPadding = 0.1, // amount of x-range to reserve to separate bars
		color = "currentColor", // bar fill color
		range = d3.schemeRdBu[3],
	} = {},
) {
	let xRange = [marginLeft, width - marginRight]; // [left, right]
	let yRange = [height - marginBottom, marginTop]; // [bottom, top]

	let mid = (yRange[0] - yRange[1]) * 0.25 + yRange[1];
	let yRangeTop = [mid, yRange[1]];
	let yRangeBottom = [yRange[0], mid + 10];

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
		groupRobust
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
		groupRobust
			? (d) =>
				d.robust === false
					? "not-robust"
					: (d.total < threshold ? "other" : d.label)
			: (d) => (d.total < threshold ? "other" : d.label),
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
	let yScale = d3.scaleLinear(yDomain, yRangeTop);
	let yAxis = d3.axisLeft(yScale).ticks(height / 100);

	let svg = (svgElement ? d3.select(svgElement) : d3.create("svg"))
		.attr("width", width)
		.attr("height", height)
		.attr("viewBox", [0, 0, width, height])
		.attr("style", "max-width: 100%; height: auto; height: intrinsic;");

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
		.scaleBand([...annotation.markers].reverse(), yRangeBottom)
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
				return count / counts[i].total; // -1 - 1
				// return (x + 1) / 2; // 0 - 1
			});
			return normalized;
		})
		.join("rect")
		.attr("fill", (normed) => colorScale((normed + 1) / 2)) // 0 - 1
		.attr("y", (_, i) => yScale(annotation.markers[i]))
		.attr("x", 0)
		.attr("height", yScale.bandwidth())
		.attr("width", (normed) => lastScaleX(Math.abs(normed)));

	return svg.node();
}

export function render(root, { dataJson, width, height }) {
  let data = JSON.parse(dataJson);
	let labelLevel = 0;
	let props = { threshold: 10, groupRobust: false, width, height };

	function trimLabels(data, level) {
		if (level === 0) return data;
		return data.map((d) => {
			let parts = d.label.split(/(\w+[\-|\+])/).filter(Boolean);
			let label = parts.slice(0, -level).join("");
			return { label, count: d.count };
		});
	}

	function redraw() {
		if (data.length === 0) return;
		let svg = CompareAnnotation(trimLabels(data, labelLevel), props);
		root.lastChild?.tagName === "svg" && root.removeChild(root.lastChild);
		root.appendChild(svg);
	}

	let s1 = Object.assign(document.createElement("input"), {
		type: "range",
		min: 0,
		max: Math.max.apply(null, data.map((d) => d.count)) + 1,
		value: props.threshold,
	});

	let s2 = Object.assign(document.createElement("input"), {
		type: "range",
		min: 0,
		max: data.length === 0 ? 0 : labelParts(data[0].label).length - 1,
		value: 0,
	});

	let toggle = Object.assign(document.createElement("button"), {
		innerHTML: "Robust Only",
	});

	root.appendChild(s1);
	root.appendChild(s2);
	root.appendChild(toggle);

	redraw();

	s1.addEventListener("input", (e) => {
		props.threshold = +e.target.value;
		redraw();
	});

	s2.addEventListener("input", (e) => {
		labelLevel = +e.target.value;
		redraw();
	});

	toggle.addEventListener("click", (e) => {
		props.groupRobust = !props.groupRobust;
		toggle.innerHTML = toggle.innerHTML === "Robust Only"
			? "All"
			: "Robust Only";
		redraw();
	});
}
