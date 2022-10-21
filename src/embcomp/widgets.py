import itertools

import ipywidgets
import jscatter
import numpy as np
import numpy.typing as npt
import pandas as pd

import dataclasses

from embcomp.logo import AnnotationLogo

Embedding = tuple[npt.ArrayLike, npt.NDArray]  # x y  # knn_indices


@dataclasses.dataclass
class PairwiseComponent:
    scatter: jscatter.Scatter
    logo: AnnotationLogo

    def __post_init__(self):
        self.link = ipywidgets.link(
            (self.scatter.widget, "selection"), (self.logo, "selection")
        )

    def show(self):
        return ipywidgets.VBox([self.scatter.show(), self.logo])


def pairwise(
    A: Embedding,
    B: Embedding,
    labels: pd.Series,
    robust: npt.NDArray[np.bool_],
):
    row_height = 600

    left, right = (
        PairwiseComponent(
            scatter=jscatter.Scatter(
                data=pd.DataFrame({"x": np.array(xy)[:, 0], "y": np.array(xy)[:, 1]}),
                x="x",
                y="y",
                background_color="black",
                axes=False,
                opacity_unselected=0.05,
                height=row_height,
            ),
            logo=AnnotationLogo(labels),
        )
        for xy in (A[0], B[0])
    )

    selection_type = ipywidgets.RadioButtons(
        options=["synced", "neighbors"],
        value="synced",
        description="selection",
    )

    link = ipywidgets.link(
        (left.logo, "selection"), (right.logo, "selection")
    )

    def handle_selection_change(change):
        nonlocal link
        if change.new == "synced":
            link.link()
        else:
            link.unlink()

    selection_type.observe(handle_selection_change, names="value")  # type: ignore

    def apply_colormap(new_labels: pd.Series):
        non_robust_label = "0_0_0_0_0"
        robust_labels = pd.Series(
            np.where(robust, new_labels, non_robust_label),  # type: ignore
            dtype="category",
        )
        colormap = dict(
            zip(robust_labels.cat.categories, itertools.cycle(jscatter.glasbey_dark))
        )
        colormap.update({non_robust_label: "#333333"})
        for cmp in (left, right):
            assert cmp.scatter._data is not None
            cmp.scatter._data["label"] = robust_labels
            cmp.scatter.color(by="label", map=colormap)

    # TODO: sync move label trimmer up one level
    def on_labels_change(change):
        new_labels = change.new if hasattr(change, "new") else labels
        apply_colormap(new_labels)

    main = ipywidgets.GridBox(
        children=[left.show(), right.show()],
        layout=ipywidgets.Layout(
            grid_template_columns="1fr 1fr",
            grid_template_rows=f"{row_height * 2}px",
        ),
    )

    on_labels_change(None)
    return ipywidgets.VBox([selection_type, main])
