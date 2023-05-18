import pandas as pd
from cev._widget_utils import trim_label_series


def test_trim_label_series():
    labels = pd.Series(
        ["CD8+CD4-CD3+", "CD8+CD4+CD3+", "CD8-CD4+CD3-", "CD8-CD4-CD3+"],
        dtype="category",
    )
    expected = pd.Series(
        ["CD8+CD3+", "CD8+CD3+", "CD8-CD3-", "CD8-CD3+"], dtype="category"
    )
    trimmed = trim_label_series(labels, {"CD8", "CD3"})
    assert trimmed.cat.categories.tolist() == expected.cat.categories.tolist()
    assert trimmed.tolist() == expected.tolist()
