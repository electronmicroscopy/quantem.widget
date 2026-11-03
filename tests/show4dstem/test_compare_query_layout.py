"""Comparison batching preserves each acquisition's encoded query layout."""

from types import SimpleNamespace

import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.adapters import gpu as gpu_adapter


def _source(layout: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        representation=SimpleNamespace(value=layout), dtype="uint16", index=index
    )


@pytest.mark.parametrize(
    "layouts",
    [
        ("encoded", "encoded"),
        ("paired", "paired"),
        ("encoded", "paired"),
        ("paired", "encoded"),
    ],
)
def test_comparison_preserves_each_source_query_layout(monkeypatch, layouts):
    """Each layout keeps its own decoder, including its detector-pixel index."""
    sources = [_source(layout, index) for index, layout in enumerate(layouts)]
    monkeypatch.setattr(
        gpu_adapter, "is_acquisition", lambda source: source in sources
    )

    series = object.__new__(gpu_adapter.AcquisitionSeries)
    series.frames = [SimpleNamespace(source=source) for source in sources]
    images = [
        torch.tensor([[4.0, 8.0], [12.0, 16.0]]).numpy(),
        torch.tensor([[20.0, 24.0], [28.0, 32.0]]).numpy(),
    ]
    queried, batched = [], []

    def joint_session(selected):
        assert layouts[0] == layouts[1], "Mixed layouts reached the joint dispatcher"
        batched.append(selected)
        return SimpleNamespace(
            masked_sum=lambda mask: torch.stack([
                torch.from_numpy(images[source.index]) for source in selected
            ]).numpy()
        )

    monkeypatch.setattr(gpu_adapter, "prepare_session", joint_session)

    def session(index):
        queried.append(index)
        return SimpleNamespace(masked_sum=lambda mask: images[index])

    viewer = SimpleNamespace(
        _data=series, _session=session,
        _compare_session=None, _compare_session_indices=(),
    )
    mask = torch.tensor([[True, False], [False, True]]).numpy()
    actual = Show4DSTEM._compare_images(viewer, (1, 0), mask)
    expected = torch.tensor([
        [[10.0, 12.0], [14.0, 16.0]], [[2.0, 4.0], [6.0, 8.0]]
    ])
    if layouts[0] == layouts[1]:
        assert batched == [[sources[1], sources[0]]]
        assert queried == []
    else:
        assert batched == []
        assert queried == [1, 0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=0, atol=0)
