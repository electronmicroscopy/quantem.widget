"""Show2D saved inspection states.

The UI presents these as microscope-stage bookmarks: a scientist explores a
view, saves a lightweight named state, then restores or updates it later
without embedding another copy of the raw dataset.
"""

import json

import numpy as np

from quantem.widget import Show2D


def _gallery():
    rng = np.random.default_rng(4)
    return [rng.random((32, 32), dtype=np.float32), rng.random((32, 32), dtype=np.float32)]


def test_save_view_state_keeps_lightweight_named_snapshots():
    w = Show2D(_gallery(), labels=["raw", "filtered"], verbose=False)
    w.roi_active = True
    w.add_roi(row=12, col=14, shape="circle")
    w.set_padding(0.25, fill="median")
    w.show_fft = True
    first = w.save_view_state("defect A")

    w.selected_idx = 1
    w.hidden_panels = [0]
    second = w.save_view_state("filtered comparison")

    assert [entry["name"] for entry in w.saved_view_states] == ["defect A", "filtered comparison"]
    assert first["id"] != second["id"]
    assert "ROI 1" in first["summary"]
    assert "FFT" in first["summary"]
    assert "pad 25%" in first["summary"]
    assert "saved_view_states" not in first["state"]
    assert "frame_bytes" not in first["state"]
    assert "panel_stack_bytes" not in first["state"]


def test_load_view_state_restores_roi_padding_fft_and_panel_selection():
    w = Show2D(_gallery(), verbose=False)
    w.roi_active = True
    w.add_roi(row=10, col=11, shape="square")
    w.set_padding(0.2, fill="mean", panels=[1])
    w.show_fft = True
    w.selected_idx = 1
    w.hidden_panels = [0]
    w.save_view_state("stage position")

    w.roi_active = False
    w.roi_list = []
    w.reset_view_ops()
    w.show_fft = False
    w.selected_idx = 0
    w.hidden_panels = []

    w.load_view_state("stage position")
    assert w.roi_active is True
    assert len(w.roi_list) == 1
    assert w.roi_list[0]["shape"] == "square"
    assert w.pad_ratios == [0.0, 0.2]
    assert w.pad_fill_modes == ["min", "mean"]
    assert w.show_fft is True
    assert w.selected_idx == 1
    assert w.hidden_panels == [0]


def test_update_delete_and_clear_saved_view_states():
    w = Show2D(_gallery(), verbose=False)
    w.selected_idx = 0
    first = w.save_view_state("candidate")
    w.selected_idx = 1
    updated = w.save_view_state("candidate", update=True)
    assert updated["id"] == first["id"]
    assert len(w.saved_view_states) == 1

    w.selected_idx = 0
    w.load_view_state("candidate")
    assert w.selected_idx == 1

    w.save_view_state("second")
    assert len(w.saved_view_states) == 2
    w.delete_view_state("candidate")
    assert [entry["name"] for entry in w.saved_view_states] == ["second"]
    w.clear_view_states()
    assert w.saved_view_states == []


def test_saved_view_states_survive_state_dict_round_trip():
    data = _gallery()
    w = Show2D(data, verbose=False)
    w.roi_active = True
    w.add_roi(row=16, col=16)
    w.save_view_state("ROI review")

    restored = Show2D(data, verbose=False)
    restored.load_state_dict(w.state_dict())
    assert [entry["name"] for entry in restored.saved_view_states] == ["ROI review"]

    restored.roi_list = []
    restored.roi_active = False
    restored.load_view_state("ROI review")
    assert restored.roi_active is True
    assert len(restored.roi_list) == 1


def test_saved_view_request_trait_drives_ui_actions():
    w = Show2D(_gallery(), verbose=False)
    w.selected_idx = 1
    w.saved_view_request = json.dumps({"action": "save", "name": "manual bookmark", "request_id": "1"})
    assert [entry["name"] for entry in w.saved_view_states] == ["manual bookmark"]

    w.selected_idx = 0
    w.saved_view_request = json.dumps({"action": "load", "id": w.saved_view_states[0]["id"], "request_id": "2"})
    assert w.selected_idx == 1

    w.saved_view_request = json.dumps({"action": "delete_all", "request_id": "3"})
    assert w.saved_view_states == []
