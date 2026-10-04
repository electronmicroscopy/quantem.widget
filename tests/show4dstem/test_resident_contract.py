"""CPU-only metadata contract; no accelerator or viewer is constructed."""

import copy

import numpy as np
import traitlets

from quantem.widget.show4dstem import Show4DSTEM


def test_resident_metadata_defaults_preserve_legacy_comparison() -> None:
    traits = Show4DSTEM.class_traits()
    for name in ("resident_batch_info", "resident_stream"):
        assert traits[name].metadata["sync"] is True
        # Dict defaults are dynamically produced by traitlets, not stored in
        # TraitType.default_value. Verify the actual descriptor behavior.
        probe_type = type("ResidentDefaultsProbe", (traitlets.HasTraits,),
                          {name: copy.copy(traits[name])})
        first, second = probe_type(), probe_type()
        assert getattr(first, name) == {}
        getattr(first, name)["request_id"] = 1
        assert getattr(second, name) == {}
    assert traits["compare_virtual_image_bytes"].metadata["sync"] is True


def test_resident_traits_keep_uint32_bytes_and_request_metadata() -> None:
    # Exercise the real trait definitions without constructing a GPU-owning UI.
    probe_type = type("ResidentTraitProbe", (traitlets.HasTraits,), {
        name: copy.copy(Show4DSTEM.class_traits()[name])
        for name in ("resident_batch_info", "resident_stream", "compare_virtual_image_bytes")
    })
    probe = probe_type()
    original = np.array([0, 65535, 65536, 16777217, 4294967295], dtype="<u4")
    probe.compare_virtual_image_bytes = original.tobytes()
    probe.resident_batch_info = {
        "dtype": "<u4", "shape": [1, 1, 5], "bytes": original.nbytes,
        "request_id": 71, "generation": "cpu-contract", "mask_area": 0,
    }
    assert probe.resident_batch_info["request_id"] == 71
    assert probe.resident_batch_info["mask_area"] == 0
    np.testing.assert_array_equal(
        np.frombuffer(probe.compare_virtual_image_bytes, dtype="<u4"), original
    )
