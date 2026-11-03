"""dataset.yaml reader: schema 1 and quantem.dataset/2 files give the same condition tags; other layouts refuse."""

import pytest

pytest.importorskip("yaml")

from quantem.widget.io.metadata.dataset_yaml import load_dataset_yaml, resolve_condition

CONDITIONS = """
conditions:
  light: {description: "LED on"}
  dark:  {description: "dark reference", skip: true}
folders:
  light1: {condition: light}
  dark1:  {condition: dark}
files:
  '07':  {condition: dark}
  3-11:  {mag: mag_3p6}
"""
# the session and microscope blocks of a file quantem.live writes today (June 2026 monolayer layout)
SCHEMA_2 = """schema: quantem.dataset/2
session:
  name: "MoS2 monolayer"
  date: 2026-06-26
microscope:
  voltage_kV: 300
  semiangle_mrad: 30
  scan_detector_rotation_deg: 176
calibrations:
  mag_3p6:  {field_of_view_nm: [26.9, 26.9], scan_sampling_A: [0.5254, 0.5254]}
data:
  scan_shape: [512, 512]
  detector_shape: [192, 192]
""" + CONDITIONS
SCHEMA_1 = """schema_version: 1
session:
  name: mos2
  date: 2026-03-06
""" + CONDITIONS


def write(tmp_path, text):
    (tmp_path / "dataset.yaml").write_text(text)
    for folder in ("light1", "dark1"):
        (tmp_path / folder).mkdir(exist_ok=True)
    return tmp_path


@pytest.mark.parametrize("text", [SCHEMA_1, SCHEMA_2], ids=["schema_version 1", "quantem.dataset/2"])
def test_both_schemas_resolve_the_same_conditions(tmp_path, text):
    session = write(tmp_path, text)
    data = load_dataset_yaml(session)
    folder_tag = resolve_condition(data, session / "light1" / "scan_003_master.h5", session)
    file_tag = resolve_condition(data, session / "light1" / "scan_07_master.h5", session)
    untagged = resolve_condition(data, session / "scan_012_master.h5", session)
    assert (folder_tag.label, folder_tag.source, folder_tag.folder, folder_tag.skip) == ("light", "folders", "light1", False)
    assert (file_tag.label, file_tag.source, file_tag.skip, file_tag.description) == ("dark", "files", True, "dark reference")
    assert (untagged.label, untagged.source) == (None, "none")


def test_schema_2_keeps_its_blocks(tmp_path):
    data = load_dataset_yaml(write(tmp_path, SCHEMA_2))
    assert data["schema"] == "quantem.dataset/2"
    assert data["microscope"]["scan_detector_rotation_deg"] == 176
    assert data["calibrations"]["mag_3p6"]["scan_sampling_A"] == [0.5254, 0.5254]


@pytest.mark.parametrize("text, message", [
    ("schema: quantem.dataset/3\n" + CONDITIONS, "has schema 'quantem.dataset/3'"),
    ("schema: quantem.screen-acquisition/2\n", "has schema"),
    (CONDITIONS, "missing schema_version"),
])
def test_other_layouts_refuse(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_dataset_yaml(write(tmp_path, text))


def test_missing_file_reads_empty(tmp_path):
    assert load_dataset_yaml(tmp_path) == {}
