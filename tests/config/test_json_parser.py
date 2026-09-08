import os

import pytest

from tyssue import config
from tyssue.config.json_parser import load_spec, save_spec

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TESTCONFIG = os.path.join(CURRENT_DIR, "test_config.json")


def test_load_spec():
    config = load_spec(TESTCONFIG)
    assert "face" in config
    assert config["face"]["num_sides"] == 6


def test_save_spec(tmp_path):
    # NamedTemporaryFile holds an exclusive handle on Windows, so save_spec()
    # could not reopen the path by name; tmp_path just yields a directory.
    config = load_spec(TESTCONFIG)
    fname = str(tmp_path / "spec.json")
    save_spec(config, fname, overwrite=True)
    saved_config = load_spec(fname)
    assert saved_config["face"]["num_sides"] == 6
    with pytest.raises(IOError):
        save_spec(config, fname, False)


def test_default():
    spec = config.geometry.cylindrical_sheet()
    assert spec["face"]["x"] == 0.0
