"""Tests for settings persistence and validation."""

from __future__ import annotations

import json

import pytest

from dragon_ball_powers import config
from dragon_ball_powers.config import (
    Calibration,
    Settings,
    field_constraints,
    get_setting,
    load_calibration,
    load_settings,
    save_calibration,
    save_settings,
    set_setting,
)


def test_defaults_are_valid_and_round_trip(tmp_config_dir):
    path = tmp_config_dir / "settings.json"
    settings = Settings()
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.to_dict() == settings.to_dict()
    assert loaded.abilities.selected == "kamehameha"
    assert loaded.effects.energy_color == "blue"


def test_missing_file_creates_defaults(tmp_config_dir):
    path = tmp_config_dir / "settings.json"
    assert not path.exists()
    settings = load_settings(path)
    assert path.exists()
    assert settings.to_dict() == Settings().to_dict()


def test_invalid_values_fall_back_or_clamp(tmp_config_dir):
    path = tmp_config_dir / "settings.json"
    path.write_text(json.dumps({
        "camera": {"index": "three", "width": 99999, "fps": 5, "mirror": "yes"},
        "effects": {"energy_color": "GOLD", "quality": "insane", "bloom_strength": float("nan")},
        "gestures": {"sensitivity": 7, "together_distance": 2.5, "release_distance": 1.0},
        "unknown_section": {"x": 1},
        "ui": "not a dict",
    }), encoding="utf-8")
    s = load_settings(path)
    assert s.camera.index == 0                # wrong type -> default
    assert s.camera.width == 3840             # clamped to max
    assert s.camera.fps == 10                 # clamped to min
    assert s.camera.mirror is True            # "yes" -> True
    assert s.effects.energy_color == "gold"   # case-insensitive choice
    assert s.effects.quality == "auto"        # invalid choice -> default
    assert s.effects.bloom_strength == 1.0    # NaN -> default
    assert s.gestures.sensitivity == 1.0      # clamped
    assert s.gestures.release_distance > s.gestures.together_distance  # cross-field fix
    assert s.ui.window_width == 1280          # bad section -> defaults


def test_corrupt_json_is_moved_aside(tmp_config_dir):
    path = tmp_config_dir / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    s = load_settings(path)
    assert s.to_dict() == Settings().to_dict()
    assert (tmp_config_dir / "settings.json.invalid").exists()


def test_bool_from_number_and_bad_bool():
    s = Settings.from_dict({"audio": {"enabled": 0}, "ui": {"fullscreen": "maybe"}})
    assert s.audio.enabled is False
    assert s.ui.fullscreen is False  # default


def test_set_and_get_setting_validates():
    s = Settings()
    assert set_setting(s, "gestures.sensitivity", 0.8) == pytest.approx(0.8)
    assert get_setting(s, "gestures.sensitivity") == pytest.approx(0.8)
    assert set_setting(s, "gestures.sensitivity", 5) == 1.0
    assert set_setting(s, "effects.energy_color", "purple") == "purple"
    assert set_setting(s, "effects.energy_color", "green") == "purple"  # rejected, unchanged
    with pytest.raises(KeyError):
        set_setting(s, "effects.nope", 1)
    with pytest.raises(KeyError):
        get_setting(s, "nosection.x")


def test_field_constraints():
    info = field_constraints("gestures.sensitivity")
    assert info["type"] is float and info["min"] == 0.0 and info["max"] == 1.0
    assert "gold" in field_constraints("effects.energy_color")["choices"]


def test_copy_is_independent():
    s = Settings()
    c = s.copy()
    c.camera.index = 3
    assert s.camera.index == 0


def test_calibration_round_trip(tmp_config_dir):
    path = tmp_config_dir / "calibration.json"
    assert load_calibration(path).valid is False
    cal = Calibration(valid=True, shoulder_width_px=210.5, together_distance=0.45,
                      hand_size_ratio=0.31, created_at=Calibration.timestamp())
    save_calibration(cal, path)
    loaded = load_calibration(path)
    assert loaded.valid and loaded.shoulder_width_px == pytest.approx(210.5)
    assert loaded.together_distance == pytest.approx(0.45)


def test_calibration_rejects_negative():
    cal = Calibration.from_dict({"valid": True, "rest_jitter": -3, "thrust_peak": "x"})
    assert cal.rest_jitter == 0.0 and cal.thrust_peak == 0.0


def test_paths_are_pathlib():
    assert config.PROJECT_ROOT.joinpath("src").is_dir()
    assert config.settings_path().name == "settings.json"
