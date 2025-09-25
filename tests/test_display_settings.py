import os
import subprocess

import pytest

from backend import display_settings


@pytest.fixture(autouse=True)
def reset_display_settings_path(tmp_path, monkeypatch):
    settings_path = tmp_path / "display_settings.json"
    monkeypatch.setattr(display_settings, "_SETTINGS_PATH", str(settings_path))
    yield
    if settings_path.exists():
        settings_path.unlink()


def test_describe_display_settings_defaults(monkeypatch):
    for key in ("PRIMARY_DISPLAY_TARGET", "SECONDARY_DISPLAY_TARGET", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)

    info = display_settings.describe_display_settings()

    assert info["assistant"]["stored"] == ""
    assert info["assistant"]["effective"] == ""
    assert "systemets standard" in info["assistant"]["message"].lower()

    assert info["display"]["stored"] == ""
    assert info["display"]["effective"] == ""
    assert "systemets standard" in info["display"]["message"].lower()


def test_describe_display_settings_uses_stored(monkeypatch):
    settings = display_settings.DisplaySettings(
        assistant_display=":0.1",
        presentation_display=":0.2",
    )
    display_settings.save_display_settings(settings)

    info = display_settings.describe_display_settings()

    assert info["assistant"]["effective"] == ":0.1"
    assert "sparad skärm" in info["assistant"]["message"].lower()
    assert info["display"]["effective"] == ":0.2"
    assert "sparad skärm" in info["display"]["message"].lower()


def test_discover_display_targets(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    def fake_listdir(path):
        assert path == "/tmp/.X11-unix"
        return ["X0"]

    def fake_check_output(cmd, text, stderr, env):  # noqa: ARG001
        if env.get("DISPLAY") == ":0":
            return "Monitors: 1\n 0: +*HDMI-1 1920/598x1080/336+0+0  HDMI-1\n"
        return "Monitors: 0\n"

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(display_settings.subprocess, "check_output", fake_check_output)

    targets, warnings = display_settings.discover_display_targets()

    assert warnings == []
    values = [entry["value"] for entry in targets]
    assert values[0] == ":99"
    assert ":0" in values
    zero_entry = next(entry for entry in targets if entry["value"] == ":0")
    assert "HDMI-1" in zero_entry["label"]
