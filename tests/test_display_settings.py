import json
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
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

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
    assert zero_entry["monitors"][0]["name"] == "HDMI-1"


def test_discover_display_targets_wayland(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/fake-runtime")

    def fake_listdir(path):
        if path == "/fake-runtime":
            return ["wayland-1"]
        if path == "/tmp/.X11-unix":
            raise FileNotFoundError
        raise AssertionError(f"Unexpected listdir path: {path}")

    def fake_check_output(cmd, text, stderr, env):  # noqa: ARG001
        if cmd and cmd[0] == "wayland-info":
            assert env.get("WAYLAND_DISPLAY") == "wayland-1"
            return json.dumps(
                {
                    "globals": [
                        {
                            "interface": "wl_output",
                            "output": {
                                "name": "WL-1",
                                "make": "Make",
                                "model": "Model",
                                "description": "Desk Monitor",
                                "scale": 2,
                                "modes": [
                                    {
                                        "width": 1920,
                                        "height": 1080,
                                        "refresh": 60000,
                                        "flags": ["current", "preferred"],
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        raise FileNotFoundError

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(display_settings.subprocess, "check_output", fake_check_output)

    targets, warnings = display_settings.discover_display_targets()

    assert warnings == []
    wayland_entry = next(entry for entry in targets if entry["value"] == "wayland-1")
    assert wayland_entry["monitors"]
    monitor = wayland_entry["monitors"][0]
    assert monitor["name"] == "WL-1"
    assert monitor["width"] == 1920
    assert "WL-1" in wayland_entry["label"]
    assert "1920x1080" in wayland_entry["label"]


def test_discover_display_targets_wayland_registry(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-2")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/fake-runtime")

    def fake_listdir(path):
        if path == "/fake-runtime":
            return ["wayland-2"]
        if path == "/tmp/.X11-unix":
            raise FileNotFoundError
        raise AssertionError(f"Unexpected listdir path: {path}")

    def fake_check_output(cmd, text, stderr, env):  # noqa: ARG001
        if cmd and cmd[0] == "wayland-info":
            assert env.get("WAYLAND_DISPLAY") == "wayland-2"
            return json.dumps(
                {
                    "registry": {
                        "globals": [
                            {
                                "interface": "wl_output",
                                "output": {
                                    "name": "WL-2",
                                    "make": "RegMake",
                                    "model": "RegModel",
                                    "description": "Registry Monitor",
                                    "scale": 1,
                                    "modes": [
                                        {
                                            "width": 2560,
                                            "height": 1440,
                                            "refresh": 144000,
                                            "flags": ["current"],
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                }
            )
        raise FileNotFoundError

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(display_settings.subprocess, "check_output", fake_check_output)

    targets, warnings = display_settings.discover_display_targets()

    assert warnings == []
    wayland_entry = next(entry for entry in targets if entry["value"] == "wayland-2")
    assert wayland_entry["monitors"]
    monitor = wayland_entry["monitors"][0]
    assert monitor["name"] == "WL-2"
    assert monitor["width"] == 2560
    assert "WL-2" in wayland_entry["label"]
    assert "2560x1440" in wayland_entry["label"]


def test_discover_display_targets_multiple_x11_monitors(monkeypatch):
    """Test that multiple X11 monitors generate separate .screen entries."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    def fake_listdir(path):
        assert path == "/tmp/.X11-unix"
        return ["X0"]

    def fake_check_output(cmd, text, stderr, env):  # noqa: ARG001
        if env.get("DISPLAY") == ":0":
            # Simulate two monitors like XWAYLAND0 and XWAYLAND1
            return "Monitors: 2\n 0: +*XWAYLAND0 1920/598x1080/336+0+0  XWAYLAND0\n 1: +XWAYLAND1 1024/300x600/200+1920+0  XWAYLAND1\n"
        return "Monitors: 0\n"

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(display_settings.subprocess, "check_output", fake_check_output)

    targets, warnings = display_settings.discover_display_targets()

    assert warnings == []
    
    # Find :0 entry
    zero_entry = next(entry for entry in targets if entry["value"] == ":0")
    assert "XWAYLAND0" in zero_entry["label"]
    assert "XWAYLAND1" in zero_entry["label"]
    assert len(zero_entry["monitors"]) == 2
    
    # Find :0.0 entry (first monitor)
    zero_zero_entry = next((entry for entry in targets if entry["value"] == ":0.0"), None)
    assert zero_zero_entry is not None
    assert "XWAYLAND0" in zero_zero_entry["label"]
    assert "1920x1080" in zero_zero_entry["label"]
    assert "primär" in zero_zero_entry["label"]
    
    # Find :0.1 entry (second monitor)
    zero_one_entry = next((entry for entry in targets if entry["value"] == ":0.1"), None)
    assert zero_one_entry is not None
    assert "XWAYLAND1" in zero_one_entry["label"]
    assert "1024x600" in zero_one_entry["label"]
    assert "primär" not in zero_one_entry["label"]
