import pytest

from backend import audio_settings


def _override_settings_path(monkeypatch, tmp_path):
    path = tmp_path / "audio_settings.json"
    monkeypatch.setattr(audio_settings, "_SETTINGS_PATH", str(path), raising=False)
    return path


def test_normalize_input_device_selection():
    assert audio_settings.normalize_input_device_selection(None) == ""
    assert audio_settings.normalize_input_device_selection(" auto ") == ""
    assert audio_settings.normalize_input_device_selection("index:2") == "index:2"
    assert audio_settings.normalize_input_device_selection(" 5 ") == "index:5"
    assert audio_settings.normalize_input_device_selection("manual: hw:2,0 ") == "manual:hw:2,0"

    with pytest.raises(ValueError):
        audio_settings.normalize_input_device_selection("index:")

    with pytest.raises(ValueError):
        audio_settings.normalize_input_device_selection("manual:   ")


def test_set_and_get_selected_input_device(monkeypatch, tmp_path):
    path = _override_settings_path(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_settings, "INPUT_DEVICE", "", raising=False)

    audio_settings.set_selected_input_device("index:3")
    assert path.is_file()
    assert audio_settings.get_selected_input_device() == 3

    audio_settings.set_selected_input_device("manual:hw:1,0")
    assert audio_settings.get_selected_input_device() == "hw:1,0"

    settings = audio_settings.load_audio_settings()
    assert settings.input_device == "manual:hw:1,0"

    audio_settings.set_selected_input_device("auto")
    assert audio_settings.load_audio_settings().input_device == ""


def test_get_selected_input_device_env_fallback(monkeypatch, tmp_path):
    _override_settings_path(monkeypatch, tmp_path)
    monkeypatch.setattr(audio_settings, "INPUT_DEVICE", "hw:9,0", raising=False)
    assert audio_settings.get_selected_input_device() == "hw:9,0"


def test_extract_helpers(monkeypatch, tmp_path):
    _override_settings_path(monkeypatch, tmp_path)
    assert audio_settings.extract_manual_value("manual:hw:2,0") == "hw:2,0"
    assert audio_settings.extract_manual_value("index:4") == ""
    assert audio_settings.extract_index("index:7") == 7
    assert audio_settings.extract_index("manual:hw:2,0") is None
    assert audio_settings.serialize_device_spec(5) == "index:5"
    assert audio_settings.serialize_device_spec("hw:1,0") == "hw:1,0"
