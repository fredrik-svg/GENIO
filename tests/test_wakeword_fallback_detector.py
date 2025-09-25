import pytest

from backend import wakeword


def test_energy_detector_triggers_after_sustained_energy():
    clock = [0.0]

    def advance(delta: float) -> None:
        clock[0] += delta

    def fake_time() -> float:
        return clock[0]

    detector = wakeword._EnergyWakeWordDetector(
        energy_threshold=0.2,
        required_consecutive_blocks=3,
        cooldown=1.0,
        time_source=fake_time,
    )

    low = [0.0] * 512
    high = [0.5] * 512

    assert detector.process(low) is False

    advance(0.05)
    assert detector.process(high) is False

    advance(0.05)
    assert detector.process(high) is False

    advance(0.05)
    assert detector.process(high) is True

    # Cooldown aktiv → ingen ny trigger direkt
    assert detector.process(high) is False

    advance(1.1)

    # Låg energi nollställer räknaren
    assert detector.process(low) is False

    advance(0.05)
    assert detector.process(high) is False

    advance(0.05)
    assert detector.process(high) is False

    advance(0.05)
    assert detector.process(high) is True


def test_listener_uses_energy_fallback_when_openwakeword_missing(monkeypatch):
    monkeypatch.setattr(wakeword, "_OpenWakeWordModel", None)

    listener = wakeword.WakeWordListener(on_detect=lambda: None, engine="openwakeword")

    assert listener.detector_name == "energy"


def test_listener_uses_openwakeword_when_available(monkeypatch):
    class DummyModel:
        sample_rate = 16000
        frame_length = 256

        def __init__(self, *, wakeword_models=None, custom_models=None):
            self.calls = 0
            self.received = {
                "wakeword_models": list(wakeword_models or []),
                "custom_models": list(custom_models or []),
            }

        def predict(self, _values):
            self.calls += 1
            score = 0.9 if self.calls >= 2 else 0.1
            return {"hej": score}

        def reset(self):  # pragma: no cover - exercised indirectly
            self.calls = 0

    monkeypatch.setattr(wakeword, "_OpenWakeWordModel", DummyModel)
    monkeypatch.setattr(wakeword, "_openwakeword_download_models", lambda model_names: None)
    monkeypatch.setattr(wakeword, "WAKEWORD_MODELS", "hej")
    monkeypatch.setattr(wakeword, "WAKEWORD_MODEL_PATHS", "")

    listener = wakeword.WakeWordListener(
        on_detect=lambda: None,
        detection_threshold=0.5,
        engine="openwakeword",
        min_activations=2,
    )

    assert listener.detector_name == "openwakeword"
    detector = listener._detector
    assert isinstance(detector, wakeword._OpenWakeWordDetector)

    assert detector.process([0.0] * detector.frame_length) is False
    assert detector.process([0.0] * detector.frame_length) is False
    assert detector.process([0.0] * detector.frame_length) is True

    # Ensure labels picked up from env configuration
    assert detector._labels == ["hej"]
