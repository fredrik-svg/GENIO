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


def test_listener_uses_energy_fallback_when_model_missing(monkeypatch):
    monkeypatch.setattr(wakeword, "Model", None)

    listener = wakeword.WakeWordListener(on_detect=lambda: None)

    assert listener.detector_name == "energy"


def test_listener_uses_openwakeword_when_available(monkeypatch):
    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, audio):
            return {"hej": 0.7}

    monkeypatch.setattr(wakeword, "Model", DummyModel)

    listener = wakeword.WakeWordListener(on_detect=lambda: None, detection_threshold=0.5)

    assert listener.detector_name == "openwakeword"

