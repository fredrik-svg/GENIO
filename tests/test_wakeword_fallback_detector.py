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


def test_listener_uses_energy_fallback_when_module_missing(monkeypatch):
    monkeypatch.setattr(wakeword, "pvporcupine", None)

    listener = wakeword.WakeWordListener(on_detect=lambda: None)

    assert listener.detector_name == "energy"


def test_listener_uses_porcupine_when_available(monkeypatch):
    class DummyPorcupine:
        sample_rate = 16000
        frame_length = 256

        def __init__(self):
            self.deleted = False

        def process(self, pcm):
            return -1

        def delete(self):  # pragma: no cover - exercised indirectly when stop is called
            self.deleted = True

    class DummyModule:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return DummyPorcupine()

    dummy_module = DummyModule()
    monkeypatch.setattr(wakeword, "pvporcupine", dummy_module)
    monkeypatch.delenv("PORCUPINE_KEYWORDS", raising=False)
    monkeypatch.delenv("PORCUPINE_KEYWORD_PATHS", raising=False)
    monkeypatch.delenv("PICOVOICE_ACCESS_KEY", raising=False)

    listener = wakeword.WakeWordListener(on_detect=lambda: None, detection_threshold=0.7)

    assert listener.detector_name == "porcupine"
    assert dummy_module.kwargs["keywords"] == ["porcupine"]
    assert dummy_module.kwargs["sensitivities"] == [0.7]

