from types import SimpleNamespace

from backend import wakeword


def test_parse_env_list_expands_user(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    value = " first , , ~/second "

    result = wakeword._parse_env_list(value.replace("~", str(home)))

    assert result == ["first", str(home / "second")]


def test_resolve_openwakeword_labels_prefers_model_attribute():
    model = SimpleNamespace(wakeword_names=["hej", "kompis"])
    labels = wakeword._resolve_openwakeword_labels(model, ["ignored"], [])
    assert labels == ["hej", "kompis"]


def test_extract_prediction_scores_from_iterable():
    predictions = [("hej", 0.8), ("kompis", 0.2)]
    scores = wakeword._extract_prediction_scores(predictions, ["hej", "kompis"])
    assert scores == [0.8, 0.2]


def test_openwakeword_detector_requires_positive_min_activation():
    model = SimpleNamespace(sample_rate=16000, frame_length=256, predict=lambda data: {"hej": 1.0})
    try:
        wakeword._OpenWakeWordDetector(
            model,
            labels=["hej"],
            detection_threshold=0.5,
            min_activations=0,
            cooldown=1.0,
        )
    except ValueError as exc:
        assert "min_activations" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError")


def test_stream_fallback_runs_even_when_support_check_rejects(monkeypatch):
    attempts = []

    class DummyPortAudioError(Exception):
        pass

    class DummyStream:
        def __init__(self, *, samplerate, blocksize, dtype, **_kwargs):
            attempts.append((samplerate, blocksize, dtype))
            if samplerate != 48000:
                raise DummyPortAudioError("unsupported sample rate")
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.started = False
            self.dtype = dtype

        def start(self):
            self.started = True

    dummy_sd = SimpleNamespace(
        InputStream=DummyStream,
        PortAudioError=DummyPortAudioError,
    )

    monkeypatch.setattr(
        wakeword,
        "_supports_input_sample_rate",
        lambda device, channels, dtype, rate: False,
    )
    monkeypatch.setattr(
        wakeword,
        "_gather_fallback_sample_rates",
        lambda device, excluded: [48000],
    )

    (
        stream,
        effective_rate,
        frames_per_read,
        dtype,
    ) = wakeword._open_wake_word_input_stream(
        dummy_sd,
        samplerate=16000,
        blocksize=512,
        device=None,
    )

    assert effective_rate == 48000
    assert frames_per_read == 1536
    assert stream.started is True
    assert dtype == "float32"
    assert attempts == [
        (16000, 512, "float32"),
        (16000, 512, "int16"),
        (48000, 1536, "float32"),
    ]


def test_stream_fallback_uses_int16_when_float32_fails(monkeypatch):
    attempts = []

    class DummyPortAudioError(Exception):
        pass

    class DummyStream:
        def __init__(self, *, samplerate, blocksize, dtype, **_kwargs):
            attempts.append((samplerate, blocksize, dtype))
            if dtype == "float32":
                raise DummyPortAudioError("float unsupported")
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.dtype = dtype
            self.started = False

        def start(self):
            self.started = True

    dummy_sd = SimpleNamespace(
        InputStream=DummyStream,
        PortAudioError=DummyPortAudioError,
    )

    monkeypatch.setattr(
        wakeword,
        "_supports_input_sample_rate",
        lambda device, channels, dtype, rate: True,
    )
    monkeypatch.setattr(
        wakeword,
        "_gather_fallback_sample_rates",
        lambda device, excluded: [],
    )

    (
        stream,
        effective_rate,
        frames_per_read,
        dtype,
    ) = wakeword._open_wake_word_input_stream(
        dummy_sd,
        samplerate=16000,
        blocksize=256,
        device=None,
    )

    assert effective_rate == 16000
    assert frames_per_read == 256
    assert stream.started is True
    assert dtype == "int16"
    assert attempts == [
        (16000, 256, "float32"),
        (16000, 256, "int16"),
    ]
