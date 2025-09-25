from types import SimpleNamespace

from backend import wakeword


def test_parse_env_list_splits_and_expands_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    value = " first.ppn , , ~/second.ppn "

    result = wakeword._parse_env_list(value)

    assert result == ["first.ppn", str(tmp_path / "second.ppn")]


def test_porcupine_kwargs_default_keyword():
    kwargs = wakeword._porcupine_create_kwargs(
        0.3,
        keywords_env=None,
        keyword_paths_env=None,
        access_key="abc",
    )

    assert kwargs["keywords"] == ["porcupine"]
    assert kwargs["sensitivities"] == [0.3]
    assert kwargs["access_key"] == "abc"


def test_porcupine_kwargs_prefers_keyword_paths(tmp_path):
    first = tmp_path / "hej.ppn"
    second = tmp_path / "kompis.ppn"
    kwargs = wakeword._porcupine_create_kwargs(
        1.4,
        keywords_env="ignored",
        keyword_paths_env=f"{first},{second}",
        access_key="abc",
    )

    assert kwargs["keyword_paths"] == [str(first), str(second)]
    assert kwargs["sensitivities"] == [1.0, 1.0]
    assert kwargs["access_key"] == "abc"
    assert "keywords" not in kwargs


def test_ensure_int16_clamps_samples():
    pcm = wakeword._ensure_int16([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    assert list(map(int, pcm)) == [
        -32767,
        -32767,
        -16383,
        0,
        16383,
        32767,
        32767,
    ]


def test_stream_fallback_runs_even_when_support_check_rejects(monkeypatch):
    attempts = []

    class DummyPortAudioError(Exception):
        pass

    class DummyStream:
        def __init__(self, *, samplerate, blocksize, **_kwargs):
            attempts.append((samplerate, blocksize))
            if samplerate != 48000:
                raise DummyPortAudioError("unsupported sample rate")
            self.samplerate = samplerate
            self.blocksize = blocksize
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
        lambda device, channels, dtype, rate: False,
    )
    monkeypatch.setattr(
        wakeword,
        "_gather_fallback_sample_rates",
        lambda device, excluded: [48000],
    )

    stream, effective_rate, frames_per_read = wakeword._open_wake_word_input_stream(
        dummy_sd,
        samplerate=16000,
        blocksize=512,
        device=None,
    )

    assert effective_rate == 48000
    assert frames_per_read == 1536
    assert stream.started is True
    assert attempts == [(16000, 512), (48000, 1536)]
