from types import SimpleNamespace

import logging

import pytest

np = pytest.importorskip("numpy")

import backend.audio as audio


class DummyPortAudioError(Exception):
    pass


def _make_dummy_sd(
    query_result=None,
    query_exception=None,
    default_samplerate=None,
    device_list=None,
):
    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            if query_exception is not None:
                raise query_exception
            return list(device_list or [])
        if query_exception is not None:
            raise query_exception
        assert kind == "input"
        return query_result or {}

    return SimpleNamespace(
        PortAudioError=DummyPortAudioError,
        query_devices=query_devices,
        default=SimpleNamespace(samplerate=default_samplerate),
    )


def test_open_stream_uses_device_default(monkeypatch):
    attempts = []

    def fake_open_stream(rate):
        attempts.append(rate)
        if rate != 44100:
            raise DummyPortAudioError("boom")
        return "stream"

    dummy_sd = _make_dummy_sd(query_result={"default_samplerate": 44100.0})
    monkeypatch.setattr(audio, "sd", dummy_sd)

    stream, rate = audio._open_stream_with_fallback(fake_open_stream, None, {audio.SAMPLE_RATE})

    assert stream == "stream"
    assert rate == 44100
    assert attempts == [44100]


def test_open_stream_tries_known_rates(monkeypatch):
    attempts = []

    def fake_open_stream(rate):
        attempts.append(rate)
        if rate == 48000:
            return "ok"
        raise DummyPortAudioError("nope")

    dummy_sd = _make_dummy_sd(query_result={"default_samplerate": None})
    monkeypatch.setattr(audio, "sd", dummy_sd)

    stream, rate = audio._open_stream_with_fallback(fake_open_stream, None, {audio.SAMPLE_RATE})

    assert stream == "ok"
    assert rate == 48000
    assert attempts[0] == 48000


def test_open_stream_honours_configured_rates(monkeypatch):
    attempts = []

    def fake_open_stream(rate):
        attempts.append(rate)
        if rate == 44100:
            return "ok"
        raise DummyPortAudioError("nope")

    dummy_sd = _make_dummy_sd(query_result={"default_samplerate": None})
    monkeypatch.setattr(audio, "sd", dummy_sd)
    monkeypatch.setattr(audio, "_COMMON_SAMPLE_RATES", (96000, 44100))

    stream, rate = audio._open_stream_with_fallback(
        fake_open_stream, None, {audio.SAMPLE_RATE}
    )

    assert stream == "ok"
    assert rate == 44100
    assert attempts == [96000, 44100]


def test_gather_fallback_filters_invalid(monkeypatch):
    dummy_sd = _make_dummy_sd(
        query_result={"default_samplerate": float("nan")},
        default_samplerate=44100.0,
    )
    monkeypatch.setattr(audio, "sd", dummy_sd)

    rates = audio._gather_fallback_sample_rates(None, {audio.SAMPLE_RATE, 44100})

    assert 44100 not in rates
    assert rates[0] == 48000
    assert 48000 in rates


def test_open_stream_raises_last_error(monkeypatch):
    attempts = []

    def fake_open_stream(rate):
        attempts.append(rate)
        raise DummyPortAudioError(f"fail {rate}")

    dummy_sd = _make_dummy_sd(query_result={"default_samplerate": None})
    monkeypatch.setattr(audio, "sd", dummy_sd)

    with pytest.raises(DummyPortAudioError):
        audio._open_stream_with_fallback(fake_open_stream, None, {audio.SAMPLE_RATE})

    assert attempts  # should have tried fallback sample rates


def test_open_stream_skips_unsupported_rates(monkeypatch):
    attempts = []

    def fake_open_stream(rate):
        attempts.append(rate)
        return f"stream-{rate}"

    dummy_sd = _make_dummy_sd(query_result={"default_samplerate": None})
    monkeypatch.setattr(audio, "sd", dummy_sd)

    stream, rate = audio._open_stream_with_fallback(
        fake_open_stream,
        None,
        {audio.SAMPLE_RATE},
        supports_rate=lambda candidate: candidate == 32000,
    )

    assert stream == "stream-32000"
    assert rate == 32000
    assert attempts == [32000]


def test_gather_handles_query_errors(monkeypatch):
    dummy_sd = _make_dummy_sd(query_result=None, query_exception=RuntimeError("no device"))
    monkeypatch.setattr(audio, "sd", dummy_sd)

    rates = audio._gather_fallback_sample_rates(None, {audio.SAMPLE_RATE})

    assert 48000 in rates


def test_record_until_silence_returns_silence_when_input_unavailable(monkeypatch, caplog):
    class DummyStreamError(Exception):
        pass

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return []
        assert kind == "input"
        return {}

    def make_dummy_input_stream(**_kwargs):  # pragma: no cover - sanity check
        raise AssertionError("InputStream should not be created when no devices are present")

    dummy_sd = SimpleNamespace(
        PortAudioError=DummyStreamError,
        InputStream=lambda *args, **kwargs: make_dummy_input_stream(**kwargs),
        query_devices=query_devices,
        default=SimpleNamespace(samplerate=None),
    )

    monkeypatch.setattr(audio, "sd", dummy_sd)

    caplog.set_level(logging.DEBUG)

    audio_data = audio.record_until_silence()

    assert isinstance(audio_data, np.ndarray)
    assert audio_data.size == 0
    assert all(
        record.levelno < logging.ERROR
        for record in caplog.records
        if record.name == audio.__name__
    )
    assert any(
        "No audio input devices detected." in record.getMessage()
        for record in caplog.records
        if record.name == audio.__name__
    )
