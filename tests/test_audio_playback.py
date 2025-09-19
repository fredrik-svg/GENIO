import io
import struct
import wave

import numpy as np
import pytest

from backend import audio


class DummySoundDevice:
    def __init__(self):
        self.play_calls = []
        self.wait_calls = 0

    def play(self, data, samplerate):
        self.play_calls.append((np.array(data, copy=True), samplerate))

    def wait(self):
        self.wait_calls += 1


@pytest.fixture
def sounddevice_stub(monkeypatch):
    stub = DummySoundDevice()
    monkeypatch.setattr(audio, "sd", stub)
    return stub


def _make_pcm16_wav(samples, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        ints = (np.array(samples, dtype=np.float32) * 32767.0).clip(-32768, 32767).astype("<i2")
        wav_file.writeframes(ints.tobytes())
    return buf.getvalue()


def _make_float32_wav(samples, sample_rate=24000):
    data = struct.pack("<" + "f" * len(samples), *samples)
    byte_rate = sample_rate * 4
    block_align = 4
    bits_per_sample = 32
    chunk_size = 4 + (8 + 16) + (8 + len(data))
    header = b"RIFF" + struct.pack("<I", chunk_size) + b"WAVE"
    fmt_chunk = b"fmt " + struct.pack("<I", 16)
    fmt_chunk += struct.pack("<HHIIHH", 3, 1, sample_rate, byte_rate, block_align, bits_per_sample)
    data_chunk = b"data" + struct.pack("<I", len(data)) + data
    return header + fmt_chunk + data_chunk


def test_play_wav_bytes_handles_pcm16(sounddevice_stub):
    samples = [0.0, 0.5, -0.5, 0.25]
    wav_bytes = _make_pcm16_wav(samples, sample_rate=16000)
    audio.play_wav_bytes(wav_bytes)
    assert sounddevice_stub.play_calls, "sd.play should have been invoked"
    played_audio, samplerate = sounddevice_stub.play_calls[0]
    np.testing.assert_allclose(played_audio, np.array(samples, dtype=np.float32), atol=1e-4)
    assert samplerate == 16000
    assert sounddevice_stub.wait_calls == 1


def test_play_wav_bytes_handles_float32(sounddevice_stub):
    samples = [0.0, 0.5, -0.25, 0.75]
    wav_bytes = _make_float32_wav(samples, sample_rate=22050)
    audio.play_wav_bytes(wav_bytes)
    assert sounddevice_stub.play_calls, "sd.play should have been invoked"
    played_audio, samplerate = sounddevice_stub.play_calls[-1]
    np.testing.assert_allclose(played_audio, np.array(samples, dtype=np.float32))
    assert samplerate == 22050
    assert sounddevice_stub.wait_calls == 1
