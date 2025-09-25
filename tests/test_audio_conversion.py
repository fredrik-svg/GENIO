import io
import struct
import wave

import numpy as np

from backend import audio


def _make_pcm16_wav(samples, sample_rate=16000, channels=1):
    samples = np.asarray(samples, dtype=np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    ints = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(ints.reshape(-1).tobytes())
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


def test_ensure_wav_pcm16_converts_float32():
    samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    wav_bytes = _make_float32_wav(samples, sample_rate=22050)

    converted = audio.ensure_wav_pcm16(wav_bytes)

    assert converted != wav_bytes
    with wave.open(io.BytesIO(converted), "rb") as wav_file:
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 22050
        assert wav_file.getnchannels() == 1
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    np.testing.assert_allclose(decoded, samples, atol=1e-4)


def test_ensure_wav_pcm16_leaves_pcm16_untouched():
    samples = np.array([0.0, -0.25, 0.75], dtype=np.float32)
    wav_bytes = _make_pcm16_wav(samples, sample_rate=16000)

    converted = audio.ensure_wav_pcm16(wav_bytes)

    assert converted is wav_bytes

