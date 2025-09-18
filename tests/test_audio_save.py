import io
import wave

import pytest

np = pytest.importorskip("numpy")

from backend.audio import save_wav_mono16


def test_save_wav_to_file_like_preserves_buffer_state():
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    buffer = io.BytesIO()

    save_wav_mono16(buffer, audio, sample_rate=8000)

    assert buffer.closed is False
    assert buffer.tell() == 0

    with wave.open(io.BytesIO(buffer.getvalue()), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 8000
        frames = wav.readframes(wav.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        assert np.allclose(pcm[: audio.size], audio, atol=1e-4)


def test_save_wav_path_roundtrip(tmp_path):
    path = tmp_path / "sample.wav"
    audio = np.linspace(-1.0, 1.0, num=16, dtype=np.float32)

    save_wav_mono16(path, audio, sample_rate=44100)

    with wave.open(path, "rb") as wav:
        assert wav.getframerate() == 44100
        frames = wav.readframes(wav.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        assert np.allclose(pcm[: audio.size], audio, atol=1e-4)
