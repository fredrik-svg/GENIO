
import contextlib
import io
import logging
import math
import queue
import time
import wave

import numpy as np
import sounddevice as sd

from .config import (
    ENERGY_THRESHOLD,
    INPUT_DEVICE,
    MAX_RECORD_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
)


logger = logging.getLogger(__name__)

_COMMON_SAMPLE_RATES = (48000, 44100, 32000, 24000, 22050, 16000, 11025, 8000)


def _gather_fallback_sample_rates(device, excluded_rates):
    """Return an ordered list of candidate sample rates excluding *excluded_rates*."""

    candidates = []
    seen = set(excluded_rates)

    def add(rate):
        if rate is None:
            return
        try:
            numeric = float(rate)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric) or numeric <= 0:
            return
        rounded = int(round(numeric))
        if rounded in seen:
            return
        seen.add(rounded)
        candidates.append(rounded)

    try:
        device_info = sd.query_devices(device, "input")
    except Exception as info_exc:  # pragma: no cover - defensive
        logger.error("Could not query input device info: %s", info_exc)
    else:
        add(device_info.get("default_samplerate"))

    default_samplerate = getattr(getattr(sd, "default", None), "samplerate", None)
    add(default_samplerate)

    for rate in _COMMON_SAMPLE_RATES:
        add(rate)

    return candidates


def _open_stream_with_fallback(open_stream, device, excluded_rates):
    """Try to open *open_stream* with fallback sample rates.

    Returns a tuple ``(stream, sample_rate)``.
    """

    last_error = None
    for candidate_rate in _gather_fallback_sample_rates(device, excluded_rates):
        try:
            stream = open_stream(candidate_rate)
        except sd.PortAudioError as exc:
            last_error = exc
            logger.warning(
                "Failed to open input stream at %d Hz: %s", candidate_rate, exc
            )
            continue
        logger.info("Recording using fallback sample rate %d Hz", candidate_rate)
        return stream, candidate_rate

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to open audio input stream")

def rms_energy(audio: np.ndarray) -> float:
    """Returnera RMS-energi (0..1 ungefär)."""
    return float(np.sqrt(np.mean(np.square(audio))))

def record_until_silence() -> np.ndarray:
    """Spelar in mono, 16kHz tills tystnad eller maxlängd."""
    q = queue.Queue()
    duration_limit = MAX_RECORD_SECONDS
    silence_hang = SILENCE_DURATION
    channels = 1
    blocksize = 1024

    def callback(indata, frames, time_info, status):
        if status:
            # print(status)
            pass
        q.put(indata.copy())

    device = INPUT_DEVICE if INPUT_DEVICE else None

    def open_stream(sample_rate: int) -> sd.InputStream:
        return sd.InputStream(
            device=device,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )

    stream = None
    effective_sample_rate = SAMPLE_RATE
    try:
        stream = open_stream(SAMPLE_RATE)
    except sd.PortAudioError as exc:
        logger.warning("Failed to open input stream at %d Hz: %s", SAMPLE_RATE, exc)
        try:
            stream, effective_sample_rate = _open_stream_with_fallback(
                open_stream, device, {SAMPLE_RATE}
            )
        except sd.PortAudioError as fallback_exc:
            logger.error(
                "Could not open any audio input stream (PortAudioError): %s",
                fallback_exc,
            )
            return np.zeros((0,), dtype=np.float32)
        except Exception as fallback_exc:  # pragma: no cover - defensive
            logger.error("Could not open any audio input stream: %s", fallback_exc)
            return np.zeros((0,), dtype=np.float32)

    if stream is None:
        logger.error("Audio input stream could not be created; returning silence.")
        return np.zeros((0,), dtype=np.float32)
    audio_chunks = []
    with stream:
        start = time.time()
        last_voice_time = start
        while True:
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                data = None
            if data is not None:
                audio_chunks.append(data[:,0])  # mono
                energy = rms_energy(data[:,0])
                if energy > ENERGY_THRESHOLD:
                    last_voice_time = time.time()

            now = time.time()
            if (now - start) > duration_limit:
                break
            if (now - last_voice_time) > silence_hang and (now - start) > 1.0:
                # minst 1s inspelat + tystnad
                break

    if not audio_chunks:
        return np.zeros((0,), dtype=np.float32)
    audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)

    if effective_sample_rate != SAMPLE_RATE and audio.size > 0:
        duration = audio.size / float(effective_sample_rate)
        target_length = max(1, int(round(duration * SAMPLE_RATE)))
        if target_length > 0:
            old_times = np.linspace(0.0, duration, num=audio.size, endpoint=False, dtype=np.float64)
            new_times = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float64)
            audio = np.interp(new_times, old_times, audio).astype(np.float32)
            logger.info("Resampled audio from %d Hz to %d Hz", effective_sample_rate, SAMPLE_RATE)
    # normalisera lätt
    if len(audio) > 0:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.97
    return audio

def save_wav_mono16(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Sparar float32 [-1..1] som 16-bit PCM WAV."""

    wav = wave.open(path, 'wb')
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    # konvertera till int16
    ints = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    wav.writeframes(ints.tobytes())
    wav.close()


def play_wav_bytes(data: bytes) -> None:
    """Spela upp WAV-data med sounddevice som blockar tills färdigt."""

    if not data:
        return

    with contextlib.closing(wave.open(io.BytesIO(data), 'rb')) as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    if not raw:
        return

    if sampwidth == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        audio = np.reshape(audio, (-1, channels))

    sd.play(audio, samplerate=sample_rate)
    sd.wait()
