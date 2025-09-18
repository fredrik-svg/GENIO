
import logging
import queue
import time
import math

import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, MAX_RECORD_SECONDS, SILENCE_DURATION, ENERGY_THRESHOLD, INPUT_DEVICE


logger = logging.getLogger(__name__)

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

    effective_sample_rate = SAMPLE_RATE
    try:
        stream = open_stream(SAMPLE_RATE)
    except sd.PortAudioError as exc:
        logger.warning("Failed to open input stream at %d Hz: %s", SAMPLE_RATE, exc)
        fallback_rate = None
        try:
            device_info = sd.query_devices(device, "input")
            fallback_rate = device_info.get("default_samplerate")
        except Exception as info_exc:  # pragma: no cover - defensive
            logger.error("Could not query input device info: %s", info_exc)
        if fallback_rate and math.isfinite(fallback_rate) and fallback_rate > 0:
            effective_sample_rate = int(round(fallback_rate))
            logger.info("Retrying recording with fallback sample rate %d Hz", effective_sample_rate)
            try:
                stream = open_stream(effective_sample_rate)
            except sd.PortAudioError:
                logger.error(
                    "Fallback sample rate %d Hz also failed; giving up.",
                    effective_sample_rate,
                )
                raise
        else:
            raise
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
    import wave, struct
    wav = wave.open(path, 'wb')
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    # konvertera till int16
    ints = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    wav.writeframes(ints.tobytes())
    wav.close()
