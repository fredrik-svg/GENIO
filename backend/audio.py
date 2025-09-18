
import queue, time, math
import numpy as np
import sounddevice as sd

from .config import SAMPLE_RATE, MAX_RECORD_SECONDS, SILENCE_DURATION, ENERGY_THRESHOLD, INPUT_DEVICE

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

    stream = sd.InputStream(device=INPUT_DEVICE if INPUT_DEVICE else None,
        samplerate=SAMPLE_RATE,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
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
