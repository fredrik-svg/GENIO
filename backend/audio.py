
import io
import logging
import math
import os
import queue
import struct
import time
import wave

from typing import BinaryIO, cast

import numpy as np
if not getattr(wave.open, "__genio_pathlike__", False):  # pragma: no cover - exercised in tests
    _original_wave_open = wave.open

    def _wave_open(path, mode=None):
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        return _original_wave_open(path, mode)

    _wave_open.__genio_pathlike__ = True
    wave.open = _wave_open
try:
    import sounddevice as sd
except OSError as exc:  # pragma: no cover - exercised in environments without PortAudio
    class _SoundDeviceStub:
        PortAudioError = RuntimeError

        def __init__(self) -> None:
            self.default = type("_Default", (), {"samplerate": None})()
            self._sounddevice_stub = True

        def __getattr__(self, name):  # pragma: no cover - defensive
            raise RuntimeError("sounddevice library unavailable")

    sd = _SoundDeviceStub()
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None


def _sounddevice_error():
    if getattr(sd, "_sounddevice_stub", False):
        return _SOUNDDEVICE_IMPORT_ERROR
    return None


def _sounddevice_available() -> bool:
    return _sounddevice_error() is None

from .audio_settings import get_selected_input_device
from .config import (
    ENERGY_THRESHOLD,
    MAX_RECORD_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
)


logger = logging.getLogger(__name__)

_COMMON_SAMPLE_RATES = (48000, 44100, 32000, 24000, 22050, 16000, 11025, 8000)

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_KSDATAFORMAT_SUBTYPE_PCM = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = b"\x03\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def _detect_input_device(device):
    """Return ``(has_device, message)`` for *device*.

    The first element indicates whether an audio input device is available.
    When ``has_device`` is ``False`` the second element contains a human
    readable message describing the reason, which can be logged by the caller.
    """

    error = _sounddevice_error()
    if error is not None:
        return False, f"sounddevice library unavailable: {error}"

    try:
        if device not in (None, ""):
            sd.query_devices(device, "input")
            return True, None
    except Exception as exc:  # pragma: no cover - defensive, exercised via tests
        return False, f"Configured audio input device {device!r} is unavailable: {exc}"

    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Unable to query audio input devices: {exc}"

    for info in devices:
        channels = None
        if isinstance(info, dict):
            channels = info.get("max_input_channels")
        else:
            channels = getattr(info, "max_input_channels", None)
        if isinstance(channels, (int, float)) and channels > 0:
            return True, None

    return False, "No audio input devices detected."


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

    if _sounddevice_available():
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


def _read_chunks(buffer: io.BytesIO):
    while True:
        header = buffer.read(8)
        if len(header) < 8:
            return
        chunk_id, chunk_size = struct.unpack("<4sI", header)
        data = buffer.read(chunk_size)
        if len(data) < chunk_size:
            raise ValueError("Truncated WAV chunk")
        if chunk_size % 2 == 1:
            buffer.seek(1, io.SEEK_CUR)
        yield chunk_id, data


def _parse_wav_metadata(data: bytes):
    stream = io.BytesIO(data)
    if stream.read(4) != b"RIFF":
        raise ValueError("Not a RIFF container")
    size_bytes = stream.read(4)
    if len(size_bytes) < 4:
        raise ValueError("Truncated RIFF header")
    if stream.read(4) != b"WAVE":
        raise ValueError("Not a WAVE file")

    fmt_chunk = None
    data_chunk = None
    for chunk_id, chunk_data in _read_chunks(stream):
        if chunk_id == b"fmt " and fmt_chunk is None:
            fmt_chunk = chunk_data
        elif chunk_id == b"data" and data_chunk is None:
            data_chunk = chunk_data
        if fmt_chunk is not None and data_chunk is not None:
            break

    if fmt_chunk is None or data_chunk is None:
        raise ValueError("Missing fmt or data chunk")
    if len(fmt_chunk) < 16:
        raise ValueError("Invalid fmt chunk")

    format_tag, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
        "<HHIIHH", fmt_chunk[:16]
    )
    extra = fmt_chunk[16:]

    if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(extra) >= 2:
        cb_size = struct.unpack("<H", extra[:2])[0]
        extension = extra[2 : 2 + cb_size]
        if len(extension) >= 22:
            subformat = extension[6:22]
            if subformat == _KSDATAFORMAT_SUBTYPE_PCM:
                format_tag = _WAVE_FORMAT_PCM
            elif subformat == _KSDATAFORMAT_SUBTYPE_IEEE_FLOAT:
                format_tag = _WAVE_FORMAT_IEEE_FLOAT

    return {
        "format_tag": format_tag,
        "channels": channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "block_align": block_align,
        "data": data_chunk,
    }


def _pcm_bytes_to_float(raw: bytes, channels: int, bits_per_sample: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Invalid channel count")
    if bits_per_sample <= 0:
        raise ValueError("Invalid bits per sample")

    sample_width = (bits_per_sample + 7) // 8
    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 3:
        as_uint8 = np.frombuffer(raw, dtype=np.uint8)
        trim = as_uint8.size - (as_uint8.size % 3)
        if trim != as_uint8.size:
            as_uint8 = as_uint8[:trim]
        reshaped = as_uint8.reshape(-1, 3)
        ints = (
            reshaped[:, 0].astype(np.int32)
            | (reshaped[:, 1].astype(np.int32) << 8)
            | (reshaped[:, 2].astype(np.int32) << 16)
        )
        mask = (1 << bits_per_sample) - 1
        sign_bit = 1 << (bits_per_sample - 1)
        ints = ints & mask
        ints = (ints ^ sign_bit) - sign_bit
        audio = ints.astype(np.float32) / float(sign_bit)
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width}")

    if channels > 1:
        frame_count = audio.size // channels
        audio = audio[: frame_count * channels]
        audio = audio.reshape(-1, channels)

    return audio.astype(np.float32, copy=False)


def _decode_wav_audio(data: bytes) -> tuple[np.ndarray, int]:
    metadata = _parse_wav_metadata(data)
    channels = metadata["channels"]
    sample_rate = metadata["sample_rate"]
    bits_per_sample = metadata["bits_per_sample"]
    block_align = metadata["block_align"] or channels * ((bits_per_sample + 7) // 8)
    raw = metadata["data"]

    if channels <= 0 or sample_rate <= 0:
        raise ValueError("Invalid WAV metadata")
    if not raw:
        return np.zeros((0,), dtype=np.float32), sample_rate

    frame_size = block_align or channels * ((bits_per_sample + 7) // 8)
    if frame_size <= 0:
        raise ValueError("Invalid frame size")
    frame_count = len(raw) // frame_size
    raw = raw[: frame_count * frame_size]

    format_tag = metadata["format_tag"]
    if format_tag == _WAVE_FORMAT_PCM:
        audio = _pcm_bytes_to_float(raw, channels, bits_per_sample)
    elif format_tag == _WAVE_FORMAT_IEEE_FLOAT:
        sample_width = (bits_per_sample + 7) // 8
        if sample_width != 4:
            raise ValueError("Unsupported float sample width")
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        if channels > 1:
            frame_count = audio.size // channels
            audio = audio[: frame_count * channels]
            audio = audio.reshape(-1, channels)
    else:
        raise ValueError(f"Unsupported WAV format tag: {format_tag}")

    return audio.astype(np.float32, copy=False), sample_rate


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

    device = get_selected_input_device()

    has_device, message = _detect_input_device(device)
    if not has_device:
        if message:
            logger.warning("%s Returning silence.", message)
        else:  # pragma: no cover - defensive
            logger.warning("No usable audio input device detected; returning silence.")
        return np.zeros((0,), dtype=np.float32)

    def open_stream(sample_rate: int):
        if not _sounddevice_available():  # pragma: no cover - defensive
            raise RuntimeError("sounddevice is unavailable")
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
            logger.warning(
                "Could not open any audio input stream (PortAudioError): %s",
                fallback_exc,
            )
            return np.zeros((0,), dtype=np.float32)
        except Exception as fallback_exc:  # pragma: no cover - defensive
            logger.warning("Could not open any audio input stream: %s", fallback_exc)
            return np.zeros((0,), dtype=np.float32)

    if stream is None:
        logger.warning("Audio input stream could not be created; returning silence.")
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

def save_wav_mono16(
    destination: str | os.PathLike[str] | BinaryIO,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Spara ``audio`` som mono 16-bit PCM WAV.

    ``destination`` kan vara en sökväg eller ett filobjekt med ett ``write``-
    gränssnitt. För filobjekt försöker vi lämna strömmen öppen efteråt så att
    anroparen kan läsa resultatet utan att behöva skriva till disk.
    """

    float_audio = np.asarray(audio, dtype=np.float32)
    if float_audio.ndim > 1:
        float_audio = float_audio.reshape(-1)
    ints = np.clip(float_audio * 32767.0, -32768, 32767).astype(np.int16)

    def _write(wav_file: wave.Wave_write) -> None:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(ints.tobytes())

    if hasattr(destination, "write") and not isinstance(destination, (str, os.PathLike)):
        binary = cast(BinaryIO, destination)

        class _NonClosingAdapter:
            def __init__(self, raw: BinaryIO) -> None:
                self._raw = raw

            def write(self, data: bytes) -> int:
                return self._raw.write(data)

            def tell(self) -> int:
                return self._raw.tell()

            def seek(self, offset: int, whence: int = io.SEEK_SET):
                return self._raw.seek(offset, whence)

            def flush(self) -> None:
                flush = getattr(self._raw, "flush", None)
                if flush is not None:
                    flush()
                return None

            def close(self) -> None:
                # ``wave`` försöker stänga strömmen – ignorera för att låta
                # anroparen hantera livscykeln.
                return None

        adapter = _NonClosingAdapter(binary)
        wav_file = wave.open(adapter, "wb")
        try:
            _write(wav_file)
        finally:
            wav_file.close()

        try:
            binary.flush()
        except AttributeError:
            pass
        except OSError:
            logger.debug("Failed to flush WAV destination", exc_info=True)
        try:
            binary.seek(0)
        except (AttributeError, OSError, ValueError):
            pass
        return

    path_like = os.fspath(destination) if isinstance(destination, os.PathLike) else destination

    with wave.open(path_like, "wb") as wav_file:
        _write(wav_file)


def play_wav_bytes(data: bytes) -> None:
    """Spela upp WAV-data med sounddevice som blockar tills färdigt."""

    if not data:
        return

    audio, sample_rate = _decode_wav_audio(data)
    if audio.size == 0:
        return

    error = _sounddevice_error()
    if error is not None:
        logger.warning(
            "Cannot play audio because sounddevice is unavailable: %s",
            error,
        )
        return

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def list_input_devices():
    """Returnera (lista, felmeddelande)."""

    error = _sounddevice_error()
    if error is not None:
        return [], f"sounddevice library unavailable: {error}"

    try:
        raw_devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - defensiv loggning
        logger.error("Could not query audio input devices: %s", exc)
        return [], f"Kunde inte läsa ljudenheter: {exc}"

    hostapi_names: dict[int, str] = {}
    try:
        raw_hostapis = sd.query_hostapis()
    except Exception:  # pragma: no cover - saknar stöd för host API-lista
        raw_hostapis = []
    for idx, info in enumerate(raw_hostapis):
        if isinstance(info, dict):
            index_value = info.get("index", idx)
            name_value = info.get("name")
        else:
            index_value = getattr(info, "index", idx)
            name_value = getattr(info, "name", None)
        try:
            index_int = int(index_value)
        except Exception:  # pragma: no cover - defensiv
            continue
        if not name_value:
            continue
        hostapi_names[index_int] = str(name_value)

    default_input_index = None
    try:
        default_device = getattr(getattr(sd, "default", None), "device", None)
        if isinstance(default_device, (tuple, list)) and default_device:
            default_input_index = int(default_device[0])
        elif isinstance(default_device, (int, float)):
            default_input_index = int(default_device)
    except Exception:  # pragma: no cover - defensiv
        default_input_index = None

    devices: list[dict[str, object]] = []
    for idx, info in enumerate(raw_devices):
        if isinstance(info, dict):
            channels = info.get("max_input_channels")
            index_value = info.get("index", idx)
            name_value = info.get("name")
            default_rate = info.get("default_samplerate")
            hostapi_index = info.get("hostapi")
        else:
            channels = getattr(info, "max_input_channels", None)
            index_value = getattr(info, "index", idx)
            name_value = getattr(info, "name", None)
            default_rate = getattr(info, "default_samplerate", None)
            hostapi_index = getattr(info, "hostapi", None)

        if not isinstance(channels, (int, float)) or channels <= 0:
            continue

        try:
            index_int = int(index_value)
        except Exception:
            index_int = idx

        sample_rate = None
        if isinstance(default_rate, (int, float)) and math.isfinite(float(default_rate)):
            sample_rate = float(default_rate)

        hostapi_name = None
        if isinstance(hostapi_index, (int, float)):
            hostapi_name = hostapi_names.get(int(hostapi_index))

        name_str = str(name_value) if name_value else f"Enhet {index_int}"

        entry = {
            "index": index_int,
            "name": name_str,
            "maxInputChannels": int(channels),
            "defaultSampleRate": sample_rate,
            "hostapi": hostapi_name,
            "isDefault": default_input_index is not None and index_int == default_input_index,
            "value": f"index:{index_int}",
        }

        devices.append(entry)

    return devices, None
