from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from openwakeword import Model as _OpenWakeWordModel
except ImportError:  # pragma: no cover - optional dependency missing in tests
    _OpenWakeWordModel = None

try:  # pragma: no cover - optional dependency
    from openwakeword.utils import download_models as _openwakeword_download_models
except Exception:  # pragma: no cover - optional dependency missing in tests
    _openwakeword_download_models = None

from .audio import _gather_fallback_sample_rates, _supports_input_sample_rate
from .audio_settings import get_selected_input_device
from .config import (
    WAKEWORD_COOLDOWN,
    WAKEWORD_ENGINE,
    WAKEWORD_MIN_ACTIVATIONS,
    WAKEWORD_MODELS,
    WAKEWORD_MODEL_PATHS,
)

logger = logging.getLogger(__name__)


def _open_wake_word_input_stream(sd_module, samplerate: int, blocksize: int, device):
    """Open an ``InputStream`` for wake word detection with graceful fallbacks."""

    def rate_supported(rate: int) -> bool:
        try:
            return _supports_input_sample_rate(device, 1, "float32", rate)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Failed to determine whether %d Hz is supported for wake word input.",
                rate,
                exc_info=True,
            )
            return True

    last_error: Optional[Exception] = None
    attempts = 0

    def try_start(rate: int, *, is_primary: bool):
        nonlocal last_error, attempts

        supported = rate_supported(rate)
        if not supported:
            logger.debug(
                "Wake word input device %r reported %d Hz as unsupported; trying anyway.",
                device,
                rate,
            )

        adjusted_blocksize = max(1, int(round(blocksize * rate / float(samplerate))))
        attempts += 1

        for dtype in ("float32", "int16"):
            try:
                stream = sd_module.InputStream(
                    device=device,
                    channels=1,
                    samplerate=rate,
                    blocksize=adjusted_blocksize,
                    dtype=dtype,
                )
                stream.start()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to start wake word audio stream at %d Hz with dtype %s: %s",
                    rate,
                    dtype,
                    exc,
                )
                continue

            if not is_primary:
                logger.info(
                    "Wake word listener using fallback sample rate %d Hz",
                    rate,
                )
            if dtype != "float32":
                logger.info(
                    "Wake word listener using %s audio format for wake word input",
                    dtype,
                )

            return stream, rate, adjusted_blocksize, dtype

        return None

    result = try_start(samplerate, is_primary=True)
    if result is not None:
        return result

    for candidate_rate in _gather_fallback_sample_rates(device, {samplerate}):
        result = try_start(candidate_rate, is_primary=False)
        if result is not None:
            return result

    if last_error is None:
        if attempts:
            last_error = RuntimeError("Unable to open audio input stream")
        else:
            last_error = RuntimeError("No usable wake word sample rates")

    raise last_error


class WakeWordListener:
    def __init__(
        self,
        on_detect: Callable[[], None],
        detection_threshold: float = 0.5,
        *,
        engine: Optional[str] = None,
        fallback_energy_threshold: float = 0.2,
        fallback_required_blocks: int = 15,
        min_activations: Optional[int] = None,
        cooldown: Optional[float] = None,
    ):
        self.on_detect = on_detect
        self.detection_threshold = detection_threshold
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._detector: Optional[_BaseWakeWordDetector] = None
        self._detector_name = "none"
        self._cooldown = cooldown if cooldown is not None else WAKEWORD_COOLDOWN
        self._min_activations = max(1, min_activations or WAKEWORD_MIN_ACTIVATIONS)

        requested_engine = (engine or WAKEWORD_ENGINE or "openwakeword").strip().lower()
        candidates: List[str]
        if requested_engine in {"", "auto", "any"}:
            candidates = ["openwakeword", "energy"]
        else:
            candidates = [requested_engine]
            if requested_engine != "energy":
                candidates.append("energy")

        failure_reasons: List[str] = []
        for candidate in candidates:
            if candidate == "openwakeword":
                detector, reason = self._setup_openwakeword_detector(detection_threshold)
                if detector:
                    self._detector = detector
                    self._detector_name = "openwakeword"
                    break
                if reason:
                    failure_reasons.append(reason)
                    logger.warning("%s; använder energibaserad fallback för wakeword.", reason)
            elif candidate == "energy":
                self._detector = _EnergyWakeWordDetector(
                    energy_threshold=fallback_energy_threshold,
                    required_consecutive_blocks=fallback_required_blocks,
                    cooldown=self._cooldown,
                )
                self._detector_name = "energy"
                break
            else:
                message = f"Unknown wake word engine '{candidate}' requested"
                failure_reasons.append(message)
                logger.warning(message)

        if self._detector is None:
            reason = "; ".join(failure_reasons) or "No wake word detector could be initialised"
            logger.warning(
                "%s; falling back to simple energy-based trigger.",
                reason,
            )
            self._detector = _EnergyWakeWordDetector(
                energy_threshold=fallback_energy_threshold,
                required_consecutive_blocks=fallback_required_blocks,
                cooldown=self._cooldown,
            )
            self._detector_name = "energy"

    def _setup_openwakeword_detector(
        self, detection_threshold: float
    ) -> tuple[Optional[_BaseWakeWordDetector], Optional[str]]:
        if _OpenWakeWordModel is None:
            return None, "openwakeword är inte installerat"

        builtin_models = _parse_env_list(WAKEWORD_MODELS)
        custom_paths = _parse_env_list(WAKEWORD_MODEL_PATHS)

        if not builtin_models and not custom_paths:
            builtin_models = ["hey_mycroft"]

        if builtin_models and _openwakeword_download_models is not None:
            try:  # pragma: no cover - nätverksberoende
                _openwakeword_download_models(model_names=builtin_models)
            except Exception as exc:  # pragma: no cover - bästa försök
                logger.debug("Could not download OpenWakeWord models: %s", exc, exc_info=True)

        kwargs: dict[str, Any] = {}
        if builtin_models:
            kwargs["wakeword_models"] = builtin_models
        if custom_paths:
            kwargs["custom_models"] = custom_paths

        try:
            model = _OpenWakeWordModel(**kwargs)
        except TypeError:
            # fallback for äldre versioner som använder ``custom_paths``
            if "custom_models" in kwargs:
                alt_kwargs = dict(kwargs)
                paths = alt_kwargs.pop("custom_models")
                alt_kwargs["custom_paths"] = paths
            else:
                alt_kwargs = kwargs
            try:
                model = _OpenWakeWordModel(**alt_kwargs)
            except Exception as exc:
                return None, f"Kunde inte starta OpenWakeWord: {exc}"
        except Exception as exc:
            return None, f"Kunde inte starta OpenWakeWord: {exc}"

        labels = _resolve_openwakeword_labels(model, builtin_models, custom_paths)
        detector = _OpenWakeWordDetector(
            model,
            labels=labels,
            detection_threshold=detection_threshold,
            min_activations=self._min_activations,
            cooldown=self._cooldown,
        )
        return detector, None

    def start(self):
        if self._detector is None:
            logger.info("Wake word listener not started because no detector is available.")
            return False

        if self.is_running():
            return True

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        if not self.is_running():
            return

        thread = self._thread
        if thread is None:
            return

        self._stop.set()
        if threading.get_ident() != thread.ident:
            thread.join(timeout=0.5)
        self._thread = None

        detector = self._detector
        if detector is not None:
            try:
                detector.close()
            except Exception:  # pragma: no cover - defensivt
                logger.debug("Wake word detector close failed", exc_info=True)

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def suspend(self, timeout: float = 1.0) -> bool:
        """Temporarily stop listening so that the microphone can be reused."""

        if not self.is_running():
            return False

        thread = self._thread
        if thread is None:
            return False

        if threading.get_ident() == thread.ident:
            logger.debug(
                "Refusing to suspend wake word listener from within its own thread.",
            )
            return False

        self._stop.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Wake word listener thread did not stop within %.1fs when suspending.",
                timeout,
            )
            return False

        self._thread = None
        self._stop = threading.Event()
        return True

    def resume(self) -> bool:
        """Restart the listener after :meth:`suspend`."""

        return self.start()

    def _run(self):
        if self._detector is None:
            return
        import sounddevice as sd
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy bör finnas i produktion
            np = None  # type: ignore[assignment]

        samplerate = getattr(self._detector, "sample_rate", 16000)
        blocksize = getattr(self._detector, "frame_length", 512)

        cooldown = getattr(self._detector, "cooldown", self._cooldown)
        device = get_selected_input_device()

        def resample_block(values, from_rate):
            if from_rate == samplerate or blocksize <= 0:
                return values
            length = len(values)
            if length == 0:
                return values

            duration = length / float(from_rate)
            target_length = blocksize if blocksize > 0 else int(round(duration * samplerate))
            target_length = max(1, target_length)

            if np is not None:
                old_times = np.linspace(
                    0.0,
                    duration,
                    num=length,
                    endpoint=False,
                    dtype=np.float64,
                )
                new_times = np.linspace(
                    0.0,
                    duration,
                    num=target_length,
                    endpoint=False,
                    dtype=np.float64,
                )
                return np.interp(new_times, old_times, values).astype(np.float32, copy=False)

            if length == 1:
                return [float(values[0])] * target_length

            original_step = duration / length
            target_step = duration / target_length
            result: List[float] = []
            for i in range(target_length):
                t = i * target_step
                position = t / original_step
                lower = int(math.floor(position))
                if lower >= length - 1:
                    result.append(float(values[-1]))
                    continue
                upper = lower + 1
                fraction = position - lower
                lower_value = float(values[lower])
                upper_value = float(values[upper])
                result.append(lower_value * (1.0 - fraction) + upper_value * fraction)
            return result

        def start_stream():
            return _open_wake_word_input_stream(
                sd,
                samplerate=samplerate,
                blocksize=blocksize,
                device=device,
            )

        stream = None
        effective_sample_rate = samplerate
        frames_per_read = blocksize
        stream_dtype = "float32"
        try:
            try:
                stream, effective_sample_rate, frames_per_read, stream_dtype = start_stream()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Could not start wake word audio stream: %s", exc)
                return

            while not self._stop.is_set():
                try:
                    audio_block, _ = stream.read(frames_per_read)
                except Exception as exc:  # pragma: no cover - defensive
                    if self._stop.is_set():
                        break
                    logger.error("Wake word audio stream failed: %s", exc)
                    break

                if np is not None:
                    y = audio_block[:, 0].astype(np.float32, copy=False)
                    if stream_dtype.startswith("int"):
                        y *= 1.0 / 32768.0
                else:  # pragma: no cover - numpy saknas endast i testmiljöer
                    y = [float(sample[0]) for sample in audio_block]
                    if stream_dtype.startswith("int"):
                        y = [value / 32768.0 for value in y]

                if effective_sample_rate != samplerate:
                    y = resample_block(y, effective_sample_rate)

                if not self._detector.process(y):
                    continue

                try:
                    stream.stop()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("Failed to stop wake word stream", exc_info=True)
                try:
                    stream.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("Failed to close wake word stream", exc_info=True)
                stream = None

                try:
                    self.on_detect()
                except Exception:  # pragma: no cover - defensive guard
                    logger.exception("Wake word callback raised an exception")

                if self._stop.is_set():
                    break

                time.sleep(cooldown)

                if self._stop.is_set():
                    break

                try:
                    stream, effective_sample_rate, frames_per_read, stream_dtype = start_stream()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.error("Could not restart wake word audio stream: %s", exc)
                    break
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
                try:
                    stream.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass

    @property
    def detector_name(self) -> str:
        return self._detector_name


class _BaseWakeWordDetector:
    cooldown: float = WAKEWORD_COOLDOWN
    sample_rate: int = 16000
    frame_length: int = 512

    def process(self, audio) -> bool:  # pragma: no cover - interface definition only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        return


class _OpenWakeWordDetector(_BaseWakeWordDetector):
    def __init__(
        self,
        model: Any,
        *,
        labels: Sequence[str],
        detection_threshold: float,
        min_activations: int,
        cooldown: float,
    ) -> None:
        if min_activations <= 0:
            raise ValueError("min_activations must be positive")
        self._model = model
        self._labels = [str(label) for label in labels] or ["wakeword"]
        self._threshold = float(detection_threshold)
        self._min_activations = int(min_activations)
        self.cooldown = float(cooldown)
        self.sample_rate = int(getattr(model, "sample_rate", 16000))
        frame_length = (
            getattr(model, "frame_length", None)
            or getattr(model, "chunk_size", None)
            or getattr(model, "samples_per_frame", None)
            or getattr(model, "samples_per_inference", None)
        )
        self.frame_length = int(frame_length or 512)
        self._streak = 0

    def process(self, audio) -> bool:
        values = _ensure_float32(audio)
        if _audio_length(values) == 0:
            self._streak = 0
            return False
        try:
            predictions = self._model.predict(values)
        except Exception:  # pragma: no cover - defensivt skydd
            logger.exception("OpenWakeWord prediction misslyckades")
            return False

        scores = _extract_prediction_scores(predictions, self._labels)
        if any(score >= self._threshold for score in scores):
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self._min_activations:
            self._streak = 0
            return True
        return False

    def close(self) -> None:  # pragma: no cover - best effort cleanup
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()
        close = getattr(self._model, "close", None)
        if callable(close):
            close()


class _EnergyWakeWordDetector(_BaseWakeWordDetector):
    """Naiv reservlösning: trigga om rösten är tillräckligt stark en stund."""

    def __init__(
        self,
        *,
        energy_threshold: float = 0.2,
        required_consecutive_blocks: int = 15,
        cooldown: float = WAKEWORD_COOLDOWN,
        time_source: Optional[Callable[[], float]] = None,
    ) -> None:
        if required_consecutive_blocks <= 0:
            raise ValueError("required_consecutive_blocks must be positive")
        self.energy_threshold = energy_threshold
        self.required_consecutive_blocks = required_consecutive_blocks
        self.cooldown = cooldown
        self._time = time_source or time.monotonic
        self._consecutive = 0
        self._last_trigger = float("-inf")

    def process(self, audio) -> bool:
        now = self._time()
        if (now - self._last_trigger) < self.cooldown:
            return False

        if _audio_length(audio) == 0:
            self._consecutive = 0
            return False

        energy = _rms_energy(audio)
        if energy >= self.energy_threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= self.required_consecutive_blocks:
            self._consecutive = 0
            self._last_trigger = now
            return True
        return False


def _resolve_openwakeword_labels(
    model: Any, builtin_models: Sequence[str], custom_paths: Sequence[str]
) -> List[str]:
    for attr in ("labels", "wakeword_names", "model_names", "keywords"):
        value = getattr(model, attr, None)
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]
    labels = [str(name) for name in builtin_models]
    labels.extend(os.path.splitext(os.path.basename(path))[0] for path in custom_paths)
    if not labels:
        labels = ["wakeword"]
    return labels


def _extract_prediction_scores(predictions: Any, labels: Sequence[str]) -> List[float]:
    if isinstance(predictions, dict):
        return [float(predictions.get(label, 0.0)) for label in labels]

    try:
        items = dict(predictions)
    except Exception:
        return [0.0 for _ in labels]
    return [float(items.get(label, 0.0)) for label in labels]


def _audio_length(audio) -> int:
    try:
        return len(audio)
    except TypeError:  # pragma: no cover - defensive fallback
        return 0


def _rms_energy(audio) -> float:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - testmiljö utan numpy
        total = 0.0
        count = 0
        for sample in audio:
            value = float(sample)
            total += value * value
            count += 1
        if count == 0:
            return 0.0
        return math.sqrt(total / count)

    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def _ensure_float32(audio: Sequence[float]):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy saknas endast i testmiljöer
        return [float(sample) for sample in audio]

    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr


def _ensure_int16(audio: Sequence[float]):  # pragma: no cover - kvar för bakåtkompatibilitet
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy saknas endast i testmiljöer
        pcm: List[int] = []
        for sample in audio:
            value = max(min(float(sample), 1.0), -1.0)
            pcm.append(int(value * 32767))
        return pcm

    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int16)
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16, copy=False)


def _parse_env_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    items: List[str] = []
    for entry in value.split(","):
        item = entry.strip()
        if not item:
            continue
        if item.startswith("~"):
            item = os.path.expanduser(item)
        items.append(item)
    return items


__all__ = ["WakeWordListener"]
