from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:
    import pvporcupine
except ImportError:  # pragma: no cover - optional dependency on Pi
    pvporcupine = None  # type: ignore[assignment]


from .audio_settings import get_selected_input_device


DEFAULT_PORCUPINE_KEYWORDS: Tuple[str, ...] = ("porcupine",)

logger = logging.getLogger(__name__)


class WakeWordListener:
    def __init__(
        self,
        on_detect: Callable[[], None],
        detection_threshold: float = 0.5,
        *,
        fallback_energy_threshold: float = 0.2,
        fallback_required_blocks: int = 15,
    ):
        self.on_detect = on_detect
        self.detection_threshold = detection_threshold
        self._stop = threading.Event()
        self._porcupine: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._detector: Optional[_BaseWakeWordDetector] = None
        self._detector_name = "none"

        def configure_fallback_detector(reason: str) -> None:
            logger.warning(
                "%s; falling back to simple energy-based trigger.",
                reason,
            )
            self._detector = _EnergyWakeWordDetector(
                energy_threshold=fallback_energy_threshold,
                required_consecutive_blocks=fallback_required_blocks,
            )
            self._detector_name = "energy"

        if pvporcupine is None:
            configure_fallback_detector("pvporcupine is not installed")
            return

        try:
            kwargs = _porcupine_create_kwargs(
                detection_threshold,
                keywords_env=os.getenv("PORCUPINE_KEYWORDS"),
                keyword_paths_env=os.getenv("PORCUPINE_KEYWORD_PATHS"),
                access_key=os.getenv("PICOVOICE_ACCESS_KEY"),
            )
        except ValueError as config_error:
            configure_fallback_detector(str(config_error))
            return

        try:
            self._porcupine = pvporcupine.create(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive guard against optional dependency issues
            configure_fallback_detector(f"Porcupine could not be initialised: {exc}")
            return

        self._detector = _PorcupineWakeWordDetector(self._porcupine)
        self._detector_name = "porcupine"

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

        cooldown = getattr(self._detector, "cooldown", 2.0)

        def open_stream():
            stream = sd.InputStream(
                device=get_selected_input_device(),
                channels=1,
                samplerate=samplerate,
                blocksize=blocksize,
                dtype="float32",
            )
            stream.start()
            return stream

        stream = None
        try:
            try:
                stream = open_stream()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Could not start wake word audio stream: %s", exc)
                return

            while not self._stop.is_set():
                try:
                    audio_block, _ = stream.read(blocksize)
                except Exception as exc:  # pragma: no cover - defensive
                    if self._stop.is_set():
                        break
                    logger.error("Wake word audio stream failed: %s", exc)
                    break

                if np is not None:
                    y = audio_block[:, 0].astype(np.float32, copy=False)
                else:  # pragma: no cover - numpy saknas endast i testmiljöer
                    y = [float(sample[0]) for sample in audio_block]

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
                    stream = open_stream()
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
    cooldown: float = 2.0
    sample_rate: int = 16000
    frame_length: int = 512

    def process(self, audio) -> bool:  # pragma: no cover - interface definition only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        return


class _PorcupineWakeWordDetector(_BaseWakeWordDetector):
    def __init__(self, porcupine_instance: Any) -> None:
        self._porcupine = porcupine_instance
        self.sample_rate = int(getattr(porcupine_instance, "sample_rate", 16000))
        self.frame_length = int(getattr(porcupine_instance, "frame_length", 512))
        self.cooldown = 1.0

    def process(self, audio) -> bool:
        pcm = _ensure_int16(audio)
        if _audio_length(pcm) != self.frame_length:
            return False
        result = self._porcupine.process(pcm)
        try:
            detected = int(result) >= 0
        except (TypeError, ValueError):  # pragma: no cover - defensive conversion guard
            detected = False
        return detected

    def close(self) -> None:  # pragma: no cover - best effort cleanup
        delete = getattr(self._porcupine, "delete", None)
        if callable(delete):
            delete()


class _EnergyWakeWordDetector(_BaseWakeWordDetector):
    """Naiv reservlösning: trigga om rösten är tillräckligt stark en stund."""

    def __init__(
        self,
        *,
        energy_threshold: float = 0.2,
        required_consecutive_blocks: int = 15,
        cooldown: float = 2.0,
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


def _ensure_int16(audio: Sequence[float]):
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


def _clamp_sensitivity(value: float) -> float:
    if math.isnan(value):  # pragma: no cover - defensive guard
        return 0.5
    return float(min(max(value, 0.0), 1.0))


def _porcupine_create_kwargs(
    detection_threshold: float,
    *,
    keywords_env: Optional[str],
    keyword_paths_env: Optional[str],
    access_key: Optional[str],
) -> dict[str, Any]:
    keyword_paths = _parse_env_list(keyword_paths_env)
    keywords = _parse_env_list(keywords_env)

    kwargs: dict[str, Any] = {}

    if not access_key:
        raise ValueError(
            "PICOVOICE_ACCESS_KEY must be set to use Porcupine wake word detection"
        )

    if keyword_paths:
        kwargs["keyword_paths"] = keyword_paths
        count = len(keyword_paths)
    else:
        resolved_keywords = keywords or list(DEFAULT_PORCUPINE_KEYWORDS)
        if not resolved_keywords:
            raise ValueError("No Porcupine keywords configured")
        kwargs["keywords"] = resolved_keywords
        count = len(resolved_keywords)

    sensitivity = _clamp_sensitivity(detection_threshold)
    kwargs["sensitivities"] = [sensitivity] * count

    kwargs["access_key"] = access_key

    return kwargs
