from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import math

try:
    from openwakeword import Model
except ImportError:  # pragma: no cover - optional dependency on Pi
    Model = None  # type: ignore[assignment]

# Enkel wrapper för att köra openWakeWord i bakgrunden och trigga callback.
# Standardfras: "Hej kompis" via en allmän modell med fraströskel.
# För bästa resultat, träna/finjustera egen modell och ange filväg här.
WW_PHRASES: List[str] = ["hej kompis", "hejkompis"]

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
        self.model: Optional[Model] = None
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

        # Laddar standardmodeller (engelska/svenska kan fungera okej för enkla fraser).
        if Model is None:
            configure_fallback_detector("openwakeword is not installed")
            return
        try:
            self.model = Model(enable_speex_noise_suppression=True)
        except Exception as exc:  # pragma: no cover - defensive guard against optional dependency issues
            missing_path = _extract_missing_model_path(exc)
            if missing_path and _try_recover_missing_model(missing_path):
                try:
                    self.model = Model(enable_speex_noise_suppression=True)
                except Exception as second_exc:  # pragma: no cover - still failing → log original context
                    logger.warning(
                        "Wake word model could not be loaded even after attempting recovery: %s",
                        second_exc,
                    )
                    self.model = None
            if self.model is None:
                configure_fallback_detector(
                    f"Wake word model could not be loaded: {exc}"
                )
                return

        assert self.model is not None  # För typkontroll
        self._detector = _OpenWakeWordDetector(self.model, detection_threshold)
        self._detector_name = "openwakeword"

    def start(self):
        if self._detector is None:
            logger.info("Wake word listener not started because no detector is available.")
            return False

        if self._thread and self._thread.is_alive():
            return True

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.1)

    def _run(self):
        if self._detector is None:
            return
        import sounddevice as sd
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy bör finnas i produktion
            np = None  # type: ignore[assignment]
        samplerate = 16000
        blocksize = 512
        with sd.InputStream(channels=1, samplerate=samplerate, blocksize=blocksize, dtype="float32") as stream:
            while not self._stop.is_set():
                audio_block, _ = stream.read(blocksize)
                if np is not None:
                    y = audio_block[:, 0].astype(np.float32)
                else:  # pragma: no cover - numpy saknas endast i testmiljöer
                    y = [float(sample[0]) for sample in audio_block]
                if self._detector.process(y):
                    self.on_detect()
                    time.sleep(getattr(self._detector, "cooldown", 2.0))

    @property
    def detector_name(self) -> str:
        return self._detector_name


class _BaseWakeWordDetector:
    cooldown: float = 2.0

    def process(self, audio) -> bool:  # pragma: no cover - interface definition only
        raise NotImplementedError


class _OpenWakeWordDetector(_BaseWakeWordDetector):
    def __init__(self, model: Model, detection_threshold: float) -> None:
        self.model = model
        self.detection_threshold = detection_threshold
        self.cooldown = 2.0

    def process(self, audio) -> bool:
        scores = self.model.predict(audio)
        max_score = 0.0
        for _, score in scores.items():
            max_score = max(max_score, float(score))
        return max_score >= self.detection_threshold


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


def _extract_missing_model_path(exc: Exception) -> Optional[Path]:
    """Försök hitta sökvägen till en saknad TFLite-modell i felmeddelandet."""

    message = str(exc)
    match = re.search(r"'([^']+\.tflite)'", message)
    if not match:
        return None
    return Path(match.group(1))


MODEL_BASE_URLS: Tuple[str, ...] = (
    "https://huggingface.co/dscripka/openwakeword/resolve/main/models/",
    "https://raw.githubusercontent.com/dscripka/openwakeword/main/openwakeword/resources/models/",
    "https://github.com/dscripka/openwakeword/raw/main/openwakeword/resources/models/",
)


def _try_recover_missing_model(target_path: Path) -> bool:
    """Försök ladda ner standardmodellen från openwakewords GitHub om den saknas."""

    if target_path.exists():
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)

    download_error: Optional[Exception] = None
    tmp_name: Optional[str] = None
    for base_url in MODEL_BASE_URLS:
        url = base_url + target_path.name
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GENIO/1.0 wakeword recovery"})
            with urllib.request.urlopen(request, timeout=30) as response:  # nosec - kontrollerad URL
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    shutil.copyfileobj(response, tmp_file)
                    tmp_name = tmp_file.name
        except HTTPError as http_error:
            download_error = http_error
            if http_error.code == 404:
                logger.debug("Wake word model %s not found at %s", target_path.name, url)
                continue
            logger.warning(
                "Wake word model %s could not be downloaded from %s: %s",
                target_path.name,
                url,
                http_error,
            )
            return False
        except Exception as generic_error:  # pragma: no cover - nätverksfel bör endast loggas
            download_error = generic_error
            logger.warning(
                "Wake word model %s could not be downloaded from %s: %s",
                target_path.name,
                url,
                generic_error,
            )
            return False
        else:
            break
    else:
        logger.warning(
            "Wake word model %s could not be downloaded from any known source: %s",
            target_path.name,
            download_error,
        )
        return False

    if tmp_name is None:  # pragma: no cover - defensive guard
        return False

    try:
        shutil.move(tmp_name, target_path)
    except Exception as move_error:  # pragma: no cover - oskrivbara kataloger etc.
        logger.warning(
            "Wake word model %s could not be placed at %s: %s",
            target_path.name,
            target_path,
            move_error,
        )
        return False

    logger.info("Downloaded missing wake word model: %s", target_path)
    return True
