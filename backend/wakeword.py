
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
    def __init__(self, on_detect: Callable[[], None], detection_threshold: float = 0.5):
        self.on_detect = on_detect
        self.detection_threshold = detection_threshold
        self._stop = threading.Event()
        self.model: Optional[Model] = None
        self._thread: Optional[threading.Thread] = None

        # Laddar standardmodeller (engelska/svenska kan fungera okej för enkla fraser).
        if Model is None:
            logger.warning(
                "openwakeword is not installed; wake word detection disabled. Install the optional dependency to enable it."
            )
            return
        try:
            self.model = Model(enable_speex_noise_suppression=True)
        except Exception as exc:  # pragma: no cover - defensive guard against optional dependency issues
            recovered = False
            missing_path = _extract_missing_model_path(exc)
            if missing_path:
                recovered = _try_recover_missing_model(missing_path)
                if recovered:
                    try:
                        self.model = Model(enable_speex_noise_suppression=True)
                    except Exception as second_exc:  # pragma: no cover - still failing → log original context
                        logger.warning(
                            "Wake word model could not be loaded even after attempting recovery: %s",
                            second_exc,
                        )
                        return
            if not recovered:
                logger.warning(
                    "Wake word model could not be loaded, disabling wake word detection: %s",
                    exc,
                )

    def start(self):
        if self.model is None:
            logger.info("Wake word listener not started because no model is available.")
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
        if self.model is None:
            return
        import sounddevice as sd
        import numpy as np
        samplerate = 16000
        blocksize = 512
        with sd.InputStream(channels=1, samplerate=samplerate, blocksize=blocksize, dtype="float32") as stream:
            while not self._stop.is_set():
                audio_block, _ = stream.read(blocksize)
                y = audio_block[:,0].astype(float)
                # Skicka till openwakeword (BERÄKNING)
                scores = self.model.predict(y)
                # 'scores' är dict över kända nyckelord i modellen. Vi approximerar här:
                # Om någon score går över tröskeln och vi hittar textlikhet mot vår fraslista → trigga.
                # (För robusthet kan man byta till en tränad modell för "hej kompis".)
                max_score = 0.0
                for _, s in scores.items():
                    max_score = max(max_score, float(s))
                if max_score >= self.detection_threshold:
                    # Debounce
                    self.on_detect()
                    time.sleep(2.0)  # undvik retrigger direkt


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
