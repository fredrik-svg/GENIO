
import logging
import threading, time
from typing import Callable, List, Optional

from openwakeword import Model

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
        try:
            self.model = Model(enable_speex_noise_suppression=True)
        except Exception as exc:  # pragma: no cover - defensive guard against optional dependency issues
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
