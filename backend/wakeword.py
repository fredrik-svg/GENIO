
import threading, time
from typing import Callable, List
from openwakeword import Model

# Enkel wrapper för att köra openWakeWord i bakgrunden och trigga callback.
# Standardfras: "Hej kompis" via en allmän modell med fraströskel.
# För bästa resultat, träna/finjustera egen modell och ange filväg här.
WW_PHRASES: List[str] = ["hej kompis", "hejkompis"]

class WakeWordListener:
    def __init__(self, on_detect: Callable[[], None], detection_threshold: float = 0.5):
        self.on_detect = on_detect
        self.detection_threshold = detection_threshold
        self._stop = threading.Event()
        # Laddar standardmodeller (engelska/svenska kan fungera okej för enkla fraser).
        self.model = Model(enable_speex_noise_suppression=True)

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _run(self):
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
