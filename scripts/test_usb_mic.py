#!/usr/bin/env python3
# Records 3s from the selected INPUT_DEVICE (or default) and prints RMS
import os, time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE","16000"))
INPUT_DEVICE = os.getenv("INPUT_DEVICE","")

duration = 3
print(f"Recording {duration}s at {SAMPLE_RATE} Hz from device='{INPUT_DEVICE or 'default'}' ...")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    device=INPUT_DEVICE if INPUT_DEVICE else None):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

rms = float(np.sqrt(np.mean(np.square(audio[:,0]))))
print(f"RMS energy: {rms:.4f}. If near 0.0, check mic levels / device selection.")
