#!/usr/bin/env python3
import sounddevice as sd
print("=== INPUT DEVICES (Mics) ===")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"[{i}] {dev['name']}  ({dev['hostapi']})")
print("\n=== OUTPUT DEVICES (Speakers) ===")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_output_channels'] > 0:
        print(f"[{i}] {dev['name']}  ({dev['hostapi']})")
