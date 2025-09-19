#!/usr/bin/env python3
"""Utility script for listing the available audio devices.

The original script unconditionally imported :mod:`sounddevice`, which caused
the program to exit with a ``ModuleNotFoundError`` when the optional
dependency was not installed.  The test-suite (and general usability) expects
the script to fail gracefully in that situation, so we dynamically check for
the module and provide a helpful message instead.
"""

from __future__ import annotations

from importlib import import_module, util


def _load_sounddevice():
    """Return the ``sounddevice`` module if it is available.

    ``sounddevice`` is an optional dependency.  By checking availability via
    :func:`importlib.util.find_spec` we avoid importing it when the package is
    missing, which keeps the script runnable on systems without audio
    hardware or the dependency installed.
    """

    spec = util.find_spec("sounddevice")
    if spec is None:
        return None
    try:
        return import_module("sounddevice")
    except OSError:
        return None


def _print_devices(
    devices,
    *,
    channel_key: str,
    header: str,
    empty_message: str = "(no devices detected)",
) -> None:
    print(header)
    printed_any = False
    for index, device in enumerate(devices):
        if device.get(channel_key, 0) > 0:
            print(f"[{index}] {device['name']}  ({device['hostapi']})")
            printed_any = True
    if not printed_any:
        print(empty_message)


def main() -> int:
    sounddevice = _load_sounddevice()
    if sounddevice is None:
        print("sounddevice is not installed or could not be initialized; unable to list audio devices.")
        devices = []
    else:
        devices = sounddevice.query_devices()

    _print_devices(devices, channel_key="max_input_channels", header="=== INPUT DEVICES (Mics) ===")
    print()
    _print_devices(devices, channel_key="max_output_channels", header="=== OUTPUT DEVICES (Speakers) ===")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
