import json
import os

from pydantic import BaseModel, Field

from .config import INPUT_DEVICE


_INDEX_PREFIX = "index:"
_MANUAL_PREFIX = "manual:"
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "audio_settings.json")


class AudioSettings(BaseModel):
    """Persisted ljudinställningar för backend."""

    input_device: str = Field(default="", alias="inputDevice")


def _ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def load_audio_settings() -> AudioSettings:
    """Hämta sparade ljudinställningar."""

    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return AudioSettings.model_validate(data)
        except Exception:
            # Skadad fil eller ogiltig struktur – fall tillbaka till standardvärden.
            return AudioSettings()
    return AudioSettings()


def save_audio_settings(settings: AudioSettings) -> None:
    """Skriv ljudinställningar till disk."""

    _ensure_directory_exists(_SETTINGS_PATH)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings.model_dump(by_alias=True), fh, indent=2, ensure_ascii=False)


def normalize_input_device_selection(raw: str | None) -> str:
    """Normalisera användarens val till lagringsformat."""

    if raw is None:
        return ""

    value = str(raw).strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered in {"auto", "default"}:
        return ""

    if value.startswith(_INDEX_PREFIX):
        rest = value[len(_INDEX_PREFIX) :].strip()
        if not rest:
            raise ValueError("Index för ljudkälla saknas.")
        try:
            index = int(rest)
        except ValueError as exc:  # pragma: no cover - validering
            raise ValueError("Ogiltigt index för ljudkälla.") from exc
        if index < 0:
            raise ValueError("Index för ljudkälla måste vara 0 eller större.")
        return f"{_INDEX_PREFIX}{index}"

    if value.startswith(_MANUAL_PREFIX):
        rest = value[len(_MANUAL_PREFIX) :].strip()
        if not rest:
            raise ValueError("Ingen manuell ljudkälla angavs.")
        return f"{_MANUAL_PREFIX}{rest}"

    if value.isdigit():
        return f"{_INDEX_PREFIX}{int(value)}"

    return value


def set_selected_input_device(selection: str | None) -> AudioSettings:
    """Spara valt ljudingångsvärde."""

    normalized = normalize_input_device_selection(selection)
    settings = load_audio_settings()
    settings.input_device = normalized
    save_audio_settings(settings)
    return settings


def _decode_input_device(value: str | None) -> str | int | None:
    if not value:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    lowered = stripped.lower()
    if lowered in {"auto", "default"}:
        return None

    if stripped.startswith(_INDEX_PREFIX):
        rest = stripped[len(_INDEX_PREFIX) :].strip()
        try:
            return int(rest)
        except ValueError:
            return None

    if stripped.startswith(_MANUAL_PREFIX):
        rest = stripped[len(_MANUAL_PREFIX) :].strip()
        return rest or None

    if stripped.isdigit():
        try:
            return int(stripped)
        except ValueError:
            return None

    return stripped


def get_selected_input_device() -> str | int | None:
    """Returnera det faktiska värdet som ska användas av ljudinspelningen."""

    stored = _decode_input_device(load_audio_settings().input_device)
    if stored is not None:
        return stored

    fallback = _decode_input_device(INPUT_DEVICE)
    if fallback is not None:
        return fallback

    return None


def get_raw_input_device_selection() -> str:
    """Returnera sparat råvärde (kan vara tom sträng)."""

    return load_audio_settings().input_device.strip()


def extract_manual_value(selection: str | None) -> str:
    """Plocka ut manuellt värde från lagrad sträng."""

    if not selection:
        return ""
    selection = selection.strip()
    if selection.startswith(_MANUAL_PREFIX):
        return selection[len(_MANUAL_PREFIX) :].strip()
    return ""


def serialize_device_spec(spec: str | int | None) -> str:
    """Konvertera ett värde till strängrepresentation för API-svar."""

    if spec is None:
        return ""
    if isinstance(spec, int):
        return f"index:{spec}"
    return str(spec)


def extract_index(selection: str | None) -> int | None:
    """Returnera index om lagrad sträng syftar på en sifferidentitet."""

    decoded = _decode_input_device(selection)
    if isinstance(decoded, int):
        return decoded
    return None


__all__ = [
    "AudioSettings",
    "extract_index",
    "extract_manual_value",
    "get_raw_input_device_selection",
    "get_selected_input_device",
    "load_audio_settings",
    "normalize_input_device_selection",
    "save_audio_settings",
    "serialize_device_spec",
    "set_selected_input_device",
]
