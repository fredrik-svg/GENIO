import json
import os
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from .config import WAKE_WORD_ENABLED, WAKE_WORDS, WAKE_WORD_TIMEOUT, WAKE_WORD_COOLDOWN


class WakeWordSettings(BaseModel):
    """Model for wake word configuration settings."""

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = Field(
        default=WAKE_WORD_ENABLED,
        description="Enable or disable wake word detection.",
    )
    wake_words: list[str] = Field(
        default_factory=lambda: WAKE_WORDS.copy(),
        alias="wakeWords",
        description="List of wake words to listen for.",
    )
    timeout: float = Field(
        default=WAKE_WORD_TIMEOUT,
        description="Timeout in seconds for wake word detection.",
    )
    cooldown: float = Field(
        default=WAKE_WORD_COOLDOWN,
        description="Cooldown in seconds between wake word detections.",
    )

    def merged_with(self, data: dict[str, Any]) -> "WakeWordSettings":
        """Return a new model with incoming data updating existing values."""
        base = self.model_dump(by_alias=True)
        base.update({k: v for k, v in data.items() if v is not None})
        return WakeWordSettings.model_validate(base)


_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "wake_word_settings.json")


def _ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def load_wake_word_settings() -> WakeWordSettings:
    """Load wake word settings from file or return defaults."""
    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return WakeWordSettings.model_validate(data)
        except Exception:
            # On corrupt file or validation error, fall back to defaults
            return WakeWordSettings()
    return WakeWordSettings()


def save_wake_word_settings(settings: WakeWordSettings) -> None:
    """Save wake word settings to file."""
    _ensure_directory_exists(_SETTINGS_PATH)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings.model_dump(by_alias=True), fh, indent=2, ensure_ascii=False)