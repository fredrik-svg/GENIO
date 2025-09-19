import json
import os
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

_DEFAULT_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Arial'


class UISettings(BaseModel):
    """Model som beskriver anpassningsbara utseendeinställningar."""

    model_config = ConfigDict(populate_by_name=True)

    assistant_name: str = Field(
        default="Pi5 Röstassistent",
        alias="assistantName",
        description="Visningsnamn för assistenten.",
    )
    background_image: str = Field(
        default="",
        alias="backgroundImage",
        description="URL till bakgrundsbild.",
    )
    font_family: str = Field(
        default=_DEFAULT_FONT,
        alias="fontFamily",
        description="CSS-font stack.",
    )
    primary_button_color: str = Field(
        default="#1f2a44",
        alias="primaryButtonColor",
        description="Bakgrundsfärg för huvudknappar.",
    )
    secondary_button_color: str = Field(
        default="#13233d",
        alias="secondaryButtonColor",
        description="Bakgrundsfärg för sekundära knappar.",
    )
    button_text_color: str = Field(
        default="#ffffff",
        alias="buttonTextColor",
        description="Textfärg för knappar.",
    )

    def merged_with(self, data: dict[str, Any]) -> "UISettings":
        """Returnera en ny modell där inkommande data uppdaterar befintliga värden."""

        base = self.model_dump(by_alias=True)
        base.update({k: v for k, v in data.items() if v is not None})
        return UISettings.model_validate(base)


_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "ui_settings.json")


def _ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def load_ui_settings() -> UISettings:
    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return UISettings.model_validate(data)
        except Exception:
            # Vid korrupt fil eller valideringsfel återgå till standardvärden
            return UISettings()
    return UISettings()


def save_ui_settings(settings: UISettings) -> None:
    _ensure_directory_exists(_SETTINGS_PATH)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings.model_dump(by_alias=True), fh, indent=2, ensure_ascii=False)
