import json
import os
import re
import subprocess
from typing import Any, Iterable, Tuple

from pydantic import BaseModel, ConfigDict, Field


_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "display_settings.json")
_MONITOR_LINE_RE = re.compile(r"^\s*(\d+):\s+([+*]*)(\S+)")


class DisplaySettings(BaseModel):
    """Persisted inställningar för vilka skärmar som används."""

    model_config = ConfigDict(populate_by_name=True)

    assistant_display: str = Field(default="", alias="assistantDisplay")
    presentation_display: str = Field(default="", alias="displayDisplay")

    def merged_with(self, data: dict[str, Any]) -> "DisplaySettings":
        base = self.model_dump(by_alias=True)
        base.update({k: v for k, v in data.items() if v is not None})
        return DisplaySettings.model_validate(base)


def _ensure_directory_exists(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def normalize_display_target(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    return cleaned


def load_display_settings() -> DisplaySettings:
    if os.path.isfile(_SETTINGS_PATH):
        try:
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return DisplaySettings.model_validate(data)
        except Exception:
            return DisplaySettings()
    return DisplaySettings()


def save_display_settings(settings: DisplaySettings) -> None:
    _ensure_directory_exists(_SETTINGS_PATH)
    with open(_SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(settings.model_dump(by_alias=True), fh, indent=2, ensure_ascii=False)


def _resolve_target(stored: str, fallbacks: Iterable[Tuple[str, str]]) -> tuple[str, str]:
    stored_clean = normalize_display_target(stored)
    if stored_clean:
        return stored_clean, "stored"

    for value, source in fallbacks:
        normalized = normalize_display_target(value)
        if normalized:
            return normalized, source

    return "", "default"


def describe_display_settings() -> dict[str, Any]:
    settings = load_display_settings()
    env_primary = os.getenv("PRIMARY_DISPLAY_TARGET", "")
    env_secondary = os.getenv("SECONDARY_DISPLAY_TARGET", "")
    env_display = os.getenv("DISPLAY", "")
    env_wayland = os.getenv("WAYLAND_DISPLAY", "")

    assistant_effective, assistant_source = _resolve_target(
        settings.assistant_display,
        [
            (env_primary, "env_primary"),
            (env_display, "env_display"),
            (env_wayland, "env_wayland"),
        ],
    )
    display_effective, display_source = _resolve_target(
        settings.presentation_display,
        [
            (env_secondary, "env_secondary"),
            (env_primary, "env_primary"),
            (env_display, "env_display"),
            (env_wayland, "env_wayland"),
        ],
    )

    return {
        "settings": settings.model_dump(by_alias=True),
        "assistant": {
            "stored": normalize_display_target(settings.assistant_display),
            "effective": assistant_effective,
            "effectiveSource": assistant_source,
            "message": _format_display_message(
                role="assistant",
                effective=assistant_effective,
                source=assistant_source,
            ),
        },
        "display": {
            "stored": normalize_display_target(settings.presentation_display),
            "effective": display_effective,
            "effectiveSource": display_source,
            "message": _format_display_message(
                role="display",
                effective=display_effective,
                source=display_source,
            ),
        },
    }


def _format_display_message(role: str, *, effective: str, source: str) -> str:
    noun_lower, noun_upper = ("assistenten", "Assistenten") if role == "assistant" else ("visningsläget", "Visningsläget")

    if not effective:
        return f"Inga specifika displayinställningar – {noun_lower} använder systemets standard."

    if source == "stored":
        return f"{noun_upper} använder sparad skärm: {effective}."

    if source == "env_secondary":
        return f"{noun_upper} följer miljövariabeln SECONDARY_DISPLAY_TARGET ({effective})."

    if source == "env_primary":
        return f"{noun_upper} följer miljövariabeln PRIMARY_DISPLAY_TARGET ({effective})."

    if source == "env_display":
        return f"{noun_upper} använder aktuell DISPLAY: {effective}."

    if source == "env_wayland":
        return f"{noun_upper} använder aktuell WAYLAND_DISPLAY: {effective}."

    return f"{noun_upper} använder skärm: {effective}."


def discover_display_targets() -> tuple[list[dict[str, Any]], list[str]]:
    targets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    warnings: list[str] = []

    def add_target(value: str, label: str, source: str) -> None:
        normalized = normalize_display_target(value)
        if not normalized:
            return
        entry = targets.get(normalized)
        if entry is None:
            entry = {"value": normalized, "label": label or normalized, "sources": [source]}
            targets[normalized] = entry
            order.append(normalized)
        else:
            entry.setdefault("sources", []).append(source)
            if label and (entry["label"] == entry["value"] or source.startswith("env")):
                entry["label"] = label

    env_display = os.getenv("DISPLAY", "")
    env_wayland = os.getenv("WAYLAND_DISPLAY", "")

    if env_display:
        add_target(env_display, f"{env_display} (aktiv DISPLAY)", "env_display")
    if env_wayland:
        add_target(env_wayland, f"{env_wayland} (Wayland)", "env_wayland")

    x11_dir = "/tmp/.X11-unix"
    try:
        for entry in sorted(os.listdir(x11_dir)):
            if not entry.startswith("X"):
                continue
            suffix = entry[1:]
            label = f":{suffix}"
            add_target(f":{suffix}", f"{label} (X11-display)", "x11")
    except FileNotFoundError:
        pass

    def enrich_with_monitors(value: str) -> None:
        entry = targets.get(value)
        if entry is None or not value.startswith(":"):
            return
        env = os.environ.copy()
        env["DISPLAY"] = value
        try:
            output = subprocess.check_output(
                ["xrandr", "--listmonitors"],
                text=True,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError:
            warnings.append("xrandr saknas – kunde inte lista aktiva monitorer.")
            return
        except subprocess.CalledProcessError as exc:
            message = exc.output.strip() if exc.output else str(exc)
            if message:
                warnings.append(message)
            return

        monitors: list[dict[str, Any]] = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("monitors:"):
                continue
            match = _MONITOR_LINE_RE.match(line)
            if not match:
                continue
            index = int(match.group(1))
            flags = match.group(2) or ""
            name = match.group(3)
            monitors.append(
                {
                    "index": index,
                    "name": name,
                    "primary": "*" in flags,
                }
            )
        if monitors:
            entry["monitors"] = monitors
            label_parts: list[str] = []
            for monitor in monitors:
                monitor_name = monitor["name"]
                if monitor.get("primary"):
                    label_parts.append(f"{monitor_name} (primär)")
                else:
                    label_parts.append(monitor_name)
            entry["label"] = f"{value} – {', '.join(label_parts)}"

    for candidate in list(order):
        enrich_with_monitors(candidate)

    result = [targets[value] for value in order]
    for entry in result:
        entry.pop("sources", None)
    return result, warnings


__all__ = [
    "DisplaySettings",
    "discover_display_targets",
    "describe_display_settings",
    "load_display_settings",
    "normalize_display_target",
    "save_display_settings",
]
