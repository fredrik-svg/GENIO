import json
import os
import re
import subprocess
from typing import Any, Iterable, Tuple

from pydantic import BaseModel, ConfigDict, Field


_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "display_settings.json")
_MONITOR_LINE_RE = re.compile(r"^\s*(\d+):\s+([+*]*)(\S+)")
_MONITOR_RESOLUTION_RE = re.compile(
    r"(?P<width>\d+)(?:/\d+)?x(?P<height>\d+)(?:/\d+)?",
)


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
    runtime_dir = os.getenv("XDG_RUNTIME_DIR", "")

    if env_display:
        add_target(env_display, f"{env_display} (aktiv DISPLAY)", "env_display")
    if env_wayland:
        add_target(env_wayland, f"{env_wayland} (Wayland)", "env_wayland")

    if runtime_dir:
        try:
            for entry in sorted(os.listdir(runtime_dir)):
                if entry.startswith("wayland-"):
                    add_target(entry, f"{entry} (Wayland)", "wayland_runtime")
        except FileNotFoundError:
            pass

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

    x11_cache: dict[str, list[dict[str, Any]] | None] = {}
    wayland_cache: dict[str, list[dict[str, Any]] | None] = {}

    def _format_refresh(value: Any) -> str | None:
        if isinstance(value, (int, float)):
            refresh = float(value)
            if refresh <= 0:
                return None
            if refresh > 1000:
                refresh = refresh / 1000.0
            return f"{refresh:.1f} Hz"
        return None

    def _format_wayland_label(value: str, monitors: list[dict[str, Any]]) -> str:
        label_parts: list[str] = []
        for monitor in monitors:
            name = monitor.get("name") or monitor.get("description") or monitor.get("model")
            name = str(name) if name else "Wayland-utgång"
            details: list[str] = []
            maker = " ".join(
                part
                for part in (
                    monitor.get("make") if isinstance(monitor.get("make"), str) else None,
                    monitor.get("model") if isinstance(monitor.get("model"), str) else None,
                )
                if part
            )
            if maker and maker not in name:
                details.append(maker)
            width = monitor.get("width")
            height = monitor.get("height")
            if isinstance(width, int) and isinstance(height, int):
                details.append(f"{width}x{height}")
            refresh_label = _format_refresh(monitor.get("refresh")) or _format_refresh(
                monitor.get("refreshHz")
            )
            if refresh_label:
                details.append(refresh_label)
            scale = monitor.get("scale")
            if isinstance(scale, (int, float)):
                details.append(f"skala {scale}")
            description = monitor.get("description")
            if isinstance(description, str) and description and description not in details and description != name:
                details.append(description)
            if details:
                label_parts.append(f"{name} ({', '.join(details)})")
            else:
                label_parts.append(name)
        if label_parts:
            return f"{value} – {', '.join(label_parts)}"
        return value

    def _discover_x11_monitors(value: str) -> list[dict[str, Any]] | None:
        cached = x11_cache.get(value)
        if cached is not None:
            return cached
        if not value.startswith(":"):
            x11_cache[value] = []
            return []
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
            x11_cache[value] = []
            return []
        except subprocess.CalledProcessError as exc:
            message = exc.output.strip() if exc.output else str(exc)
            if message:
                warnings.append(message)
            x11_cache[value] = []
            return []

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
            rest = line[match.end() :].strip()
            width: int | None = None
            height: int | None = None
            if rest:
                resolution_match = _MONITOR_RESOLUTION_RE.search(rest)
                if resolution_match:
                    try:
                        width = int(resolution_match.group("width"))
                        height = int(resolution_match.group("height"))
                    except (TypeError, ValueError):
                        width = None
                        height = None
            monitors.append(
                {
                    "index": index,
                    "name": name,
                    "primary": "*" in flags,
                    "width": width,
                    "height": height,
                }
            )
        x11_cache[value] = monitors
        return monitors

    def _discover_wayland_monitors(value: str, sources: list[str]) -> list[dict[str, Any]] | None:
        cached = wayland_cache.get(value)
        if cached is not None:
            return cached
        if not any("wayland" in source for source in sources):
            wayland_cache[value] = []
            return []
        env = os.environ.copy()
        if value:
            env["WAYLAND_DISPLAY"] = value
        try:
            output = subprocess.check_output(
                ["wayland-info", "--json"],
                text=True,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError:
            warnings.append("wayland-info saknas – kunde inte lista Wayland-utgångar.")
            wayland_cache[value] = []
            return []
        except subprocess.CalledProcessError as exc:
            message = exc.output.strip() if exc.output else str(exc)
            if message:
                warnings.append(message)
            wayland_cache[value] = []
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            warnings.append("wayland-info gav ogiltig JSON – kunde inte tolka skärminformation.")
            wayland_cache[value] = []
            return []

        globals_sources: list[list[dict[str, Any]]] = []

        def _collect_globals(candidate: Any) -> None:
            if isinstance(candidate, list):
                globals_sources.append([item for item in candidate if isinstance(item, dict)])

        _collect_globals(data.get("globals"))
        registry_info = data.get("registry")
        if isinstance(registry_info, dict):
            _collect_globals(registry_info.get("globals"))

        if not globals_sources:
            wayland_cache[value] = []
            return []

        monitors: list[dict[str, Any]] = []
        for globals_info in globals_sources:
            for index, item in enumerate(globals_info):
                if not isinstance(item, dict):
                    continue
                if item.get("interface") != "wl_output":
                    continue
                output_info = item.get("output")
                if not isinstance(output_info, dict):
                    continue
                name = output_info.get("name")
                description = output_info.get("description") if isinstance(output_info.get("description"), str) else None
                make = output_info.get("make") if isinstance(output_info.get("make"), str) else None
                model = output_info.get("model") if isinstance(output_info.get("model"), str) else None
                scale = output_info.get("scale")
                modes = output_info.get("modes")
                current_mode: dict[str, Any] | None = None
                if isinstance(modes, list):
                    for mode in modes:
                        if not isinstance(mode, dict):
                            continue
                        flags = mode.get("flags")
                        if isinstance(flags, list) and "current" in flags:
                            current_mode = mode
                            break
                    if current_mode is None and modes:
                        first_mode = modes[0]
                        if isinstance(first_mode, dict):
                            current_mode = first_mode

                width = current_mode.get("width") if isinstance(current_mode, dict) else None
                height = current_mode.get("height") if isinstance(current_mode, dict) else None
                refresh = current_mode.get("refresh") if isinstance(current_mode, dict) else None
                refresh_hz = None
                if isinstance(refresh, (int, float)):
                    refresh_hz = float(refresh)
                    if refresh_hz > 1000:
                        refresh_hz = refresh_hz / 1000.0

                monitors.append(
                    {
                        "index": index,
                        "name": name if isinstance(name, str) else None,
                        "description": description,
                        "make": make,
                        "model": model,
                        "scale": scale,
                        "width": width if isinstance(width, int) else None,
                        "height": height if isinstance(height, int) else None,
                        "refresh": refresh,
                        "refreshHz": refresh_hz,
                        "active": current_mode is not None,
                    }
                )

        wayland_cache[value] = monitors
        return monitors

    for candidate in list(order):
        entry = targets.get(candidate)
        if not entry:
            continue
        sources = [s for s in entry.get("sources", []) if isinstance(s, str)]
        monitors: list[dict[str, Any]] = []
        x11_monitors = _discover_x11_monitors(candidate)
        if x11_monitors:
            monitors = x11_monitors
            names: list[str] = []
            for monitor in x11_monitors:
                monitor_name = monitor.get("name")
                if not monitor_name:
                    continue
                if monitor.get("primary"):
                    names.append(f"{monitor_name} (primär)")
                else:
                    names.append(str(monitor_name))
            if names:
                entry["label"] = f"{candidate} – {', '.join(names)}"
            
            # Add separate entries for each X11 monitor using .screen notation
            if len(x11_monitors) > 1 and candidate.startswith(":"):
                for monitor in x11_monitors:
                    monitor_index = monitor.get("index")
                    monitor_name = monitor.get("name")
                    if monitor_index is None or not monitor_name:
                        continue
                    
                    # Create display target like :0.0, :0.1, etc.
                    screen_value = f"{candidate}.{monitor_index}"
                    screen_label_parts = [monitor_name]
                    
                    width = monitor.get("width")
                    height = monitor.get("height")
                    if isinstance(width, int) and isinstance(height, int):
                        screen_label_parts.append(f"{width}x{height}")
                    
                    if monitor.get("primary"):
                        screen_label_parts.append("primär")
                    
                    screen_label = f"{screen_value} – {', '.join(screen_label_parts)}"
                    add_target(screen_value, screen_label, "x11_screen")
        else:
            wayland_monitors = _discover_wayland_monitors(candidate, sources)
            if wayland_monitors:
                monitors = wayland_monitors
                entry["label"] = _format_wayland_label(candidate, wayland_monitors)
        if monitors:
            entry["monitors"] = monitors

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
