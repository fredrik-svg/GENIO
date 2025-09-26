
import io, os, subprocess, logging, shlex, socket, uuid, asyncio
from contextlib import suppress
from ipaddress import ip_address
from typing import Literal

from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, ConfigDict
from fastapi.staticfiles import StaticFiles

from .config import (
    HOST, PORT, SAMPLE_RATE, PLAY_CMD, OUTPUT_WAV_PATH, INPUT_DEVICE, OUTPUT_DEVICE,
    WAKE_WORD_ENABLED, WAKE_WORDS
)
from .audio import (
    record_until_silence,
    save_wav_mono16,
    play_wav_bytes,
    ensure_wav_pcm16,
    list_input_devices,
    list_output_devices,
)
from .ai import stt_transcribe_wav, tts_speak_sv
from .rag.service import ingest_sources as ingest_rag_sources, rag_answer, reset_index as reset_rag_index
from .ui_settings import load_ui_settings, save_ui_settings, UISettings
from .display_settings import (
    describe_display_settings,
    discover_display_targets,
    load_display_settings,
    normalize_display_target,
    save_display_settings,
)
from .audio_settings import (
    extract_index,
    extract_manual_value,
    get_raw_input_device_selection,
    get_raw_output_device_selection,
    get_selected_input_device,
    get_selected_output_device,
    serialize_device_spec,
    set_selected_input_device,
    set_selected_output_device,
)
from .wake_word import get_wake_word_detector

logger = logging.getLogger(__name__)

app = FastAPI(title="Pi5 Swedish Voice Assistant")

static_dir = os.path.join(os.path.dirname(__file__), "static")
_background_upload_dir = os.path.abspath(os.path.join(static_dir, "uploads", "backgrounds"))
_ALLOWED_IMAGE_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/jpg": ".jpg",
}
_MAX_BACKGROUND_SIZE = 6 * 1024 * 1024  # 6 MB

# serve frontend
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


def _collect_non_loopback_addresses() -> list[str]:
    """Returnera unika IP-adresser som kan nås inom samma nätverk."""

    addresses: list[str] = []
    seen: set[str] = set()

    def add_address(raw: str | None) -> None:
        if not raw:
            return
        candidate = raw.strip()
        if not candidate:
            return
        candidate = candidate.split("%", 1)[0]
        with suppress(ValueError):
            parsed = ip_address(candidate)
            if parsed.is_loopback or parsed.is_unspecified:
                return
            key = parsed.compressed
            if key in seen:
                return
            seen.add(key)
            addresses.append(key)

    try:
        output = subprocess.check_output(
            ["ip", "-o", "addr", "show", "scope", "global"],
            text=True,
        )
    except Exception:  # pragma: no cover - saknar ip-kommandot eller annat fel
        output = ""

    if output:
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ip_with_prefix = parts[3]
                addr = ip_with_prefix.split("/", 1)[0]
                add_address(addr)

    hostnames = {socket.gethostname(), socket.getfqdn()}
    for name in hostnames:
        if not name or name.lower() == "localhost":
            continue
        with suppress(socket.gaierror):
            resolved = socket.gethostbyname_ex(name)[2]
            for addr in resolved:
                add_address(addr)

    with suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            addr = sock.getsockname()[0]
            add_address(addr)

    add_address(HOST)

    return addresses


def _unique(sequence: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in sequence:
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_items.append(item)
    return unique_items


def _format_base_url(host: str, scheme: str, port: int) -> str:
    host_part = host.strip()
    if not host_part:
        host_part = "localhost"
    bare_host = host_part.split("%", 1)[0]
    url_host = bare_host
    with suppress(ValueError):
        parsed = ip_address(bare_host)
        if parsed.version == 6:
            url_host = f"[{parsed.compressed}]"
        else:
            url_host = parsed.compressed
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    port_suffix = "" if default_port else f":{port}"
    return f"{scheme}://{url_host}{port_suffix}"


def _format_device_display(spec: str | int | None, devices: list[dict[str, object]]) -> str:
    if spec is None:
        return "Systemets standard"

    if isinstance(spec, int):
        for device in devices:
            try:
                device_index = int(device.get("index"))  # type: ignore[arg-type]
            except Exception:
                continue
            if device_index == spec:
                name = device.get("name")
                name_str = str(name) if name else f"Enhet {spec}"
                return f"{name_str} (index {spec})"
        return f"Ljudenhet index {spec}"

    return str(spec)


def _audio_devices_payload(kind: Literal["input", "output"]):
    if kind == "input":
        devices, devices_error = list_input_devices()
        stored_raw = get_raw_input_device_selection()
        effective_spec = get_selected_input_device()
        fallback_env = (INPUT_DEVICE or "").strip()
        noun = "ljudkälla"
    else:
        devices, devices_error = list_output_devices()
        stored_raw = get_raw_output_device_selection()
        effective_spec = get_selected_output_device()
        fallback_env = (OUTPUT_DEVICE or "").strip()
        noun = "ljudutgång"

    manual_value = extract_manual_value(stored_raw)
    selected_option = "auto"
    if stored_raw:
        if stored_raw.startswith("manual:"):
            selected_option = "manual"
        else:
            selected_option = stored_raw

    effective_display = _format_device_display(effective_spec, devices)
    effective_serialized = serialize_device_spec(effective_spec)

    if stored_raw:
        effective_source = "stored"
    elif fallback_env:
        effective_source = "env"
    else:
        effective_source = "system"

    selected_available = True
    selected_index = extract_index(stored_raw)
    if selected_index is not None:
        selected_available = any(
            isinstance(device.get("index"), (int, float))
            and int(device.get("index")) == selected_index
            for device in devices
        )
    elif stored_raw and selected_option not in {"auto", "manual"}:
        selected_available = any(stored_raw == device.get("value") for device in devices)

    default_message = f"Ingen specifik {noun} vald – systemets standard används."
    if devices_error:
        message = devices_error
    elif effective_source == "stored" and stored_raw:
        message = f"Aktiv {noun}: {effective_display}."
    elif effective_source == "env" and fallback_env:
        message = f"Aktiv {noun} via miljövariabel: {effective_display}."
    else:
        message = default_message

    if not devices_error and not selected_available and stored_raw:
        message = f"{message.rstrip('.')} (tidigare vald enhet hittades inte)."

    payload: dict[str, object] = {
        "kind": kind,
        "devices": devices,
        "devicesOk": devices_error is None,
        "selected": stored_raw,
        "selectedOption": selected_option,
        "selectedAvailable": selected_available,
        "manualValue": manual_value,
        "effective": effective_serialized,
        "effectiveDisplay": effective_display,
        "effectiveSource": effective_source,
        "fallback": fallback_env,
        "message": message,
    }

    if devices_error:
        payload["devicesError"] = devices_error

    return payload, devices_error

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/display", response_class=HTMLResponse)
async def display_page() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "display.html"))


@app.get("/admin", response_class=HTMLResponse)
async def admin_page() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "admin.html"))


@app.get("/api/network/lan-info")
async def lan_info(request: Request) -> JSONResponse:
    scheme = request.url.scheme or "http"
    request_port = request.url.port
    port = PORT or request_port or (443 if scheme == "https" else 80)

    discovered_addresses = _collect_non_loopback_addresses()
    request_host = request.url.hostname or ""

    hostname_candidates = [request_host, socket.gethostname(), socket.getfqdn()]
    extra_hostnames: list[str] = []
    for name in hostname_candidates:
        if not name:
            continue
        clean = name.strip()
        if not clean or clean.lower() == "localhost":
            continue
        extra_hostnames.append(clean)
        if "." not in clean:
            extra_hostnames.append(f"{clean}.local")

    base_hosts = _unique(discovered_addresses + extra_hostnames + ["localhost"])
    base_urls = [_format_base_url(host, scheme, port) for host in base_hosts]

    if not base_urls:
        base_urls = [_format_base_url("localhost", scheme, port)]

    admin_urls = [f"{url}/admin" for url in base_urls]
    display_urls = [f"{url}/display" for url in base_urls]

    note = (
        "Öppna länkarna från en annan dator, platta eller mobil på samma nätverk."
        if discovered_addresses
        else "Inga nätverksadresser hittades automatiskt – använd adressen du redan anslutit med eller kontrollera nätverket."
    )

    return JSONResponse(
        {
            "ok": True,
            "scheme": scheme,
            "port": port,
            "lanAddresses": discovered_addresses,
            "baseUrls": base_urls,
            "adminUrls": admin_urls,
            "displayUrls": display_urls,
            "note": note,
        }
    )


@app.get("/api/display/targets")
async def get_display_targets() -> JSONResponse:
    targets, warnings = discover_display_targets()
    note = (
        "Välj skärm från listan eller ange manuellt värde (t.ex. :0)."
        if targets
        else "Inga skärmar hittades automatiskt – ange värdet manuellt (t.ex. :0)."
    )

    payload: dict[str, object] = {"ok": True, "targets": targets, "note": note}
    if warnings:
        payload["warnings"] = warnings
    return JSONResponse(payload)


@app.get("/api/display/settings")
async def get_display_settings() -> JSONResponse:
    description = describe_display_settings()
    description["ok"] = True
    return JSONResponse(description)


@app.get("/rag")
async def rag_page_redirect():
    return RedirectResponse(url="/admin", status_code=307)

# simple status broadcast (optional)
clients = set()

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # keep alive
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        clients.remove(websocket)

async def notify(msg: str):
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        clients.discard(d)

async def full_converse_flow(trigger: str = "touch") -> dict:
    await notify(f"status: Lyssnar ({trigger}) ...")
    audio = record_until_silence()
    if audio.size == 0:
        await notify("status: Hörde inget – försök igen.")
        return {"ok": False, "error": "no_audio"}

    await notify("status: Transkriberar ...")
    buf = io.BytesIO()
    save_wav_mono16(buf, audio)
    wav_bytes = buf.getvalue()

    text = await stt_transcribe_wav(wav_bytes, language="sv")
    text = (text or "").strip()
    await notify(f"du: {text}")

    if not text:
        await notify("status: Hörde inget – försök igen.")
        return {"ok": False, "error": "empty_transcription"}

    await notify("status: Söker i kunskapsbas ...")
    rag = await rag_answer(text)
    if rag.used_rag:
        await notify("status: Hittade relevanta källor.")
    else:
        await notify("status: Inga träffar – använder generell kunskap.")
    await notify(f"context: {rag.context_payload()}")
    reply = rag.answer
    await notify(f"assistent: {reply}")

    # Optimering: Starta TTS och filoperationer parallellt
    await notify("status: Skapar tal ...")
    
    # Kör TTS och förbered filhantering parallellt
    async def generate_audio():
        wav_reply = await tts_speak_sv(reply)
        return ensure_wav_pcm16(wav_reply)
    
    async def setup_playback():
        # Förbereder playback-konfiguration medan TTS körs
        prefer_sounddevice = get_selected_output_device() is not None
        play_cmd = (PLAY_CMD or "").strip()
        return prefer_sounddevice, play_cmd
    
    # Kör båda operationerna parallellt
    wav_reply, (prefer_sounddevice, play_cmd) = await asyncio.gather(
        generate_audio(),
        setup_playback()
    )

    # Spara och spela upp
    out_path = OUTPUT_WAV_PATH
    with open(out_path, "wb") as f:
        f.write(wav_reply)

    played = False
    if play_cmd and not prefer_sounddevice:
        try:
            subprocess.run(shlex.split(play_cmd) + [out_path], check=True)
            played = True
        except FileNotFoundError:
            logger.warning(
                "Playback command '%s' was not found; falling back to sounddevice.",
                play_cmd,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Playback command '%s' failed with exit code %s; falling back to sounddevice.",
                play_cmd,
                exc.returncode,
            )
        except Exception as exc:
            logger.error("Playback command error (%s); falling back to sounddevice.", exc)

    if not played:
        try:
            play_wav_bytes(wav_reply)
            played = True
        except Exception as exc:
            logger.error("Could not play audio with sounddevice: %s", exc)

    if not played:
        await notify("status: Kunde inte spela upp ljudet.")

    return {
        "ok": True,
        "question": text,
        "answer": reply,
        "used_rag": rag.used_rag,
        "contexts": rag.contexts_for_client(),
    }

@app.post("/api/converse")
async def converse():
    data = await full_converse_flow(trigger="touch")
    return JSONResponse(data)


@app.post("/api/wake-word/start")
async def start_wake_word():
    """Start wake word detection."""
    detector = get_wake_word_detector()
    
    async def on_wake_word():
        """Callback when wake word is detected."""
        await notify("status: Wake word detected!")
        data = await full_converse_flow(trigger="wake-word")
        # Optionally restart listening after conversation
        if detector.is_listening:
            asyncio.create_task(detector.start_listening(on_wake_word))
    
    try:
        await detector.start_listening(on_wake_word)
        await notify("status: Wake word detection started")
        return JSONResponse({"ok": True, "message": "Wake word detection started"})
    except Exception as e:
        logger.error("Failed to start wake word detection: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/wake-word/stop")
async def stop_wake_word():
    """Stop wake word detection."""
    detector = get_wake_word_detector()
    try:
        await detector.stop_listening()
        await notify("status: Wake word detection stopped")
        return JSONResponse({"ok": True, "message": "Wake word detection stopped"})
    except Exception as e:
        logger.error("Failed to stop wake word detection: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/wake-word/status")
async def wake_word_status():
    """Get wake word detection status."""
    detector = get_wake_word_detector()
    
    return JSONResponse({
        "enabled": WAKE_WORD_ENABLED,
        "wake_words": WAKE_WORDS,
        "is_listening": detector.is_listening,
    })


class AskPayload(BaseModel):
    question: str


class AudioInputSettingsPayload(BaseModel):
    input_device: str | None = Field(default=None, alias="inputDevice")
    manual_value: str | None = Field(default=None, alias="manualValue")
    output_device: str | None = Field(default=None, alias="outputDevice")
    output_manual_value: str | None = Field(default=None, alias="outputManualValue")


class DisplaySettingsPayload(BaseModel):
    assistant_choice: str | None = Field(default=None, alias="assistantChoice")
    assistant_manual: str | None = Field(default=None, alias="assistantManual")
    display_choice: str | None = Field(default=None, alias="displayChoice")
    display_manual: str | None = Field(default=None, alias="displayManual")


@app.post("/api/ask")
async def ask(payload: AskPayload):
    question = (payload.question or "").strip()
    if not question:
        return JSONResponse({"ok": False, "error": "Ingen fråga angavs."}, status_code=400)

    await notify("status: Söker i kunskapsbas ...")
    await notify(f"du: {question}")

    rag = await rag_answer(question)
    if rag.used_rag:
        await notify("status: Hittade relevanta källor.")
    else:
        await notify("status: Inga träffar – använder generell kunskap.")
    await notify(f"context: {rag.context_payload()}")
    reply = rag.answer

    await notify(f"assistent: {reply}")
    await notify("status: Svar klart.")

    return JSONResponse(
        {
            "ok": True,
            "question": question,
            "answer": reply,
            "used_rag": rag.used_rag,
            "contexts": rag.contexts_for_client(),
        }
    )


@app.get("/api/audio/input-devices")
async def get_audio_input_devices():
    payload, error = _audio_devices_payload("input")
    devices_ok = bool(payload.get("devicesOk"))
    payload["ok"] = devices_ok
    if error and "error" not in payload:
        payload["error"] = error
    status_code = 200 if devices_ok else 503
    return JSONResponse(payload, status_code=status_code)


@app.get("/api/audio/output-devices")
async def get_audio_output_devices():
    payload, error = _audio_devices_payload("output")
    devices_ok = bool(payload.get("devicesOk"))
    payload["ok"] = devices_ok
    if error and "error" not in payload:
        payload["error"] = error
    status_code = 200 if devices_ok else 503
    return JSONResponse(payload, status_code=status_code)


@app.post("/api/audio/settings")
async def update_audio_settings(payload: AudioInputSettingsPayload):
    def _prepare_selection(choice: str | None, manual_value: str | None) -> str | None:
        provided = choice is not None or manual_value is not None
        if not provided:
            return None

        raw_choice = (choice or "").strip()
        manual = (manual_value or "").strip()

        if raw_choice == "manual":
            raw_choice = f"manual:{manual}"
        elif raw_choice.startswith("manual:") and manual:
            raw_choice = f"manual:{manual}"
        elif not raw_choice and manual:
            raw_choice = f"manual:{manual}"

        return raw_choice

    input_selection = _prepare_selection(payload.input_device, payload.manual_value)
    if input_selection is not None:
        try:
            set_selected_input_device(input_selection)
        except ValueError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc), "field": "input"}, status_code=400
            )

    output_selection = _prepare_selection(
        payload.output_device, payload.output_manual_value
    )
    if output_selection is not None:
        try:
            set_selected_output_device(output_selection)
        except ValueError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc), "field": "output"}, status_code=400
            )

    input_payload, input_error = _audio_devices_payload("input")
    input_payload["ok"] = bool(input_payload.get("devicesOk"))
    if input_error and "error" not in input_payload:
        input_payload["error"] = input_error

    output_payload, output_error = _audio_devices_payload("output")
    output_payload["ok"] = bool(output_payload.get("devicesOk"))
    if output_error and "error" not in output_payload:
        output_payload["error"] = output_error

    return JSONResponse({"ok": True, "input": input_payload, "output": output_payload})


@app.post("/api/display/settings")
async def update_display_settings(payload: DisplaySettingsPayload) -> JSONResponse:
    def _prepare_display_selection(choice: str | None, manual: str | None, *, field: str) -> str | None:
        if choice is None and manual is None:
            return None

        raw_choice = (choice or "").strip()
        manual_value = normalize_display_target(manual)
        lowered = raw_choice.lower()

        if lowered in {"", "auto"}:
            return manual_value if (not raw_choice and manual_value) else ""

        if lowered == "manual":
            if not manual_value:
                raise ValueError(f"Ange ett värde för {field}.")
            return manual_value

        return normalize_display_target(raw_choice)

    try:
        assistant_value = _prepare_display_selection(
            payload.assistant_choice,
            payload.assistant_manual,
            field="assistentens skärm",
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc), "field": "assistant"}, status_code=400)

    try:
        display_value = _prepare_display_selection(
            payload.display_choice,
            payload.display_manual,
            field="visningslägets skärm",
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc), "field": "display"}, status_code=400)

    settings = load_display_settings()
    changed = False

    if assistant_value is not None and settings.assistant_display != assistant_value:
        settings.assistant_display = assistant_value
        changed = True

    if display_value is not None and settings.presentation_display != display_value:
        settings.presentation_display = display_value
        changed = True

    if changed:
        save_display_settings(settings)

    description = describe_display_settings()
    description["ok"] = True
    return JSONResponse(description)


class RAGIngestPayload(BaseModel):
    sources: list[str] = Field(default_factory=list, description="Lista av URL:er eller filvägar som ska indexeras.")


class UISettingsUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    assistant_name: str | None = Field(default=None, alias="assistantName")
    background_image: str | None = Field(default=None, alias="backgroundImage")
    font_family: str | None = Field(default=None, alias="fontFamily")
    primary_button_color: str | None = Field(default=None, alias="primaryButtonColor")
    secondary_button_color: str | None = Field(default=None, alias="secondaryButtonColor")
    button_text_color: str | None = Field(default=None, alias="buttonTextColor")


@app.get("/api/ui-settings")
async def get_ui_settings():
    settings = load_ui_settings()
    return JSONResponse({"ok": True, "settings": settings.model_dump(by_alias=True)})


@app.post("/api/ui-settings")
async def update_ui_settings(payload: UISettingsUpdate):
    current = load_ui_settings()
    updated = current.merged_with(payload.model_dump(exclude_unset=True, by_alias=True))
    save_ui_settings(updated)
    return JSONResponse({"ok": True, "settings": updated.model_dump(by_alias=True)})


@app.post("/api/ui-settings/background-image")
async def upload_background_image(file: UploadFile = File(...)):
    if not file or not file.filename:
        return JSONResponse({"ok": False, "error": "Ingen fil skickades."}, status_code=400)

    content_type = (file.content_type or "").lower()
    extension = _ALLOWED_IMAGE_TYPES.get(content_type)
    if not extension:
        allowed_extensions = ", ".join(
            sorted({ext.lstrip(".").upper() for ext in _ALLOWED_IMAGE_TYPES.values()})
        )
        return JSONResponse(
            {
                "ok": False,
                "error": f"Ogiltig filtyp. Tillåtna format: {allowed_extensions}.",
            },
            status_code=400,
        )

    raw_data = await file.read()
    if not raw_data:
        return JSONResponse({"ok": False, "error": "Filen var tom."}, status_code=400)

    if len(raw_data) > _MAX_BACKGROUND_SIZE:
        size_mb = _MAX_BACKGROUND_SIZE // (1024 * 1024)
        return JSONResponse(
            {
                "ok": False,
                "error": f"Filen är för stor. Maxstorlek är {size_mb} MB.",
            },
            status_code=400,
        )

    os.makedirs(_background_upload_dir, exist_ok=True)

    filename = f"background-{uuid.uuid4().hex}{extension}"
    destination_path = os.path.abspath(os.path.join(_background_upload_dir, filename))
    with open(destination_path, "wb") as out:
        out.write(raw_data)

    await file.close()

    public_path = f"/static/uploads/backgrounds/{filename}"

    current = load_ui_settings()
    previous = current.background_image or ""
    current.background_image = public_path
    save_ui_settings(current)

    if previous.startswith("/static/"):
        relative = previous[len("/static/"):]
        previous_path = os.path.abspath(os.path.join(static_dir, relative))
        try:
            if os.path.isfile(previous_path) and os.path.commonpath(
                [previous_path, _background_upload_dir]
            ) == _background_upload_dir:
                os.remove(previous_path)
        except Exception:
            logger.warning("Kunde inte ta bort tidigare bakgrundsbild: %s", previous_path)

    return JSONResponse({"ok": True, "backgroundImage": public_path})


@app.post("/api/rag/ingest")
async def rag_ingest(payload: RAGIngestPayload):
    if not payload.sources:
        return JSONResponse({"ok": False, "error": "Minst en källa krävs."}, status_code=400)
    try:
        result = await ingest_rag_sources(payload.sources)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:  # pragma: no cover - oväntat fel
        logging.exception("RAG-ingestering misslyckades")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    await notify("status: Kunskapsbas uppdaterad.")
    return JSONResponse({"ok": True, **result})


@app.post("/api/rag/reset")
async def rag_reset():
    reset_rag_index()
    await notify("status: Kunskapsbas rensad.")
    return JSONResponse({"ok": True})

