
import asyncio, io, os, subprocess, logging, shlex, socket, uuid
from contextlib import suppress
from ipaddress import ip_address
from typing import Literal

from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, ConfigDict
from fastapi.staticfiles import StaticFiles

from .config import HOST, PORT, SAMPLE_RATE, PLAY_CMD, OUTPUT_WAV_PATH, INPUT_DEVICE, OUTPUT_DEVICE
from .audio import (
    record_until_silence,
    save_wav_mono16,
    play_wav_bytes,
    ensure_wav_pcm16,
    list_input_devices,
    list_output_devices,
)
from .openai_client import stt_transcribe_wav, tts_speak_sv
from .wakeword import WakeWordListener
from .rag.service import ingest_sources as ingest_rag_sources, rag_answer, reset_index as reset_rag_index
from .ui_settings import load_ui_settings, save_ui_settings, UISettings
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

async def full_converse_flow(trigger: str = "touch", *, suspend_wakeword: bool = True) -> dict:
    should_resume = False
    try:
        if suspend_wakeword and ww_listener is not None:
            try:
                should_resume = ww_listener.suspend()
                if not should_resume:
                    logger.debug(
                        "Wake word listener was not running when attempting to suspend before %s trigger.",
                        trigger,
                    )
            except Exception:  # pragma: no cover - defensive guard
                logger.exception("Could not suspend wake word listener before recording")
                should_resume = False

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

        await notify("status: Skapar tal ...")
        wav_reply = await tts_speak_sv(reply)
        wav_reply = ensure_wav_pcm16(wav_reply)

        # Spara och spela upp
        out_path = OUTPUT_WAV_PATH
        with open(out_path, "wb") as f:
            f.write(wav_reply)

        played = False
        play_cmd = (PLAY_CMD or "").strip()
        if play_cmd:
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
    finally:
        if should_resume and ww_listener is not None:
            if not ww_listener.resume():
                logger.warning(
                    "Wake word listener could not be restarted after %s trigger.",
                    trigger,
                )

@app.post("/api/converse")
async def converse():
    data = await full_converse_flow(trigger="touch", suspend_wakeword=True)
    return JSONResponse(data)


class AskPayload(BaseModel):
    question: str


class AudioInputSettingsPayload(BaseModel):
    input_device: str | None = Field(default=None, alias="inputDevice")
    manual_value: str | None = Field(default=None, alias="manualValue")
    output_device: str | None = Field(default=None, alias="outputDevice")
    output_manual_value: str | None = Field(default=None, alias="outputManualValue")


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

# Start wakeword on startup
ww_listener = None
@app.on_event("startup")
async def startup_event():
    async def on_detect():
        await notify("status: Wakeword detekterat!")
        try:
            await full_converse_flow(trigger="wakeword", suspend_wakeword=False)
        except Exception as e:
            await notify(f"fel: {e}")

    def wrap():
        asyncio.run(on_detect())

    global ww_listener
    ww_listener = WakeWordListener(on_detect=wrap, detection_threshold=0.6)
    if not ww_listener.start():
        logging.warning("Wake word listener could not be started; voice activation disabled.")

@app.on_event("shutdown")
def shutdown_event():
    global ww_listener
    if ww_listener:
        ww_listener.stop()
