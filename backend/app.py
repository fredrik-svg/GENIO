
import asyncio, io, os, subprocess, logging, shlex
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .config import HOST, PORT, SAMPLE_RATE, PLAY_CMD, OUTPUT_WAV_PATH
from .audio import record_until_silence, save_wav_mono16, play_wav_bytes
from .openai_client import stt_transcribe_wav, tts_speak_sv
from .wakeword import WakeWordListener
from .rag.service import ingest_sources as ingest_rag_sources, rag_answer, reset_index as reset_rag_index

logger = logging.getLogger(__name__)

app = FastAPI(title="Pi5 Swedish Voice Assistant")

static_dir = os.path.join(os.path.dirname(__file__), "static")

# serve frontend
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "index.html"))

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
        await notify(f"du: {text}")

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


class RAGIngestPayload(BaseModel):
    sources: list[str] = Field(default_factory=list, description="Lista av URL:er eller filvägar som ska indexeras.")


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
