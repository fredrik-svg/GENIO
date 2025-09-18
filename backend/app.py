
import asyncio, os, subprocess, base64, logging
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import HOST, PORT, SAMPLE_RATE, PLAY_CMD, OUTPUT_WAV_PATH
from .audio import record_until_silence, save_wav_mono16
from .openai_client import stt_transcribe_wav, chat_reply_sv, tts_speak_sv
from .wakeword import WakeWordListener

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

async def full_converse_flow(trigger: str = "touch") -> dict:
    await notify(f"status: Lyssnar ({trigger}) ...")
    audio = record_until_silence()
    if audio.size == 0:
        await notify("status: Hörde inget – försök igen.")
        return {"ok": False, "error": "no_audio"}

    await notify("status: Transkriberar ...")
    import io
    buf = io.BytesIO()
    save_wav_mono16(buf, audio)  # tweak save_wav to accept file-like
    # fix: rewrite save_wav to accept path only; so write to tmp
    # Workaround: write to /tmp/audio.wav
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    save_wav_mono16(tmp_path, audio)
    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    os.unlink(tmp_path)

    text = await stt_transcribe_wav(wav_bytes, language="sv")
    await notify(f"du: {text}")

    await notify("status: Frågar OpenAI ...")
    reply = await chat_reply_sv(text)
    await notify(f"assistent: {reply}")

    await notify("status: Skapar tal ...")
    wav_reply = await tts_speak_sv(reply)

    # Spara och spela upp
    out_path = OUTPUT_WAV_PATH
    with open(out_path,"wb") as f:
        f.write(wav_reply)
    # play (blocking)
    try:
        subprocess.run(PLAY_CMD.split() + [out_path], check=False)
    except Exception:
        pass

    return {"ok": True, "question": text, "answer": reply}

@app.post("/api/converse")
async def converse():
    data = await full_converse_flow(trigger="touch")
    return JSONResponse(data)

# Start wakeword on startup
ww_listener = None
@app.on_event("startup")
async def startup_event():
    async def on_detect():
        await notify("status: Wakeword detekterat!")
        try:
            await full_converse_flow(trigger="wakeword")
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
