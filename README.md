
# Pi5 Swedish Voice Assistant (Touch + Wakeword)

En lättviktig röstassistent för **Raspberry Pi 5** som pratar **svenska** och använder **OpenAI** för STT (tal→text), chatt och TTS (text→tal).
- **Touch UI**: Enkel webbsida med stor knapp som startar lyssning.
- **Wakeword**: Lokal "Hej kompis" via `openwakeword` (låg resursförbrukning).
- **Naturligt tal**: TTS via OpenAI (t.ex. `gpt-4o-mini-tts`) – låter naturtroget och körs i molnet för att spara Pi-resurser.
- **Minimal belastning** på Pi: STT/TTS och språkmodell körs i molnet.
- **RAG-kunskapsbas**: Indexera webbsidor och dokument lokalt och återanvänd innehållet vid svar.

> Testat på Raspberry Pi OS Bookworm (64-bit). Antag Python 3.11+.

## Snabbstart

1) **Förbered ljud**
- Anslut **USB-mikrofon** och **högtalare**/3.5mm/HDMI-ljud.
- Ställ in standardenheter (se `alsamixer`, `arecord -l`, `aplay -l`).

2) **Hämta projektet**
```bash
# på din Pi
mkdir -p ~/apps && cd ~/apps
# ladda upp zip: pi5-assistant.zip via SCP / webbläsarens nedladdningslänk
unzip pi5-assistant.zip -d pi5-assistant
cd pi5-assistant
```

3) **Installera**
```bash
chmod +x install.sh
./install.sh
```

4) **Miljövariabler**
```bash
cp .env.sample .env
nano .env
# Sätt OPENAI_API_KEY och justera röst/modeller vid behov
```

5) **Kör**
```bash
# manuellt
source .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8080
# öppna i webbläsare på Pi: http://localhost:8080
```

6) **(Valfritt) Kör som tjänst**
```bash
sudo cp systemd/pi5-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi5-assistant
sudo systemctl start pi5-assistant
# status: sudo systemctl status pi5-assistant
```

## Wakeword

- Standardfras: **"Hej kompis"**
- Byt fras i `backend/wakeword.py` (modell `openwakeword` + konfig i `WW_PHRASES`).
- Wakeword körs i en bakgrundstråd och triggar samma flöde som touch-knappen.

## Arkitektur

- **FastAPI backend** (`backend/app.py`): Endpoints för samtal + websockets för status.
- **Mic**: `sounddevice` (alsa) med enkel energibaserad VAD + timeout.
- **STT**: OpenAI Audio Transcriptions (`whisper-1` eller nyare).
- **Chat**: `gpt-4o-mini` (snabbt/billigt) med svensk systemprompt.
- **TTS**: OpenAI TTS (`gpt-4o-mini-tts`) → WAV → uppspelning med `aplay`.
- **RAG**: Egen SQLite-baserad vektorstore + OpenAI embeddings (`text-embedding-3-small`).
- **Wakeword**: `openwakeword` (lokalt, låg CPU).
- **Wakeword-modeller**: `openwakeword` laddar automatiskt hem standardmodellerna vid första körningen (cache i `~/.cache/openwakeword`).
- **Frontend**: Statisk HTML/JS med stor touchknapp, status och text.

## Kunskapsbas (RAG)

Assistenten kan indexera lokala dokument eller webbsidor för att svara baserat på eget material.

1. **Inmatning** – Lägg till källor via webbgränssnittet (fältet "Klistra in URL...") eller REST-endpointen `/api/rag/ingest`.
2. **Extraktion** – HTML och PDF/TXT konverteras till text, normaliseras och delas upp i bitar om cirka 400 ord.
3. **Indexering** – Bitarna lagras i en lokal SQLite-databas (`RAG_DB_PATH`). Embeddings skapas via `EMBEDDING_MODEL`.
4. **Sökning** – Varje fråga gör en semantisk sökning (`RAG_TOP_K` toppträffar). Endast träffar över `RAG_MIN_SCORE` skickas vidare.

> **Obs!** PDF-stöd kräver att Python-paketet `pypdf` installeras. Utan det indexeras endast textbaserade filer.

### API-exempel

```bash
# Lägg till en webbsida och ett lokalt dokument
curl -X POST http://localhost:8080/api/rag/ingest \
  -H 'Content-Type: application/json' \
  -d '{"sources": ["https://exempel.se/faq", "/home/pi/docs/manual.pdf"]}'

# Rensa hela indexet
curl -X POST http://localhost:8080/api/rag/reset
```

När en fråga besvaras returneras `contexts` i HTTP-svaret och visas i webbgränssnittet under rubriken **Källor**. Om ingen relevant information hittas används generellt svar utan RAG.

## Konfig

Se `.env.sample`:
- `OPENAI_API_KEY` – **obligatorisk**
- `CHAT_MODEL=gpt-4o-mini`
- `TTS_MODEL=gpt-4o-mini-tts`
- `TTS_VOICE=alloy`  (alternativ: "verse", "aria" m.fl. beroende på tillgänglighet)
- `STT_MODEL=whisper-1`
- `SAMPLE_RATE=16000`
- `MAX_RECORD_SECONDS=12`
- `SILENCE_DURATION=1.0`
- `ENERGY_THRESHOLD=0.015`
- `EMBEDDING_MODEL=text-embedding-3-small`
- `RAG_ENABLED=1`
- `RAG_DB_PATH=./rag_store`
- `RAG_TOP_K=4`
- `RAG_MIN_SCORE=0.35`
- `RAG_CHUNK_SIZE=400`
- `RAG_CHUNK_OVERLAP=80`

## Kända tips

- Om inget ljud hörs, testa: `aplay /usr/share/sounds/alsa/Front_Center.wav`
- Justera `ENERGY_THRESHOLD` och mikrofonens nivå i `alsamixer`.
- Om wakeword triggar för lätt, höj `DETECTION_THRESHOLD` i `backend/wakeword.py`.

## Licenser och beroenden

Öppen källkod där möjligt (openwakeword, FastAPI). OpenAI är moln-API (kommersiellt). Se `requirements.txt`.


## Installation via Git (GitHub)

1) **Kräver Raspberry Pi OS Bookworm 64-bit** (rekommenderas). Kör `uname -m` → bör visa `aarch64`.
   - Uppdatera systemet: `sudo apt update && sudo apt full-upgrade -y && sudo reboot`

2) **Klona repot**
```bash
mkdir -p ~/apps && cd ~/apps
git clone https://github.com/<ditt-konto>/pi5-assistant.git
cd pi5-assistant
```

3) **Installera beroenden**
```bash
chmod +x install.sh
./install.sh
```

4) **Miljövariabler**
```bash
cp .env.sample .env
nano .env  # sätt OPENAI_API_KEY
```

5) **Starta**
```bash
source .venv/bin/activate
uvicorn backend.app:app --host 0.0.0.0 --port 8080
# Öppna http://localhost:8080 på Pi
```

### Systemd-tjänst (Git-version)

Just nu pekar servicefilen på katalogen `/home/pi/apps/pi5-assistant`. Om du klonade någon annanstans, **uppdatera** `WorkingDirectory` och `ExecStart` i `systemd/pi5-assistant.service`.

```bash
sudo cp systemd/pi5-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pi5-assistant
sudo systemctl start pi5-assistant
```

## Raspberry Pi OS 64-bit?

Ja – projektet är avsett för **Raspberry Pi OS Bookworm 64-bit**. Det fungerar bäst där eftersom några paket
(PortAudio, openwakeword m.fl.) är mest vältestade på 64-bit och vi vill ha bästa prestanda på Pi 5.


## USB-mikrofon och Bluetooth-högtalare

### Välj mikrofon (USB)
Lista enheter:
```bash
source .venv/bin/activate
python scripts/list_audio.py
```
Sätt i `.env`:
```
INPUT_DEVICE="hw:2,0"   # eller exakt namn/index från listan (sounddevice accepterar båda)
```

Snabbtest mic:
```bash
python scripts/test_usb_mic.py
```

### Anslut Bluetooth-högtalare
Följ `scripts/bluetooth.md` för att para och sätta standard-sink.
Testa uppspelning:
```bash
scripts/test_bt_speaker.sh
```
Starta appen med Pulse/PipeWire:
```bash
export PLAY_CMD="paplay"
uvicorn backend.app:app --host 0.0.0.0 --port 8080
```


## Docker (lokalt på Pi eller via GHCR)

### Bygg lokalt på Pi 5 (arm64)
```bash
docker build -t pi5-assistant:local .
# Kör med ALSA (enkelt):
docker run --rm -it \
  --device /dev/snd \
  --group-add audio \
  -p 8080:8080 \
  --env-file .env \
  -e PLAY_CMD="aplay -q" \
  pi5-assistant:local
# Öppna: http://localhost:8080
```

### Använd Bluetooth/PulseAudio från host
1) Se till att din BT-högtalare är standard i hostens Pulse/ PipeWire.
2) Kör containern mot hostens Pulse-socket:
```bash
PULSE_DIR="/run/user/$(id -u)/pulse"
docker run --rm -it \
  --device /dev/snd \
  --group-add audio \
  -p 8080:8080 \
  -v $PULSE_DIR:$PULSE_DIR \
  -e PULSE_SERVER=$PULSE_DIR/native \
  --env-file .env \
  -e PLAY_CMD="paplay" \
  pi5-assistant:local
```

### Dra färdig image från GHCR (efter att du pushat workflow)
```bash
docker pull ghcr.io/<ditt-konto>/pi5-assistant:main
docker run --rm -it \
  --device /dev/snd --group-add audio \
  -p 8080:8080 \
  --env OPENAI_API_KEY=sk-... \
  -e PLAY_CMD="paplay" \
  ghcr.io/<ditt-konto>/pi5-assistant:main
```
