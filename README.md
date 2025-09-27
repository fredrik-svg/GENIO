
# Pi5 Swedish Voice Assistant

En lättviktig röstassistent för **Raspberry Pi 5** som pratar **svenska** och använder **OpenAI Voice API** för STT (tal→text), chatt och TTS (text→tal).
- **Touch UI**: Enkel webbsida med stor knapp som startar lyssning.
- **Naturligt tal**: TTS via vald provider (standard `gpt-4o-mini-tts` på OpenAI Voice API) – låter naturtroget och körs i molnet för att spara Pi-resurser.
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
# Sätt OPENAI_API_KEY eller justera AI_PROVIDER vid behov
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

7) **(Valfritt) Genväg i kioskläge**
```bash
# gör skriptet körbart (om det inte redan är det)
chmod +x scripts/pi5-assistant-kiosk.sh

# kopiera .desktop-filen till din användare (Pi OS använder /home/pi)
install -Dm644 desktop/pi5-assistant-kiosk.desktop \
  ~/.local/share/applications/pi5-assistant-kiosk.desktop

# uppdatera genvägen om projektet ligger på en annan sökväg
sed -i "s|/home/pi/apps/pi5-assistant|$PWD|" \
  ~/.local/share/applications/pi5-assistant-kiosk.desktop
```

> Genvägen startar backend om den inte redan kör och öppnar sedan Chromium i helskärmsläge på `http://localhost:8080`.

#### Två skärmar (kiosk + visning)

Kioskskriptet kan även öppna **två** Chromium-fönster: huvudassistenten och visningsläget (`/display`).

- Standardläget (`SECONDARY_DISPLAY_MODE=auto`) försöker upptäcka minst två skärmar via `xrandr` innan ett extra fönster startas.
- Sätt `SECONDARY_DISPLAY_MODE=always` om du vill tvinga två fönster även utan automatisk upptäckt.
- Välj X11-skärm (t.ex. `:0.0` och `:0.1`) med `PRIMARY_DISPLAY_TARGET` och `SECONDARY_DISPLAY_TARGET` om du använder separata display-heads.
- Varje Chromium-instans använder en egen profilkatalog under `.chromium-profiles/` (kan ändras med `CHROMIUM_PROFILE_BASE`).
- Extra argument kan skickas via `PRIMARY_BROWSER_EXTRA_ARGS` respektive `SECONDARY_BROWSER_EXTRA_ARGS` (blankstegsseparerade flaggor).

Exempel – starta två fönster där `/display` hamnar på skärm `:0.1`:

```bash
SECONDARY_DISPLAY_MODE=always \
PRIMARY_DISPLAY_TARGET=:0.0 \
SECONDARY_DISPLAY_TARGET=:0.1 \
SECONDARY_BROWSER_EXTRA_ARGS="--window-position=0,0" \
~/apps/pi5-assistant/scripts/pi5-assistant-kiosk.sh
```

## Arkitektur

- **FastAPI backend** (`backend/app.py`): Endpoints för samtal + websockets för status.
- **Mic**: `sounddevice` (alsa) med enkel energibaserad VAD + timeout.
- **AI-provider**: `backend/ai.py` laddar OpenAI som standard men kan bytas ut via `AI_PROVIDER`.
- **STT**: Standard `gpt-4o-mini-transcribe` via OpenAI Voice API (kan justeras via miljövariabel).
- **Chat**: Standard `gpt-4o-mini` med svensk systemprompt (går att ersätta via provider-konfiguration).
- **TTS**: Standard `gpt-4o-mini-tts` → WAV → uppspelning med `aplay`.
- **RAG**: Egen SQLite-baserad vektorstore + embeddings från vald provider (default OpenAI `text-embedding-3-small`).
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
- `STT_MODEL=gpt-4o-mini-transcribe`
- `SAMPLE_RATE=16000`
- `FALLBACK_SAMPLE_RATES=48000,44100,32000,24000,22050,16000,11025,8000`
- `MAX_RECORD_SECONDS=12`
- `SILENCE_DURATION=1.0`
- `ENERGY_THRESHOLD=0.015`
- `EMBEDDING_MODEL=text-embedding-3-small`
- `RAG_ENABLED=1`
- `RAG_DB_PATH=./rag_store`
- `RAG_TOP_K=3`
- `RAG_MIN_SCORE=0.4`
- `RAG_CHUNK_SIZE=400`
- `RAG_CHUNK_OVERLAP=80`
- `AI_PROVIDER=openai` (alias eller modulväg, t.ex. `backend.ai:EchoProvider`)
- `AI_PROVIDER_CONFIG={}` (JSON-objekt med extra inställningar till vald provider)

### Wake word-funktionalitet

Systemet stöder kontinuerlig wake word-detektering:

- `WAKE_WORD_ENABLED=0` – Aktivera/avaktivera wake word-detektering
- `WAKE_WORDS=hej genio,genio,hej assistant` – Kommaseparerad lista över wake words (svenska)
- `WAKE_WORD_TIMEOUT=5.0` – Sekunder att lyssna för wake word-detektering
- `WAKE_WORD_COOLDOWN=1.0` – Paus mellan wake word-detekteringar

API-endpoints för wake word:
- `POST /api/wake-word/start` – Starta wake word-lyssning
- `POST /api/wake-word/stop` – Stoppa wake word-lyssning  
- `GET /api/wake-word/status` – Kontrollera wake word-status

När wake word detekteras startas automatiskt en konversation.

### Byt AI-leverantör

`backend/ai.py` hanterar alla samtal mot språk-/talmodeller. Standardaliaset är `openai`, men du kan:
- Ange ett eget modulnamn + klass i `AI_PROVIDER` (t.ex. `my_package.provider:CustomProvider`).
- Välja `backend.ai:EchoProvider` för ett enkelt offline-läge eller som exempel på egen implementation.
- Skicka extra inställningar som JSON via `AI_PROVIDER_CONFIG`, exempelvis `{"base_url": "https://...", "api_key": "..."}`.

Alla endpoints (`/api/converse`, RAG och TTS) använder samma provider, så ett byte slår igenom i hela backend.

### Prestandaoptimering

För att minska responstiden från mikrofon-aktivering till svar kan följande parametrar justeras:

**Ljudinspelning:**
- `SILENCE_DURATION=1.0` – Tid att vänta på tystnad innan inspelning avslutas (standard: 1.0s)
- `AUDIO_BLOCKSIZE=512` – Mindre blockstorlek ger lägre latens men mer CPU-användning
- `USE_WEBRTC_VAD=1` – Använd WebRTC Voice Activity Detection för bättre röstigenkänning
- `ENERGY_THRESHOLD=0.015` – Känslighetsnivå för ljuddetektering

**RAG-sökning:**
- `RAG_TOP_K=3` – Färre sökresultat = snabbare bearbetning (standard: 3)  
- `RAG_MIN_SCORE=0.4` – Högre tröskel = färre men mer relevanta träffar

Systemet använder också parallell bearbetning där TTS-generering körs samtidigt som andra operationer för maximal prestanda.

## Kända tips

- Om inget ljud hörs, testa: `aplay /usr/share/sounds/alsa/Front_Center.wav`
- Justera `ENERGY_THRESHOLD` och mikrofonens nivå i `alsamixer`.

## Licenser och beroenden

Öppen källkod där möjligt (FastAPI m.fl.). OpenAI är moln-API (kommersiellt). Se `requirements.txt`.


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
nano .env  # sätt OPENAI_API_KEY eller justera AI_PROVIDER vid behov
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

Ja – projektet är avsett för **Raspberry Pi OS Bookworm 64-bit**. Det fungerar bäst där eftersom flera paket
(PortAudio m.fl.) är mest vältestade på 64-bit och vi vill ha bästa prestanda på Pi 5.


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

### Installera Docker på Raspberry Pi OS

På en ny installation av **Raspberry Pi OS Bookworm 64-bit** installeras Docker enklast via det officiella
installationsscriptet:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

Efter installationen, lägg till din användare (t.ex. `pi`) i `docker`-gruppen för att slippa köra allt som `sudo` och
aktivera tjänsten:

```bash
sudo usermod -aG docker $USER
newgrp docker
sudo systemctl enable docker
sudo systemctl start docker
```

Kontrollera att allt fungerar:

```bash
docker run --rm hello-world
```

Om kommandot inte hittar `hello-world` första gången laddas bilden automatiskt ned.

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
