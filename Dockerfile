
# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for audio + bluetooth + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 libasound2-dev ffmpeg bluez bluez-tools pulseaudio-utils \
    pipewire-audio wireplumber \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deps first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY backend ./backend
COPY scripts ./scripts
COPY .env.sample ./
COPY pyproject.toml ./pyproject.toml

EXPOSE 8080

# Default env (can be overridden)
ENV HOST=0.0.0.0 PORT=8080 PLAY_CMD="aplay -q"

# For ALSA access inside container you may need to pass --device /dev/snd and group perms.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
