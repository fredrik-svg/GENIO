# Bluetooth-högtalare på Raspberry Pi OS (Bookworm)

1) Installera Bluetooth-verktyg (om saknas)
```bash
sudo apt update
sudo apt install -y bluez bluez-tools pulseaudio-utils pipewire-audio wireplumber
sudo reboot
```

2) Para och koppla
```bash
bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
[bluetooth]# scan on     # vänta tills du ser din högtalare
[bluetooth]# pair XX:XX:XX:XX:XX:XX
[bluetooth]# connect XX:XX:XX:XX:XX:XX
[bluetooth]# trust XX:XX:XX:XX:XX:XX
[bluetooth]# exit
```

3) Sätt som standard-utgång
- Via grafiskt gränssnitt (volymikonen) välj din BT-högtalare som output.
- Eller via `pactl`:
```bash
pactl list short sinks
pactl set-default-sink <sink_name>
```

4) Testa
```bash
scripts/test_bt_speaker.sh
```

5) Kör appen så att den använder *paplay* (PipeWire/Pulse) istället för aplay (ALSA)
```bash
export PLAY_CMD="paplay"
uvicorn backend.app:app --host 0.0.0.0 --port 8080
```
