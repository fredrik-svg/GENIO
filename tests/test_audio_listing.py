
import subprocess, sys

def test_list_audio_script_runs():
    # Script should run and exit 0 even if no devices present.
    r = subprocess.run([sys.executable, "scripts/list_audio.py"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "INPUT DEVICES" in r.stdout
