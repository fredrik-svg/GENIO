import io
from pathlib import Path
from typing import List
from urllib.error import HTTPError

from backend import wakeword


def test_extract_missing_model_path_from_error_message():
    error = ValueError("Could not open '/tmp/fake/models/alexa_v0.1.tflite'")

    result = wakeword._extract_missing_model_path(error)

    assert result == Path("/tmp/fake/models/alexa_v0.1.tflite")


def test_extract_missing_model_path_when_not_present():
    error = RuntimeError("some other failure")

    result = wakeword._extract_missing_model_path(error)

    assert result is None


class _DummyResponse(io.BytesIO):
    def __enter__(self):  # pragma: no cover - simple helper
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple helper
        self.close()


def test_try_recover_missing_model_uses_fallback_url(monkeypatch, tmp_path):
    model_name = "alexa_v0.1.tflite"
    target_path = tmp_path / "models" / model_name
    attempted_urls: List[str] = []

    monkeypatch.setattr(
        wakeword,
        "MODEL_BASE_URLS",
        ("https://primary.example/", "https://fallback.example/"),
    )

    def fake_urlopen(request, timeout=0):
        url = request.full_url if hasattr(request, "full_url") else request
        attempted_urls.append(url)
        if len(attempted_urls) == 1:
            raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return _DummyResponse(b"wakeword")

    monkeypatch.setattr(wakeword.urllib.request, "urlopen", fake_urlopen)

    recovered = wakeword._try_recover_missing_model(target_path)

    assert recovered is True
    assert target_path.exists()
    assert attempted_urls == [
        "https://primary.example/" + model_name,
        "https://fallback.example/" + model_name,
    ]
