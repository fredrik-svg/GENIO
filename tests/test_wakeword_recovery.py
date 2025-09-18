from pathlib import Path

from backend import wakeword


def test_extract_missing_model_path_from_error_message():
    error = ValueError("Could not open '/tmp/fake/models/alexa_v0.1.tflite'")

    result = wakeword._extract_missing_model_path(error)

    assert result == Path("/tmp/fake/models/alexa_v0.1.tflite")


def test_extract_missing_model_path_when_not_present():
    error = RuntimeError("some other failure")

    result = wakeword._extract_missing_model_path(error)

    assert result is None
