import pytest
from backend.wake_word_settings import WakeWordSettings


def test_wake_word_normalization():
    """Test that wake words are normalized to lowercase."""
    settings = WakeWordSettings(
        enabled=True,
        wake_words=["Hej Genio", "GENIO", "hej Assistant"],
        timeout=5.0,
        cooldown=1.0
    )
    
    assert settings.wake_words == ["hej genio", "genio", "hej assistant"]


def test_wake_word_normalization_with_whitespace():
    """Test that wake words have whitespace stripped."""
    settings = WakeWordSettings(
        enabled=True,
        wake_words=["  Hej Genio  ", "GENIO  ", "  hej Assistant"],
        timeout=5.0,
        cooldown=1.0
    )
    
    assert settings.wake_words == ["hej genio", "genio", "hej assistant"]


def test_wake_word_normalization_filters_empty():
    """Test that empty wake words are filtered out."""
    settings = WakeWordSettings(
        enabled=True,
        wake_words=["Hej Genio", "", "  ", "GENIO"],
        timeout=5.0,
        cooldown=1.0
    )
    
    assert settings.wake_words == ["hej genio", "genio"]


def test_wake_word_normalization_via_alias():
    """Test normalization when using alias (wakeWords)."""
    data = {
        "enabled": True,
        "wakeWords": ["Hej Genio", "GENIO"],
        "timeout": 5.0,
        "cooldown": 1.0
    }
    settings = WakeWordSettings.model_validate(data)
    
    assert settings.wake_words == ["hej genio", "genio"]


def test_wake_word_normalization_via_merged_with():
    """Test normalization when using merged_with method."""
    settings = WakeWordSettings()
    merged = settings.merged_with({
        "wakeWords": ["HEJ GENIO", "Genio", "  Hej Assistant  "]
    })
    
    assert merged.wake_words == ["hej genio", "genio", "hej assistant"]


def test_wake_word_case_insensitive_matching():
    """Test that wake word matching is case-insensitive."""
    settings = WakeWordSettings(
        enabled=True,
        wake_words=["Hej Genio", "GENIO"],
        timeout=5.0,
        cooldown=1.0
    )
    
    # All wake words should be normalized to lowercase
    text_lower = "hej genio hur mår du"
    for wake_word in settings.wake_words:
        assert wake_word.lower() == wake_word  # Already lowercase
        assert wake_word in text_lower
